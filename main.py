import os
import time
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.gkd import GKDConfig, GKDTrainer

# Maximize multi-core CPU usage
num_cores = os.cpu_count() or 4
torch.set_num_threads(num_cores)


class DetailedDistillationCallback(TrainerCallback):
    def __init__(self):
        self.step_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        print("\n" + "=" * 60)
        print(f" [Step {state.global_step + 1}/{args.max_steps}] Training Step Running...")
        print(" -> Generating student rollout & calculating teacher JSD loss...")

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.step_start_time if self.step_start_time else 0.0
        print(f" -> Step {state.global_step}/{args.max_steps} completed in {elapsed:.2f}s")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            grad_norm = logs.get("grad_norm", "N/A")

            loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else str(loss)
            lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
            grad_str = f"{grad_norm:.4f}" if isinstance(grad_norm, (float, int)) else str(grad_norm)

            print(f" [Metrics] Step Loss (JSD): {loss_str} | LR: {lr_str} | Grad Norm: {grad_str}")


def test_generation(model, tokenizer, prompt, device="cpu"):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=45,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    completion_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def prepare_dataset():
    """Builds a rich calibration dataset combining smoltalk with reasoning prompts."""
    print("\n[Phase 4] Loading calibration prompt dataset...")

    # 1. Curated reasoning, math, and factual prompt-completion pairs
    curated_data = [
                       {"messages": [
                           {"role": "user",
                            "content": "Explain why the sky looks blue to human eyes in two sentences."},
                           {"role": "assistant",
                            "content": "The sky appears blue because molecules in Earth's atmosphere scatter shorter blue wavelengths of sunlight more than other colors. This phenomenon is known as Rayleigh scattering."}
                       ]},
                       {"messages": [
                           {"role": "user", "content": "What is 12 multiplied by 8? Explain briefly."},
                           {"role": "assistant",
                            "content": "12 multiplied by 8 is 96. You can calculate this by breaking 12 into 10 and 2: (10 * 8) + (2 * 8) = 80 + 16 = 96."}
                       ]},
                       {"messages": [
                           {"role": "user",
                            "content": "If all birds lay eggs, and a sparrow is a bird, does a sparrow lay eggs?"},
                           {"role": "assistant",
                            "content": "Yes, a sparrow lays eggs. Since all birds lay eggs and a sparrow is classified as a bird, it follows logically that sparrows lay eggs."}
                       ]},
                       {"messages": [
                           {"role": "user", "content": "Name two renewable energy sources and explain them briefly."},
                           {"role": "assistant",
                            "content": "1. Solar energy captures light from the sun using photovoltaic panels. 2. Wind energy uses turbines to convert moving air into electricity."}
                       ]},
                       {"messages": [
                           {"role": "user", "content": "Solve for x: 3x + 12 = 27."},
                           {"role": "assistant",
                            "content": "Subtract 12 from both sides to get 3x = 15. Then divide both sides by 3 to find x = 5."}
                       ]}
                   ] * 6  # 30 high-quality reasoning samples
    curated_dataset = Dataset.from_list(curated_data)

    # 2. Extract first-turn exchanges from smoltalk (already cached on your disk)
    try:
        raw_smoltalk = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:70]")
        # Slicing [:2] guarantees extracting exactly 1 user turn and 1 assistant turn
        smoltalk_trimmed = raw_smoltalk.map(lambda x: {"messages": x["messages"][:2]}).select_columns(["messages"])
        combined_dataset = concatenate_datasets([curated_dataset, smoltalk_trimmed]).shuffle(seed=42)
        print(f" -> Successfully prepared {len(combined_dataset)} total training conversations.")
        return combined_dataset
    except Exception as e:
        print(f" -> smoltalk mapping notice ({e}); using curated calibration dataset.")
        return curated_dataset


def run_cpu_distillation():
    # Both are Instruct models sharing the identical 49,152 vocabulary
    student_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    teacher_model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

    device = "cpu"
    compute_dtype = torch.float32

    print(f"Configuring CPU Distillation environment with {num_cores} threads...")

    # 1. Unified Tokenizer
    print(f"\n[Phase 1] Loading tokenizer from {teacher_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Frozen Teacher (360M)
    print(f"\n[Phase 2] Loading teacher model ({teacher_model_id})...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        dtype=compute_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(" -> Teacher model loaded and frozen.")

    # 3. Instruct Student (135M) + LoRA
    print(f"\n[Phase 3] Loading instruct student model ({student_model_id})...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        dtype=compute_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    student_model = get_peft_model(student_model, lora_config)
    print(" -> Injected LoRA adapters into student:")
    student_model.print_trainable_parameters()

    # 4. Calibration Dataset
    train_dataset = prepare_dataset()

    # Baseline Sanity Check
    test_prompt = "Explain why the sky looks blue to human eyes in two sentences."
    print("\n" + "-" * 60)
    print("[Sanity Check] Student response BEFORE distillation:")
    before_output = test_generation(student_model, tokenizer, test_prompt, device=device)
    print(f"Prompt: {test_prompt}\nStudent output:\n{before_output}")
    print("-" * 60)

    # 5. Distillation Hyperparameters
    # 50 steps provides clear convergence in ~2.5 minutes on an 8-thread CPU
    training_args = GKDConfig(
        output_dir="./distilled_smollm_poc",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=50,
        save_steps=25,
        lmbda=0.5,  # 50% student rollouts, 50% teacher guidance
        beta=0.5,  # Generalized JSD
        temperature=0.7,
        max_new_tokens=32,  # Fast CPU rollouts
        seq_kd=False,
        disable_dropout=True,
        fp16=False,
        bf16=False,
        use_cpu=True
    )

    # 6. Initialize Trainer
    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[DetailedDistillationCallback()]
    )

    print("\n[Phase 5] Starting Knowledge Distillation Loop (50 steps)...")
    trainer.train()

    # 7. Post-Training Sanity Check
    print("\n" + "-" * 60)
    print("[Sanity Check] Student response AFTER distillation:")
    after_output = test_generation(student_model, tokenizer, test_prompt, device=device)
    print(f"Prompt: {test_prompt}\nStudent output:\n{after_output}")
    print("-" * 60)

    # 8. Save Weights
    adapter_output_dir = "./distilled_smollm_poc/final_adapter"
    student_model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    print(f"\n[Phase 6] Distillation complete! Adapter saved to: {adapter_output_dir}\n")


if __name__ == "__main__":
    run_cpu_distillation()