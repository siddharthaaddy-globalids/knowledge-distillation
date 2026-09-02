import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models():
    # Matches the model architecture used during training
    student_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    teacher_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "./distilled_smollm_scaled/final_adapter"

    print("Loading shared tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Original Student (135M-Instruct before distillation)
    print("\n1. Loading Original Student (135M-Instruct)...")
    base_student = AutoModelForCausalLM.from_pretrained(
        student_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to("cpu")
    base_student.eval()

    # 2. Distilled Student (135M-Instruct + Trained LoRA Adapter)
    # NOTE: PeftModel.from_pretrained injects LoRA layers into the model object it is
    # handed. A *second, independent* base instance is loaded here so that `base_student`
    # above stays a true pre-distillation baseline instead of aliasing the distilled one.
    print("2. Loading Distilled Student (135M-Instruct + Distilled LoRA)...")
    distilled_base = AutoModelForCausalLM.from_pretrained(
        student_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to("cpu")
    distilled_student = PeftModel.from_pretrained(
        distilled_base,
        adapter_path
    ).to("cpu")
    distilled_student.eval()

    # 3. Teacher Model (360M-Instruct reference)
    print("3. Loading Teacher Model (360M-Instruct)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to("cpu")
    teacher.eval()

    return tokenizer, base_student, distilled_student, teacher


def generate_response(model, tokenizer, prompt, max_new_tokens=60):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15
        )
    # Strip the input prompt from the output tokens
    completion_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def run_test():
    tokenizer, base_student, distilled_student, teacher = load_models()

    test_queries = [
        "Explain why the sky looks blue to human eyes in two sentences.",
        "What is 12 multiplied by 8? Explain briefly.",
        "Name two renewable energy sources and explain them briefly.",
        "Solve for x: 3x + 12 = 27."
    ]

    print("\n" + "=" * 70)
    print(" 3-WAY BENCHMARK: ORIGINAL STUDENT vs. DISTILLED STUDENT vs. TEACHER")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Prompt {i}]: {query}\n" + "-" * 70)

        # 1. Original Student (pre-distillation baseline)
        base_ans = generate_response(base_student, tokenizer, query)
        print(f"[Original 135M-Instruct]:\n{base_ans}\n")

        # 2. Distilled Student
        distilled_ans = generate_response(distilled_student, tokenizer, query)
        print(f"[Distilled 135M + Adapter]:\n{distilled_ans}\n")

        # 3. Teacher
        teacher_ans = generate_response(teacher, tokenizer, query)
        print(f"[Teacher 360M-Instruct]:\n{teacher_ans}\n")

    # Interactive prompt mode
    print("=" * 70)
    print("Interactive Evaluation Mode (Type 'exit' or 'quit' to stop):")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter prompt > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            output = generate_response(distilled_student, tokenizer, user_input)
            print(f"\nDistilled Model Response:\n{output}")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_test()