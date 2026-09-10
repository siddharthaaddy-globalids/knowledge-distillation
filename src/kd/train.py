"""
On-policy knowledge distillation (GKD): a large teacher into a small LoRA student.

Fully config-driven. Models, dataset, LoRA shape, schedule and hardware all come
from a YAML file, so retargeting to a different teacher, student or dataset never
requires a code edit:

    kd train --config configs/finance.yaml
    kd train --config configs/finance.yaml --set training.max_steps=600

What the run does, in order:
  1. tokenizer - student and teacher must share a vocabulary for standard GKD
  2. dataset   - kd.data assembles the balanced split and holds out a validation set
  3. teacher   - loaded frozen, then pre-flighted (kd.teacher) before anything is
                 spent on training, because GKD trains the student to match the
                 teacher's distribution and a broken teacher yields a broken student
  4. student   - LoRA injected across the configured projections
  5. training  - live console telemetry: step time, running JSD loss, grad norm, LR
  6. samples   - benchmark generations at intervals, so quality drift is visible
  7. save      - adapter and tokenizer into the run bundle

Everything this produces lands in the run directory (kd.runlog), so a run is a
single self-contained thing to inspect, archive or upload.
"""

import os
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from .data import build_datasets
from .teacher import load_teacher, verify_teacher

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
from trl.experimental.gkd import GKDConfig, GKDTrainer  # noqa: E402


def resolve_tokenizer_source(config):
    """models.tokenizer is 'teacher', 'student', or an explicit hub id / path."""
    choice = str(config["models"].get("tokenizer", "teacher"))
    return {
        "teacher": config["models"]["teacher"],
        "student": config["models"]["student"],
    }.get(choice, choice)


# --------------------------------------------------------------------------- #
# Generation helper
# --------------------------------------------------------------------------- #
def generate_sample(model, tokenizer, prompt, device, max_new_tokens=512):
    """Sampled generation, used for the periodic quality probes during training.

    Two things here exist because of models that open with a reasoning block.

    The rendering comes from kd.teacher.render_prompt, which positions the
    prompt so the next token is an ANSWER rather than `<think>`. Without it a
    Qwen3-family student spends the whole sample budget deliberating and the
    probe shows you none of the thing it was printed to show.

    And the default is 512 rather than 48. Forty-eight was enough when a sample
    began at the answer; against a model that thinks first it is not enough to
    escape the preamble, so every probe in the log ended mid-sentence and told
    you nothing about whether the student was learning the target format.
    """
    from .teacher import render_prompt

    inputs = tokenizer(render_prompt(tokenizer, prompt), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def _fmt(value, spec=".4f"):
    if isinstance(value, (int, float)):
        try:
            return format(float(value), spec)
        except (ValueError, OverflowError):
            return str(value)
    return str(value)


def _fmt_eta(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
class TelemetryCallback(TrainerCallback):
    """Console telemetry: step time, running JSD loss, grad norm, LR, periodic samples.

    Also mirrors every step into the run's events.jsonl, which is what makes a run
    readable by anything other than a person - a cost graph, a CI gate, a dashboard.
    """

    def __init__(self, model, tokenizer, device, benchmark_prompts,
                 eval_every=100, window=20, run=None, sample_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.benchmark_prompts = list(benchmark_prompts or [])
        self.eval_every = eval_every
        self.window = window
        self.run = run
        self.sample_tokens = int(sample_tokens)
        self.step_start = None
        self.run_start = None
        self.losses = []
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.run_start = time.time()
        print("\n" + "=" * 78)
        print(f" Distillation started - {args.max_steps} steps, "
              f"effective batch {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print("=" * 78)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is not None:
            self.step_times.append(time.time() - self.step_start)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        loss = logs.get("loss")
        if isinstance(loss, (int, float)):
            self.losses.append(float(loss))
        recent = self.losses[-self.window:]
        running = sum(recent) / len(recent) if recent else float("nan")

        lr = logs.get("learning_rate", float("nan"))
        grad_norm = logs.get("grad_norm", float("nan"))
        step_time = self.step_times[-1] if self.step_times else float("nan")

        avg_step = (sum(self.step_times) / len(self.step_times)) if self.step_times else 0.0
        remaining = max(0, args.max_steps - state.global_step) * avg_step

        print(
            f" [step {state.global_step:>4}/{args.max_steps}] "
            f"jsd={_fmt(loss)} run{self.window}={_fmt(running)} "
            f"grad={_fmt(grad_norm)} lr={_fmt(lr, '.2e')} "
            f"{step_time:5.1f}s/step  eta={_fmt_eta(remaining)}"
        )
        if self.run:
            self.run.event("train", "step", step=state.global_step,
                           max_steps=args.max_steps, loss=loss, running_loss=running,
                           grad_norm=grad_norm, learning_rate=lr,
                           step_seconds=round(step_time, 3),
                           eta_seconds=round(remaining, 1))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f" [eval @ step {state.global_step}] "
                  f"held-out loss = {metrics['eval_loss']:.4f}")
            if self.run:
                self.run.event("train", "eval", step=state.global_step,
                               eval_loss=metrics["eval_loss"])

    def on_save(self, args, state, control, **kwargs):
        print(f" [checkpoint] saved at step {state.global_step} -> {args.output_dir}")
        if self.run:
            self.run.event("train", "checkpoint", step=state.global_step,
                           path=args.output_dir)

    @property
    def average_step_seconds(self):
        return (sum(self.step_times) / len(self.step_times)) if self.step_times else 0.0

    def maybe_sample(self, args, state):
        """Generate on the benchmark prompts to make quality drift visible."""
        if not self.benchmark_prompts:
            return
        print("\n" + "-" * 78)
        print(f" [benchmark @ step {state.global_step}] student generations")
        print("-" * 78)
        was_training = self.model.training
        self.model.eval()
        try:
            for prompt in self.benchmark_prompts:
                text = generate_sample(self.model, self.tokenizer, prompt,
                                       self.device,
                                       max_new_tokens=self.sample_tokens)
                print(f"  Q: {prompt}\n  A: {text}\n")
        except Exception as exc:
            print(f"  !! benchmark generation failed: {exc}")
        finally:
            if was_training:
                self.model.train()
        print("-" * 78 + "\n")


class BenchmarkCallback(TrainerCallback):
    """Runs the benchmark prompts every `eval_every` steps and once at the end."""

    def __init__(self, parent, eval_every=100):
        self.parent = parent
        self.eval_every = eval_every

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_every and state.global_step > 0 and state.global_step % self.eval_every == 0:
            self.parent.maybe_sample(args, state)

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - (self.parent.run_start or time.time())
        print(f"\n Training wall clock: {_fmt_eta(total)}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(config, hardware, run, dry_run=False, allow_bad_teacher=False,
          callbacks=None, checkpoints=None):
    """Distil the teacher into a LoRA student. Returns a summary dict.

    config      resolved configuration mapping
    hardware    the resolved device/dtype block from kd.config.resolve_device
    run         a kd.runlog.Run - owns the output directory and the event stream
    dry_run     2 steps and no final save; validates the pipeline end to end
    callbacks   extra TrainerCallbacks, e.g. the pipeline's limit enforcement
    checkpoints where to write resumable checkpoints; defaults to the run bundle's
                checkpoints/. The smoke stage points this elsewhere so its throwaway
                two steps do not sit alongside the real run's.
    """
    training_cfg = config["training"]
    gkd_cfg = config["gkd"]
    lora_cfg = config["lora"]

    device = hardware["device"]
    dtype = hardware["dtype"]
    seed = int(config["project"]["seed"])
    benchmark_prompts = list(config.get("benchmark_prompts") or [])
    sample_tokens = int(training_cfg.get("benchmark_max_new_tokens") or 512)

    teacher_id = config["models"]["teacher"]
    student_id = config["models"]["student"]
    teacher_adapter = config["models"].get("teacher_adapter") or None
    tokenizer_source = resolve_tokenizer_source(config)

    max_steps = 2 if dry_run else int(training_cfg["max_steps"])
    eval_every = 1 if dry_run else int(training_cfg["benchmark_every"])
    save_steps = 1 if dry_run else int(training_cfg["save_steps"])
    use_eval = bool(training_cfg["eval_enabled"])
    if dry_run:
        print(" MODE: DRY RUN (2 steps, no final save)")

    # 1. Tokenizer. Student and teacher must share a vocabulary for standard GKD.
    print(f"\n[Phase 1] Loading tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f" -> vocab size {len(tokenizer)}")

    # 2. Dataset
    train_dataset, eval_dataset = build_datasets(tokenizer, config)
    if dry_run:
        # Keep the validation pass cheap while still exercising the eval code path.
        eval_dataset = eval_dataset.select(range(min(4, len(eval_dataset))))
    run.event("train", "dataset", train_rows=len(train_dataset),
              eval_rows=len(eval_dataset))

    # 3. Teacher (frozen)
    print(f"\n[Phase 2] Loading frozen teacher ({teacher_id})...")
    teacher_model, loading_info = load_teacher(
        teacher_id, teacher_adapter, dtype=dtype, device=device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(" -> teacher frozen")
    probe = benchmark_prompts[0] if benchmark_prompts else "Explain compound interest."
    verify_teacher(teacher_model, tokenizer, device, probe, loading_info,
                   strict=not allow_bad_teacher)

    # 4. Student + LoRA
    print(f"\n[Phase 3] Loading student ({student_id}) and injecting LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_id, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)

    # target_modules may be a list of suffixes (Llama-style models) or a single regex
    # string. The regex form matters for hybrid architectures such as Qwen3.5, whose
    # checkpoint also contains a vision tower and an MTP head that share module names
    # with the language model - a plain suffix list would inject adapters into those too.
    targets = lora_cfg["target_modules"]
    targets = targets if isinstance(targets, str) else list(targets)
    lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=targets,
    )
    if lora_cfg.get("exclude_modules"):
        excludes = lora_cfg["exclude_modules"]
        lora_kwargs["exclude_modules"] = excludes if isinstance(excludes, str) else list(excludes)
    student_model = get_peft_model(student_model, LoraConfig(**lora_kwargs))
    student_model.print_trainable_parameters()
    trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    run.event("train", "student_ready", trainable_params=trainable)

    # 5. Baseline sample before training
    print("\n[Phase 4] Student output BEFORE distillation:")
    for prompt in benchmark_prompts[:1]:
        print(f"  Q: {prompt}\n  A: "
              f"{generate_sample(student_model, tokenizer, prompt, device, sample_tokens)}\n")

    # 6. Training configuration
    # NOTE: transformers 5.x removed `warmup_ratio`; `warmup_steps` accepts a float in
    # [0, 1) and is interpreted as a ratio of total steps (see
    # TrainingArguments.get_warmup_steps).
    config_kwargs = dict(
        output_dir=checkpoints or run.checkpoint_dir,
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        warmup_steps=training_cfg["warmup"],
        max_steps=max_steps,
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=int(training_cfg["save_total_limit"]),
        max_grad_norm=float(training_cfg["max_grad_norm"]),
        seed=seed,
        lmbda=float(gkd_cfg["lmbda"]),
        beta=float(gkd_cfg["beta"]),
        temperature=float(gkd_cfg["temperature"]),
        max_new_tokens=int(gkd_cfg["max_new_tokens"]),
        max_length=int(config["dataset"]["max_total_tokens"]),
        seq_kd=bool(gkd_cfg["seq_kd"]),
        disable_dropout=True,
        # Device and precision flags come from the resolved hardware layer, so the same
        # config runs unchanged on CPU, Apple Silicon MPS and CUDA.
        fp16=hardware["fp16"],
        bf16=hardware["bf16"],
        use_cpu=hardware["use_cpu"],
        dataloader_pin_memory=hardware["pin_memory"],
        report_to=[],
        dataloader_num_workers=0,
    )
    if use_eval:
        config_kwargs.update(
            eval_strategy="steps",
            eval_steps=save_steps,
            per_device_eval_batch_size=int(training_cfg["batch_size"]),
        )

    training_args = GKDConfig(**config_kwargs)

    # 7. Trainer
    telemetry = TelemetryCallback(student_model, tokenizer, device, benchmark_prompts,
                                  eval_every=eval_every, run=run,
                                  sample_tokens=sample_tokens)
    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        processing_class=tokenizer,
        callbacks=[telemetry, BenchmarkCallback(telemetry, eval_every=eval_every)]
                  + list(callbacks or []),
    )

    print(f"\n[Phase 5] Starting GKD training for {max_steps} steps "
          f"(cosine schedule, warmup {training_args.warmup_steps}, "
          f"lr {training_args.learning_rate})...")
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    # 8. Post-training samples
    print("\n[Phase 6] Student output AFTER distillation:")
    for prompt in benchmark_prompts:
        print(f"  Q: {prompt}\n  A: "
              f"{generate_sample(student_model, tokenizer, prompt, device, sample_tokens)}\n")

    summary = {
        "steps_completed": int(trainer.state.global_step),
        "steps_requested": max_steps,
        "train_seconds": round(elapsed, 1),
        "seconds_per_step": round(telemetry.average_step_seconds, 3),
        "trainable_params": trainable,
        "final_loss": float(result.training_loss) if result else None,
        "dry_run": dry_run,
    }

    # 9. Save
    if dry_run:
        print("[Phase 7] Dry run - skipping final adapter save.")
        summary["adapter"] = None
        return summary

    os.makedirs(run.adapter_dir, exist_ok=True)
    student_model.save_pretrained(run.adapter_dir)
    tokenizer.save_pretrained(run.adapter_dir)
    print(f"\n[Phase 7] Complete. Adapter + tokenizer saved to: {run.adapter_dir}\n")
    summary["adapter"] = run.adapter_dir
    return summary
