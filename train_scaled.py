"""
Scaled on-policy knowledge distillation: SmolLM2-360M-Instruct -> SmolLM2-135M-Instruct.

Upgrades the 50-step PoC in main.py to a production-shaped run:
  * 1,800+ multi-domain calibration trajectories (smoltalk + synthetic reasoning)
  * a 50-prompt held-out validation split for generalization tracking
  * higher-capacity LoRA (r=32, alpha=64) across all attention + MLP projections
  * 300 optimizer steps, cosine LR schedule with warmup, effective batch size 4
  * live console logging of step time / running JSD loss / grad norm / LR
  * benchmark generations every 100 steps so quality drift is visible during training

CPU-only, float32. Usage:
    python train_scaled.py                 # full run
    python train_scaled.py --dry-run       # dataset build + 2 steps, no checkpointing
    python train_scaled.py --max-steps 50  # short run
"""

import argparse
import os
import random
import time

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
from trl.experimental.gkd import GKDConfig, GKDTrainer  # noqa: E402

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
NUM_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CORES)

STUDENT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEACHER_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_DIR = "./distilled_smollm_scaled"
ADAPTER_DIR = f"{OUTPUT_DIR}/final_adapter"

DEVICE = "cpu"
COMPUTE_DTYPE = torch.float32
SEED = 42

# Prompt-length budget. The collator builds the prompt from messages[:-1]; keeping it
# short is what keeps on-policy CPU rollouts affordable.
MAX_PROMPT_TOKENS = 128
MAX_TOTAL_TOKENS = 384

VALIDATION_SIZE = 50

# Domain quotas. `pool` oversamples the source because the 128-token prompt filter is
# aggressive on the long-document domains (only ~6% of smol-summarize prompts survive).
SMOLTALK_DOMAINS = [
    {"name": "everyday-conversations", "config": "everyday-conversations", "quota": 320, "pool": 1800},
    # Only ~2% of summarize prompts survive the 128-token cap (long source documents),
    # so this pool is deliberately oversampled to reach quota.
    {"name": "summarization",          "config": "smol-summarize",         "quota": 240, "pool": 14000},
    {"name": "rewriting",              "config": "smol-rewrite",           "quota": 240, "pool": 2500},
    {"name": "constraints",            "config": "smol-constraints",       "quota": 320, "pool": 900},
    {"name": "factual-qa",             "config": "openhermes-100k",        "quota": 380, "pool": 1500},
]

BENCHMARK_PROMPTS = [
    "Explain why the sky looks blue to human eyes in two sentences.",
    "Solve for x: 3x + 12 = 27.",
    "Name two renewable energy sources and explain them briefly.",
]


# --------------------------------------------------------------------------- #
# Synthetic reasoning / instruction-following prompts
# --------------------------------------------------------------------------- #
def build_synthetic_samples(rng):
    """Programmatically varied reasoning, science, list and constraint prompts."""
    samples = []

    # --- 1. Step-by-step linear equations ---------------------------------- #
    for _ in range(70):
        a = rng.randint(2, 12)
        x = rng.randint(2, 15)
        b = rng.randint(3, 40)
        c = a * x + b
        samples.append({
            "domain": "reasoning-algebra",
            "messages": [
                {"role": "user", "content": f"Solve for x: {a}x + {b} = {c}. Show your steps."},
                {"role": "assistant", "content": (
                    f"Step 1: Subtract {b} from both sides, giving {a}x = {c - b}.\n"
                    f"Step 2: Divide both sides by {a}, giving x = {x}.\n"
                    f"Check: {a} times {x} plus {b} equals {c}, so x = {x} is correct."
                )},
            ],
        })

    # --- 2. Arithmetic decomposition --------------------------------------- #
    for _ in range(60):
        p = rng.randint(6, 24)
        q = rng.randint(4, 19)
        tens, ones = (p // 10) * 10, p % 10
        samples.append({
            "domain": "reasoning-arithmetic",
            "messages": [
                {"role": "user", "content": f"What is {p} multiplied by {q}? Explain briefly."},
                {"role": "assistant", "content": (
                    f"{p} multiplied by {q} is {p * q}. Break {p} into {tens} and {ones}: "
                    f"({tens} x {q}) + ({ones} x {q}) = {tens * q} + {ones * q} = {p * q}."
                )},
            ],
        })

    # --- 3. Percentage and unit reasoning ---------------------------------- #
    for _ in range(40):
        if rng.random() < 0.5:
            pct = rng.choice([10, 20, 25, 50, 75])
            total = rng.choice([40, 60, 80, 120, 200, 240])
            samples.append({
                "domain": "reasoning-arithmetic",
                "messages": [
                    {"role": "user", "content": f"What is {pct}% of {total}? Explain the calculation."},
                    {"role": "assistant", "content": (
                        f"{pct}% of {total} is {total * pct // 100}. Convert {pct}% to the fraction "
                        f"{pct}/100, then multiply: {total} x {pct}/100 = {total * pct // 100}."
                    )},
                ],
            })
        else:
            km = rng.randint(2, 40)
            samples.append({
                "domain": "reasoning-arithmetic",
                "messages": [
                    {"role": "user", "content": f"Convert {km} kilometres into metres and explain."},
                    {"role": "assistant", "content": (
                        f"{km} kilometres is {km * 1000} metres. One kilometre equals 1,000 metres, "
                        f"so multiply {km} by 1,000 to get {km * 1000}."
                    )},
                ],
            })

    # --- 4. Science explanations ------------------------------------------- #
    science = [
        ("why the sky looks blue to human eyes",
         "Molecules in Earth's atmosphere scatter the shorter blue wavelengths of sunlight far more "
         "than the longer red ones. This effect is called Rayleigh scattering, so we see blue light "
         "arriving from every direction in the sky."),
        ("what photosynthesis does",
         "Photosynthesis lets plants convert sunlight, water and carbon dioxide into glucose and oxygen. "
         "The glucose stores chemical energy for the plant, and the oxygen is released into the air."),
        ("why ice floats on water",
         "Water expands as it freezes because its molecules lock into an open hexagonal lattice. "
         "That makes ice less dense than liquid water, so it floats."),
        ("what causes the seasons on Earth",
         "Earth's rotational axis is tilted about 23.5 degrees relative to its orbit. As the planet orbits "
         "the Sun, each hemisphere alternately tilts toward and away from it, changing sunlight intensity."),
        ("how vaccines protect the body",
         "A vaccine introduces a harmless piece or weakened form of a pathogen. The immune system learns "
         "to recognise it and builds memory cells, so a real infection is neutralised quickly."),
        ("why metals conduct electricity",
         "Metals have delocalised outer electrons that are not bound to any single atom. When a voltage is "
         "applied these free electrons drift through the lattice, carrying an electric current."),
        ("what gravity is",
         "Gravity is the mutual attraction between objects that have mass. The more massive an object and "
         "the closer you are to it, the stronger the pull you feel."),
        ("why the Moon has phases",
         "The Moon is lit from one side by the Sun while it orbits Earth. We see varying fractions of that "
         "lit half depending on the Moon's position, which produces the phases."),
        ("how sound travels through air",
         "A vibrating object compresses and rarefies the air around it, creating a pressure wave. That wave "
         "propagates outward and your eardrum converts it back into vibration."),
        ("what DNA does in a cell",
         "DNA stores the genetic instructions a cell needs to build proteins. Sections called genes are "
         "transcribed into RNA, which the cell then translates into specific proteins."),
        ("why we see lightning before we hear thunder",
         "Light travels roughly a million times faster than sound in air. The flash reaches you almost "
         "instantly while the thunder takes about three seconds per kilometre."),
        ("what evaporation is",
         "Evaporation happens when faster-moving molecules at a liquid's surface escape into the air as "
         "vapour. It cools the remaining liquid because the most energetic molecules leave first."),
    ]
    phrasings = [
        "Explain {topic} in two sentences.",
        "In simple terms, explain {topic}.",
        "Briefly explain {topic} to someone with no science background.",
        "Give a short, clear explanation of {topic}.",
    ]
    for topic, answer in science:
        for template in phrasings:
            samples.append({
                "domain": "science-explanation",
                "messages": [
                    {"role": "user", "content": template.format(topic=topic)},
                    {"role": "assistant", "content": answer},
                ],
            })

    # --- 5. List formatting ------------------------------------------------- #
    list_tasks = [
        ("three renewable energy sources",
         "1. Solar power - photovoltaic panels convert sunlight directly into electricity.\n"
         "2. Wind power - turbines convert the kinetic energy of moving air into electricity.\n"
         "3. Hydropower - flowing water spins turbines connected to generators."),
        ("three prime numbers between 10 and 30",
         "1. 13\n2. 17\n3. 23\nEach of these is divisible only by 1 and itself."),
        ("three states of matter",
         "1. Solid - fixed shape and volume.\n2. Liquid - fixed volume, takes the shape of its container.\n"
         "3. Gas - expands to fill the available space."),
        ("three planets in our solar system",
         "1. Mercury - the smallest planet and closest to the Sun.\n2. Earth - the only known planet with life.\n"
         "3. Jupiter - the largest planet, a gas giant."),
        ("three uses for a spreadsheet",
         "1. Tracking a budget or expenses.\n2. Sorting and filtering lists of records.\n"
         "3. Producing charts from tabular data."),
        ("three ways to reduce household energy use",
         "1. Switch to LED lighting.\n2. Improve insulation around windows and doors.\n"
         "3. Run washing machines only with full loads."),
        ("three programming data types",
         "1. Integer - whole numbers such as 42.\n2. String - text such as \"hello\".\n"
         "3. Boolean - either true or false."),
        ("three common musical instruments",
         "1. Piano - a keyboard instrument with hammered strings.\n2. Guitar - a plucked string instrument.\n"
         "3. Flute - a woodwind instrument played by blowing across an opening."),
    ]
    list_phrasings = [
        "List {topic}. Use a numbered list.",
        "Name {topic}, formatted as a numbered list with one item per line.",
        "Give me {topic}. Answer only with the numbered list.",
        "What are {topic}? Present them as a numbered list.",
    ]
    for topic, answer in list_tasks:
        for template in list_phrasings:
            samples.append({
                "domain": "list-formatting",
                "messages": [
                    {"role": "user", "content": template.format(topic=topic)},
                    {"role": "assistant", "content": answer},
                ],
            })

    # --- 6. Constraint adherence -------------------------------------------- #
    constraint_tasks = [
        ("Describe the ocean in exactly one sentence.",
         "The ocean is a vast body of saltwater that covers most of Earth's surface and regulates its climate."),
        ("Answer in lowercase letters only: what is the capital of France?",
         "the capital of france is paris."),
        ("Explain what a computer does in no more than 20 words.",
         "A computer accepts input, processes it using stored instructions, and produces useful output such as text or images."),
        ("Summarise the water cycle in exactly two sentences.",
         "Water evaporates from oceans and lakes, rises, and condenses into clouds. It then falls back as "
         "precipitation and flows toward the sea, restarting the cycle."),
        ("Answer with a single word: what colour is a ripe banana?",
         "Yellow."),
        ("Reply with exactly three bullet points about exercise.",
         "- Improves cardiovascular health.\n- Strengthens muscles and bones.\n- Supports better sleep and mood."),
        ("Explain gravity without using the word 'force'.",
         "Gravity is the mutual attraction between objects with mass, pulling them toward one another. "
         "The larger the mass and the shorter the distance, the stronger that pull becomes."),
        ("Answer in one sentence, and do not use any numbers: how do plants get energy?",
         "Plants capture sunlight and use it to convert water and carbon dioxide into sugars that store energy."),
        ("Describe a bicycle in under 15 words.",
         "A two-wheeled, pedal-powered vehicle steered with handlebars and balanced by the rider."),
        ("Answer only 'yes' or 'no': is the Sun a star?",
         "Yes."),
    ]
    constraint_prefixes = [
        "",
        "Follow the instruction exactly. ",
        "Be precise and obey the constraint. ",
        "Read the requirement carefully before answering. ",
    ]
    for question, answer in constraint_tasks:
        for prefix in constraint_prefixes:
            samples.append({
                "domain": "constraint-following",
                "messages": [
                    {"role": "user", "content": prefix + question},
                    {"role": "assistant", "content": answer},
                ],
            })

    # --- 7. Short logical inference ------------------------------------------ #
    syllogisms = [
        ("All birds lay eggs, and a sparrow is a bird. Does a sparrow lay eggs?",
         "Yes. Every bird lays eggs, and a sparrow is a bird, so it follows that a sparrow lays eggs."),
        ("All squares are rectangles. Is every rectangle a square?",
         "No. Every square is a rectangle, but a rectangle only counts as a square when all four sides are equal."),
        ("If it rains the ground gets wet, and the ground is dry. Did it rain?",
         "No. If it had rained the ground would be wet, and the ground is dry, so it did not rain."),
        ("Every mammal breathes air, and a whale is a mammal. Does a whale breathe air?",
         "Yes. All mammals breathe air and a whale is a mammal, so a whale breathes air."),
        ("Anna is taller than Ben, and Ben is taller than Carl. Who is tallest?",
         "Anna is tallest. She is taller than Ben, and Ben is taller than Carl, so Anna is above both."),
        ("A train leaves at 2pm and takes 3 hours. When does it arrive?",
         "It arrives at 5pm, because adding the three-hour journey to the 2pm departure gives 5pm."),
        ("If no fish are birds, and a salmon is a fish, is a salmon a bird?",
         "No. No fish is a bird and a salmon is a fish, so a salmon is not a bird."),
        ("Some flowers are red. Are all flowers red?",
         "No. Knowing that some flowers are red says nothing about the rest, so not all flowers are red."),
    ]
    logic_phrasings = [
        "{q}",
        "{q} Explain your reasoning.",
        "Think step by step. {q}",
        "{q} Answer briefly and justify it.",
    ]
    for question, answer in syllogisms:
        for template in logic_phrasings:
            samples.append({
                "domain": "reasoning-logic",
                "messages": [
                    {"role": "user", "content": template.format(q=question)},
                    {"role": "assistant", "content": answer},
                ],
            })

    rng.shuffle(samples)
    return samples


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def within_budget(tokenizer, turns):
    """Enforce the prompt and total token budgets exactly as the collator would see them."""
    try:
        prompt_text = tokenizer.apply_chat_template(
            turns[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            turns, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return False

    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    if prompt_len >= MAX_PROMPT_TOKENS:
        return False
    total_len = len(tokenizer(full_text, add_special_tokens=False).input_ids)
    return total_len <= MAX_TOTAL_TOKENS


def select_exchange(tokenizer, messages):
    """Pick the longest conversation prefix ending in an assistant turn that fits budget.

    The GKD ChatML collator treats messages[:-1] as the prompt and the final message as the
    completion. Taking the *longest fitting* prefix rather than the first exchange matters for
    multi-turn sources: every everyday-conversations dialogue opens with "Hi"/"Hi there", so
    first-turn extraction yields near-identical prompts that collapse under deduplication.
    """
    if not messages:
        return None

    system = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if system is not None else messages

    def clean(turns):
        return [{"role": m["role"], "content": str(m["content"])} for m in turns]

    # Candidate cut points: assistant turns that have at least one user turn before them.
    cuts = [
        i for i, m in enumerate(body)
        if m.get("role") == "assistant"
        and any(t.get("role") == "user" for t in body[:i])
        and str(m.get("content", "")).strip()
    ]

    for i in reversed(cuts):
        prefix = body[:i + 1]
        if not all(str(m.get("content", "")).strip() for m in prefix):
            continue
        turns = clean(([system] if system is not None else []) + prefix)
        if within_budget(tokenizer, turns):
            return turns
    return None


def collect_smoltalk_domain(tokenizer, spec, seen_prompts):
    """Pull `quota` length-filtered single-turn samples from one smoltalk config."""
    name, config, quota, pool = spec["name"], spec["config"], spec["quota"], spec["pool"]
    print(f"  - {name:24} (smoltalk/{config}) target={quota} ...", end=" ", flush=True)
    try:
        raw = load_dataset("HuggingFaceTB/smoltalk", config, split=f"train[:{pool}]")
    except Exception as exc:
        print(f"SKIPPED ({type(exc).__name__}: {str(exc)[:80]})")
        return []

    kept, scanned = [], 0
    for messages in raw["messages"]:
        scanned += 1
        if len(kept) >= quota:
            break
        turns = select_exchange(tokenizer, messages)
        if turns is None:
            continue
        key = turns[-2]["content"].strip()[:200]
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        kept.append({"domain": name, "messages": turns})

    rate = (len(kept) / scanned * 100) if scanned else 0.0
    print(f"kept {len(kept)}/{quota} (scanned {scanned}, {rate:.1f}% pass)")
    return kept


def build_datasets(tokenizer):
    """Assemble the balanced multi-domain train split plus a held-out validation split."""
    print("\n[Phase A] Building multi-domain calibration dataset...")
    rng = random.Random(SEED)
    seen_prompts = set()
    records = []

    for spec in SMOLTALK_DOMAINS:
        records.extend(collect_smoltalk_domain(tokenizer, spec, seen_prompts))

    print("  - synthetic-reasoning     (generated)             ...", end=" ", flush=True)
    synthetic_kept = []
    for sample in build_synthetic_samples(rng):
        if not within_budget(tokenizer, sample["messages"]):
            continue
        key = sample["messages"][-2]["content"].strip()[:200]
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        synthetic_kept.append(sample)
    records.extend(synthetic_kept)
    print(f"kept {len(synthetic_kept)}")

    if not records:
        raise RuntimeError("Dataset construction produced zero samples.")

    dataset = Dataset.from_list(records).shuffle(seed=SEED)

    val_size = min(VALIDATION_SIZE, max(1, len(dataset) // 10))
    eval_dataset = dataset.select(range(val_size))
    train_dataset = dataset.select(range(val_size, len(dataset)))

    print(f"\n  Total collected : {len(dataset)}")
    print(f"  Train split     : {len(train_dataset)}")
    print(f"  Validation split: {len(eval_dataset)}")
    print("  Domain balance  :")
    counts = {}
    for domain in dataset["domain"]:
        counts[domain] = counts.get(domain, 0) + 1
    for domain, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = count / len(dataset) * 100
        print(f"    {domain:24} {count:5d}  ({share:4.1f}%)")

    # The trainer only consumes "messages"; drop the bookkeeping column.
    return (
        train_dataset.select_columns(["messages"]),
        eval_dataset.select_columns(["messages"]),
    )


# --------------------------------------------------------------------------- #
# Logging callback
# --------------------------------------------------------------------------- #
class ScaledDistillationCallback(TrainerCallback):
    """Console telemetry: step time, running JSD loss, grad norm, LR, periodic samples."""

    def __init__(self, model, tokenizer, eval_every=100, window=20):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.window = window
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

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f" [eval @ step {state.global_step}] "
                  f"held-out loss = {metrics['eval_loss']:.4f}")

    def on_save(self, args, state, control, **kwargs):
        print(f" [checkpoint] saved at step {state.global_step} -> {args.output_dir}")

    def maybe_sample(self, args, state):
        """Generate on the benchmark prompts to make quality drift visible."""
        print("\n" + "-" * 78)
        print(f" [benchmark @ step {state.global_step}] student generations")
        print("-" * 78)
        was_training = self.model.training
        self.model.eval()
        try:
            for prompt in BENCHMARK_PROMPTS:
                text = generate_sample(self.model, self.tokenizer, prompt)
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
# Generation helper
# --------------------------------------------------------------------------- #
def generate_sample(model, tokenizer, prompt, max_new_tokens=48):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Scaled GKD distillation for SmolLM2-135M")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the dataset and run 2 steps to validate the pipeline.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable held-out validation loss (saves CPU time).")
    parser.add_argument("--eval-every", type=int, default=100)
    return parser.parse_args()


def run(args):
    max_steps = 2 if args.dry_run else args.max_steps
    eval_every = 1 if args.dry_run else args.eval_every
    use_eval = not args.no_eval

    print("=" * 78)
    print(f" Scaled CPU distillation | {NUM_CORES} threads | {COMPUTE_DTYPE}")
    print(f" Student: {STUDENT_MODEL_ID}")
    print(f" Teacher: {TEACHER_MODEL_ID}")
    if args.dry_run:
        print(" MODE: DRY RUN (2 steps, no final save)")
    print("=" * 78)

    # 1. Tokenizer (shared 49,152-token vocabulary)
    print("\n[Phase 1] Loading shared tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f" -> vocab size {len(tokenizer)}")

    # 2. Dataset
    train_dataset, eval_dataset = build_datasets(tokenizer)
    if args.dry_run:
        # Keep the validation pass cheap while still exercising the eval code path.
        eval_dataset = eval_dataset.select(range(min(4, len(eval_dataset))))

    # 3. Teacher (frozen)
    print(f"\n[Phase 2] Loading frozen teacher ({TEACHER_MODEL_ID})...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(" -> teacher frozen")

    # 4. Student + high-capacity LoRA
    print(f"\n[Phase 3] Loading student ({STUDENT_MODEL_ID}) and injecting LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID, dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # 5. Baseline sample before training
    print("\n[Phase 4] Student output BEFORE distillation:")
    for prompt in BENCHMARK_PROMPTS[:1]:
        print(f"  Q: {prompt}\n  A: {generate_sample(student_model, tokenizer, prompt)}\n")

    # 6. Training configuration
    # NOTE: transformers 5.x removed `warmup_ratio`; `warmup_steps` accepts a float in
    # [0, 1) and is interpreted as a ratio of total steps (see TrainingArguments.get_warmup_steps).
    save_steps = 1 if args.dry_run else 100
    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=2,
        max_grad_norm=1.0,
        seed=SEED,
        lmbda=0.5,
        beta=0.5,
        temperature=0.7,
        max_new_tokens=40,
        max_length=MAX_TOTAL_TOKENS,
        seq_kd=False,
        disable_dropout=True,
        fp16=False,
        bf16=False,
        use_cpu=True,
        report_to=[],
        dataloader_num_workers=0,
    )
    if use_eval:
        config_kwargs.update(
            eval_strategy="steps",
            eval_steps=save_steps,
            per_device_eval_batch_size=1,
        )

    training_args = GKDConfig(**config_kwargs)

    # 7. Trainer
    telemetry = ScaledDistillationCallback(student_model, tokenizer, eval_every=eval_every)
    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        processing_class=tokenizer,
        callbacks=[telemetry, BenchmarkCallback(telemetry, eval_every=eval_every)],
    )

    print(f"\n[Phase 5] Starting GKD training for {max_steps} steps "
          f"(cosine schedule, warmup {training_args.warmup_steps}, lr {training_args.learning_rate})...")
    trainer.train()

    # 8. Post-training samples
    print("\n[Phase 6] Student output AFTER distillation:")
    for prompt in BENCHMARK_PROMPTS:
        print(f"  Q: {prompt}\n  A: {generate_sample(student_model, tokenizer, prompt)}\n")

    # 9. Save
    if args.dry_run:
        print("[Phase 7] Dry run - skipping final adapter save.")
        return

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    student_model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n[Phase 7] Complete. Adapter + tokenizer saved to: {ADAPTER_DIR}\n")


if __name__ == "__main__":
    run(parse_args())
