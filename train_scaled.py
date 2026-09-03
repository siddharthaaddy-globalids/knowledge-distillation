"""
Scaled on-policy knowledge distillation: SmolLM2-360M-Instruct -> SmolLM2-135M-Instruct.

Upgrades the 50-step PoC in main.py to a production-shaped run:
  * 1,800+ multi-domain calibration trajectories (smoltalk + synthetic reasoning)
  * a 50-prompt held-out validation split for generalization tracking
  * higher-capacity LoRA (r=32, alpha=64) across all attention + MLP projections
  * 300 optimizer steps, cosine LR schedule with warmup, effective batch size 4
  * live console logging of step time / running JSD loss / grad norm / LR
  * benchmark generations every 100 steps so quality drift is visible during training

Fully config-driven: models, dataset, LoRA, schedule and hardware all come from a YAML
file, so retargeting to a different teacher/student/dataset never requires a code edit.

    python train_scaled.py --config configs/default.yaml    # Windows / Linux CPU
    python train_scaled.py --config configs/mac.yaml        # Apple Silicon (MPS)
    python train_scaled.py --config configs/mac.yaml --print-config   # resolve only
    python train_scaled.py --dry-run                        # dataset build + 2 steps

Retarget without touching YAML, via CLI or KD_* environment variables:

    python train_scaled.py --teacher HuggingFaceTB/SmolLM2-1.7B-Instruct
    KD_TEACHER_MODEL=... KD_MAX_STEPS=600 python train_scaled.py
"""

import argparse
import math
import os
import random
import time

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

import kd_config

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
from trl.experimental.gkd import GKDConfig, GKDTrainer  # noqa: E402

# --------------------------------------------------------------------------- #
# Runtime state, populated from the YAML config by apply_config().
#
# These stay module-level globals so the dataset helpers below remain importable and
# unit-testable in isolation; nothing here is read before apply_config() has run.
# --------------------------------------------------------------------------- #
CONFIG = None
HARDWARE = None

STUDENT_MODEL_ID = None
TEACHER_MODEL_ID = None
TOKENIZER_SOURCE = None
OUTPUT_DIR = None
ADAPTER_DIR = None

DEVICE = "cpu"
COMPUTE_DTYPE = torch.float32
SEED = 42

# Prompt-length budget. The collator builds the prompt from messages[:-1]; keeping it
# short is what keeps on-policy rollouts affordable.
MAX_PROMPT_TOKENS = 128
MAX_TOTAL_TOKENS = 384
VALIDATION_SIZE = 50

DATASET_SOURCE = "HuggingFaceTB/smoltalk"
DATASET_DOMAINS = []
INCLUDE_SYNTHETIC = True
BENCHMARK_PROMPTS = []


def apply_config(config, hardware):
    """Bind the resolved config and hardware onto the module globals."""
    global CONFIG, HARDWARE, STUDENT_MODEL_ID, TEACHER_MODEL_ID, TOKENIZER_SOURCE
    global OUTPUT_DIR, ADAPTER_DIR, DEVICE, COMPUTE_DTYPE, SEED
    global MAX_PROMPT_TOKENS, MAX_TOTAL_TOKENS, VALIDATION_SIZE
    global DATASET_SOURCE, DATASET_DOMAINS, INCLUDE_SYNTHETIC, BENCHMARK_PROMPTS

    CONFIG, HARDWARE = config, hardware

    STUDENT_MODEL_ID = config["models"]["student"]
    TEACHER_MODEL_ID = config["models"]["teacher"]
    TEACHER_ADAPTER = config["models"].get("teacher_adapter") or None

    tokenizer_choice = str(config["models"].get("tokenizer", "teacher"))
    TOKENIZER_SOURCE = {
        "teacher": TEACHER_MODEL_ID,
        "student": STUDENT_MODEL_ID,
    }.get(tokenizer_choice, tokenizer_choice)

    OUTPUT_DIR = config["project"]["output_dir"]
    ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
    SEED = int(config["project"]["seed"])

    DEVICE = hardware["device"]
    COMPUTE_DTYPE = hardware["dtype"]

    dataset_cfg = config["dataset"]
    MAX_PROMPT_TOKENS = int(dataset_cfg["max_prompt_tokens"])
    MAX_TOTAL_TOKENS = int(dataset_cfg["max_total_tokens"])
    VALIDATION_SIZE = int(dataset_cfg["validation_size"])
    DATASET_SOURCE = dataset_cfg["source"]
    DATASET_DOMAINS = list(dataset_cfg.get("domains") or [])
    INCLUDE_SYNTHETIC = bool(dataset_cfg.get("include_synthetic", True))

    BENCHMARK_PROMPTS = list(config.get("benchmark_prompts") or [])


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


def alpaca_to_turns(row, spec):
    """Convert one Alpaca-style row (instruction / input / output) into chat turns.

    Alpaca datasets such as gbharti/finance-alpaca carry no `messages` column, so the
    prompt is assembled here. A non-empty `input` is appended to the instruction, which
    is the convention those datasets were written against.
    """
    instruction = str(row.get(spec.get("instruction_column", "instruction")) or "").strip()
    context = str(row.get(spec.get("input_column", "input")) or "").strip()
    output = str(row.get(spec.get("output_column", "output")) or "").strip()
    if not instruction or not output:
        return None
    user = f"{instruction}\n\n{context}" if context else instruction
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def collect_domain(tokenizer, spec, seen_prompts):
    """Pull `quota` length-filtered samples from one dataset config.

    Handles two source layouts, selected by `format` in the domain spec:
      messages (default) - conversational datasets like smoltalk
      alpaca             - instruction/input/output datasets like finance-alpaca
    """
    name, config, quota, pool = spec["name"], spec.get("config"), spec["quota"], spec["pool"]
    label = f"{DATASET_SOURCE.split('/')[-1]}/{config}" if config else DATASET_SOURCE
    print(f"  - {name:24} ({label}) target={quota} ...", end=" ", flush=True)
    try:
        split = spec.get("split", f"train[:{pool}]")
        if config:
            raw = load_dataset(DATASET_SOURCE, config, split=split)
        else:
            raw = load_dataset(DATASET_SOURCE, split=split)
    except Exception as exc:
        print(f"SKIPPED ({type(exc).__name__}: {str(exc)[:80]})")
        return []

    fmt = str(spec.get("format", "messages")).lower()
    if fmt == "alpaca":
        required = spec.get("instruction_column", "instruction")
        if required not in raw.column_names:
            print(f"SKIPPED (no '{required}' column; found {raw.column_names})")
            return []
        rows = raw
    else:
        column = spec.get("messages_column", "messages")
        if column not in raw.column_names:
            print(f"SKIPPED (no '{column}' column; found {raw.column_names})")
            return []
        rows = raw[column]

    kept, scanned = [], 0
    for row in rows:
        scanned += 1
        if len(kept) >= quota:
            break
        if fmt == "alpaca":
            turns = alpaca_to_turns(row, spec)
            if turns is not None and not within_budget(tokenizer, turns):
                turns = None
        else:
            turns = select_exchange(tokenizer, row)
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

    for spec in DATASET_DOMAINS:
        records.extend(collect_domain(tokenizer, spec, seen_prompts))

    if INCLUDE_SYNTHETIC:
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
    parser = argparse.ArgumentParser(
        description="Config-driven GKD distillation (CPU / Apple Silicon MPS / CUDA)"
    )
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="YAML config file, e.g. configs/default.yaml or configs/mac.yaml")
    parser.add_argument("--print-config", action="store_true",
                        help="Resolve and print the effective config, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the dataset and run 2 steps to validate the pipeline.")

    # Ad-hoc overrides. Anything left as None does not override the config file.
    parser.add_argument("--student", default=None, help="Override models.student")
    parser.add_argument("--teacher", default=None, help="Override models.teacher")
    parser.add_argument("--teacher-adapter", default=None,
                        help="LoRA adapter to merge into the teacher "
                             "(use when the teacher is base + adapter rather "
                             "than a merged checkpoint)")
    parser.add_argument("--allow-bad-teacher", action="store_true",
                        help="Train even if the teacher fails its pre-flight "
                             "sanity check (not recommended)")
    parser.add_argument("--dataset", default=None, help="Override dataset.source")
    parser.add_argument("--output-dir", default=None, help="Override project.output_dir")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"],
                        help="Override hardware.device")
    parser.add_argument("--dtype", default=None,
                        choices=["auto", "float32", "bfloat16", "float16"],
                        help="Override hardware.dtype")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable held-out validation loss (saves time).")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Override training.benchmark_every")
    return parser.parse_args()


def build_overrides(args):
    """Translate CLI flags into a nested override mapping for load_config()."""
    overrides = {}

    def put(section, key, value):
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    put("models", "student", args.student)
    put("models", "teacher", args.teacher)
    put("models", "teacher_adapter", args.teacher_adapter)
    put("dataset", "source", args.dataset)
    put("project", "output_dir", args.output_dir)
    put("hardware", "device", args.device)
    put("hardware", "dtype", args.dtype)
    put("training", "max_steps", args.max_steps)
    put("training", "batch_size", args.batch_size)
    put("training", "benchmark_every", args.eval_every)
    if args.no_eval:
        put("training", "eval_enabled", False)
    return overrides


# ---------------------------------------------------------------------------
# Teacher pre-flight
# ---------------------------------------------------------------------------
# GKD trains the student to match the teacher's output distribution, so a broken
# teacher silently produces a broken student: the run "succeeds", burns the whole
# step budget, and only the final samples reveal it.
#
# The failure that motivated this check was a merged checkpoint whose key layout
# did not match the architecture transformers built for it. from_pretrained does
# not raise in that case - it randomly initialises whatever it could not map and
# prints a warning that scrolls past. A partly random teacher emits near-uniform
# token salad, and the student faithfully learns to reproduce it.
# ---------------------------------------------------------------------------
def verify_teacher(model, tokenizer, loading_info=None, strict=True):
    """Fail fast when the teacher is not a usable distillation target."""
    problems = []

    # 1. Weights that were fabricated rather than loaded from the checkpoint.
    if loading_info:
        missing = list(loading_info.get("missing_keys") or [])
        unexpected = list(loading_info.get("unexpected_keys") or [])
        # Derived buffers and tied heads go missing legitimately; real parameter
        # tensors do not.
        ignorable = ("rotary_emb.inv_freq", ".attn_bias", "lm_head.weight")
        real = [k for k in missing if not k.endswith(ignorable)]
        if real:
            problems.append(
                "%d weight(s) were randomly initialised instead of loaded - the "
                "teacher is partly untrained. First few: %s" % (len(real), real[:6])
            )
        if unexpected:
            print(" !! %d checkpoint tensor(s) went unused: %s"
                  % (len(unexpected), unexpected[:4]))

    # 2. Non-finite parameters, e.g. a bad merge or an overflowed save.
    bad = [n for n, t in model.named_parameters() if not torch.isfinite(t).all()]
    if bad:
        problems.append("%d parameter tensor(s) contain NaN/Inf: %s" % (len(bad), bad[:4]))

    # 3. The behavioural test. A healthy instruct teacher is confident about its
    #    next token; a randomly initialised one is near-uniform over ~248k tokens,
    #    which is exactly what produces multilingual token salad downstream.
    probe = BENCHMARK_PROMPTS[0] if BENCHMARK_PROMPTS else "Explain compound interest."
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": probe}], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top_p = float(probs.max())
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    uniform = math.log(logits.numel())

    print(" -> teacher sanity: top-1 prob %.4f, entropy %.2f (uniform would be %.2f)"
          % (top_p, entropy, uniform))
    print("    teacher says: %r" % (generate_sample(model, tokenizer, probe,
                                                    max_new_tokens=32)[:160],))

    if entropy > 0.80 * uniform or top_p < 0.02:
        problems.append(
            "teacher output is near-uniform (entropy %.2f of a possible %.2f); it is "
            "not predicting language" % (entropy, uniform)
        )

    if not problems:
        print(" -> teacher pre-flight OK")
        return

    bar = "=" * 78
    print("\n" + bar)
    print("  TEACHER PRE-FLIGHT FAILED")
    print(bar)
    for item in problems:
        print("  * " + item)
    print("""
  Distilling from this teacher would produce a broken student, because GKD trains
  the student to match whatever distribution the teacher emits.

  Common causes:
    * the checkpoint's key layout does not match the architecture transformers
      built for it (merged or converted models often keep an older prefix
      convention, e.g. language_model.model.* vs model.language_model.*);
    * config.json declares components the checkpoint does not contain;
    * the merge was saved from a quantised or partially loaded model.

  Re-run with --allow-bad-teacher to proceed anyway.
""")
    print(bar)
    if strict:
        raise SystemExit(2)


def run(args):
    config = kd_config.load_config(args.config, build_overrides(args))
    hardware = kd_config.resolve_device(config)
    apply_config(config, hardware)

    print(kd_config.describe(config, hardware))

    if args.print_config:
        printable = {k: v for k, v in config.items() if k != "_meta"}
        print("\n--- effective config ---")
        print(yaml.safe_dump(printable, sort_keys=False, default_flow_style=False))
        return

    training_cfg = config["training"]
    gkd_cfg = config["gkd"]
    lora_cfg = config["lora"]

    max_steps = 2 if args.dry_run else int(training_cfg["max_steps"])
    eval_every = 1 if args.dry_run else int(training_cfg["benchmark_every"])
    use_eval = bool(training_cfg["eval_enabled"])
    if args.dry_run:
        print(" MODE: DRY RUN (2 steps, no final save)")

    # 1. Tokenizer. Student and teacher must share a vocabulary for standard GKD.
    print(f"\n[Phase 1] Loading tokenizer from {TOKENIZER_SOURCE}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
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
    teacher_model, loading_info = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if TEACHER_ADAPTER:
        # The teacher is a base model plus a LoRA adapter rather than a merged
        # checkpoint. This is the safer form: merged checkpoints written by other
        # frameworks often use that framework's key layout, which plain
        # transformers may not map back onto the architecture. Merging here, from
        # the canonical base, sidesteps that entirely.
        print(f" -> applying teacher LoRA adapter: {TEACHER_ADAPTER}")
        teacher_model = PeftModel.from_pretrained(teacher_model, TEACHER_ADAPTER)
        teacher_model = teacher_model.merge_and_unload()
        print(" -> adapter merged into the teacher")
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print(" -> teacher frozen")
    verify_teacher(teacher_model, tokenizer, loading_info,
                   strict=not getattr(args, "allow_bad_teacher", False))

    # 4. Student + high-capacity LoRA
    print(f"\n[Phase 3] Loading student ({STUDENT_MODEL_ID}) and injecting LoRA...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID, dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)

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
    lora_config = LoraConfig(**lora_kwargs)
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # 5. Baseline sample before training
    print("\n[Phase 4] Student output BEFORE distillation:")
    for prompt in BENCHMARK_PROMPTS[:1]:
        print(f"  Q: {prompt}\n  A: {generate_sample(student_model, tokenizer, prompt)}\n")

    # 6. Training configuration
    # NOTE: transformers 5.x removed `warmup_ratio`; `warmup_steps` accepts a float in
    # [0, 1) and is interpreted as a ratio of total steps (see TrainingArguments.get_warmup_steps).
    save_steps = 1 if args.dry_run else int(training_cfg["save_steps"])
    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
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
        seed=SEED,
        lmbda=float(gkd_cfg["lmbda"]),
        beta=float(gkd_cfg["beta"]),
        temperature=float(gkd_cfg["temperature"]),
        max_new_tokens=int(gkd_cfg["max_new_tokens"]),
        max_length=MAX_TOTAL_TOKENS,
        seq_kd=bool(gkd_cfg["seq_kd"]),
        disable_dropout=True,
        # Device and precision flags come from the resolved hardware layer, so the same
        # config runs unchanged on CPU, Apple Silicon MPS and CUDA.
        fp16=HARDWARE["fp16"],
        bf16=HARDWARE["bf16"],
        use_cpu=HARDWARE["use_cpu"],
        dataloader_pin_memory=HARDWARE["pin_memory"],
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
