"""
Dataset assembly for distillation.

Two sources feed one balanced split:

  * real conversations pulled from a Hugging Face dataset, one block per domain
    with its own quota, so no single domain dominates the calibration set;
  * programmatically varied synthetic prompts covering reasoning, science, list
    formatting and constraint following, which are the behaviours a short run
    actually transfers.

Everything is filtered against the same token budget the GKD collator will apply,
so a sample that survives here cannot be silently truncated later. The split is
seeded from project.seed, which is what lets kd.evaluate rebuild the identical
held-out rows the student never trained on.

Nothing here reads module state: every function takes what it needs. The previous
version bound a dozen globals in an apply_config() step, and a name that step
forgot to declare became a local, so the failure surfaced hundreds of lines later
- after the dataset build and the teacher download had already been paid for.
"""

import random
from typing import NamedTuple

from datasets import Dataset, load_dataset


class Budget(NamedTuple):
    """The token budget every sample is filtered against.

    Carried as one value rather than two loose ints because it is threaded through
    four call levels, and a transposed pair of numbers there would silently change
    which samples survive rather than raise anything.
    """

    max_prompt_tokens: int
    max_total_tokens: int

    @classmethod
    def from_config(cls, config):
        dataset = config["dataset"]
        return cls(int(dataset["max_prompt_tokens"]), int(dataset["max_total_tokens"]))


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
def within_budget(tokenizer, turns, budget):
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
    if prompt_len >= budget.max_prompt_tokens:
        return False
    total_len = len(tokenizer(full_text, add_special_tokens=False).input_ids)
    return total_len <= budget.max_total_tokens


def select_exchange(tokenizer, messages, budget):
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
        if within_budget(tokenizer, turns, budget):
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


def collect_domain(tokenizer, spec, seen_prompts, source, budget):
    """Pull `quota` length-filtered samples from one dataset config.

    Handles two source layouts, selected by `format` in the domain spec:
      messages (default) - conversational datasets like smoltalk
      alpaca             - instruction/input/output datasets like finance-alpaca
    """
    name, config, quota, pool = spec["name"], spec.get("config"), spec["quota"], spec["pool"]
    label = f"{source.split('/')[-1]}/{config}" if config else source
    print(f"  - {name:24} ({label}) target={quota} ...", end=" ", flush=True)
    try:
        split = spec.get("split", f"train[:{pool}]")
        if config:
            raw = load_dataset(source, config, split=split)
        else:
            raw = load_dataset(source, split=split)
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
            if turns is not None and not within_budget(tokenizer, turns, budget):
                turns = None
        else:
            turns = select_exchange(tokenizer, row, budget)
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


def build_datasets(tokenizer, config):
    """Assemble the balanced multi-domain train split plus a held-out validation split.

    Seeded from project.seed, so kd.evaluate can call this with the same config and
    get back byte-identical held-out rows - the ones the student never trained on.
    """
    dataset_cfg = config["dataset"]
    seed = int(config["project"]["seed"])
    budget = Budget.from_config(config)
    source = dataset_cfg["source"]

    print("\n[Phase A] Building multi-domain calibration dataset...")
    rng = random.Random(seed)
    seen_prompts = set()
    records = []

    for spec in dataset_cfg.get("domains") or []:
        records.extend(collect_domain(tokenizer, spec, seen_prompts, source, budget))

    if bool(dataset_cfg.get("include_synthetic", True)):
        print("  - synthetic-reasoning     (generated)             ...", end=" ", flush=True)
        synthetic_kept = []
        for sample in build_synthetic_samples(rng):
            if not within_budget(tokenizer, sample["messages"], budget):
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

    dataset = Dataset.from_list(records).shuffle(seed=seed)

    val_size = min(int(dataset_cfg["validation_size"]), max(1, len(dataset) // 10))
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

