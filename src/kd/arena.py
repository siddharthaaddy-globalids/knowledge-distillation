"""
Head-to-head scoring: accuracy on a held-out multiple-choice set, and Elo.

    python -m kd arena --config configs/enlibraQ3-8B.yaml

`kd evaluate` asks how faithfully the student reproduces the TEACHER - top-1
agreement, KL divergence, perplexity. That is the right question for
distillation in the abstract, and it is the wrong one here: these are questions
with a known correct letter, so a student that mirrors a mediocre teacher
perfectly scores well there and badly on the only number anyone will ask about.

Three players, because two is not enough to learn anything from:

    base       the stock student, no adapter. The control.
    distilled  the same base plus the trained LoRA.
    teacher    what was distilled from. The ceiling.

Reading the result:
  * distilled below base    - training made it worse. Stop and look at the loss.
  * distilled at base       - the format transferred, the capability did not.
  * distilled between them  - distillation worked; the gap left is the headroom.
  * distilled above teacher - possible on a narrow set, and worth distrusting
                              until it survives a second eval file.

WHY ELO AND NOT JUST ACCURACY
-----------------------------
Accuracy answers "how often is it right". Elo answers "how often is it right
*where the others are not*", which is a different and more useful question when
three models share most of their answers. Two models both at 60% may agree on
every item or disagree on half of them; accuracy cannot tell those apart and Elo
can, because it is computed from per-question outcomes rather than from totals.

Each question is a round-robin: for every pair of players, whoever answered it
correctly beats whoever did not, and matching outcomes are a draw. That grounds
every match in the answer key, so no judge model is needed and nothing here is a
matter of opinion.

Sequential Elo depends on the order the matches are played, which is an artifact
of the algorithm rather than a fact about the models. `rounds` replays the whole
schedule over that many seeded shuffles and averages, and the spread across
shuffles is reported alongside - a spread comparable to the gap between two
players means the gap is not real.
"""

import json
import os
import random
import re

# `<Answer>:` - the colon sits inside the tag in the curriculum exports, and the
# letter is on its own line after it. Matched loosely enough to survive a model
# dropping the colon or the newline, which they do.
ANSWER_PATTERN = re.compile(r"<Answer>\s*:?\s*\n?\s*([A-Z])\b", re.IGNORECASE)

# Standard chess constants. K is the step size per match; 400 is the rating
# difference that corresponds to a 10:1 expected score.
K_FACTOR = 24
START_RATING = 1000.0
RATING_SCALE = 400.0


def extract_answer(text):
    """The letter a completion settled on, or None when it never committed."""
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).upper() if match else None


def load_questions(path):
    """Read a held-out .jsonl into [{prompt, gold}], skipping ungradeable rows.

    A row with no answer letter - a persona exchange that found its way in - is
    dropped rather than scored, because every player would "fail" it identically
    and it would dilute the ratings toward a draw.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"evaluation.arena_file points at {path}, which does not exist.\n"
            f"  Held-out sets are written by scripts/prepare_curriculum.py; the "
            f"one it marks HELD OUT is the file to name here.")

    questions, skipped = [], 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            turns = json.loads(line)["messages"]
            gold = extract_answer(turns[-1]["content"])
            if not gold:
                skipped += 1
                continue
            questions.append({"prompt": turns[-2]["content"], "gold": gold})
    return questions, skipped


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
def outcomes(results):
    """Per-question pairwise outcomes from {player: [correct?, ...]}.

    Yields (question index, player a, player b, score for a) with 1.0, 0.5 or
    0.0 - the same encoding chess uses, so the update below is the textbook one.
    """
    players = sorted(results)
    length = len(next(iter(results.values()))) if results else 0
    for index in range(length):
        for i, a in enumerate(players):
            for b in players[i + 1:]:
                right_a, right_b = results[a][index], results[b][index]
                if right_a == right_b:
                    score = 0.5
                else:
                    score = 1.0 if right_a else 0.0
                yield index, a, b, score


def _play(matches, players, k=K_FACTOR):
    """One pass of sequential Elo over `matches`, returning final ratings."""
    rating = {name: START_RATING for name in players}
    for _index, a, b, score in matches:
        expected_a = 1.0 / (1.0 + 10 ** ((rating[b] - rating[a]) / RATING_SCALE))
        change = k * (score - expected_a)
        rating[a] += change
        # Zero-sum: what one player gains the other loses, which is what keeps
        # the mean rating fixed at START_RATING and makes the numbers comparable
        # between runs.
        rating[b] -= change
    return rating


def elo(results, rounds=25, seed=42):
    """Elo ratings from per-question correctness, averaged over shuffled orders.

    Returns {player: {"rating", "spread"}}. `spread` is the standard deviation
    across shuffles: it is the noise floor, and a gap between two players that
    does not clear it is not a gap.
    """
    players = sorted(results)
    if not players:
        return {}

    schedule = list(outcomes(results))
    if not schedule:
        return {name: {"rating": START_RATING, "spread": 0.0} for name in players}

    samples = {name: [] for name in players}
    rng = random.Random(seed)
    for _ in range(max(1, rounds)):
        order = list(schedule)
        if rounds > 1:
            rng.shuffle(order)
        for name, value in _play(order, players).items():
            samples[name].append(value)

    ratings = {}
    for name, values in samples.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        ratings[name] = {"rating": mean, "spread": variance ** 0.5}
    return ratings


def head_to_head(results):
    """{"a vs b": {win, loss, draw}} - the evidence behind the ratings.

    Worth reporting because a rating gap built entirely out of draws means the
    players disagree about almost nothing, which is a different situation from
    the same gap built out of decisive results.
    """
    table = {}
    for _index, a, b, score in outcomes(results):
        key = f"{a} vs {b}"
        entry = table.setdefault(key, {"win": 0, "loss": 0, "draw": 0})
        entry["win" if score == 1.0 else "loss" if score == 0.0 else "draw"] += 1
    return table


def summarise(results, unparsed=None, rounds=25, seed=42):
    """The whole payload: accuracy, Elo, and the pairwise record."""
    total = len(next(iter(results.values()))) if results else 0
    ratings = elo(results, rounds=rounds, seed=seed)
    unparsed = unparsed or {}

    players = {}
    for name, correct in results.items():
        hits = sum(1 for value in correct if value)
        players[name] = {
            "accuracy": (hits / total) if total else None,
            "correct": hits,
            "elo": round(ratings[name]["rating"], 1),
            "elo_spread": round(ratings[name]["spread"], 1),
            # Never answering is not the same failure as answering wrongly, and
            # collapsing them hides the one that is actually fixable: running out
            # of max_new_tokens before the <Answer> tag.
            "unanswered": unparsed.get(name, 0),
        }
    return {
        "questions": total,
        "random_baseline": 0.25,
        "elo_rounds": rounds,
        "players": players,
        "head_to_head": head_to_head(results),
    }


def render(payload):
    """The summary table, for a log or a terminal."""
    lines = [
        f"  {payload['questions']} held-out questions, "
        f"random baseline {payload['random_baseline'] * 100:.0f}%",
        "",
        f"  {'player':<12} {'accuracy':>9}  {'elo':>7}  {'+/-':>5}  {'unanswered':>10}",
        f"  {'-' * 12} {'-' * 9}  {'-' * 7}  {'-' * 5}  {'-' * 10}",
    ]
    ranked = sorted(payload["players"].items(),
                    key=lambda kv: -kv[1]["elo"])
    for name, entry in ranked:
        accuracy = f"{entry['accuracy'] * 100:.1f}%" if entry["accuracy"] is not None else "-"
        lines.append(f"  {name:<12} {accuracy:>9}  {entry['elo']:>7.0f}  "
                     f"{entry['elo_spread']:>5.0f}  {entry['unanswered']:>10}")

    lines.append("")
    for pair, record in sorted(payload["head_to_head"].items()):
        lines.append(f"  {pair:<28} {record['win']}W {record['loss']}L "
                     f"{record['draw']}D")

    # The spread is the noise floor. Saying so once, next to the numbers, is
    # cheaper than watching someone act on a 12-point difference.
    spreads = [e["elo_spread"] for e in payload["players"].values()]
    if spreads:
        lines += ["",
                  f"  +/- is the standard deviation across "
                  f"{payload['elo_rounds']} shuffled orderings. A gap smaller "
                  f"than it is noise."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Playing the matches
# --------------------------------------------------------------------------- #
def _free(model):
    """Drop a model and give the memory back before the next one is loaded.

    The three players together are far larger than the machine that trained them
    - an 8B teacher plus two students - so they are loaded one at a time. Without
    this the second load meets a still-resident first one.
    """
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _generate(model, tokenizer, prompt, device, max_new_tokens):
    """One completion, rendered through the same chat template training used."""
    import torch

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Greedy, always. An accuracy measured with sampling is a different
            # number every run, and the difference between two runs of one model
            # would be read as a difference between two models.
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def _answer_all(model, tokenizer, questions, device, max_new_tokens, label, log):
    """Every question, greedily. Returns (correct flags, unanswered count)."""
    correct, unanswered = [], 0
    for index, question in enumerate(questions, start=1):
        text = _generate(model, tokenizer, question["prompt"], device,
                         max_new_tokens)
        predicted = extract_answer(text)
        if predicted is None:
            unanswered += 1
        correct.append(predicted == question["gold"])
        if log and (index % 10 == 0 or index == len(questions)):
            hits = sum(1 for value in correct if value)
            log.info(f"      {label:<10} {index}/{len(questions)}  "
                     f"{hits / index * 100:5.1f}%")
    return correct, unanswered


def play(config, hardware, adapter, questions, max_new_tokens=512, log=None,
         players=("base", "distilled", "teacher")):
    """Load each player in turn, answer every question, return the raw results.

    Returns ({player: [correct?, ...]}, {player: unanswered count}).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = hardware["device"], hardware["dtype"]
    base_id = config["models"]["student"]

    # From the adapter directory, because kd.train saves it there beside the
    # weights. The Hub copy is usually the same file and occasionally is not - a
    # different revision, a changed template - and a template that disagrees with
    # training is an accuracy loss that looks like a bad run.
    source = adapter if adapter and os.path.isfile(
        os.path.join(str(adapter), "tokenizer_config.json")) else base_id
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if log:
        log.info(f"      tokenizer: {source}")

    results, unparsed = {}, {}

    def run(label, build):
        if log:
            log.info(f"      loading {label}")
        model = build()
        model.eval()
        try:
            correct, missing = _answer_all(model, tokenizer, questions, device,
                                           max_new_tokens, label, log)
        finally:
            _free(model)
        results[label] = correct
        unparsed[label] = missing

    if "base" in players:
        run("base", lambda: AutoModelForCausalLM.from_pretrained(
            base_id, dtype=dtype, low_cpu_mem_usage=True).to(device))

    if "distilled" in players and adapter:
        def build_distilled():
            model = AutoModelForCausalLM.from_pretrained(
                base_id, dtype=dtype, low_cpu_mem_usage=True)
            return PeftModel.from_pretrained(
                model, str(adapter)).merge_and_unload().to(device)
        run("distilled", build_distilled)

    if "teacher" in players:
        def build_teacher():
            from .teacher import load_teacher
            model, _info = load_teacher(
                config["models"]["teacher"],
                config["models"].get("teacher_adapter"),
                dtype=dtype, device=device, verbose=False)
            return model
        run("teacher", build_teacher)

    return results, unparsed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(args=None):
    import argparse
    import logging
    import sys

    if args is None:
        parser = argparse.ArgumentParser(
            prog="kd arena",
            description="Accuracy and Elo on a held-out multiple-choice set.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__)
        parser.add_argument("-c", "--config", default=None, metavar="PATH",
                            help="Profile the adapter was trained from")
        parser.add_argument("--adapter", default=None, metavar="DIR",
                            help="Adapter directory; default is the newest under "
                                 "project.runs_dir")
        parser.add_argument("--file", default=None, metavar="PATH",
                            help="Held-out .jsonl; default is evaluation.arena_file")
        parser.add_argument("--limit", type=int, default=None, metavar="N",
                            help="Score only the first N questions")
        parser.add_argument("--skip", action="append", default=[],
                            choices=["base", "distilled", "teacher"],
                            help="Leave a player out, repeatable. --skip teacher "
                                 "is the one worth knowing: it avoids loading "
                                 "8B of weights when you only want base vs "
                                 "distilled.")
        parser.add_argument("--max-new-tokens", type=int, default=None)
        parser.add_argument("--device", default=None,
                            help="auto | cpu | mps | cuda")
        parser.add_argument("--json", default=None, metavar="PATH",
                            help="Write the payload here")
        args = parser.parse_args()

    from .config import load_config, resolve_device
    from .runlog import discover_adapters, is_adapter

    log = logging.getLogger("kd.arena")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    config = load_config(args.config, use_env=False)
    if getattr(args, "device", None):
        config.setdefault("hardware", {})["device"] = args.device
    hardware = resolve_device(config)
    settings = config.get("evaluation") or {}

    path = args.file or settings.get("arena_file")
    if not path:
        log.error("xx  nothing to score: pass --file, or set "
                  "evaluation.arena_file in the config")
        return 1

    adapter = args.adapter
    if not adapter and "distilled" not in (args.skip or []):
        found = discover_adapters(config["project"].get("runs_dir") or "./runs")
        adapter = found[0] if found else None
        if adapter:
            log.info(f"==> newest adapter: {adapter}")
    if adapter and not is_adapter(adapter):
        log.error(f"xx  {adapter} has no adapter_config.json")
        return 1

    questions, skipped = load_questions(path)
    limit = args.limit or settings.get("arena_limit")
    if limit:
        questions = questions[:int(limit)]
    log.info(f"==> {len(questions)} questions from {path}"
             + (f" ({skipped} ungradeable rows skipped)" if skipped else ""))

    players = tuple(p for p in ("base", "distilled", "teacher")
                    if p not in (args.skip or []))
    results, unparsed = play(
        config, hardware, adapter, questions,
        max_new_tokens=int(args.max_new_tokens
                           or settings.get("arena_max_new_tokens") or 512),
        log=log, players=players)

    payload = summarise(results, unparsed,
                        rounds=int(settings.get("arena_elo_rounds") or 25),
                        seed=int(config["project"]["seed"]))
    payload["arena_file"] = str(path)
    payload["adapter"] = str(adapter) if adapter else None

    log.info("")
    log.info(render(payload))

    if getattr(args, "json", None):
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        log.info(f"\n  wrote {args.json}")
    return 0
