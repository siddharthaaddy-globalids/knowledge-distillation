#!/usr/bin/env python3
# ===========================================================================
#  Ask the distilled student a question, or score it on the held-out split.
#
#      python scripts/ask.py --config configs/enlibraQ3-8B.yaml \
#          --question "What is a star formed from?"
#
#      python scripts/ask.py --config configs/enlibraQ3-8B.yaml --accuracy
#
#  WHY THIS EXISTS ALONGSIDE `kd evaluate`
#  ---------------------------------------
#  kd.evaluate measures how faithfully the student reproduces the TEACHER -
#  agreement on next-token distributions, perplexity, generation similarity.
#  That is the right question for distillation in general and it is what the
#  pipeline's evaluate stage reports.
#
#  It is not the question this curriculum asks. These are multiple-choice items
#  with a known correct letter, so the number that decides whether the run
#  worked is accuracy: how often the student picks the right option. A student
#  that mirrors a mediocre teacher perfectly scores well on fidelity and badly
#  here, and it is this number you would put in front of anyone.
#
#  --accuracy rebuilds the held-out split the same way kd.evaluate does - from
#  the config and project.seed, through kd.data.build_datasets - so it scores
#  exactly the rows training never saw. Change the seed or the domains and the
#  split changes with them; that is the point.
#
#  PROMPT FORMAT
#  -------------
#  The student was trained on one shape and will be off-distribution in any
#  other. Free-text questions are wrapped to match it, and the tokenizer is
#  loaded from the adapter directory rather than from the Hub, so the chat
#  template is byte-identical to the one training rendered with.
# ===========================================================================

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# The curriculum's own tags. A prompt that does not use them is a prompt the
# student has never seen.
QUESTION_TEMPLATE = "<Question>\n{question}\n</Question>"
OPTIONS_TEMPLATE = "<Options>\n{options}\n</Options>"

# `<Answer>:` - the colon is inside the tag in the exports, and the letter sits
# on its own line after it. Matched loosely enough to survive the model dropping
# the colon or the newline, which it will occasionally do.
ANSWER_PATTERN = re.compile(r"<Answer>\s*:?\s*\n?\s*([A-Z])\b", re.IGNORECASE)


def log(message):
    print(message, file=sys.stderr)


def build_prompt(question, options=None):
    """Wrap a free-text question in the shape the student was trained on."""
    if "<Question>" in question:
        return question  # already formatted - a row from the corpus, most likely
    parts = [QUESTION_TEMPLATE.format(question=question.strip())]
    if options:
        parts.append(OPTIONS_TEMPLATE.format(options="\n".join(options)))
    return "\n".join(parts)


def extract_answer(text):
    """The letter the student settled on, or None when it never committed."""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).upper() if match else None


def load_student(config, adapter, device, dtype):
    """Base student plus the trained LoRA, and the tokenizer training used.

    The tokenizer comes from the adapter directory because kd.train saves it
    there next to the weights. Loading it from the Hub instead would usually be
    the same file and would occasionally not be - a different revision, a changed
    template - and a chat template that disagrees with training is a silent
    accuracy loss that looks like a bad run.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = config["models"]["student"]
    tokenizer_source = adapter if os.path.isfile(
        os.path.join(adapter, "tokenizer_config.json")) else base_id
    log(f"==> tokenizer : {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"==> base      : {base_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, low_cpu_mem_usage=True)
    log(f"==> adapter   : {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload().to(device)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, device, max_new_tokens=512, greedy=True):
    """One completion, rendered through the training chat template."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Greedy for scoring: accuracy measured with sampling is a different
            # number every run, and the disagreement between two runs would be
            # mistaken for a difference between two adapters.
            do_sample=not greedy,
            temperature=None if greedy else 0.7,
            top_p=None if greedy else 0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def held_out_rows(config, tokenizer):
    """The validation split, rebuilt exactly as kd.evaluate rebuilds it."""
    from kd.data import build_datasets

    _train, evaluation = build_datasets(tokenizer, config)
    rows = []
    for record in evaluation:
        turns = record["messages"]
        rows.append({"prompt": turns[-2]["content"],
                     "gold": extract_answer(turns[-1]["content"])})
    return rows


def score(model, tokenizer, rows, device, max_new_tokens, show=0):
    """Accuracy over the held-out rows, and what went wrong where."""
    graded = [row for row in rows if row["gold"]]
    skipped = len(rows) - len(graded)
    if skipped:
        log(f"!!  {skipped} held-out rows carry no answer letter (identity rows, "
            f"most likely) and are not scored")

    correct, unparsed, wrong = 0, 0, []
    for index, row in enumerate(graded, start=1):
        text = generate(model, tokenizer, row["prompt"], device,
                        max_new_tokens=max_new_tokens, greedy=True)
        predicted = extract_answer(text)
        if predicted is None:
            unparsed += 1
        elif predicted == row["gold"]:
            correct += 1
        else:
            wrong.append((row["gold"], predicted))

        if show and index <= show:
            log("-" * 70)
            log(row["prompt"][:400])
            log("  -> " + text[:400].replace("\n", "\n     "))
        state = "ok " if predicted == row["gold"] else "XX "
        print(f"\r  {state}{index}/{len(graded)}  "
              f"accuracy {correct / index * 100:5.1f}%", end="", file=sys.stderr)
    print(file=sys.stderr)

    total = len(graded)
    return {
        "scored": total,
        "correct": correct,
        # An answer the model never committed to is not a wrong answer, and
        # collapsing the two hides the failure that is actually worth fixing:
        # running out of max_new_tokens before reaching the <Answer> tag.
        "unparsed": unparsed,
        "accuracy": (correct / total) if total else None,
        "random_baseline": 0.25,
        "confusions": sorted({f"{g}->{p}" for g, p in wrong}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask the distilled student, or score it on the held-out split.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("-c", "--config", default="configs/enlibraQ3-8B.yaml",
                        help="Profile the adapter was trained from. Supplies the "
                             "base model, the domains and the seed.")
    parser.add_argument("--adapter", default=None, metavar="DIR",
                        help="Adapter directory. Default: the newest under "
                             "project.runs_dir.")
    parser.add_argument("--question", default=None, metavar="TEXT",
                        help="Ask one question. Use '-' to read it from stdin.")
    parser.add_argument("--option", action="append", default=[], metavar="TEXT",
                        help="One multiple-choice option, repeatable: "
                             "--option 'A. ...' --option 'B. ...'")
    parser.add_argument("--accuracy", action="store_true",
                        help="Score multiple-choice accuracy on the held-out split.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Score only the first N held-out rows.")
    parser.add_argument("--show", type=int, default=0, metavar="N",
                        help="Print the first N generations in full, for eyeballing.")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="The answer sits AFTER a ~300-token explanation, so a "
                             "small value here scores as 'never answered' rather "
                             "than as wrong. Default 512.")
    parser.add_argument("--device", default=None, help="auto | cpu | mps | cuda")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Write the accuracy payload here as JSON.")
    args = parser.parse_args(argv)

    if not args.accuracy and not args.question:
        parser.error("nothing to do: pass --question or --accuracy")

    from kd.config import load_config, resolve_device
    from kd.runlog import discover_adapters, is_adapter

    config = load_config(args.config, use_env=False)
    if args.device:
        config.setdefault("hardware", {})["device"] = args.device
    hardware = resolve_device(config)

    adapter = args.adapter
    if not adapter:
        found = discover_adapters(config["project"].get("runs_dir") or "./runs")
        if not found:
            raise SystemExit(
                "xx  no adapter found. Train one, or name it with --adapter DIR.")
        adapter = found[0]
        log(f"==> newest adapter: {adapter}")
    if not is_adapter(adapter):
        raise SystemExit(f"xx  {adapter} has no adapter_config.json")

    model, tokenizer = load_student(config, adapter, hardware["device"],
                                    hardware["dtype"])

    if args.question:
        question = sys.stdin.read() if args.question == "-" else args.question
        prompt = build_prompt(question, args.option or None)
        text = generate(model, tokenizer, prompt, hardware["device"],
                        max_new_tokens=args.max_new_tokens, greedy=True)
        print(text)
        letter = extract_answer(text)
        if letter:
            log(f"==> parsed answer: {letter}")
        return 0

    rows = held_out_rows(config, tokenizer)
    if args.limit:
        rows = rows[:args.limit]
    log(f"==> scoring {len(rows)} held-out rows (greedy, "
        f"max_new_tokens={args.max_new_tokens})")

    payload = score(model, tokenizer, rows, hardware["device"],
                    args.max_new_tokens, show=args.show)
    payload["adapter"] = str(adapter)
    payload["config"] = args.config

    log("")
    log(f"  accuracy   : {payload['accuracy'] * 100:.1f}%  "
        f"({payload['correct']}/{payload['scored']})")
    log(f"  random     : 25.0%   (four options)")
    if payload["unparsed"]:
        log(f"  unanswered : {payload['unparsed']}  - never emitted <Answer>; "
            f"raise --max-new-tokens if this is not zero")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        log(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
