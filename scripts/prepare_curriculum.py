#!/usr/bin/env python3
# ===========================================================================
#  Turn the enLibra curriculum JSON exports into chat JSONL the trainer reads.
#
#      python scripts/prepare_curriculum.py \
#          --out data/enlibra-curriculum \
#          ~/Downloads/curriculum_sft.json \
#          ~/Downloads/curriculum_sft_stage_2.json \
#          ~/Downloads/curriculum_rl.json
#
#  The exports are a bespoke multiple-choice shape - `question`, `answer`,
#  `explanation`, plus knowledge-graph bookkeeping - and kd.data understands
#  only chat turns or Alpaca triples. This converts once, to a file you can
#  read, diff and ship, rather than hiding the transform inside the loader.
#
#  WHAT THE SPLIT IS, and why it is not invented here
#  --------------------------------------------------
#  Every row already carries `question_and_explanation`, which is the full text
#  the curriculum authors wrote:
#
#      <Question>...</Question>
#      <Options>A. ... D. ...</Options>     <- the prompt
#      <Explanation>...</Explanation>
#      <Answer>:
#      D
#      </Answer>                            <- the completion
#
#  and it begins with `question` verbatim. So the prompt/completion boundary is
#  a prefix strip, not a format this script chose. Rows where that invariant
#  does not hold fall back to assembling the same shape from `explanation` and
#  `answer`, and are counted separately so a silent change of export format
#  shows up as a number rather than as a worse student.
#
#  Identity rows (`_identity`) are a different shape - a plain question and a
#  plain answer, no explanation - and are emitted as an ordinary exchange. The
#  exports repeat each of them eight times; duplicates are dropped, because
#  kd.data's prompt deduplication would drop them anyway and a quota measured
#  against 120 rows that are really 15 is a quota that lies.
#
#  Output is one .jsonl per input file, named after `split`/`stage` rather than
#  after the input filename, plus a manifest.json recording what was read and
#  what was dropped. Rows are sorted by item_id first: the trainer's own split
#  is seeded, and feeding it input in filesystem order would make that seed
#  meaningless.
# ===========================================================================

import argparse
import hashlib
import json
import os
import sys

# Rows carrying these `split` values never join the training corpus. They are
# written to their own file and reported as held out, so that pointing a domain
# at one takes a deliberate edit rather than a slip.
#
# `eval` only. `rl` was held out while it was the closest thing to an unseen set;
# once curriculum_verified.json supplied a real one, keeping rl out of training
# cost 13% of the corpus for nothing - there is no RL trainer here, and its rows
# are the same multiple-choice shape as the rest.
HELD_OUT_SPLITS = {"eval"}


def log(message):
    print(message, file=sys.stderr)


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"xx  {path} is not a JSON list of rows")
    return data


def completion_text(row):
    """The assistant turn, and how it was derived.

    Returns (text, source) where source is 'canonical' when the export's own
    `question_and_explanation` supplied it and 'assembled' when it had to be
    rebuilt. The caller counts those separately - see the module docstring.
    """
    question = str(row.get("question") or "")
    full = str(row.get("question_and_explanation") or "")

    if question and full.startswith(question):
        tail = full[len(question):].lstrip("\n")
        if tail:
            return tail, "canonical"

    explanation = str(row.get("explanation") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not explanation or not answer:
        return None, "incomplete"
    return (f"<Explanation>\n{explanation}\n</Explanation>\n"
            f"<Answer>:\n{answer}\n</Answer>"), "assembled"


def mcq_exchange(row, system=None):
    """One multiple-choice row as chat turns, or None when it is unusable."""
    question = str(row.get("question") or "").strip()
    if not question:
        return None, "no-question"

    completion, source = completion_text(row)
    if completion is None:
        return None, source

    # The letter in the completion has to agree with the `answer` column. A row
    # where they disagree teaches the student to contradict its own reasoning,
    # which is worse than a row that is simply missing.
    answer = str(row.get("answer") or "").strip()
    if len(answer) == 1 and f"\n{answer}\n" not in completion:
        return None, "answer-mismatch"

    turns = []
    if system:
        turns.append({"role": "system", "content": system})
    turns.append({"role": "user", "content": question})
    turns.append({"role": "assistant", "content": completion})
    return {"messages": turns}, source


def identity_exchange(row, system=None):
    """One `_identity` row - a plain question and a plain answer."""
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer:
        return None, "incomplete"
    turns = []
    if system:
        turns.append({"role": "system", "content": system})
    turns.append({"role": "user", "content": question})
    turns.append({"role": "assistant", "content": answer})
    return {"messages": turns}, "identity"


def stable_key(row):
    """A deterministic sort key, so the same inputs always produce the same file.

    item_id when the row has one. Identity rows do not, so they are keyed by a
    hash of the question - which also gives the duplicate-drop below something
    to compare on.
    """
    item_id = row.get("item_id")
    if item_id:
        return str(item_id)
    return hashlib.sha1(
        str(row.get("question", "")).encode("utf-8")).hexdigest()


def convert(path, system=None, require_verdict=True, identity_sink=None, seen=None):
    """Convert one export, grouped by its `split` column.

    Returns ({split: {"records": [...], "rows": [...]}}, stats).

    Grouping here rather than relying on one file per split is what lets a single
    combined export - curriculum_verified.json carries sft, rl and eval together -
    still produce one corpus file per split, which is the unit a domain or a
    held-out set addresses.

    Identity rows go to `identity_sink` rather than into any group, so they end up
    in a file of their own. They are a different thing from the curriculum - a
    persona, not knowledge - there are only fifteen distinct ones against a
    thousand questions, and mixing them into a curriculum file makes that ratio
    invisible. Given their own domain it is a quota in the config, which is where
    a decision about the mix belongs.

    `seen` is shared across every input, deliberately. Exports overlap:
    curriculum_verified.json is a superset of the three stage files, so processing
    both without a shared record of what has been emitted would write every
    question twice.
    """
    rows = load_rows(path)
    stats = {"read": len(rows), "written": 0, "canonical": 0, "assembled": 0,
             "identity": 0, "duplicate": 0, "unverified": 0, "dropped": {}}

    groups = {}
    seen = seen if seen is not None else set()
    for row in sorted(rows, key=stable_key):
        is_identity = "_identity" in row or not row.get("stage")

        # pair_verdict is the curriculum's own quality gate. A row it marked
        # false is one its authors did not stand behind; training on it spends
        # the budget arguing with them.
        if not is_identity and require_verdict and row.get("pair_verdict") is not True:
            stats["unverified"] += 1
            continue

        record, source = (identity_exchange(row, system) if is_identity
                          else mcq_exchange(row, system))
        if record is None:
            stats["dropped"][source] = stats["dropped"].get(source, 0) + 1
            continue

        key = record["messages"][-2]["content"].strip()

        if is_identity and identity_sink is not None:
            # The exports repeat each identity row eight times, and the same
            # persona questions appear across several exports. Deduplicated
            # against the sink rather than this file's `seen`, so the count is
            # right no matter how many exports carry them.
            if key in identity_sink:
                stats["duplicate"] += 1
                continue
            identity_sink[key] = record
            stats["identity"] += 1
            continue

        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)

        group = groups.setdefault(str(row.get("split") or "unsplit"),
                                  {"records": [], "rows": []})
        group["records"].append(record)
        group["rows"].append(row)
        stats[source] = stats.get(source, 0) + 1
        stats["written"] += 1

    return groups, stats


def output_name(path, rows):
    """Name the output after what the rows say they are, not after the input file.

    A file called `curriculum_sft_stage_2.json` whose rows say `split: sft` and
    `hop_count: 2` becomes `sft-2hop.jsonl`. The config then names a corpus by
    what it contains, which survives the next export being called something else.
    """
    splits = {str(r.get("split")) for r in rows if r.get("split")}
    hops = {int(r["hop_count"]) for r in rows if isinstance(r.get("hop_count"), int)}

    split = splits.pop() if len(splits) == 1 else None
    if not split:
        return os.path.splitext(os.path.basename(path))[0].replace("_", "-")

    if hops and hops != {1}:
        return f"{split}-{min(hops)}to{max(hops)}hop" if len(hops) > 1 \
            else f"{split}-{hops.pop()}hop"
    return f"{split}-1hop" if hops else split


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert enLibra curriculum JSON exports to chat JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("inputs", nargs="+", metavar="FILE",
                        help="curriculum_*.json exports")
    parser.add_argument("--out", default="data/enlibra-curriculum", metavar="DIR",
                        help="Directory to write the .jsonl corpus into "
                             "(default: data/enlibra-curriculum)")
    parser.add_argument("--system", default=None, metavar="TEXT",
                        help="Prepend this system turn to every exchange. Off by "
                             "default: the questions are self-describing, and a "
                             "system prompt at training time has to be repeated "
                             "at inference time or the student sees a prompt it "
                             "was never trained on.")
    parser.add_argument("--keep-unverified", action="store_true",
                        help="Keep rows whose pair_verdict is not true "
                             "(not recommended)")
    parser.add_argument("--stats-tokenizer", default=None, metavar="ID",
                        help="Measure token lengths with this tokenizer, e.g. "
                             "Qwen/Qwen3-8B, and print what the budget must be. "
                             "Needs transformers and a download of the tokenizer.")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"inputs": [], "files": {}}
    written_paths = []
    # Keyed by prompt, so the same persona question appearing in two exports is
    # kept once. Insertion order is the sorted order it was read in.
    identity_sink = {}
    seen = set()

    for path in args.inputs:
        if not os.path.isfile(path):
            raise SystemExit(f"xx  no such file: {path}")
        rows = load_rows(path)
        groups, stats = convert(path, system=args.system,
                                require_verdict=not args.keep_unverified,
                                identity_sink=identity_sink, seen=seen)
        # Zero groups is fine when the file still contributed persona rows: a
        # combined export processed first leaves the stage files with nothing new
        # to say except their identity rows, and that is the intended outcome
        # rather than a failure.
        if not groups and not stats["identity"]:
            raise SystemExit(
                f"xx  {path} produced no usable rows and no identity rows.\n"
                f"    Every question in it was already written by an earlier "
                f"input, or none passed the verdict filter.")

        manifest["inputs"].append({"path": os.path.abspath(path),
                                   "sha1": _sha1(path), "rows": len(rows)})
        log(f"==> {os.path.basename(path)}  ({len(rows)} rows)")
        if stats["unverified"]:
            log(f"      {stats['unverified']} dropped: pair_verdict not true")
        if stats["duplicate"]:
            log(f"      {stats['duplicate']} dropped: already written, or a "
                f"repeat inside this file")
        for reason, count in sorted(stats["dropped"].items()):
            log(f"      {count} dropped: {reason}")

        for split in sorted(groups):
            group = groups[split]
            name = output_name(path, group["rows"])
            target = os.path.join(args.out, f"{name}.jsonl")
            with open(target, "w", encoding="utf-8") as handle:
                for record in group["records"]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_paths.append(target)

            held_out = split in HELD_OUT_SPLITS
            manifest["files"][f"{name}.jsonl"] = {
                "written": len(group["records"]), "split": split,
                "held_out": held_out}
            log(f"      -> {target}   {len(group['records'])} rows"
                + ("   HELD OUT" if held_out else ""))
            if held_out:
                log(f"         keep this out of dataset.domains - it is what the "
                    f"student is scored on")

    if identity_sink:
        target = os.path.join(args.out, "identity.jsonl")
        with open(target, "w", encoding="utf-8") as handle:
            for record in identity_sink.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_paths.append(target)
        manifest["files"]["identity.jsonl"] = {"written": len(identity_sink),
                                               "held_out": False}
        log(f"==> identity")
        log(f"      -> {target}")
        log(f"      {len(identity_sink)} distinct persona exchanges")
        log(f"      give this its own domain and quota - fifteen rows against a "
            f"thousand questions is ~1.4% of the mix")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    log(f"==> manifest: {os.path.join(args.out, 'manifest.json')}")

    if args.stats_tokenizer:
        _report_budget(written_paths, args.stats_tokenizer)
    return 0


def _sha1(path):
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_budget(paths, tokenizer_id):
    """Print the token budget the converted corpus actually needs.

    kd.data filters every sample against dataset.max_prompt_tokens and
    max_total_tokens, and a sample over either is dropped silently. Set them
    below what this prints and the run trains on a fraction of the corpus, or
    on nothing at all.
    """
    from transformers import AutoTokenizer

    log(f"\n==> measuring with {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    prompt_lengths, total_lengths = [], []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                turns = json.loads(line)["messages"]
                prompt = tokenizer.apply_chat_template(
                    turns[:-1], tokenize=False, add_generation_prompt=True)
                full = tokenizer.apply_chat_template(
                    turns, tokenize=False, add_generation_prompt=False)
                prompt_lengths.append(
                    len(tokenizer(prompt, add_special_tokens=False).input_ids))
                total_lengths.append(
                    len(tokenizer(full, add_special_tokens=False).input_ids))

    def line(label, values):
        ordered = sorted(values)
        p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
        log(f"      {label:14} median {ordered[len(ordered) // 2]:5d}   "
            f"p99 {p99:5d}   max {ordered[-1]:5d}")

    line("prompt", prompt_lengths)
    line("prompt+answer", total_lengths)

    # A margin over the observed maximum, rounded up to a multiple of 64. The
    # filter is strict - `>=` on the prompt budget - so sitting exactly on the
    # longest sample would drop it.
    def ceiling(values):
        return ((max(values) + 32) // 64 + 1) * 64

    log(f"\n      dataset.max_prompt_tokens: {ceiling(prompt_lengths)}")
    log(f"      dataset.max_total_tokens:  {ceiling(total_lengths)}")


if __name__ == "__main__":
    sys.exit(main())
