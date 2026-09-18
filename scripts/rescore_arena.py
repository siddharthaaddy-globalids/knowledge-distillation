#!/usr/bin/env python3
"""Re-read an arena transcript with the current parser and rebuild the numbers.

The completions in a transcript are the expensive, unrepeatable part of a run -
five models over a held-out set, hours of GPU. The letters scored off them are
cheap and derived. So when the parser is wrong, nothing needs re-generating: the
text is already on disk, and every number can be rebuilt from it.

That is what this does. It re-parses every completion in a transcript through
kd.arena's own `score_all` - the same function the run used, so there is exactly
one parser in play - and writes back:

    the transcript   answer / how / correct, corrected in place
    arena.json       accuracies, Elo, agreement, head-to-head, by-hop, rebuilt
    a report         the arena sections, rendered from the rebuilt payload

It reports what moved before it writes anything, and --dry-run stops there.

    python scripts/rescore_arena.py arena-transcript.jsonl --dry-run
    python scripts/rescore_arena.py arena-transcript.jsonl \
        --arena-json arena.json --report report.rescored.html

WHAT IT CANNOT REBUILD: the token-level sections kd.evaluate adds - perplexity,
KL divergence, throughput, the training settings. Those come from logits and
config, not from parsed letters, so a parser fix does not change them - but they
are also not in the transcript, so a report rebuilt here does not carry them.
Re-run `kd evaluate --report` if you need those in one document.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd.arena import score_all, summarise  # noqa: E402


# --------------------------------------------------------------------------- #
# transcript I/O - the file is either true JSONL or concatenated pretty-printed
# records, and whichever it was on the way in is what it must be on the way out.
# --------------------------------------------------------------------------- #


def load_transcript(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    decoder = json.JSONDecoder()
    records, index, end = [], 0, len(text)
    while index < end:
        while index < end and text[index] in " \r\n\t":
            index += 1
        if index >= end:
            break
        record, index = decoder.raw_decode(text, index)
        records.append(record)
    return records


def is_indented(path):
    """True when the file is pretty-printed rather than one record per line."""
    with open(path, encoding="utf-8") as handle:
        first = handle.readline().rstrip("\n\r")
    return first.strip() == "{"


def _scalar_list(value):
    return isinstance(value, list) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value)


def dump_record(obj, indent=4, level=0):
    """Pretty-print the way the pretty-printed transcripts are, scalar lists inline."""
    pad, pad2 = " " * (indent * level), " " * (indent * (level + 1))
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        body = ",\n".join(
            f"{pad2}{json.dumps(key, ensure_ascii=False)}: "
            f"{dump_record(value, indent, level + 1)}"
            for key, value in obj.items())
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if _scalar_list(obj):
            return json.dumps(obj, ensure_ascii=False)
        body = ",\n".join(pad2 + dump_record(item, indent, level + 1) for item in obj)
        return "[\n" + body + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def save_transcript(path, records, indented):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        if indented:
            handle.write("".join(dump_record(record) for record in records))
        else:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# rescoring
# --------------------------------------------------------------------------- #


def player_names(records):
    names = []
    for record in records:
        for name in record.get("players", {}):
            if name not in names:
                names.append(name)
    return names


def questions_from(records):
    """The shape score_all, by_hop and summarise want, rebuilt from the transcript."""
    return [{
        "prompt": record.get("prompt", ""),
        "gold": record.get("gold"),
        "hop": record.get("hop_count"),
        "item_id": record.get("item_id"),
    } for record in records]


def rescore(records, names):
    """(predictions, formats, unanswered, changes) - nothing written yet."""
    questions = questions_from(records)
    predictions, formats, unanswered, changes = {}, {}, {}, {}
    for name in names:
        said = [(record.get("players", {}).get(name) or {}).get("completion") or ""
                for record in records]
        picks, how, missed = score_all(questions, said, label=name)
        predictions[name], formats[name], unanswered[name] = picks, how, missed

        moved = []
        for record, pick, shape in zip(records, picks, how):
            entry = record.get("players", {}).get(name)
            if entry is None:
                continue
            was, was_how = entry.get("answer"), entry.get("how")
            if was != pick:
                moved.append({
                    "question": record.get("question"),
                    "gold": record.get("gold"),
                    "was": was, "was_how": was_how,
                    "now": pick, "now_how": shape,
                })
        changes[name] = moved
    return predictions, formats, unanswered, changes


def apply_to_records(records, predictions, formats, names):
    for name in names:
        for record, pick, shape in zip(records, predictions[name], formats[name]):
            entry = record.get("players", {}).get(name)
            if entry is None:
                continue
            entry["answer"] = pick
            entry["how"] = shape
            entry["correct"] = pick is not None and pick == record.get("gold")
    return records


def report_changes(records, names, changes, predictions):
    golds = [record.get("gold") for record in records]
    width = max(len(n) for n in names)
    header = (f"{'player':{width}}  {'was':>9}  {'now':>9}  {'moved':>6}  "
              f"{'rescued':>8}  {'letter changed':>15}")
    print(header)
    print("-" * len(header))
    for name in names:
        before = sum(1 for record in records
                     if (record["players"].get(name) or {}).get("correct"))
        after = sum(1 for pick, gold in zip(predictions[name], golds)
                    if pick is not None and pick == gold)
        rescued = sum(1 for c in changes[name] if c["was"] is None)
        swapped = sum(1 for c in changes[name] if c["was"] is not None)
        total = len(records)
        print(f"{name:{width}}  {f'{before}/{total}':>9}  {f'{after}/{total}':>9}  "
              f"{len(changes[name]):>6}  {rescued:>8}  {swapped:>15}")
    print()
    for name in names:
        if not changes[name]:
            continue
        print(f"  {name}: {len(changes[name])} answer(s) re-read")
        for change in changes[name]:
            verdict = ("now RIGHT" if change["now"] == change["gold"]
                       else ("now wrong" if change["was"] == change["gold"]
                             else "still wrong"))
            print(f"    q{change['question']:<5} "
                  f"{str(change['was']):>4} ({change['was_how'] or '-':<8}) -> "
                  f"{str(change['now']):>4} ({change['now_how'] or '-':<8})  "
                  f"gold {change['gold']}   {verdict}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-parse an arena transcript and rebuild arena.json and the report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("transcript", help="the arena transcript to re-read")
    parser.add_argument("--arena-json", default=None,
                        help="the arena.json written alongside it, to rebuild")
    parser.add_argument("--report", default=None,
                        help="where to write a rebuilt arena report (.html or .md)")
    parser.add_argument("--out-transcript", default=None,
                        help="default: <transcript>.rescored<ext>")
    parser.add_argument("--out-json", default=None,
                        help="default: <arena-json>.rescored.json")
    parser.add_argument("--in-place", action="store_true",
                        help="overwrite the transcript and arena.json instead")
    parser.add_argument("--players", default=None,
                        help="comma-separated subset to rescore (default: all in the file)")
    parser.add_argument("--match-json", action="store_true",
                        help="rebuild arena.json with only the players the old one had, "
                             "so the two stay directly comparable; by default every "
                             "player in the transcript is carried into it")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would change and write nothing")
    parser.add_argument("--elo-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    records = load_transcript(args.transcript)
    if not records:
        raise SystemExit(f"{args.transcript} is empty")
    indented = is_indented(args.transcript)
    names = player_names(records)
    if args.players:
        wanted = [n.strip() for n in args.players.split(",") if n.strip()]
        missing = [n for n in wanted if n not in names]
        if missing:
            raise SystemExit(f"no such player(s) in the transcript: {', '.join(missing)}")
        names = wanted

    print(f"{args.transcript}")
    print(f"  {len(records)} questions, players: {', '.join(names)}")
    print(f"  format: {'pretty-printed records' if indented else 'one record per line'}\n")

    predictions, formats, unanswered, changes = rescore(records, names)
    report_changes(records, names, changes, predictions)

    if not any(changes.values()):
        print("nothing moved - the transcript already agrees with the current parser.")
        return 0

    base, ext = os.path.splitext(args.transcript)
    out_transcript = args.out_transcript or (
        args.transcript if args.in_place else f"{base}.rescored{ext or '.jsonl'}")
    out_json = None
    if args.arena_json:
        jbase, jext = os.path.splitext(args.arena_json)
        out_json = args.out_json or (
            args.arena_json if args.in_place else f"{jbase}.rescored{jext or '.json'}")

    print("will write")
    print(f"  {out_transcript}")
    if out_json:
        print(f"  {out_json}")
    if args.report:
        print(f"  {args.report}")
    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    apply_to_records(records, predictions, formats, names)
    save_transcript(out_transcript, records, indented)
    print(f"\nwrote {out_transcript}")

    payload = None
    if args.arena_json:
        with open(args.arena_json, encoding="utf-8") as handle:
            old = json.load(handle)
        # Every player in the transcript, including ones a later script added.
        # Dropping them would be the surprising choice: the transcript is the
        # record of who played, and a column that silently vanishes from a
        # rebuilt report looks like a measurement that failed.
        #
        # --match-json is the other reading - keep the rebuilt file directly
        # comparable with the one it replaces, same players, same Elo pool.
        scored = names
        if args.match_json:
            known = old.get("players") or {}
            scored = [n for n in names if n in known] or names
            dropped = [n for n in names if n not in scored]
            if dropped:
                print(f"  --match-json: leaving out {', '.join(dropped)} - "
                      f"not in {os.path.basename(args.arena_json)}")
        print(f"  arena.json players: {', '.join(scored)}")
        questions = questions_from(records)
        payload = summarise(
            {n: predictions[n] for n in scored},
            [record.get("gold") for record in records],
            formats={n: formats[n] for n in scored},
            unanswered={n: unanswered[n] for n in scored},
            rounds=old.get("elo_rounds", args.elo_rounds), seed=args.seed,
            questions=questions,
            completions={n: [(record["players"].get(n) or {}).get("completion") or ""
                             for record in records] for n in scored},
        )
        # Provenance the transcript cannot know: which eval file, which adapter,
        # which engine, what the quantizer did. Carried across untouched.
        for key, value in old.items():
            if key not in payload:
                payload[key] = copy.deepcopy(value)
        payload["rescored_from"] = os.path.basename(args.transcript)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {out_json}")

    if args.report:
        if payload is None:
            raise SystemExit("--report needs --arena-json to rebuild the numbers from")
        from kd.report import write_report

        written = write_report({
            "arena": payload,
            "student": payload.get("student"),
            "teacher": payload.get("teacher"),
            "adapter": payload.get("adapter"),
        }, args.report)
        print(f"wrote {written}")
        print("  note: perplexity, KL, cost and training sections are not in the "
              "transcript and are not in this report; a parser fix does not change "
              "them - re-run `kd evaluate --report` if you need them together.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
