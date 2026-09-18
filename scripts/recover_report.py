#!/usr/bin/env python3
"""Rebuild the arena numbers and the report from a transcript alone.

The recovery path. A transcript holds every question and every word every player
said, so everything derived from it - the letters, the accuracies, Elo, the
agreement matrix, the report - can be rebuilt without a GPU, however many
players are in the file and whoever put them there.

Use it when:

  * the parser was wrong and the numbers on disk were scored with it;
  * arena.json or report.html was lost, overwritten, or never written;
  * players were added to the transcript after the run - an API model through
    openai_arena.py, a served one through modal_arena.py - and you want ONE
    report with every column in it, scored the same way.

    python scripts/recover_report.py arena-transcript.jsonl --report report.html

Everything is re-parsed through kd.arena's own parser, so every column is read
the same way, including ones an external script wrote with a parser of its own.

    --arena-json old.json   carry provenance across (eval file, adapter, engine,
                            quantization config) and reuse its Elo settings
    --players a,b,c         choose and order the columns; default is every
                            player in the transcript
    --out-json out.json     write the rebuilt payload as well
    --out-transcript t.jsonl  write the transcript back with corrected letters
    --dry-run               print what it found and write nothing

WHAT IT CANNOT REBUILD: perplexity, KL divergence, throughput and the training
settings come from logits and config, not from the transcript. A report built
here carries the answer-key sections only, and says so rather than printing
blanks. Re-run `kd evaluate --report` if you need those in one document.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from kd.arena import closeness, summarise  # noqa: E402
from kd.report import _columns, write_report  # noqa: E402
from rescore_arena import (  # noqa: E402
    apply_to_records, load_transcript, is_indented, player_names, questions_from,
    rescore, save_transcript,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild arena.json and the report from an arena transcript.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("transcript", help="the arena transcript to rebuild from")
    parser.add_argument("--report", default=None,
                        help="where to write the report; .html or .md "
                             "(default: <transcript>.report.html)")
    parser.add_argument("--arena-json", default=None,
                        help="an existing arena.json, for provenance and Elo settings")
    parser.add_argument("--out-json", default=None,
                        help="where to write the rebuilt payload (default: none)")
    parser.add_argument("--out-transcript", default=None,
                        help="write the transcript back with the re-parsed letters")
    parser.add_argument("--players", default=None,
                        help="comma-separated players, in the order the columns print "
                             "(default: every player in the transcript)")
    parser.add_argument("--teacher", default="teacher",
                        help="the player everything is measured against")
    parser.add_argument("--student", default=None, help="student model id, for the header")
    parser.add_argument("--teacher-model", default=None,
                        help="teacher model id, for the header")
    parser.add_argument("--elo-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print what was found and write nothing")
    args = parser.parse_args(argv)

    records = load_transcript(args.transcript)
    if not records:
        raise SystemExit(f"{args.transcript} is empty")
    present = player_names(records)
    names = present
    if args.players:
        names = [n.strip() for n in args.players.split(",") if n.strip()]
        missing = [n for n in names if n not in present]
        if missing:
            raise SystemExit(
                f"no such player(s) in the transcript: {', '.join(missing)}\n"
                f"  it has: {', '.join(present)}")

    golds = [record.get("gold") for record in records]
    ungraded = sum(1 for g in golds if not g)
    partial = [n for n in names
               if sum(1 for r in records if n in r.get("players", {})) != len(records)]

    print(f"{args.transcript}")
    print(f"  questions        {len(records)}"
          + (f"   ! {ungraded} with no gold letter" if ungraded else ""))
    print(f"  players in file  {len(present)}: {', '.join(present)}")
    print(f"  columns          {len(names)}: {', '.join(names)}")
    if partial:
        for name in partial:
            answered = sum(1 for r in records if name in r.get("players", {}))
            print(f"    ! {name} is on only {answered} of {len(records)} questions; "
                  "its cells are scored over the whole set")
    if args.teacher not in names:
        print(f"  ! no '{args.teacher}' column - closeness to the teacher will be "
              "omitted, everything else still builds")
    print()

    # Re-parse every column with one parser, so an external script's own reading
    # of its own model cannot differ from how the rest of the file was read.
    predictions, formats, unanswered, changes = rescore(records, names)
    width = max(len(n) for n in names)
    header = f"{'player':{width}}  {'on disk':>9}  {'re-parsed':>11}  {'moved':>6}"
    print(header)
    print("-" * len(header))
    for name in names:
        before = sum(1 for r in records
                     if (r.get("players", {}).get(name) or {}).get("correct"))
        after = sum(1 for pick, gold in zip(predictions[name], golds)
                    if pick is not None and pick == gold)
        print(f"{name:{width}}  {f'{before}/{len(records)}':>9}  "
              f"{f'{after}/{len(records)}':>11}  {len(changes[name]):>6}")
    moved = sum(len(v) for v in changes.values())
    print(f"\n  {moved} answer(s) read differently than the transcript recorded"
          if moved else "\n  every answer agrees with what the transcript recorded")

    old = {}
    if args.arena_json:
        with open(args.arena_json, encoding="utf-8") as handle:
            old = json.load(handle)

    payload = summarise(
        {n: predictions[n] for n in names}, golds,
        formats={n: formats[n] for n in names},
        unanswered={n: unanswered[n] for n in names},
        rounds=old.get("elo_rounds", args.elo_rounds), seed=args.seed,
        questions=questions_from(records),
        completions={n: [(r.get("players", {}).get(n) or {}).get("completion") or ""
                         for r in records] for n in names},
    )
    # summarise() measures closeness against kd.arena's own REFERENCE; redo it
    # when the reference player is called something else in this file.
    if args.teacher != "teacher":
        payload["closeness"] = closeness(payload, reference=args.teacher)
    for key, value in old.items():
        if key not in payload:
            payload[key] = copy.deepcopy(value)
    payload["rebuilt_from"] = os.path.basename(args.transcript)

    report = args.report or f"{os.path.splitext(args.transcript)[0]}.report.html"
    print("\nwill write")
    print(f"  {report}")
    if args.out_json:
        print(f"  {args.out_json}")
    if args.out_transcript:
        print(f"  {args.out_transcript}")
    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    if args.out_transcript:
        apply_to_records(records, predictions, formats, names)
        save_transcript(args.out_transcript, records, is_indented(args.transcript))
        print(f"\nwrote {args.out_transcript}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.out_json}")

    written = write_report({
        "arena": payload,
        "student": args.student or old.get("student"),
        "teacher": args.teacher_model or old.get("teacher"),
        "adapter": old.get("adapter"),
    }, report)
    print(f"wrote {written}")
    # The order the report prints them in, which is not the order they appear in
    # the transcript: the teacher is held last so the tables' colouring keeps
    # meaning "started here, moved to there, aimed at that".
    rendered = [header for _name, header in _columns({"arena": payload})]
    print(f"  {len(rendered)} columns: {', '.join(rendered)}")
    print("  answer-key sections only - perplexity, KL, cost and the training "
          "settings are not in a transcript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
