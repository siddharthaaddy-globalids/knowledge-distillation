"""
Readable evaluation reports - Markdown and self-contained HTML.

Separated from the measurement code because they answer to different readers.
kd.evaluate produces numbers; this module turns them into something a person who
did not run the job can act on, leading with a plain-English summary so nobody
needs to know what perplexity is to learn whether the training worked.

Every claim in the summary is derived from the measurements rather than asserted.
The HTML is a single file with no external assets, so it can be opened offline or
sent to someone as-is.
"""

import math
import pathlib
from datetime import datetime

# --------------------------------------------------------------------------- #
# Readable report
# --------------------------------------------------------------------------- #
def _arena_summary(arena):
    """The plain-English reading when the answer key is all there is.

    `kd arena` scores who was RIGHT without loading the machinery kd.evaluate
    needs to say how closely the student tracks the teacher token by token. That
    is a smaller report, not a broken one, so it gets its own summary rather
    than a page of em-dashes where the fidelity numbers would have been.
    """
    players = arena.get("players") or {}
    base, dist = players.get("base"), players.get("distilled")
    teacher, total = players.get("teacher"), arena.get("questions", 0)
    lines = []

    # First line, before any number, when the run was limited. Everything below
    # is arithmetic on a handful of questions and would read as a result
    # otherwise - and a file on disk outlives the memory of which flags produced
    # it.
    if arena.get("available") and arena.get("limited_to"):
        lines.append(
            f"NOT THE SCORE. This run was limited to {arena['limited_to']} of "
            f"{arena['available']} held-out questions, which is enough to prove "
            f"the stage runs and far too few to rank three models. Re-run "
            f"without --limit for a number worth quoting.")

    if not (base and dist):
        return lines + ["Scored on a held-out answer key."]

    pct = lambda e: (e.get("accuracy") or 0) * 100
    moved = pct(dist) - pct(base)
    verb = ("is ahead of" if moved > 0 else
            "is level with" if moved == 0 else "is behind")
    lines.append(
        f"On {total} held-out questions the distilled student answered "
        f"{pct(dist):.1f}% correctly, against {pct(base):.1f}% for the same model "
        f"before training - it {verb} where it started, by "
        f"{abs(moved):.1f} points.")

    if teacher:
        gap = pct(teacher) - pct(base)
        if gap > 0:
            closed = moved / gap * 100
            lines.append(
                f"The teacher scores {pct(teacher):.1f}%, so the gap training had "
                f"to close was {gap:.1f} points and it closed {closed:.0f}% of it.")
        else:
            lines.append(
                f"The teacher scores {pct(teacher):.1f}%, at or below the untrained "
                f"student - so on this set there was no gap to close, and the "
                f"comparison says more about the questions than about the models.")

    # Answered-vs-correct kept separate, because a model that never produces a
    # parseable letter scores 0% for a reason that has nothing to do with what
    # it knows - and that is a fixable problem, unlike being wrong.
    silent = [n for n, e in players.items() if e.get("answered", 0) < total * 0.9]
    if silent:
        lines.append(
            "Read the answered row before the accuracy row: "
            + ", ".join(f"{n} produced a parseable answer on "
                        f"{players[n].get('answered', 0)} of {total}"
                        for n in sorted(silent))
            + ". Accuracy counts the rest as wrong.")
    return lines


def plain_summary(payload):
    """Plain-English reading of the numbers, for someone who did not run the job.

    Every claim here is derived from the measurements, not asserted: the point is
    that a reader should not have to know what perplexity is to learn whether the
    training worked.
    """
    # An arena-only payload: `kd arena --report` writes one, and it carries the
    # answer key without any of the token-level measurements below.
    if "fidelity" not in payload:
        return _arena_summary(payload.get("arena") or {})

    fid, cap = payload["fidelity"], payload["capability"]
    lift = fid["agreement_lift_pts"]
    recovered = cap.get("gap_recovered_pct")
    lines = []

    if lift > 0 and cap["perplexity_distilled"] < cap["perplexity_base"]:
        lines.append(
            f"The training worked. The student now picks the same next word as the "
            f"teacher {fid['top1_agreement_distilled_pct']:.1f}% of the time, up from "
            f"{fid['top1_agreement_base_pct']:.1f}% before training.")
    else:
        lines.append(
            "The training did not move the student toward the teacher. The numbers "
            "below are at or behind where the untrained student started.")

    if isinstance(recovered, (int, float)) and math.isfinite(recovered):
        lines.append(
            f"Think of the teacher as a finish line and the untrained student as 100 "
            f"steps behind it. Training moved the student {recovered:.0f} of those "
            f"100 steps.")

    lines.append(
        f"It is also less surprised by real text in this domain than before "
        f"(perplexity {cap['perplexity_base']:.1f} to {cap['perplexity_distilled']:.1f}; "
        f"the teacher scores {cap['perplexity_teacher']:.1f}, and lower is better).")

    sim = payload.get("generation_similarity") or {}
    bs_b, bs_d = sim.get("bertscore_f1_base"), sim.get("bertscore_f1_distilled")
    rl_b, rl_d = sim.get("rougeL_base"), sim.get("rougeL_distilled")
    if all(isinstance(v, (int, float)) for v in (bs_b, bs_d, rl_b, rl_d)) and bs_b and rl_b:
        words = (rl_d - rl_b) / rl_b * 100
        meaning = (bs_d - bs_b) / bs_b * 100
        lines.append(
            f"When both models write a full answer on their own, the student's "
            f"WORDING moved {words:+.0f}% toward the teacher's while its MEANING "
            f"moved {meaning:+.0f}%.")
        if words > 2 * meaning:
            lines.append(
                "Wording moved far more than meaning. That is the signature of "
                "distillation transferring style and structure rather than "
                "knowledge - normal for a short run, and worth knowing if you "
                "needed the student to learn facts it did not already have.")

    eff = payload["efficiency"]
    if eff["teacher_params"]:
        ratio = eff["student_params"] / eff["teacher_params"]
        speed = ((eff["distilled_tok_per_s"] / eff["teacher_tok_per_s"])
                 if eff.get("teacher_tok_per_s") else None)
        tail = f" and runs {speed:.1f}x faster" if speed else ""
        lines.append(f"It does this at {ratio:.0%} of the teacher's size{tail}.")
    return lines


# What the four columns mean unless a section says otherwise. Most sections
# compare the three models; the similarity table compares three PAIRS of them,
# and labelling its columns with model names would misdescribe every cell.
DEFAULT_HEADERS = ("Metric", "Base student", "Distilled", "Teacher")


def _sections(payload):
    """(title, rows, headers) for every section, headers defaulted."""
    return [(s[0], s[1], s[2] if len(s) > 2 else DEFAULT_HEADERS)
            for s in _report_rows(payload)]


def _report_rows(payload):
    """(section, [(label, base, distilled, teacher)]) for both report formats.

    A section may carry a third element, its own column headers, for a table
    whose columns are not the three models.
    """
    fid = payload.get("fidelity")
    cap = payload.get("capability")
    eff = payload.get("efficiency")
    close = payload.get("closeness_to_teacher") or {}
    fmt = lambda v, spec=".4f": (format(v, spec)
                                 if isinstance(v, (int, float)) and math.isfinite(v)
                                 else "-")
    sections = []

    # The answer key first when there is one. Fidelity below says how closely the
    # student tracks the teacher; this says who was RIGHT, which is the question
    # anyone outside the project asks first.
    arena = payload.get("arena") or {}
    players = arena.get("players") or {}
    if players:
        total = arena.get("questions", 0)
        order = [n for n in ("base", "distilled", "teacher") if n in players]
        pct = lambda v: (f"{v * 100:.1f}%" if isinstance(v, float) else "-")

        def three(cell):
            values = {n: cell(players[n]) for n in order}
            return (values.get("base", "-"), values.get("distilled", "-"),
                    values.get("teacher", "-"))

        rows = [
            ("Produced a parseable answer",
             *three(lambda e: f"{e['answered']} / {total}")),
            (f"Correct, counting all {total}",
             *three(lambda e: pct(e.get("accuracy")))),
            ("Correct, when it answered",
             *three(lambda e: pct(e.get("accuracy_when_answered")))),
            ("Answered in the trained <Answer> format",
             *three(lambda e: f"{e.get('in_trained_format', 0)} / {total}")),
            ("Elo", *three(lambda e: f"{e['elo']:.0f}  ±{e['elo_spread']:.0f}")),
        ]
        # Agreement is pairwise, and this table has one column per player - so
        # report the pair that matters and let the columns do the work: how
        # often each student picked the TEACHER's letter. That is what
        # distillation is supposed to move, and a row per unordered pair would
        # have left two thirds of every row empty to say it.
        agreement = arena.get("agreement") or {}

        def against_teacher(name):
            entry = (agreement.get(f"{name} vs teacher")
                     or agreement.get(f"teacher vs {name}"))
            if not entry:
                return "-"
            return f"{entry['same']} / {entry['of']}  ({pct(entry['pct'])})"

        if agreement and "teacher" in players:
            rows.append(("Chose the teacher's letter",
                         against_teacher("base"), against_teacher("distilled"),
                         f"{total} / {total}  (100.0%)"))
        subset = (f" — a SUBSET of {arena['available']}, not the score"
                  if arena.get("available") and arena.get("limited_to") else "")
        sections.append(
            (f"The answer key — {total} held-out questions{subset} "
             f"(random baseline {arena.get('random_baseline', 0.25) * 100:.0f}%)",
             rows))

    # Hop-wise cosine between the players' EXPLANATIONS. Computed by the arena,
    # and until now visible only in the terminal - which meant the one table
    # that says whether the student reasons like its teacher never reached the
    # file people actually read.
    #
    # Shaped to the four-column table like everything else: one row per hop,
    # the three columns being the three pairs rather than the three players.
    # The header is overridden below, because "Base student / Distilled /
    # Teacher" would be a lie about what these numbers compare.
    asim = arena.get("similarity") or {}
    if asim.get("pairs"):
        pairs = asim["pairs"]
        want = ("base vs teacher", "distilled vs teacher", "base vs distilled")

        def cell(pair, hop=None):
            entry = pairs.get(pair) or pairs.get(" vs ".join(reversed(pair.split(" vs "))))
            if not entry:
                return "-"
            value = (entry.get("overall") if hop is None
                     else (entry.get("by_hop", {}).get(hop) or {}).get("cosine"))
            return fmt(value, ".3f")

        rows = []
        for hop in asim.get("hops") or []:
            counted = ((pairs.get(want[0]) or {}).get("by_hop", {}).get(hop) or {})
            rows.append((f"{hop} hop  ({counted.get('n', 0)} questions)",
                         *(cell(p, hop) for p in want)))
        rows.append(("All questions", *(cell(p) for p in want)))
        sections.append((
            "How alike are the explanations? (cosine 0-1, by reasoning depth)",
            rows,
            ("Reasoning depth", "Base vs teacher", "Distilled vs teacher",
             "Base vs distilled")))

    if not (fid and cap and eff):
        return sections

    sections += [
        ("How close is the student to the teacher?", [
            ("Prediction agreement",
             fmt(close.get("prediction_agreement_base_pct"), ".2f") + "%",
             fmt(close.get("prediction_agreement_distilled_pct"), ".2f") + "%", "100%"),
            ("Perplexity retention",
             fmt(close.get("perplexity_retention_base_pct"), ".2f") + "%",
             fmt(close.get("perplexity_retention_distilled_pct"), ".2f") + "%", "100%"),
        ]),
        ("Fidelity - does it predict what the teacher predicts?", [
            ("Top-1 agreement with teacher",
             fmt(fid["top1_agreement_base_pct"], ".2f") + "%",
             fmt(fid["top1_agreement_distilled_pct"], ".2f") + "%", "100%"),
            ("KL divergence from teacher (lower is better)",
             fmt(fid["kl_base"]), fmt(fid["kl_distilled"]), "0"),
        ]),
        ("Capability - is it better at the task?", [
            ("Held-out perplexity (lower is better)",
             fmt(cap["perplexity_base"], ".3f"), fmt(cap["perplexity_distilled"], ".3f"),
             fmt(cap["perplexity_teacher"], ".3f")),
        ]),
        ("Cost", [
            ("Parameters", "-", f"{eff['student_params'] / 1e9:.3f}B",
             f"{eff['teacher_params'] / 1e9:.3f}B"),
            ("Decode throughput (tokens/sec)", "-",
             fmt(eff.get("distilled_tok_per_s"), ".1f"),
             fmt(eff.get("teacher_tok_per_s"), ".1f")),
            ("Trainable adapter parameters", "-",
             f"{eff['adapter_params'] / 1e6:.2f}M", "-"),
        ]),
    ]

    sim = payload.get("generation_similarity")
    if sim:
        rows = []
        if "bertscore_f1_base" in sim:
            rows.append(("BERTScore vs teacher (meaning)",
                         fmt(sim["bertscore_f1_base"]),
                         fmt(sim["bertscore_f1_distilled"]), "1.0"))
        if "rougeL_base" in sim:
            rows.append(("ROUGE-L vs teacher (wording)",
                         fmt(sim["rougeL_base"]), fmt(sim["rougeL_distilled"]), "1.0"))
        if rows:
            sections.insert(3, (
                f"Free-running similarity - both models writing on their own "
                f"({sim.get('prompts', '?')} prompts)", rows))
    return sections


# Model colours are semantic, not decorative: grey is where the student started,
# teal is where it moved to, indigo is the target it was moving toward. The same
# three colours carry that meaning in the gap bars and in every table.
_REPORT_CSS = """
:root{
  --paper:#FBFBFD; --ink:#14181F; --muted:#626A78; --rule:#E4E6EC; --card:#F3F4F8;
  --inert:#A6ADBA; --accent:#0F6E68; --target:#3B4A7A; --track:#EAECF1;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#12151B; --ink:#E9EBEF; --muted:#98A0AE; --rule:#262B34; --card:#191D25;
    --inert:#5C6675; --accent:#4FBFB4; --target:#8494C8; --track:#222731;
  }
}
:root[data-theme="dark"]{
  --paper:#12151B; --ink:#E9EBEF; --muted:#98A0AE; --rule:#262B34; --card:#191D25;
  --inert:#5C6675; --accent:#4FBFB4; --target:#8494C8; --track:#222731;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 16px/1.65 "IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:50rem;margin:0 auto;padding:3.5rem 1.5rem 4rem;
  display:flex;flex-direction:column;gap:2.75rem}
.eyebrow{font:500 .72rem/1 "IBM Plex Mono",ui-monospace,monospace;
  letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 .9rem}
h1{font:600 2.5rem/1.1 Newsreader,Georgia,serif;margin:0;text-wrap:balance;
  letter-spacing:-.01em}
.lede{color:var(--muted);margin:.5rem 0 0;font-size:1.02rem}
h2{font:600 1.15rem/1.3 Newsreader,Georgia,serif;margin:0 0 1rem;text-wrap:balance}
header{border-bottom:1px solid var(--rule);padding-bottom:2rem}

.verdict{display:grid;grid-template-columns:minmax(8.5rem,auto) 1fr;gap:2rem;
  align-items:start;background:var(--card);border-radius:10px;padding:1.6rem 1.7rem}
.big{font:600 3.4rem/1 "IBM Plex Mono",ui-monospace,monospace;color:var(--accent);
  font-variant-numeric:tabular-nums;letter-spacing:-.03em}
.big span{display:block;font:400 .78rem/1.4 "IBM Plex Sans",sans-serif;
  color:var(--muted);margin-top:.5rem;letter-spacing:0}
.verdict p{margin:0 0 .75rem}
.verdict p:last-child{margin-bottom:0}

.bar{margin-bottom:1.9rem}
.bar:last-child{margin-bottom:0}
.bar-h{display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:.55rem;font-size:.9rem}
.bar-h b{font-weight:500}
.bar-h em{font-style:normal;color:var(--muted);
  font:400 .82rem/1 "IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
.track{position:relative;height:.55rem;border-radius:99px;background:var(--track);
  overflow:hidden}
.seg-base{position:absolute;inset-block:0;left:0;background:var(--inert);
  border-radius:99px 0 0 99px}
.seg-gain{position:absolute;inset-block:0;background:var(--accent)}
.ticks{position:relative;height:1.35rem;margin-top:.4rem;
  font:400 .72rem/1 "IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
.tick{position:absolute;transform:translateX(-50%);white-space:nowrap}
.tick.b{color:var(--muted)} .tick.d{color:var(--accent);font-weight:500}
.tick.t{color:var(--target);right:0;transform:none}

.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.93rem}
caption{text-align:left;color:var(--muted);font-size:.85rem;padding-bottom:.6rem}
th,td{padding:.6rem .7rem;border-bottom:1px solid var(--rule);text-align:right;
  font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{font:500 .72rem/1.3 "IBM Plex Mono",ui-monospace,monospace;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  border-bottom-color:var(--ink)}
tbody td:not(:first-child){font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.88rem}
.c-b{color:var(--muted)} .c-d{color:var(--accent);font-weight:600}
.c-t{color:var(--target)}
tbody tr:last-child td{border-bottom:none}

dl{display:grid;grid-template-columns:max-content 1fr;gap:.45rem 1.4rem;
  margin:0;font-size:.89rem}
dt{color:var(--muted)}
dd{margin:0;word-break:break-word;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.84rem}
footer{border-top:1px solid var(--rule);padding-top:1.25rem;color:var(--muted);
  font-size:.85rem}
@media (max-width:34rem){
  h1{font-size:1.9rem}
  .verdict{grid-template-columns:1fr;gap:1.1rem}
  .big{font-size:2.6rem}
}
"""


def _render_html(payload, facts, sections, summary):
    """Self-contained report page. No external assets beyond Google Fonts."""
    esc = lambda t: (str(t).replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;").replace('"', "&quot;"))
    close = payload.get("closeness_to_teacher") or {}
    recovered = (payload.get("capability") or {}).get("gap_recovered_pct")

    def bar(label, base_pct, dist_pct):
        """base -> distilled -> teacher(=100) on one track.

        The teal segment is exactly the distance training moved, which is the
        quantity the whole report exists to communicate.
        """
        if not all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in (base_pct, dist_pct)):
            return ""
        lo, hi = max(0.0, min(base_pct, 100)), max(0.0, min(dist_pct, 100))
        gain = max(0.0, hi - lo)
        return (
            f'<div class="bar"><div class="bar-h"><b>{esc(label)}</b>'
            f'<em>{lo:.1f}% &rarr; {hi:.1f}%</em></div>'
            f'<div class="track"><div class="seg-base" style="width:{lo:.2f}%"></div>'
            f'<div class="seg-gain" style="left:{lo:.2f}%;width:{gain:.2f}%"></div></div>'
            f'<div class="ticks"><span class="tick b" style="left:{lo:.2f}%">base</span>'
            f'<span class="tick d" style="left:{hi:.2f}%">distilled</span>'
            f'<span class="tick t">teacher</span></div></div>')

    bars = "".join([
        bar("Predicts the same next word as the teacher",
            close.get("prediction_agreement_base_pct"),
            close.get("prediction_agreement_distilled_pct")),
        bar("Understands the domain text as well as the teacher",
            close.get("perplexity_retention_base_pct"),
            close.get("perplexity_retention_distilled_pct")),
    ])

    head = [
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600'
        '&family=Newsreader:opsz,wght@6..72,600&display=swap">',
        f"<style>{_REPORT_CSS}</style>",
    ]

    student = str(payload.get("student") or "the student").split("/")[-1]
    teacher = str(payload.get("teacher") or "").split("/")[-1]
    body = ['<div class="wrap"><header>',
            '<p class="eyebrow">Knowledge distillation &middot; evaluation</p>',
            f'<h1>{esc(student)}</h1>',
            (f'<p class="lede">Distilled from <strong>{esc(teacher)}</strong>, '
             f'then measured against it and against its own untrained self.</p>'
             if teacher else
             '<p class="lede">Measured against its own untrained self on a '
             'held-out answer key.</p>'),
            '</header>']

    # The headline number is the share of the teacher gap that training closed.
    # kd.evaluate measures it token by token; with only an arena payload the
    # same quantity is available from the accuracy columns, and the caption says
    # which one is on the page rather than leaving them to look identical.
    arena_players = (payload.get("arena") or {}).get("players") or {}
    caption = "of the distance to the teacher, closed by training"
    if not isinstance(recovered, (int, float)) or not math.isfinite(recovered):
        recovered = None
        base, dist, tea = (arena_players.get("base"), arena_players.get("distilled"),
                           arena_players.get("teacher"))
        if base and dist and tea:
            gap = (tea.get("accuracy") or 0) - (base.get("accuracy") or 0)
            if gap > 0:
                recovered = ((dist.get("accuracy") or 0)
                             - (base.get("accuracy") or 0)) / gap * 100
                caption = "of the teacher's lead on the answer key, closed by training"

    big = (f"{recovered:.0f}%" if isinstance(recovered, (int, float))
           and math.isfinite(recovered) else "&mdash;")
    body.append(f'<section class="verdict"><div><div class="big">{big}'
                f'<span>{caption}</span>'
                '</div></div><div>'
                + "".join(f"<p>{esc(line)}</p>" for line in summary)
                + '</div></section>')

    if bars:
        body.append(f'<section><h2>How far it moved</h2>{bars}</section>')

    for title, rows, headers in sections:
        cells = "".join(
            f'<tr><td>{esc(label)}</td><td class="c-b">{esc(b)}</td>'
            f'<td class="c-d">{esc(d)}</td><td class="c-t">{esc(t)}</td></tr>'
            for label, b, d, t in rows)
        head_cells = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body.append(
            f'<section><h2>{esc(title)}</h2><div class="tbl"><table><thead><tr>'
            f'{head_cells}'
            f'</tr></thead><tbody>{cells}</tbody></table></div></section>')

    body.append('<section><h2>This run</h2><dl>'
                + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts)
                + '</dl></section>')
    body.append('<footer>Every figure is also reported for the untrained base '
                'student, because that column is what separates &ldquo;distillation '
                'worked&rdquo; from &ldquo;the small model could already do this&rdquo;. '
                'Read the change, not the absolute value.</footer></div>')

    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Distillation evaluation</title>"
            + "".join(head) + "</head><body>" + "".join(body) + "</body></html>")


def write_report(payload, path):
    """Write a human-readable report. Format chosen by the file extension."""
    target = pathlib.Path(path)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = _sections(payload)
    summary = plain_summary(payload)

    # Built by appending what is present rather than by indexing what should be:
    # an arena-only payload carries the answer key and the models, and none of
    # the token-level fields kd.evaluate adds.
    arena = payload.get("arena") or {}
    facts = []
    if payload.get("teacher"):
        facts.append(("Teacher", payload["teacher"]
                      + (f"  + {payload['teacher_adapter']}"
                         if payload.get("teacher_adapter") else "")))
    if payload.get("student"):
        facts.append(("Student", payload["student"]))
    if payload.get("adapter") or arena.get("adapter"):
        facts.append(("Adapter evaluated",
                      payload.get("adapter") or arena.get("adapter")))
    if payload.get("device"):
        facts.append(("Hardware", f"{payload['device']} ({payload.get('dtype')})"))
    if payload.get("samples") is not None:
        facts.append(("Held-out samples",
                      f"{payload['samples']} "
                      f"({payload.get('completion_tokens')} scored tokens)"))
    if arena.get("questions"):
        facts.append(("Answer key", f"{arena['questions']} questions"
                      + (f" from {arena['arena_file']}"
                         if arena.get("arena_file") else "")))
    facts.append(("Generated", stamp))

    if target.suffix.lower() in (".html", ".htm"):
        target.write_text(_render_html(payload, facts, sections, summary),
                          encoding="utf-8")
    else:
        lede = (f"`{payload['student']}` distilled from `{payload['teacher']}`"
                if payload.get("student") and payload.get("teacher")
                else "Scored on a held-out answer key")
        out = ["# Distillation evaluation", "", lede, "", "## Summary", ""]
        out += [f"{line}\n" for line in summary]
        out += ["", "## Run", "", "| | |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in facts]
        for title, rows, headers in sections:
            out += ["", f"## {title}", "",
                    "| " + " | ".join(headers) + " |",
                    "|" + "---|" * len(headers)]
            out += [f"| {label} | {b} | **{d}** | {t} |" for label, b, d, t in rows]
        out += ["", "---", "",
                "Every metric is reported for the untrained base student as well, "
                "because that column is what separates \"distillation worked\" from "
                "\"the small model could already do this\". Read the change, not the "
                "absolute value."]
        target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


