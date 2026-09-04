"""Measure how much of the teacher actually transferred into the distilled student.

Two questions, measured separately, because the distillation literature is explicit
that they do not track each other (Stanton et al., "Does Knowledge Distillation
Really Work?", NeurIPS 2021):

  FIDELITY    how closely does the student reproduce the TEACHER's predictions?
              -> top-1 agreement rate, KL(teacher || student)

  CAPABILITY  is the student actually better at the task than it was?
              -> held-out perplexity, and optionally task benchmarks via
                 lm-evaluation-harness

Both are reported for the BASE student and the DISTILLED student against the same
teacher, on the same held-out split. The base student column is what makes the
numbers mean anything: without it there is no way to tell "distillation worked"
apart from "the small model could already do this".

    python evaluate.py --config configs/finance.yaml
    python evaluate.py --config configs/finance.yaml --dtype bfloat16 --samples 100
    python evaluate.py --config configs/finance.yaml --tasks ifeval,hellaswag
    python evaluate.py --config configs/mac.yaml --json results.json

The held-out split is rebuilt with the training seed, so it is exactly the split
the student never trained on.

Exit status is 0 when the distilled student improved on the base student, and 3
when it did not, so a run can be gated in a shell script.
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kd_config import load_config, resolve_device

BAR = "=" * 78

DTYPES = {
    "auto": None,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Measure teacher->student transfer: fidelity and capability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-c", "--config", default="configs/default.yaml",
                    help="Training config the adapter was produced from "
                         "(default: configs/default.yaml)")
    ap.add_argument("-a", "--adapter", default=None,
                    help="Adapter directory to evaluate "
                         "(default: <output_dir>/final_adapter from the config)")
    ap.add_argument("-t", "--teacher", default=None, help="Override models.teacher")
    ap.add_argument("-s", "--student", default=None, help="Override models.student")
    ap.add_argument("--teacher-adapter", default=None,
                    help="LoRA adapter merged into the teacher, when the teacher is "
                         "a base model plus an adapter rather than a merged checkpoint")
    ap.add_argument("-n", "--samples", type=int, default=50,
                    help="Held-out samples to score (default: 50)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"],
                    help="Override hardware.device")
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES),
                    help="Override hardware.dtype (default: auto)")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated lm-evaluation-harness tasks to run as well, "
                         "e.g. ifeval,hellaswag,arc_easy. Requires `lm-eval` "
                         "(uv sync --extra eval). Slow: it runs three models.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-task example cap passed to lm-eval (use for a quick look)")
    ap.add_argument("--no-generations", action="store_true",
                    help="Skip the qualitative side-by-side generations")
    ap.add_argument("--gen-similarity", type=int, default=0, metavar="N",
                    help="Also measure free-running generation similarity to the "
                         "teacher on N held-out prompts (BERTScore + ROUGE-L). "
                         "0 = off. Unlike agreement/KL this is NOT teacher-forced, "
                         "so it captures the student's own drift. Slow: generates "
                         "from three models. Requires `uv sync --extra eval`.")
    ap.add_argument("--no-rescale", action="store_true",
                    help="Report RAW BERTScore instead of baseline-rescaled. Raw is "
                         "what papers quote, but it is not a 0-1 similarity: two "
                         "unrelated English sentences score ~0.86, so real gains "
                         "look like rounding errors.")
    ap.add_argument("--similarity-model", default="roberta-large", metavar="ID",
                    help="Encoder for BERTScore (default: roberta-large, the "
                         "bert-score English default; ~1.4GB on first use)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="Write all metrics to a JSON file (machine-readable)")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="Write a readable report. The extension picks the format: "
                         ".md for Markdown, .html for a self-contained page you can "
                         "open in a browser or send to someone.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Held-out split
# --------------------------------------------------------------------------- #
def build_eval_samples(config, hardware, tokenizer, limit):
    """Rebuild the exact held-out split the student was never trained on.

    train_scaled.build_datasets() is reused rather than reimplemented: it is seeded
    from project.seed and applies the same length filter and dedup, so the split
    here is byte-identical to the one training held out. Reimplementing it would
    silently drift the moment either side changed.
    """
    import train_scaled

    train_scaled.apply_config(config, hardware)
    _, eval_dataset = train_scaled.build_datasets(tokenizer)
    rows = eval_dataset.select(range(min(limit, len(eval_dataset))))
    return [row["messages"] for row in rows]


def encode(tokenizer, messages, device):
    """Tokenize one sample and return (input_ids, prompt_token_count).

    The prompt/completion boundary is computed exactly as the GKD collator sees it:
    messages[:-1] rendered with a generation prompt is the prompt, the full turn
    list is prompt + completion. Only completion positions are scored.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)

    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    ids = tokenizer(full_text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    return ids, prompt_len


def completion_slice(logits, ids, prompt_len):
    """Return (logits_at_completion_positions, target_ids).

    Causal shift: logits[:, t] predicts token t+1, so the distribution over
    completion token `t` lives at logit index `t - 1`.
    """
    length = ids.shape[1]
    if prompt_len < 1 or prompt_len >= length:
        return None, None
    return logits[0, prompt_len - 1:length - 1, :], ids[0, prompt_len:length]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compare_distributions(teacher_logits, student_logits, chunk=32):
    """Top-1 agreement count and summed KL(teacher || student) over positions.

    Chunked and cast to float32 per chunk: a full [positions, 248k] float32 tensor
    is hundreds of megabytes at these vocabulary sizes, which is enough to push a
    16 GB machine into swap during what is supposed to be the cheap step.
    """
    positions = teacher_logits.shape[0]
    kl_total, agree = 0.0, 0
    for i in range(0, positions, chunk):
        t = teacher_logits[i:i + chunk].float()
        s = student_logits[i:i + chunk].float()
        t_log = F.log_softmax(t, dim=-1)
        s_log = F.log_softmax(s, dim=-1)
        # F.kl_div(input, target) computes KL(target || input), i.e. the argument
        # order is the reverse of the mathematical convention - the same quirk TRL
        # works around in generalized_jsd_loss. Passing (student, teacher) here is
        # therefore KL(teacher || student), which is what fidelity means: how much
        # information is lost when the student stands in for the teacher.
        kl_total += float(F.kl_div(s_log, t_log, log_target=True, reduction="sum"))
        agree += int((t.argmax(-1) == s.argmax(-1)).sum())
    return agree, kl_total


def summed_nll(logits, targets, chunk=32):
    """Summed negative log-likelihood of the reference completion."""
    total = 0.0
    for i in range(0, logits.shape[0], chunk):
        total += float(F.cross_entropy(
            logits[i:i + chunk].float(), targets[i:i + chunk], reduction="sum"))
    return total


def measure_latency(model, tokenizer, prompt, device, new_tokens=32):
    """Indicative decode throughput, tokens/second. Single greedy run."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(device)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():  # warm up kernels/allocator so the timed run is representative
        model.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=pad)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                             pad_token_id=pad)
    elapsed = time.time() - start
    produced = out.shape[1] - ids.input_ids.shape[1]
    return (produced / elapsed) if elapsed > 0 else float("nan")


def generate(model, tokenizer, prompt, device, new_tokens=64):
    """Greedy generation. Deterministic on purpose: sampled output is not evidence."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                             repetition_penalty=1.1,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Readable report
# --------------------------------------------------------------------------- #
def plain_summary(payload):
    """Plain-English reading of the numbers, for someone who did not run the job.

    Every claim here is derived from the measurements, not asserted: the point is
    that a reader should not have to know what perplexity is to learn whether the
    training worked.
    """
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


def _report_rows(payload):
    """(section, [(label, base, distilled, teacher)]) for both report formats."""
    fid, cap, eff = payload["fidelity"], payload["capability"], payload["efficiency"]
    close = payload.get("closeness_to_teacher") or {}
    fmt = lambda v, spec=".4f": (format(v, spec)
                                 if isinstance(v, (int, float)) and math.isfinite(v)
                                 else "-")
    sections = [
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

    body = ['<div class="wrap"><header>',
            '<p class="eyebrow">Knowledge distillation &middot; evaluation</p>',
            f'<h1>{esc(payload["student"].split("/")[-1])}</h1>',
            f'<p class="lede">Distilled from <strong>'
            f'{esc(payload["teacher"].split("/")[-1])}</strong>, then measured against '
            f'it and against its own untrained self.</p></header>']

    big = (f"{recovered:.0f}%" if isinstance(recovered, (int, float))
           and math.isfinite(recovered) else "&mdash;")
    body.append(f'<section class="verdict"><div><div class="big">{big}'
                '<span>of the distance to the teacher, closed by training</span>'
                '</div></div><div>'
                + "".join(f"<p>{esc(line)}</p>" for line in summary)
                + '</div></section>')

    if bars:
        body.append(f'<section><h2>How far it moved</h2>{bars}</section>')

    for title, rows in sections:
        cells = "".join(
            f'<tr><td>{esc(label)}</td><td class="c-b">{esc(b)}</td>'
            f'<td class="c-d">{esc(d)}</td><td class="c-t">{esc(t)}</td></tr>'
            for label, b, d, t in rows)
        body.append(
            f'<section><h2>{esc(title)}</h2><div class="tbl"><table><thead><tr>'
            '<th>Metric</th><th>Base student</th><th>Distilled</th><th>Teacher</th>'
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
    import datetime

    target = pathlib.Path(path)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = _report_rows(payload)
    summary = plain_summary(payload)
    facts = [
        ("Teacher", payload["teacher"] + (f"  + {payload['teacher_adapter']}"
                                          if payload.get("teacher_adapter") else "")),
        ("Student", payload["student"]),
        ("Adapter evaluated", payload["adapter"]),
        ("Hardware", f"{payload['device']} ({payload['dtype']})"),
        ("Held-out samples", f"{payload['samples']} "
                             f"({payload['completion_tokens']} scored tokens)"),
        ("Generated", stamp),
    ]

    if target.suffix.lower() in (".html", ".htm"):
        target.write_text(_render_html(payload, facts, sections, summary),
                          encoding="utf-8")
    else:
        out = ["# Distillation evaluation", "",
               f"`{payload['student']}` distilled from `{payload['teacher']}`", "",
               "## Summary", ""]
        out += [f"{line}\n" for line in summary]
        out += ["", "## Run", "", "| | |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in facts]
        for title, rows in sections:
            out += ["", f"## {title}", "",
                    "| Metric | Base student | Distilled | Teacher |",
                    "|---|---|---|---|"]
            out += [f"| {label} | {b} | **{d}** | {t} |" for label, b, d, t in rows]
        out += ["", "---", "",
                "Every metric is reported for the untrained base student as well, "
                "because that column is what separates \"distillation worked\" from "
                "\"the small model could already do this\". Read the change, not the "
                "absolute value."]
        target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Optional: free-running generation similarity
# --------------------------------------------------------------------------- #
def generation_similarity(student, teacher, tokenizer, prompts, device,
                          model_type="roberta-large", new_tokens=64, rescale=True):
    """Does the student SAY what the teacher says, when each writes freely?

    Agreement and KL are teacher-forced: both models read the same correct text,
    so neither ever sees the student's own drift. At inference the student runs
    free and its errors compound. This generates from each model independently on
    the same prompts and compares the resulting text, which is the behaviour a
    user actually experiences.

    BERTScore is the primary metric because these are free-text answers with many
    valid phrasings. Exact-match and n-gram metrics punish paraphrase: on two
    sentences that mean the same thing, exact_match scores 0.0 and ROUGE-2 scores
    0.0 (the shared words appear in a different order), while BERTScore - which
    compares contextual embeddings rather than spelling - scores ~0.95. ROUGE-L is
    reported beside it only as a cheap surface-overlap reference point.

    The comparison is student-vs-TEACHER, not student-vs-dataset-reference: this
    is a fidelity measurement, not a correctness one.
    """
    teacher_gen, base_gen, dist_gen = [], [], []
    for index, prompt in enumerate(prompts):
        teacher_gen.append(generate(teacher, tokenizer, prompt, device, new_tokens))
        dist_gen.append(generate(student, tokenizer, prompt, device, new_tokens))
        with student.disable_adapter():
            base_gen.append(generate(student, tokenizer, prompt, device, new_tokens))
        if (index + 1) % 5 == 0:
            print(f"   {index + 1}/{len(prompts)} prompts generated (3 models each)")

    # A model can legitimately emit nothing; both scorers choke on an empty string.
    clean = lambda texts: [t if t.strip() else "(empty)" for t in texts]
    teacher_gen, base_gen, dist_gen = (clean(teacher_gen), clean(base_gen),
                                       clean(dist_gen))

    result = {"prompts": len(prompts), "max_new_tokens": new_tokens}

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    def paired_bootstrap(base_scores, dist_scores, rounds=2000, seed=42):
        """95% CI on the per-prompt (distilled - base) difference.

        Paired and resampled over prompts, because the two columns are scored on
        the SAME prompts - the pairing removes prompt difficulty from the
        comparison. Without this there is no way to tell a real gain from the
        luck of which 50 prompts landed in the held-out split.
        """
        deltas = [d - b for b, d in zip(base_scores, dist_scores)]
        if len(deltas) < 2:
            return None
        rng = random.Random(seed)
        n = len(deltas)
        means = []
        for _ in range(rounds):
            means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        return (means[int(0.025 * rounds)], means[int(0.975 * rounds)])

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        r_base = [scorer.score(t, c)["rougeL"].fmeasure
                  for t, c in zip(teacher_gen, base_gen)]
        r_dist = [scorer.score(t, c)["rougeL"].fmeasure
                  for t, c in zip(teacher_gen, dist_gen)]
        result["rougeL_base"] = mean(r_base)
        result["rougeL_distilled"] = mean(r_dist)
        result["rougeL_ci95"] = paired_bootstrap(r_base, r_dist)
    except ImportError:
        print(" !! rouge_score not installed; skipping ROUGE-L")

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("\n !! bert-score is not installed, so only ROUGE-L was computed.")
        print("    BERTScore is the metric that actually handles paraphrase here:")
        print("      uv sync --extra eval")
        return result, {"teacher": teacher_gen, "base": base_gen, "distilled": dist_gen}

    print(f"   scoring with BERTScore ({model_type}; first run downloads it)")
    # rescale_with_baseline matters more than it looks. RAW BERTScore is not a
    # 0-1 similarity: two entirely unrelated English sentences score ~0.86, so the
    # whole meaningful range is compressed into roughly 0.85-0.96 and a genuine
    # improvement reads as a rounding error. Rescaling against bert-score's
    # random-pair baseline puts ~0 at "unrelated" and ~1 at "identical", which is
    # what makes the number legible. Raw is still what papers report, so it stays
    # available via --no-rescale.
    per_pair = {}
    for label, cands in (("base", base_gen), ("distilled", dist_gen)):
        try:
            _, _, f1 = bert_score_fn(cands, teacher_gen, model_type=model_type,
                                     lang="en", rescale_with_baseline=rescale,
                                     verbose=False, batch_size=8)
        except Exception as exc:
            # No baseline file ships for every encoder; raw is better than nothing.
            print(f"   !! rescaling unavailable ({type(exc).__name__}); using raw scores")
            rescale = False
            _, _, f1 = bert_score_fn(cands, teacher_gen, model_type=model_type,
                                     verbose=False, batch_size=8)
        per_pair[label] = [float(x) for x in f1]
        result[f"bertscore_f1_{label}"] = mean(per_pair[label])
    result["bertscore_ci95"] = paired_bootstrap(per_pair["base"], per_pair["distilled"])
    result["bertscore_rescaled"] = rescale
    result["bertscore_model"] = model_type

    return result, {"teacher": teacher_gen, "base": base_gen, "distilled": dist_gen}


def report_similarity(result):
    print("\n" + BAR)
    print("  GENERATION SIMILARITY - free-running, vs the teacher's own output")
    print(BAR)
    print(f"\n  {result['prompts']} prompts, greedy decoding, "
          f"{result['max_new_tokens']} new tokens per model\n")
    print(f"  {'metric':34} {'base':>10} {'distilled':>11} {'change':>12}")
    print("  " + "-" * 70)

    scale = "" if result.get("bertscore_rescaled", True) else " (raw)"
    rows = [(f"BERTScore F1 vs teacher{scale}", "bertscore_f1_base",
             "bertscore_f1_distilled", "bertscore_ci95"),
            ("ROUGE-L vs teacher", "rougeL_base", "rougeL_distilled", "rougeL_ci95")]
    significant = {}
    for title, base_key, dist_key, ci_key in rows:
        if base_key not in result or dist_key not in result:
            continue
        b, d = result[base_key], result[dist_key]
        ci = result.get(ci_key)
        line = f"  {title:34} {b:10.4f} {d:11.4f} {d - b:+12.4f}"
        if ci:
            # A 95% CI on the difference that excludes zero is the difference
            # being real rather than an artefact of which prompts were sampled.
            line += f"   95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            significant[title] = ci[0] > 0 or ci[1] < 0
        print(line)

    print("""
  This is the one measurement here taken with the models running FREE rather
  than reading the reference text, so it is the closest to what a user sees.
  BERTScore compares meaning, so a correct answer worded differently is not
  punished; ROUGE-L compares word overlap and is shown only for contrast.""")

    # Free-running generation is far noisier than the teacher-forced metrics: a
    # single divergent token early in a greedy decode changes the whole
    # continuation. Small prompt counts routinely produce a change of either sign.
    for title, is_sig in significant.items():
        verdict = ("REAL - the 95% interval excludes zero" if is_sig
                   else "NOT SIGNIFICANT - the 95% interval includes zero")
        print(f"\n  {title.split(' vs')[0]}: {verdict}.")

    if result["prompts"] < 20:
        print(f"\n  !! ONLY {result['prompts']} PROMPTS - treat the change column as noise.")
        print("     Greedy generation diverges on a single early token, so this")
        print("     metric needs 50+ prompts before a small difference means")
        print("     anything. Raise --gen-similarity.")

    # Worth saying out loud when it happens: the two families of fidelity metric
    # genuinely can disagree, and that disagreement is informative rather than a bug.
    b = result.get("bertscore_f1_base")
    d = result.get("bertscore_f1_distilled")
    if isinstance(b, (int, float)) and isinstance(d, (int, float)) and d < b:
        print("\n  Note: free-running similarity did NOT improve, even if the")
        print("  teacher-forced agreement above did. That combination means the")
        print("  student matches the teacher well when reading correct text, but")
        print("  still drifts when writing on its own - the gap on-policy training")
        print("  (a higher gkd.lmbda) is meant to close.")


# --------------------------------------------------------------------------- #
# Optional: lm-evaluation-harness
# --------------------------------------------------------------------------- #
def run_lm_eval(tasks, student_id, adapter_dir, teacher_id, teacher_adapter,
                dtype_name, device, limit):
    """Score base student, distilled student and teacher on standard benchmarks.

    lm-evaluation-harness is the de facto standard harness (it is what the HF Open
    LLM Leaderboard runs), so numbers produced here are comparable with published
    ones rather than only with each other.
    """
    if importlib.util.find_spec("lm_eval") is None:
        print("\n !! lm-eval is not installed; skipping --tasks.")
        print("    Install the optional extra and re-run:")
        print("      uv sync --extra eval")
        return None

    runs = {
        "base_student": f"pretrained={student_id}",
        "distilled_student": f"pretrained={student_id},peft={adapter_dir}",
        "teacher": f"pretrained={teacher_id}"
                   + (f",peft={teacher_adapter}" if teacher_adapter else ""),
    }

    results = {}
    for label, model_args in runs.items():
        print(f"\n  [lm-eval] {label} on {tasks}")
        # Invoked as a module rather than by console-script name: the entry point is
        # spelled lm-eval in some releases and lm_eval in others, and neither is
        # guaranteed to be on PATH. sys.executable always resolves to this venv.
        #
        # --apply_chat_template is set for all three models, including the base
        # student. Every model here is instruct-tuned and the adapter was trained
        # against a chat template, so this keeps the three columns internally
        # consistent, which is what a base-vs-distilled comparison needs. It does
        # mean the absolute numbers are not directly comparable to leaderboard
        # entries that scored multiple-choice tasks without a template.
        cmd = [
            sys.executable, "-m", "lm_eval", "run",
            "--model", "hf",
            "--model_args", f"{model_args},dtype={dtype_name}",
            "--tasks", tasks,
            "--device", device,
            "--batch_size", "1",
            "--seed", "42",
            "--apply_chat_template",
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        out_dir = pathlib.Path("./evals") / label
        cmd += ["--output_path", str(out_dir)]

        # lm-eval renders its summary table with Unicode arrows. On a Windows
        # console that defaults to cp1252 the print raises UnicodeEncodeError and
        # the process exits 1 - AFTER the results file has been written. Force
        # UTF-8 so it does not happen, and treat the results file as the source of
        # truth below rather than the exit status.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        completed = subprocess.run(cmd, env=env)

        # lm-eval writes results_<timestamp>.json under a model-named subdirectory.
        files = sorted(out_dir.rglob("results_*.json"))
        if not files:
            print(f"  !! lm-eval produced no results for {label} "
                  f"(exit {completed.returncode})")
            continue
        if completed.returncode != 0:
            print(f"  -- lm-eval exited {completed.returncode} for {label}, but wrote "
                  f"results; using them")
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        results[label] = payload.get("results", {})
    return results or None


def report_tasks(results):
    """Print the retention table: distilled / teacher, the standard KD headline."""
    print("\n" + BAR)
    print("  TASK BENCHMARKS - lm-evaluation-harness")
    print(BAR)

    base, dist, teach = (results.get("base_student", {}),
                         results.get("distilled_student", {}),
                         results.get("teacher", {}))
    task_names = sorted(set(base) | set(dist) | set(teach))

    def stderr_key(metric):
        """lm-eval names metrics 'acc,none' and their error 'acc_stderr,none'."""
        name, _, suffix = metric.partition(",")
        return f"{name}_stderr,{suffix}" if suffix else f"{name}_stderr"

    def is_reportable(task, metric):
        # Skip the error bars themselves (they are attached to their metric below)
        # and lm-eval's bookkeeping entries, which have no meaningful retention.
        name = metric.partition(",")[0]
        if name.endswith("_stderr") or name in ("alias", "sample_len"):
            return False
        return isinstance((dist.get(task) or {}).get(metric), (int, float))

    def cell(value, err):
        if not isinstance(value, (int, float)):
            return f"{'-':>14}"
        return f"{value:8.4f}+-{err:<4.3f}" if isinstance(err, (int, float)) \
            else f"{value:8.4f}      "

    print(f"\n  {'task / metric':30} {'base':>14} {'distilled':>14} "
          f"{'teacher':>14} {'retention':>10}")
    print("  " + "-" * 86)
    noisy = []
    for task in task_names:
        for metric in [m for m in (dist.get(task) or {}) if is_reportable(task, m)]:
            sk = stderr_key(metric)
            b, d, t = ((base.get(task) or {}).get(metric),
                       (dist.get(task) or {}).get(metric),
                       (teach.get(task) or {}).get(metric))
            be, de, te = ((base.get(task) or {}).get(sk),
                          (dist.get(task) or {}).get(sk),
                          (teach.get(task) or {}).get(sk))
            retention = (f"{d / t * 100:9.1f}%"
                         if isinstance(d, (int, float)) and isinstance(t, (int, float)) and t
                         else "        -")
            print(f"  {task + ' / ' + metric.partition(',')[0]:30} "
                  f"{cell(b, be)} {cell(d, de)} {cell(t, te)} {retention}")

            # A difference smaller than the combined error bars is not a result.
            # Saying so here is the whole point of running a standard harness.
            if all(isinstance(v, (int, float)) for v in (b, d, be, de)):
                if abs(d - b) <= (be + de):
                    noisy.append(f"{task}/{metric.partition(',')[0]}")

    print("\n  retention = distilled / teacher. The comparison that matters is")
    print("  distilled vs base: if that lift is ~0, distillation changed nothing.")
    print("  +- values are lm-eval's standard error at the sample count you ran.")
    if noisy:
        print("\n  !! NOT SIGNIFICANT - distilled vs base is within the error bars for:")
        for item in noisy:
            print(f"       {item}")
        print("     Re-run with a larger --eval-limit (or none) before drawing")
        print("     any conclusion from these.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.setdefault("hardware", {})["device"] = args.device
    if args.dtype != "auto":
        config.setdefault("hardware", {})["dtype"] = args.dtype

    hardware = resolve_device(config)
    device = hardware["device"]
    dtype = DTYPES[args.dtype] or hardware["dtype"]
    dtype_name = str(dtype).replace("torch.", "")

    teacher_id = args.teacher or config["models"]["teacher"]
    student_id = args.student or config["models"]["student"]
    teacher_adapter = args.teacher_adapter or config["models"].get("teacher_adapter")

    adapter_dir = args.adapter or os.path.join(
        config["project"]["output_dir"], "final_adapter")
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise SystemExit(
            f"No adapter_config.json in {adapter_dir}\n"
            f"Train one first, or point --adapter at the right directory."
        )

    tokenizer_choice = str(config["models"].get("tokenizer", "teacher"))
    tokenizer_id = {"teacher": teacher_id, "student": student_id}.get(
        tokenizer_choice, tokenizer_choice)

    print(BAR)
    print("  Distillation evaluation")
    print(BAR)
    print(f"   teacher   : {teacher_id}" + (f"  + {teacher_adapter}" if teacher_adapter else ""))
    print(f"   student   : {student_id}")
    print(f"   adapter   : {adapter_dir}")
    print(f"   device    : {device} ({dtype_name})")
    print(f"   samples   : {args.samples} held-out")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = build_eval_samples(config, hardware, tokenizer, args.samples)
    if not samples:
        raise SystemExit("Held-out split is empty; nothing to evaluate.")

    # --- models -------------------------------------------------------------- #
    # Teacher and student are both resident: agreement and KL need both
    # distributions for the same position at the same time.
    print(f"\n[1/4] Loading teacher ({teacher_id})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype=dtype, low_cpu_mem_usage=True)
    if teacher_adapter:
        from peft import PeftModel as _Peft
        print(f" -> merging teacher adapter: {teacher_adapter}")
        teacher = _Peft.from_pretrained(teacher, teacher_adapter).merge_and_unload()
    teacher = teacher.to(device).eval()

    # One student instance serves as both columns: PEFT's disable_adapter() context
    # turns the LoRA branches off, which is the base student exactly. Loading a
    # second copy would double peak memory for no additional information.
    print(f"[2/4] Loading student ({student_id}) + adapter...")
    from peft import PeftModel
    student = AutoModelForCausalLM.from_pretrained(
        student_id, dtype=dtype, low_cpu_mem_usage=True)
    student = PeftModel.from_pretrained(student, adapter_dir).to(device).eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    adapter_params = sum(p.numel() for n, p in student.named_parameters() if "lora_" in n)

    # --- fidelity + perplexity ----------------------------------------------- #
    print(f"[3/4] Scoring {len(samples)} held-out samples...")
    tokens = 0
    agree_base = agree_dist = 0
    kl_base = kl_dist = 0.0
    nll_teacher = nll_base = nll_dist = 0.0
    skipped = 0

    for index, messages in enumerate(samples):
        ids, prompt_len = encode(tokenizer, messages, device)
        with torch.no_grad():
            t_logits_full = teacher(ids).logits
            t_logits, targets = completion_slice(t_logits_full, ids, prompt_len)
            if t_logits is None or t_logits.shape[0] == 0:
                skipped += 1
                continue

            d_logits, _ = completion_slice(student(ids).logits, ids, prompt_len)
            with student.disable_adapter():
                b_logits, _ = completion_slice(student(ids).logits, ids, prompt_len)

        # Standard GKD compares per-token distributions, which is only meaningful
        # over a shared vocabulary. Fail loudly rather than reporting a number that
        # cannot mean anything.
        if t_logits.shape[-1] != d_logits.shape[-1]:
            raise SystemExit(
                f"Vocabulary mismatch: teacher {t_logits.shape[-1]} vs student "
                f"{d_logits.shape[-1]}. Teacher and student must share a tokenizer "
                f"for token-level distillation metrics to mean anything."
            )

        n = t_logits.shape[0]
        tokens += n
        a, k = compare_distributions(t_logits, b_logits)
        agree_base += a
        kl_base += k
        a, k = compare_distributions(t_logits, d_logits)
        agree_dist += a
        kl_dist += k
        nll_teacher += summed_nll(t_logits, targets)
        nll_base += summed_nll(b_logits, targets)
        nll_dist += summed_nll(d_logits, targets)

        del t_logits_full, t_logits, d_logits, b_logits
        if (index + 1) % 10 == 0:
            print(f"   {index + 1}/{len(samples)} scored ({tokens} completion tokens)")

    if not tokens:
        raise SystemExit("No scorable completion tokens; try --samples with a larger value.")

    agreement_base = agree_base / tokens * 100
    agreement_dist = agree_dist / tokens * 100
    kl_base_avg = kl_base / tokens
    kl_dist_avg = kl_dist / tokens
    ppl_teacher = math.exp(nll_teacher / tokens)
    ppl_base = math.exp(nll_base / tokens)
    ppl_dist = math.exp(nll_dist / tokens)

    # --- efficiency + generations -------------------------------------------- #
    print("[4/4] Measuring decode throughput...")
    probe = (config.get("benchmark_prompts") or ["Explain compound interest."])[0]
    tps_teacher = measure_latency(teacher, tokenizer, probe, device)
    tps_dist = measure_latency(student, tokenizer, probe, device)

    # ----------------------------------------------------------------------- #
    print("\n" + BAR)
    print("  FIDELITY - how much of the teacher transferred")
    print(BAR)
    print(f"  {len(samples) - skipped} held-out samples, {tokens} completion tokens, "
          f"teacher-forced\n")
    print(f"  {'metric':34} {'base':>10} {'distilled':>11} {'change':>12}")
    print("  " + "-" * 70)
    print(f"  {'top-1 agreement with teacher':34} {agreement_base:9.2f}% "
          f"{agreement_dist:10.2f}% {agreement_dist - agreement_base:+11.2f} pts")
    kl_change = ((kl_dist_avg - kl_base_avg) / kl_base_avg * 100) if kl_base_avg else 0.0
    print(f"  {'KL(teacher || student), per token':34} {kl_base_avg:10.4f} "
          f"{kl_dist_avg:11.4f} {kl_change:+11.1f}%")
    print("\n  Agreement is the fraction of positions where the student's top token")
    print("  matches the teacher's. Fidelity is routinely far below task accuracy -")
    print("  a student can score well while disagreeing with the teacher often.")

    print("\n" + BAR)
    print("  CAPABILITY - held-out perplexity (lower is better)")
    print(BAR)
    print(f"\n  {'teacher':34} {ppl_teacher:10.3f}")
    print(f"  {'base student':34} {ppl_base:10.3f}")
    print(f"  {'distilled student':34} {ppl_dist:10.3f}")
    gap = ppl_base - ppl_teacher
    recovered = ((ppl_base - ppl_dist) / gap * 100) if abs(gap) > 1e-9 else float("nan")
    if math.isfinite(recovered):
        print(f"\n  teacher-student gap recovered : {recovered:.1f}%")
        print("  (fraction of the base->teacher perplexity gap the adapter closed;")
        print("   negative means the distilled student is worse than the base)")

    # --- the single "how close is it" number ------------------------------- #
    # Two standard percentages, deliberately NOT blended into one score. There is
    # no accepted composite closeness metric, and averaging these would combine a
    # token-level agreement rate with a likelihood ratio - different units,
    # different questions. Retention on perplexity is inverted (teacher/student)
    # because lower perplexity is better.
    ppl_ret_base = (ppl_teacher / ppl_base * 100) if ppl_base else float("nan")
    ppl_ret_dist = (ppl_teacher / ppl_dist * 100) if ppl_dist else float("nan")

    print("\n" + BAR)
    print("  CLOSENESS TO TEACHER")
    print(BAR)
    print(f"\n  {'':34} {'base':>10} {'distilled':>11} {'teacher':>10}")
    print("  " + "-" * 70)
    print(f"  {'prediction agreement':34} {agreement_base:9.2f}% "
          f"{agreement_dist:10.2f}% {100.0:9.2f}%")
    print(f"  {'perplexity retention':34} {ppl_ret_base:9.2f}% "
          f"{ppl_ret_dist:10.2f}% {100.0:9.2f}%")
    if math.isfinite(recovered):
        print(f"\n  Training closed {recovered:.1f}% of the base->teacher gap.")
    print("""
  Read these as "how close", not "how good". The base student already scores
  most of this before any training, because student and teacher share an
  architecture, a tokenizer and instruction tuning - so the absolute number is
  dominated by that head start, not by distillation. What distillation bought
  is the LIFT over the base column, and the gap-recovered figure.

  The two rows answer different questions (token agreement vs likelihood) and
  are not averaged: no standard composite closeness score exists.""")

    print("\n" + BAR)
    print("  EFFICIENCY - what the retention cost")
    print(BAR)
    print(f"\n  {'':34} {'teacher':>12} {'distilled':>12}")
    print("  " + "-" * 60)
    print(f"  {'parameters':34} {teacher_params / 1e9:11.3f}B {student_params / 1e9:11.3f}B")
    print(f"  {'decode throughput (tok/s)':34} {tps_teacher:12.1f} {tps_dist:12.1f}")
    print(f"  {'size ratio':34} {'1.00x':>12} "
          f"{student_params / teacher_params:11.2f}x")
    if tps_teacher > 0:
        print(f"  {'speed ratio':34} {'1.00x':>12} {tps_dist / tps_teacher:11.2f}x")
    print(f"\n  trainable adapter parameters : {adapter_params / 1e6:.2f}M "
          f"({adapter_params / student_params * 100:.2f}% of the student)")

    generations = {}
    if not args.no_generations:
        print("\n" + BAR)
        print("  GENERATIONS - greedy, deterministic")
        print(BAR)
        for prompt in (config.get("benchmark_prompts") or [])[:3]:
            with student.disable_adapter():
                base_text = generate(student, tokenizer, prompt, device)
            dist_text = generate(student, tokenizer, prompt, device)
            teach_text = generate(teacher, tokenizer, prompt, device)
            generations[prompt] = {"base": base_text, "distilled": dist_text,
                                   "teacher": teach_text}
            print(f"\n  Q: {prompt}")
            print(f"    base      : {base_text[:220]}")
            print(f"    distilled : {dist_text[:220]}")
            print(f"    teacher   : {teach_text[:220]}")

    similarity, similarity_texts = None, None
    if args.gen_similarity > 0:
        print("\n" + BAR)
        print(f"  Generating from three models on {args.gen_similarity} held-out "
              f"prompts...")
        print(BAR)
        # The user turn of each held-out sample: real domain prompts the student
        # was never trained on, not the handful of benchmark_prompts.
        prompts = [m[-2]["content"] for m in samples[:args.gen_similarity]
                   if len(m) >= 2]
        similarity, similarity_texts = generation_similarity(
            student, teacher, tokenizer, prompts, device,
            model_type=args.similarity_model, rescale=not args.no_rescale)
        report_similarity(similarity)

    payload = {
        "teacher": teacher_id,
        "teacher_adapter": teacher_adapter,
        "student": student_id,
        "adapter": adapter_dir,
        "device": device,
        "dtype": dtype_name,
        "samples": len(samples) - skipped,
        "completion_tokens": tokens,
        "fidelity": {
            "top1_agreement_base_pct": agreement_base,
            "top1_agreement_distilled_pct": agreement_dist,
            "agreement_lift_pts": agreement_dist - agreement_base,
            "kl_base": kl_base_avg,
            "kl_distilled": kl_dist_avg,
        },
        "capability": {
            "perplexity_teacher": ppl_teacher,
            "perplexity_base": ppl_base,
            "perplexity_distilled": ppl_dist,
            "gap_recovered_pct": recovered,
        },
        "closeness_to_teacher": {
            "prediction_agreement_base_pct": agreement_base,
            "prediction_agreement_distilled_pct": agreement_dist,
            "perplexity_retention_base_pct": ppl_ret_base,
            "perplexity_retention_distilled_pct": ppl_ret_dist,
            "gap_recovered_pct": recovered,
        },
        "efficiency": {
            "teacher_params": teacher_params,
            "student_params": student_params,
            "adapter_params": adapter_params,
            "teacher_tok_per_s": tps_teacher,
            "distilled_tok_per_s": tps_dist,
        },
        "generations": generations,
        "generation_similarity": similarity,
        "generation_similarity_texts": similarity_texts,
    }

    # Free the resident models before lm-eval spawns its own processes. Not
    # cosmetic: each lm-eval run loads its own copy of the model, so on a 16 GB
    # unified-memory Mac the parent still caching ~6 GB is the difference between
    # the benchmark running and the machine swapping itself to a halt.
    del teacher, student
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    if args.tasks:
        task_results = run_lm_eval(args.tasks, student_id, adapter_dir, teacher_id,
                                   teacher_adapter, dtype_name, device, args.limit)
        if task_results:
            report_tasks(task_results)
            payload["tasks"] = task_results

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  metrics written to {args.json}")

    if args.report:
        written = write_report(payload, args.report)
        print(f"  report written to {written}")

    improved = (agreement_dist > agreement_base) and (ppl_dist < ppl_base)
    print("\n" + BAR)
    if improved:
        print("  ALL OK - the adapter moved the student toward the teacher")
        print(BAR)
        print(f"   agreement  : {agreement_base:.2f}% -> {agreement_dist:.2f}% "
              f"({agreement_dist - agreement_base:+.2f} pts)")
        print(f"   perplexity : {ppl_base:.3f} -> {ppl_dist:.3f} "
              f"(teacher {ppl_teacher:.3f})")
        print()
        return 0

    print("  NO IMPROVEMENT - the adapter did not move the student toward the teacher")
    print(BAR)
    print(f"   agreement  : {agreement_base:.2f}% -> {agreement_dist:.2f}%")
    print(f"   perplexity : {ppl_base:.3f} -> {ppl_dist:.3f}")
    print("""
  Things worth checking, in order:
    * did the teacher pass --check-teacher? distilling from a broken teacher
      trains the student toward noise;
    * were enough steps run? 100-300 steps on a 2B->0.8B pair is a small budget;
    * do the LoRA target_modules cover this architecture? a Llama-style target
      list misses linear attention in 18 of Qwen3.5's 24 layers;
    * is lmbda > 0? without on-policy rollouts this is plain off-policy KD.
""")
    return 3


if __name__ == "__main__":
    sys.exit(main())
