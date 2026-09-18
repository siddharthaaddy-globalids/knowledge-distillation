#!/usr/bin/env python3
"""Add an OpenAI model as an extra player in a distillation arena.

Reads an arena transcript (the concatenated pretty-printed JSON records written by
the eval harness), asks an OpenAI model exactly the same questions, records the
answers back into the transcript as a new player, and rebuilds report.html with the
new player added as an extra column plus a few sections of its own.

Standalone: standard library only (the `openai` package is NOT required).

    export OPENAI_API_KEY=sk-...
    python openai_arena.py --transcript arena-transcript.jsonl --report report.html

Outputs (next to the inputs unless overridden):
    arena-transcript.openai.jsonl
    report.openai.html

Re-running is cheap: every answer is cached in --cache, so only missing questions
are sent. Use --report-only to rebuild the report from an already-answered
transcript without touching the API.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# --------------------------------------------------------------------------------------
# transcript I/O
# --------------------------------------------------------------------------------------


def load_transcript(path):
    """Load records. Handles both concatenated pretty-printed JSON and true JSONL."""
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    dec = json.JSONDecoder()
    objs, i, n = [], 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \r\n\t":
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(txt, i)
        objs.append(obj)
    return objs


def _scalar_list(v):
    return isinstance(v, list) and all(
        x is None or isinstance(x, (str, int, float, bool)) for x in v
    )


def dump_record(obj, indent=4, level=0):
    """Serialise the way the harness does: 4-space indent, scalar lists on one line."""
    pad, pad2 = " " * (indent * level), " " * (indent * (level + 1))
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        body = ",\n".join(
            f"{pad2}{json.dumps(k, ensure_ascii=False)}: {dump_record(v, indent, level + 1)}"
            for k, v in obj.items()
        )
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if _scalar_list(obj):
            return json.dumps(obj, ensure_ascii=False)
        body = ",\n".join(pad2 + dump_record(v, indent, level + 1) for v in obj)
        return "[\n" + body + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def save_transcript(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(dump_record(r) for r in records))


# --------------------------------------------------------------------------------------
# answer parsing - mirrors the "how" taxonomy already in the transcript
# (tagged / labelled / restated / named), so the new column is comparable
# --------------------------------------------------------------------------------------

LETTER = r"[A-H]"


def _strip_reasoning(text):
    """Drop <think>...</think> so stray letters in it are not mistaken for an answer."""
    out = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    if "<think>" in out.lower() and "</think>" not in out.lower():
        out = re.sub(r"<think>.*$", " ", out, flags=re.S | re.I)
    return out


def parse_answer(completion, options):
    """Return (letter, how) or (None, None)."""
    if not completion:
        return None, None
    text = _strip_reasoning(completion)

    m = list(re.finditer(rf"<\s*Answer\s*>[\s:*]*\(?({LETTER})\b", text, re.I))
    if m:
        return m[-1].group(1).upper(), "tagged"

    labelled = [
        rf"\\boxed\{{\s*(?:\\text\{{)?\s*\(?({LETTER})[.)\s}}]",
        rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is)?\s*[:\-\u2014]?[\s*]*\(?({LETTER})\b",
        rf"^[\s*]*\(?({LETTER})[.)]\s",
    ]
    for pat in labelled:
        m = list(re.finditer(pat, text, re.I | re.M))
        if m:
            return m[-1].group(1).upper(), "labelled"

    # restated: the option repeated with its letter near the end, e.g. "C. Olfaction"
    tail = text[-1200:]
    for opt in options or []:
        mo = re.match(rf"\s*({LETTER})[.)]\s*(.+)", opt)
        if not mo:
            continue
        letter, body = mo.group(1).upper(), mo.group(2).strip()
        if re.search(rf"\b{letter}\s*[.)]\s*\**\s*{re.escape(body)}", tail, re.I):
            return letter, "restated"

    # named: the option's own words, without the letter
    hits = []
    for opt in options or []:
        mo = re.match(rf"\s*({LETTER})[.)]\s*(.+)", opt)
        if not mo:
            continue
        body = mo.group(2).strip().rstrip(".")
        if len(body) < 3:
            continue
        for found in re.finditer(re.escape(body), tail, re.I):
            hits.append((found.start(), mo.group(1).upper()))
    if hits:
        hits.sort()
        return hits[-1][1], "named"

    return None, None


def looks_repetitive(text, window_words=250, gram=10, threshold=3):
    words = text.split()[-window_words:]
    if len(words) < gram * threshold:
        return False
    seen = {}
    for i in range(len(words) - gram + 1):
        key = " ".join(words[i : i + gram])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= threshold:
            return True
    return False


# --------------------------------------------------------------------------------------
# OpenAI client (stdlib only)
# --------------------------------------------------------------------------------------

DEFAULT_SYSTEM = (
    "You are answering a multiple-choice exam question. Think it through, then reply "
    "with your reasoning inside <Explanation></Explanation> tags followed by the single "
    "letter of the best option inside <Answer></Answer> tags. Emit nothing after "
    "</Answer>."
)


class OpenAIClient:
    def __init__(self, api_key, model, base_url, max_tokens, temperature,
                 reasoning_effort, timeout, max_retries, verbose=False):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        # parameter support differs across model families; learn it from the first 400s
        self._lock = threading.Lock()
        self._use_max_completion_tokens = bool(
            re.match(r"^(o[1-9]|gpt-5|gpt-6)", model, re.I)
        )
        self._send_temperature = (
            temperature is not None and not self._use_max_completion_tokens
        )
        self._send_effort = bool(reasoning_effort)

    def _body(self, prompt, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": msgs}
        with self._lock:
            if self.max_tokens:
                key = "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
                body[key] = self.max_tokens
            if self._send_temperature:
                body["temperature"] = self.temperature
            if self._send_effort:
                body["reasoning_effort"] = self.reasoning_effort
        return body

    def _adapt(self, message):
        """True if a parameter was renamed or dropped and the call is worth retrying."""
        msg = (message or "").lower()
        with self._lock:
            if "max_completion_tokens" in msg and not self._use_max_completion_tokens:
                self._use_max_completion_tokens = True
                return True
            if "max_tokens" in msg and self._use_max_completion_tokens:
                self._use_max_completion_tokens = False
                return True
            if "temperature" in msg and self._send_temperature:
                self._send_temperature = False
                return True
            if "reasoning_effort" in msg and self._send_effort:
                self._send_effort = False
                return True
        return False

    def complete(self, prompt, system):
        url = f"{self.base_url}/chat/completions"
        delay, last_err = 1.5, None
        for attempt in range(self.max_retries + 1):
            payload = json.dumps(self._body(prompt, system)).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - started
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                return {
                    "completion": text,
                    "finish_reason": choice.get("finish_reason"),
                    "latency_s": round(elapsed, 3),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": details.get("reasoning_tokens"),
                    "model": data.get("model") or self.model,
                    "error": None,
                }
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", "replace")
                try:
                    message = json.loads(raw)["error"]["message"]
                except Exception:
                    message = raw[:400]
                last_err = f"HTTP {exc.code}: {message}"
                if exc.code == 400 and self._adapt(message):
                    if self.verbose:
                        print(f"    adapting parameters: {message[:120]}", file=sys.stderr)
                    continue
                if exc.code in (401, 403, 404):
                    break  # key or model problems will not fix themselves
                if exc.code < 500 and exc.code not in (408, 409, 429):
                    break
            except Exception as exc:  # timeouts, connection resets
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(delay + random.uniform(0, 0.75))
                delay = min(delay * 2, 30)
        return {
            "completion": "",
            "finish_reason": "error",
            "latency_s": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "model": self.model,
            "error": last_err or "unknown error",
        }


# --------------------------------------------------------------------------------------
# running the questions
# --------------------------------------------------------------------------------------


def cache_key(model, system, prompt):
    h = hashlib.sha1()
    for part in (model, system or "", prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_cache(path):
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            print(f"  ! cache at {path} is unreadable, starting fresh", file=sys.stderr)
    return {}


def save_cache(path, cache):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    os.replace(tmp, path)


def run_openai(records, client, system, cache, cache_path, workers, limit):
    scope = records[: limit or len(records)]
    todo = []
    for rec in scope:
        key = cache_key(client.model, system, rec["prompt"])
        if key in cache and not cache[key].get("error"):
            continue
        todo.append((rec, key))

    print(f"  {len(scope) - len(todo)} answer(s) already cached, {len(todo)} to fetch "
          f"({workers} in parallel)")
    if not todo:
        return cache

    lock = threading.Lock()
    done = [0]

    def work(item):
        rec, key = item
        out = client.complete(rec["prompt"], system)
        with lock:
            cache[key] = out
            done[0] += 1
            tag = "ERR" if out["error"] else "ok "
            print(f"  [{done[0]:>4}/{len(todo)}] q{rec.get('question')} {tag} "
                  f"{out['latency_s'] if out['latency_s'] is not None else '-'}s", flush=True)
            if out["error"] and done[0] <= 3:
                print(f"        {out['error'][:200]}", file=sys.stderr, flush=True)
            if done[0] % 10 == 0:
                save_cache(cache_path, cache)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    save_cache(cache_path, cache)
    return cache


def attach_player(records, client, system, cache, player_name, limit):
    """Fold cached OpenAI answers into the transcript as a new player."""
    attached = 0
    for rec in records[: limit or len(records)]:
        out = cache.get(cache_key(client.model, system, rec["prompt"]))
        if not out:
            continue
        completion = out.get("completion") or ""
        answer, how = parse_answer(completion, rec.get("options"))
        entry = {
            "answer": answer,
            "how": how,
            "correct": bool(answer) and answer == rec.get("gold"),
            "completion": completion,
            "completion_chars": len(completion),
            "model": out.get("model"),
            "latency_s": out.get("latency_s"),
            "prompt_tokens": out.get("prompt_tokens"),
            "completion_tokens": out.get("completion_tokens"),
            "reasoning_tokens": out.get("reasoning_tokens"),
            "finish_reason": out.get("finish_reason"),
        }
        if out.get("error"):
            entry["error"] = out["error"]
        rec.setdefault("players", {})[player_name] = entry
        attached += 1
    return attached


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def player_stats(records, name, teacher="teacher"):
    rows = [r["players"][name] for r in records if name in r.get("players", {})]
    if not rows:
        return None
    n = len(rows)

    parseable = sum(1 for p in rows if p.get("answer"))
    correct = sum(1 for p in rows if p.get("correct"))
    tagged = sum(1 for p in rows if p.get("how") == "tagged")

    if name == teacher:
        agree = n  # the teacher trivially agrees with itself
    else:
        agree = sum(
            1 for r in records
            if name in r.get("players", {})
            and r["players"][name].get("answer")
            and r["players"][name]["answer"] == r["players"].get(teacher, {}).get("answer")
        )

    truncated = repetition = 0
    think_tokens = []
    for p in rows:
        c = p.get("completion") or ""
        low = c.lower()
        if "<think>" in low and "</think>" not in low:
            truncated += 1
        elif p.get("finish_reason") == "length" and not p.get("answer"):
            truncated += 1
        if looks_repetitive(c):
            repetition += 1
        if p.get("reasoning_tokens") is not None:
            think_tokens.append(p["reasoning_tokens"])

    by_hop = {}
    for r in records:
        if name not in r.get("players", {}):
            continue
        slot = by_hop.setdefault(r.get("hop_count"), {"total": 0, "answered": 0, "correct": 0})
        slot["total"] += 1
        p = r["players"][name]
        if p.get("answer"):
            slot["answered"] += 1
        if p.get("correct"):
            slot["correct"] += 1

    lat = [p["latency_s"] for p in rows if p.get("latency_s") is not None]
    ptok = [p["prompt_tokens"] for p in rows if p.get("prompt_tokens") is not None]
    ctok = [p["completion_tokens"] for p in rows if p.get("completion_tokens") is not None]

    return {
        "name": name,
        "n": n,
        "parseable": parseable,
        "correct": correct,
        "wrong_answered": parseable - correct,
        "no_commit": n - parseable,
        "acc_all": correct / n,
        "acc_answered": correct / parseable if parseable else 0.0,
        "agree_teacher": agree,
        "agree_teacher_pct": agree / n,
        "tagged": tagged,
        "truncated": truncated,
        "repetition": repetition,
        "mean_think_tokens": (sum(think_tokens) / len(think_tokens)) if think_tokens else None,
        "by_hop": by_hop,
        "mean_latency": statistics.mean(lat) if lat else None,
        "p95_latency": sorted(lat)[max(0, math.ceil(0.95 * len(lat)) - 1)] if lat else None,
        "prompt_tokens": sum(ptok) if ptok else None,
        "completion_tokens": sum(ctok) if ctok else None,
        "errors": sum(1 for p in rows if p.get("error")),
        "how_mix": {
            k: sum(1 for p in rows if p.get("how") == k)
            for k in ("tagged", "labelled", "restated", "named")
        },
    }


def agreement(records, a, b):
    """Questions where a and b committed to the same letter."""
    both = sum(
        1 for r in records
        if r.get("players", {}).get(a, {}).get("answer")
        and r["players"][a]["answer"] == r["players"].get(b, {}).get("answer")
    )
    return both, len(records)


def head_to_head(records, a, b):
    """(a right & b wrong, b right & a wrong, both right, both wrong)."""
    aw = bw = both = neither = 0
    for r in records:
        pa = bool(r.get("players", {}).get(a, {}).get("correct"))
        pb = bool(r.get("players", {}).get(b, {}).get("correct"))
        if pa and pb:
            both += 1
        elif pa:
            aw += 1
        elif pb:
            bw += 1
        else:
            neither += 1
    return aw, bw, both, neither


def bradley_terry_elo(records, names, iters=500, lr=0.6, anchor=1000.0, bootstrap=24):
    """Elo from per-question pairwise outcomes (correct beats incorrect, ties draw)."""
    idx = {n: i for i, n in enumerate(names)}
    per_q = []
    for r in records:
        players = r.get("players", {})
        block = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if a not in players or b not in players:
                    continue
                ca, cb = bool(players[a].get("correct")), bool(players[b].get("correct"))
                block.append((idx[a], idx[b], 0.5 if ca == cb else (1.0 if ca else 0.0)))
        if block:
            per_q.append(block)
    if not per_q:
        return {n: (anchor, 0.0) for n in names}

    scale = 400.0 / math.log(10)

    def fit(blocks):
        pairs = [p for block in blocks for p in block]
        theta = [0.0] * len(names)
        for _ in range(iters):
            grad = [0.0] * len(names)
            for ia, ib, s in pairs:
                g = s - 1.0 / (1.0 + math.exp(-(theta[ia] - theta[ib])))
                grad[ia] += g
                grad[ib] -= g
            for i in range(len(theta)):
                theta[i] += lr * grad[i] / len(pairs)
            mean = sum(theta) / len(theta)
            theta = [t - mean for t in theta]
        return [anchor + scale * t for t in theta]

    point = fit(per_q)
    rng = random.Random(20260917)
    draws = [fit([rng.choice(per_q) for _ in per_q]) for _ in range(bootstrap)]
    return {
        n: (point[i], statistics.pstdev([d[i] for d in draws]) if len(draws) > 1 else 0.0)
        for i, n in enumerate(names)
    }


# --------------------------------------------------------------------------------------
# report: inject the new column into the existing report.html
# --------------------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")
DASH = "\u2014"
MID = "\u00b7"
PM = "\u00b1"

EXTRA_CSS = (
    "<style>:root{--openai:#8A4B0B}"
    '@media (prefers-color-scheme:dark){:root:not([data-theme="light"])'
    "{--openai:#E0A25E}}"
    ':root[data-theme="dark"]{--openai:#E0A25E}'
    ".c-o{color:var(--openai);font-weight:600}"
    "th.h-o{color:var(--openai)}</style>"
)


def strip_tags(s):
    return html_mod.unescape(TAG_RE.sub("", s)).strip()


def norm_label(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9<>/ ]", "", strip_tags(s).lower())).strip()


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def cell_value(label, section_title, st, records, teacher):
    """The OpenAI cell for one existing report row, or an em dash where it does not apply."""
    lab = norm_label(label)
    sect = norm_label(section_title)
    n = st["n"]

    if lab.startswith("hop "):
        m = re.search(r"hop (\d+)", lab)
        slot = st["by_hop"].get(int(m.group(1))) if m else None
        if not slot:
            return DASH
        if "accuracy when answered" in sect:
            share = slot["correct"] / slot["answered"] if slot["answered"] else 0.0
            return f"{slot['correct']}/{slot['answered']} {MID} {pct(share)}"
        if "answer rate" in sect:
            share = slot["answered"] / slot["total"] if slot["total"] else 0.0
            return f"{slot['answered']} {MID} {pct(share)}"
        share = slot["correct"] / slot["total"] if slot["total"] else 0.0
        return f"{slot['correct']} {MID} {pct(share)}"

    if "gave the teachers answer" in lab:
        return f"{st['agree_teacher']} / {n}  ({pct(st['agree_teacher_pct'])})"
    if "accuracy as a share of the teachers" in lab:
        t = player_stats(records, teacher, teacher)
        if not t or not t["acc_all"]:
            return DASH
        return f"{st['acc_all'] / t['acc_all'] * 100:.0f}%"
    if "produced a parseable answer" in lab:
        return f"{st['parseable']} / {n}"
    if "correct counting all" in lab:
        return pct(st["acc_all"])
    if "correct when it answered" in lab:
        return pct(st["acc_answered"])
    if "answered in the trained" in lab:
        return f"{st['tagged']} / {n}"
    if lab == "elo":
        return DASH  # see the recomputed Elo section further down
    if lab == "correct":
        return f"{st['correct']} / {n}"
    if "answered but wrong" in lab:
        return f"{st['wrong_answered']} / {n}"
    if "never committed" in lab:
        return f"{st['no_commit']} / {n}"
    if "ran out of tokens" in lab:
        return str(st["truncated"])
    if "mean tokens spent thinking" in lab:
        v = st["mean_think_tokens"]
        return f"{v:.0f}" if v is not None else DASH
    if "repetition loop" in lab:
        return f"{st['repetition']} / {n}"
    return DASH


def inject_column(html, st, records, label, teacher):
    """Append an OpenAI column to every per-player table in the report."""
    if "</head>" in html and ".c-o{" not in html:
        html = html.replace("</head>", EXTRA_CSS + "</head>", 1)

    touched = [0]

    def fix_section(sec_match):
        block = sec_match.group(0)
        tm = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
        title = tm.group(1) if tm else ""

        def fix_table(match):
            table = match.group(0)
            head = re.search(r"<thead>.*?</thead>", table, re.S)
            if not head or "Base student" not in head.group(0):
                return table  # not a per-player table (e.g. the W4A16 comparison)
            touched[0] += 1
            table = table.replace(
                "</tr></thead>",
                f'<th class="h-o">{html_mod.escape(label)}</th></tr></thead>',
                1,
            )

            def fix_row(rm):
                row = rm.group(0)
                first = re.search(r"<td[^>]*>(.*?)</td>", row, re.S)
                value = cell_value(first.group(1) if first else "", title, st, records, teacher)
                return row[: -len("</tr>")] + f'<td class="c-o">{html_mod.escape(value)}</td></tr>'

            body = re.search(r"<tbody>.*?</tbody>", table, re.S)
            if body:
                new_body = re.sub(r"<tr>.*?</tr>", fix_row, body.group(0), flags=re.S)
                table = table.replace(body.group(0), new_body, 1)
            return table

        return re.sub(r"<table>.*?</table>", fix_table, block, flags=re.S)

    html = re.sub(r"<section\b.*?</section>", fix_section, html, flags=re.S)
    return html, touched[0]


def tbl(caption, headers, rows):
    head = "".join(f"<th>{html_mod.escape(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="{cls}">{html_mod.escape(str(val))}</td>' if cls
            else f"<td>{html_mod.escape(str(val))}</td>"
            for val, cls in row
        )
        body += f"<tr>{cells}</tr>"
    cap = f"<caption>{html_mod.escape(caption)}</caption>" if caption else ""
    return f'<div class="tbl"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section(title, inner):
    return f"<section><h2>{html_mod.escape(title)}</h2>{inner}</section>"


def new_sections(records, st, player, label, elo, teacher, others,
                 price_in, price_out, system):
    n = st["n"]
    parts = []

    # --- narrative ---------------------------------------------------------------------
    dist = player_stats(records, "distilled", teacher)
    tea = player_stats(records, teacher, teacher)
    lines = [
        f"<p><strong>{html_mod.escape(label)}</strong> answered the same {n} questions from "
        "the same prompt, with no access to the curriculum. It scores "
        f"<strong>{pct(st['acc_all'])}</strong> on the answer key and gives the teacher's "
        f"answer {pct(st['agree_teacher_pct'])} of the time.</p>"
    ]
    if dist and tea:
        delta = st["acc_all"] - dist["acc_all"]
        verdict = "ahead of" if delta > 0.005 else ("behind" if delta < -0.005 else "level with")
        lines.append(
            f"<p>That is {verdict} the distilled student ({pct(dist['acc_all'])}) by "
            f"{abs(delta) * 100:.1f} points, and "
            f"{'ahead of' if st['acc_all'] > tea['acc_all'] else 'behind'} the teacher "
            f"({pct(tea['acc_all'])}).</p>"
        )
        aw, bw, both, neither = head_to_head(records, player, "distilled")
        lines.append(
            f"<p>Question by question against the distilled student: both right on {both}, "
            f"both wrong on {neither}, {html_mod.escape(label)} alone right on {aw}, the "
            f"distilled student alone right on {bw}.</p>"
        )
    lines.append(
        "<p>This column is an external reference point, not a distillation target - "
        "nothing here was trained on it.</p>"
    )
    parts.append(
        '<section class="verdict"><div><div class="big">'
        f'{pct(st["acc_all"], 0)}<span>the OpenAI reference on the same answer key</span>'
        f'</div></div><div>{"".join(lines)}</div></section>'
    )

    # --- the reference model itself -----------------------------------------------------
    rows = [
        [("Questions asked", ""), (n, "c-o")],
        [("Produced a parseable answer", ""), (f"{st['parseable']} / {n}", "c-o")],
        [("Correct, counting all", ""), (pct(st["acc_all"]), "c-o")],
        [("Correct, when it answered", ""), (pct(st["acc_answered"]), "c-o")],
        [("Gave the teacher's answer", ""),
         (f"{st['agree_teacher']} / {n}  ({pct(st['agree_teacher_pct'])})", "c-o")],
        [("Answered in the trained <Answer> format", ""), (f"{st['tagged']} / {n}", "c-o")],
        [("Fell into a repetition loop", ""), (f"{st['repetition']} / {n}", "c-o")],
        [("API errors after retries", ""), (f"{st['errors']} / {n}", "c-o")],
    ]
    if st["mean_latency"] is not None:
        rows.append([("Mean wall-clock per question", ""), (f"{st['mean_latency']:.2f}s", "c-o")])
        rows.append([("95th percentile", ""), (f"{st['p95_latency']:.2f}s", "c-o")])
    if st["prompt_tokens"] is not None:
        rows.append([("Prompt tokens, total", ""), (f"{st['prompt_tokens']:,}", "c-o")])
    if st["completion_tokens"] is not None:
        rows.append([("Completion tokens, total", ""), (f"{st['completion_tokens']:,}", "c-o")])
    if st["mean_think_tokens"] is not None:
        rows.append([("Mean tokens spent thinking", ""),
                     (f"{st['mean_think_tokens']:.0f}", "c-o")])
    if price_in is not None and price_out is not None and st["prompt_tokens"] is not None:
        cost = (st["prompt_tokens"] / 1e6) * price_in + \
               ((st["completion_tokens"] or 0) / 1e6) * price_out
        rows.append([("Cost for this run", ""), (f"${cost:.4f}", "c-o")])
        rows.append([("Cost per question", ""), (f"${cost / n:.5f}", "c-o")])
    mix = ", ".join(f"{k} {v}" for k, v in st["how_mix"].items() if v)
    rows.append([("How its answer was read off", ""), (mix or DASH, "c-o")])
    parts.append(section(
        f"The OpenAI reference {MID} {label}",
        tbl("Measured on the same held-out questions, in the same run.",
            ["Measure", "Value"], rows),
    ))

    # --- agreement matrix ---------------------------------------------------------------
    pool = others + [player]
    rows = []
    for a in pool:
        row = [(a, "")]
        for b in pool:
            k, total = agreement(records, a, b)
            row.append((pct(k / total) if total else DASH,
                        "c-o" if player in (a, b) else "c-x"))
        rows.append(row)
    parts.append(section(
        "Who answers like whom",
        tbl("Share of questions where the two committed to the same letter. A blank "
            "answer never counts as agreement, so a row need not reach 100% against itself.",
            ["Agreement"] + pool, rows),
    ))

    # --- head to head ---------------------------------------------------------------------
    rows = []
    for opp in others:
        aw, bw, both, neither = head_to_head(records, player, opp)
        rows.append([(opp, ""), (both, "c-x"), (aw, "c-o"), (bw, "c-d"), (neither, "c-x")])
    parts.append(section(
        f"{label} against each player, question by question",
        tbl(f"Out of all {n} questions.",
            ["Opponent", "Both right", f"{label} only", "Opponent only", "Both wrong"], rows),
    ))

    # --- Elo ---------------------------------------------------------------------------------
    rows = []
    for name in sorted(elo, key=lambda k: -elo[k][0]):
        rating, err = elo[name]
        cls = "c-o" if name == player else ("c-d" if name == "distilled" else "c-x")
        rows.append([(name, ""), (f"{rating:.0f}  {PM}{err:.0f}", cls)])
    parts.append(section(
        "Elo, recomputed with the OpenAI reference in the pool",
        tbl("Bradley-Terry fit over per-question pairwise outcomes (correct beats "
            "incorrect, equal outcomes draw), anchored to a mean of 1000, "
            f"{PM} from {24} bootstrap resamples over questions. Recomputed here because "
            "adding a player moves every rating - these are not comparable with the Elo "
            "row in the tables above.",
            ["Player", "Elo"], rows),
    ))

    # --- how it was asked ---------------------------------------------------------------------
    parts.append(section(
        "How the OpenAI reference was asked",
        "<p>The user message is byte-for-byte the <code>prompt</code> field already in the "
        "transcript - the same question, options and formatting every other player saw. "
        "The only addition is a system message asking for the answer format the student "
        "was trained to emit, so the letter can be read off the same way:</p>"
        f'<pre class="cmd">{html_mod.escape(system or "(no system message)")}</pre>'
    ))
    return "".join(parts)


STANDALONE_CSS = """
:root{--paper:#FBFBFD;--ink:#14181F;--muted:#626A78;--rule:#E4E6EC;--card:#F3F4F8;
--accent:#0F6E68;--target:#3B4A7A;--openai:#8A4B0B}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--paper:#12151B;--ink:#E9EBEF;--muted:#98A0AE;--rule:#262B34;--card:#191D25;
--accent:#4FBFB4;--target:#8494C8;--openai:#E0A25E}}
:root[data-theme="dark"]{--paper:#12151B;--ink:#E9EBEF;--muted:#98A0AE;--rule:#262B34;
--card:#191D25;--accent:#4FBFB4;--target:#8494C8;--openai:#E0A25E}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:400 16px/1.65 "IBM Plex Sans","Segoe UI",system-ui,sans-serif}
.wrap{max-width:80rem;margin:0 auto;padding:3.5rem 1.5rem 4rem;display:flex;
flex-direction:column;gap:2.75rem}
.eyebrow{font:500 .72rem/1 ui-monospace,monospace;letter-spacing:.14em;
text-transform:uppercase;color:var(--muted);margin:0 0 .9rem}
h1{font:600 2.5rem/1.1 Georgia,serif;margin:0}
h2{font:600 1.15rem/1.3 Georgia,serif;margin:0 0 1rem}
.lede{color:var(--muted);margin:.5rem 0 0}
header{border-bottom:1px solid var(--rule);padding-bottom:2rem}
.verdict{display:grid;grid-template-columns:minmax(8.5rem,auto) 1fr;gap:2rem;
align-items:start;background:var(--card);border-radius:10px;padding:1.6rem 1.7rem}
.big{font:600 3.4rem/1 ui-monospace,monospace;color:var(--openai);
font-variant-numeric:tabular-nums}
.big span{display:block;font:400 .78rem/1.4 "IBM Plex Sans",sans-serif;
color:var(--muted);margin-top:.5rem}
.verdict p{margin:0 0 .75rem}.verdict p:last-child{margin-bottom:0}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.93rem}
caption{text-align:left;color:var(--muted);font-size:.85rem;padding-bottom:.6rem}
th,td{padding:.6rem .7rem;border-bottom:1px solid var(--rule);text-align:right;
font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{font:500 .72rem/1.3 ui-monospace,monospace;letter-spacing:.08em;
text-transform:uppercase;color:var(--muted);border-bottom-color:var(--ink)}
tbody td:not(:first-child){font-family:ui-monospace,monospace;font-size:.88rem}
.c-b{color:var(--muted)}.c-d{color:var(--accent);font-weight:600}
.c-t{color:var(--target)}.c-x{color:var(--muted)}
.c-o{color:var(--openai);font-weight:600}
.cmd{margin:0;padding:.7rem .9rem;background:var(--card);border-radius:8px;
font:400 .82rem/1.5 ui-monospace,monospace;overflow-x:auto;white-space:pre-wrap}
footer{border-top:1px solid var(--rule);padding-top:1.25rem;color:var(--muted);
font-size:.85rem}
@media (max-width:34rem){h1{font-size:1.9rem}.verdict{grid-template-columns:1fr}}
"""


def standalone_report(records, st, player, label, elo, teacher, others,
                      price_in, price_out, system):
    """Used when no existing report.html was supplied: everything the transcript can prove."""
    pool = others + [player]
    cls_of = {"base": "c-b", "distilled": "c-d", teacher: "c-t", player: "c-o"}
    stats = {name: player_stats(records, name, teacher) for name in pool}

    def row(title, fn):
        return [(title, "")] + [
            (fn(stats[p]) if stats[p] else DASH, cls_of.get(p, "c-x")) for p in pool
        ]

    core = [
        row("Produced a parseable answer", lambda s: f"{s['parseable']} / {s['n']}"),
        row("Correct, counting all", lambda s: pct(s["acc_all"])),
        row("Correct, when it answered", lambda s: pct(s["acc_answered"])),
        row("Gave the teacher's answer",
            lambda s: f"{s['agree_teacher']} / {s['n']}  ({pct(s['agree_teacher_pct'])})"),
        row("Answered in the trained <Answer> format", lambda s: f"{s['tagged']} / {s['n']}"),
        row("Answered, but wrong", lambda s: f"{s['wrong_answered']} / {s['n']}"),
        row("Never committed to a letter", lambda s: f"{s['no_commit']} / {s['n']}"),
        row("Ran out of tokens mid-<think>", lambda s: str(s["truncated"])),
        row("Fell into a repetition loop", lambda s: f"{s['repetition']} / {s['n']}"),
    ]

    hop_rows = []
    for hop in sorted({r.get("hop_count") for r in records if r.get("hop_count") is not None}):
        total = sum(1 for r in records if r.get("hop_count") == hop)
        cells = [(f"hop {hop}  ({total} questions)", "")]
        for p in pool:
            slot = (stats[p] or {}).get("by_hop", {}).get(hop)
            if not slot:
                cells.append((DASH, cls_of.get(p, "c-x")))
            else:
                share = slot["correct"] / slot["total"] if slot["total"] else 0
                cells.append((f"{slot['correct']} {MID} {pct(share)}", cls_of.get(p, "c-x")))
        hop_rows.append(cells)

    body = (
        section("The answer key", tbl(f"{len(records)} held-out questions.",
                                      ["Metric"] + pool, core))
        + section("Accuracy by reasoning depth",
                  tbl("Correct out of everything asked.", ["Reasoning depth"] + pool, hop_rows))
        + new_sections(records, st, player, label, elo, teacher, others,
                       price_in, price_out, system)
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Arena {MID} OpenAI reference</title><style>{STANDALONE_CSS}</style>"
        '</head><body><div class="wrap"><header>'
        '<p class="eyebrow">Knowledge distillation &middot; arena</p>'
        f"<h1>{html_mod.escape(label)} on the held-out set</h1>"
        '<p class="lede">Rebuilt from the arena transcript alone. No existing report.html '
        "was supplied, so figures that need the model weights (perplexity, KL divergence, "
        "throughput) are not shown.</p></header>"
        f"{body}"
        f"<footer>Generated {datetime.now():%Y-%m-%d %H:%M} by openai_arena.py."
        "</footer></div></body></html>"
    )


def rebuild_report(report_path, records, player, label, out_path,
                   price_in, price_out, system, teacher):
    st = player_stats(records, player, teacher)
    if not st:
        raise SystemExit(f"no player named {player!r} in the transcript")

    others = [n for n in records[0].get("players", {}) if n != player]
    elo = bradley_terry_elo(records, others + [player])

    if report_path and os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            html = fh.read()
        if 'class="h-o"' in html:
            raise SystemExit(
                f"{report_path} already carries an injected OpenAI column. Point --report "
                "at the original report so the column is added once, not twice."
            )
        html, touched = inject_column(html, st, records, label, teacher)
        extra = new_sections(records, st, player, label, elo, teacher, others,
                             price_in, price_out, system)
        anchor = "<section><h2>How it was trained</h2>"
        if anchor in html:
            html = html.replace(anchor, extra + anchor, 1)
        else:
            html = html.replace("</div></body>", extra + "</div></body>", 1)
        html = html.replace(
            "</footer>",
            f" The {html_mod.escape(label)} column was added on "
            f"{datetime.now():%Y-%m-%d %H:%M} by openai_arena.py; it is an external "
            "reference, not part of the distillation.</footer>",
            1,
        )
    else:
        if report_path:
            print(f"  ! {report_path} not found, writing a standalone report instead")
        html, touched = standalone_report(records, st, player, label, elo, teacher, others,
                                          price_in, price_out, system), 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    return st, touched


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Ask an OpenAI model the arena questions and rebuild the report with it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--transcript", default="arena-transcript.jsonl",
                    help="input arena transcript")
    ap.add_argument("--report", default="report.html",
                    help="existing report to extend; if it is missing, a standalone one is written")
    ap.add_argument("--out-transcript", default=None,
                    help="where to write the transcript with the new player "
                         "(default: <transcript>.openai<ext>)")
    ap.add_argument("--out-report", default=None,
                    help="where to write the new report (default: <report>.openai.html)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input transcript and report instead")
    ap.add_argument("--api-key", default=None, help="OpenAI API key (default: $OPENAI_API_KEY)")
    ap.add_argument("--model", default="gpt-5", help="OpenAI model id")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--player-name", default=None,
                    help="key for the new player in the transcript (default: openai-<model>)")
    ap.add_argument("--label", default=None,
                    help="column heading in the report (default: the model id)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM,
                    help="system message; pass '' to send the bare prompt")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="only sent to models that accept it")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    ap.add_argument("--cache", default=None,
                    help="answer cache (default: <out-transcript>.cache.json)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the API and rebuild the report from a transcript that already "
                         "has the player")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per 1M input tokens, to show a cost line")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per 1M output tokens")
    ap.add_argument("--teacher", default="teacher", help="player key used as the teacher")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    player = args.player_name or f"openai-{args.model}"
    label = args.label or args.model

    base, ext = os.path.splitext(args.transcript)
    out_transcript = args.out_transcript or (
        args.transcript if args.in_place else f"{base}.openai{ext or '.jsonl'}"
    )
    rbase, rext = os.path.splitext(args.report)
    out_report = args.out_report or (
        args.report if args.in_place else f"{rbase}.openai{rext or '.html'}"
    )
    cache_path = args.cache or f"{os.path.splitext(out_transcript)[0]}.cache.json"

    print(f"reading {args.transcript}")
    records = load_transcript(args.transcript)
    if not records:
        raise SystemExit("transcript is empty")
    print(f"  {len(records)} questions, players: {', '.join(records[0].get('players', {}))}")

    if args.report_only:
        if player not in records[0].get("players", {}):
            raise SystemExit(
                f"--report-only, but there is no player {player!r} in {args.transcript}. "
                "Point --transcript at the transcript this script wrote earlier."
            )
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "no API key: pass --api-key or set OPENAI_API_KEY "
                "(or use --report-only to skip the API)"
            )
        client = OpenAIClient(
            api_key=api_key, model=args.model, base_url=args.base_url,
            max_tokens=args.max_tokens, temperature=args.temperature,
            reasoning_effort=args.reasoning_effort, timeout=args.timeout,
            max_retries=args.max_retries, verbose=args.verbose,
        )
        print(f"asking {args.model} the same {args.limit or len(records)} questions")
        cache = load_cache(cache_path)
        run_openai(records, client, args.system, cache, cache_path, args.workers, args.limit)
        attached = attach_player(records, client, args.system, cache, player, args.limit)
        print(f"  attached {attached} answers as player {player!r}")
        save_transcript(out_transcript, records)
        print(f"wrote {out_transcript}")

    scored = [r for r in records if player in r.get("players", {})]
    st, touched = rebuild_report(args.report, scored, player, label, out_report,
                                 args.price_in, args.price_out, args.system, args.teacher)
    print(f"wrote {out_report}  ({touched} existing tables extended)")
    print(
        f"\n{label}: {st['correct']}/{st['n']} correct ({pct(st['acc_all'])}), "
        f"parseable {st['parseable']}/{st['n']}, "
        f"agrees with the teacher {pct(st['agree_teacher_pct'])}"
        + (f", {st['errors']} API error(s)" if st["errors"] else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
