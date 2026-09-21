#!/usr/bin/env python3
"""Add a Modal-served model as an extra player in a distillation arena.

The companion to openai_arena.py. Where that one asks a hosted OpenAI model the
arena questions, this one asks a model you are serving yourself on Modal - a vLLM
container behind a Modal web endpoint, speaking the OpenAI wire format - and folds
its answers back into the same transcript and the same report.html.

It is chain-aware. If openai_arena.py has already run, this script finds the
transcript and report it wrote and extends *those*, so the report ends up with both
extra columns side by side. If it has not run, the original transcript and report
are used instead. Nothing is guessed silently: every run prints what it found and
what it is about to write, and --dry-run stops right there.

Standalone: standard library only, plus openai_arena.py beside it for the parsing,
metrics and Elo it already implements.

    export MODAL_BASE_URL=https://<workspace>--<app>-serve.modal.run/v1
    export MODAL_API_KEY=...            # or MODAL_KEY / MODAL_SECRET for proxy auth
    python modal_arena.py --transcript arena-transcript.jsonl --report report.html --dry-run
    python modal_arena.py --transcript arena-transcript.jsonl --report report.html

Outputs (next to the inputs unless overridden):
    arena-transcript.modal.jsonl
    report.modal.html

Re-running is cheap: every answer is cached in --cache, so only missing questions
are sent. Use --report-only to rebuild the report from an already-answered
transcript without touching the endpoint.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import openai_arena as oa
except ImportError as exc:  # pragma: no cover - a missing sibling is a setup problem
    raise SystemExit(
        f"modal_arena.py needs openai_arena.py beside it (import failed: {exc})"
    )

DASH, MID, PM = oa.DASH, oa.MID, oa.PM
pct = oa.pct
player_stats = oa.player_stats
section, tbl = oa.section, oa.tbl

# The local arena asks its players with no system message at all - the bare prompt
# through the chat template. A model served on Modal is usually one of those same
# players, so the default here is no system message: byte-for-byte the conditions
# the base, distilled and teacher columns were measured under. Pass --system to
# hand it the answer-format instruction openai_arena.py uses instead.
DEFAULT_SYSTEM = ""

# Everything the pipeline itself can produce. Anything else in a transcript was
# put there by a script like this one, and is what "an earlier pass" means.
CORE_PLAYERS = ("base", "distilled", "distilled-w4a16", "teacher-base")


# --------------------------------------------------------------------------------------
# the endpoint
# --------------------------------------------------------------------------------------


def build_ssl_context(ca_bundle=None):
    """(context, what it trusts).

    Windows builds its trust from the OS certificate store, and that store goes
    stale: a root expires, nobody notices, and every https call fails with
    "certificate has expired" pointing at a server whose own certificate is
    perfectly valid. certifi ships a current bundle and is already in this
    project's environment, so it is preferred when present - the same thing
    requests has always done. --ca-bundle overrides both, for a corporate root.
    """
    explicit = ca_bundle or os.environ.get("SSL_CERT_FILE")
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"--ca-bundle {explicit} does not exist")
        return ssl.create_default_context(cafile=explicit), explicit
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where()), \
            f"certifi {certifi.__version__}"
    except ImportError:
        return ssl.create_default_context(), "the system certificate store"


def parse_header(raw):
    if ":" not in raw:
        raise SystemExit(f"--header wants 'Name: value', got {raw!r}")
    name, value = raw.split(":", 1)
    return name.strip(), value.strip()


def auth_headers(api_key, modal_key, modal_secret, extra):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if modal_key:
        headers["Modal-Key"] = modal_key
    if modal_secret:
        headers["Modal-Secret"] = modal_secret
    headers.update(extra or {})
    return headers


class ModalClient(oa.OpenAIClient):
    """An OpenAI-compatible client that can carry Modal's proxy-auth headers.

    vLLM speaks /v1/chat/completions, so the request body, the retry loop and the
    400-driven parameter adaptation are all inherited. What differs: authentication
    is optional and may be header-based, extra sampling knobs (seed, top_p) and an
    arbitrary --extra-body go out with every call, and a reasoning model's
    `reasoning_content` is folded back into the completion as a <think> block so the
    answer parser and the truncation checks see what they saw for the local players.
    """

    def __init__(self, headers, seed=None, top_p=None, extra_body=None,
                 ssl_context=None, **kw):
        kw.setdefault("api_key", "")
        super().__init__(**kw)
        self.headers = headers
        self.ssl_context = ssl_context
        # An authentication failure is the same on question 1 and question 142.
        # After a few of them the run is over, and the only thing left to decide
        # is how many pointless requests to send first. The answer is none: the
        # rest return the stored error without touching the network, so the run
        # ends in seconds with every row marked and the reason printed once.
        self._auth_halt = None
        self._auth_failures = 0
        self.seed = seed
        self.top_p = top_p
        self.extra_body = extra_body or {}
        # vLLM takes max_tokens whatever the model is called; the inherited name
        # sniffing only exists for OpenAI's own families.
        self._use_max_completion_tokens = False
        self._send_temperature = self.temperature is not None

    def _body(self, prompt, system):
        body = super()._body(prompt, system)
        if self.seed is not None:
            body["seed"] = self.seed
        if self.top_p is not None:
            body["top_p"] = self.top_p
        body.update(self.extra_body)
        return body

    def _halted(self):
        return {
            "completion": "", "finish_reason": "error", "latency_s": None,
            "prompt_tokens": None, "completion_tokens": None,
            "reasoning_tokens": None, "model": self.model, "error": self._auth_halt,
        }

    def complete(self, prompt, system):
        if self._auth_halt:
            return self._halted()
        url = f"{self.base_url}/chat/completions"
        delay, last_err = 1.5, None
        for attempt in range(self.max_retries + 1):
            payload = json.dumps(self._body(prompt, system)).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload, headers=self.headers, method="POST"
            )
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout,
                                            context=self.ssl_context) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - started
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                thinking = message.get("reasoning_content") or ""
                if thinking and "<think>" not in text.lower():
                    text = f"<think>{thinking}</think>\n{text}"
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
                    detail = json.loads(raw)["error"]["message"]
                except Exception:
                    detail = raw[:400]
                last_err = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and self._adapt(detail):
                    if self.verbose:
                        print(f"    adapting parameters: {detail[:120]}", file=sys.stderr)
                    continue
                if exc.code in (401, 403):
                    last_err += "  (check --api-key / --modal-key / --modal-secret)"
                    with self._lock:
                        self._auth_failures += 1
                        if self._auth_failures >= 3 and not self._auth_halt:
                            self._auth_halt = last_err
                            print(f"\n  !! giving up on the rest: the endpoint refuses these "
                                  f"credentials for {self.model}\n     {last_err}\n",
                                  file=sys.stderr, flush=True)
                    break
                if exc.code == 404:
                    last_err += "  (is --base-url the /v1 root of the Modal endpoint?)"
                    break
                if exc.code < 500 and exc.code not in (408, 409, 429):
                    break
            except Exception as exc:  # timeouts, cold starts, connection resets
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


def list_models(base_url, headers, timeout=60.0, context=None):
    """Ask the endpoint what it is serving. Returns (records, error).

    The whole record, not just the id: a served model may describe itself with
    fields OpenAI never defined - who owns it, what domain it was trained for,
    what card it is on - and those belong in the report, because "which model
    was this column" is the first thing anyone asks six months later.
    """
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m for m in (data.get("data") or []) if m.get("id")], None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200]
        return [], f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


# Fields worth printing when the endpoint volunteers them, and what to call them.
MODEL_FIELDS = (("owned_by", "owner"), ("domain", "domain"), ("gpu", "gpu"),
                ("created", "created"), ("max_model_len", "context"),
                ("quantization", "quantization"))


def run_modal(records, client, system, cache, cache_path, workers, limit):
    """Fetch the missing answers, remembering how long the fetch took."""
    before = sum(1 for k in cache if k != "__meta__")
    started = time.time()
    oa.run_openai(records, client, system, cache, cache_path, workers, limit)
    fetched = sum(1 for k in cache if k != "__meta__") - before
    if fetched > 0:
        cache["__meta__"] = {
            "wall_s": round(time.time() - started, 2),
            "fetched": fetched,
            "workers": workers,
            "when": datetime.now().isoformat(timespec="seconds"),
        }
        oa.save_cache(cache_path, cache)
    return cache


# --------------------------------------------------------------------------------------
# what is already on disk - the verification the whole script hangs off
# --------------------------------------------------------------------------------------


def all_players(records):
    """Every player key that appears anywhere in the transcript, in first-seen order."""
    seen = []
    for rec in records:
        for name in rec.get("players", {}):
            if name not in seen:
                seen.append(name)
    return seen


def extra_players(records, teacher="teacher"):
    """Players the local arena did not produce - i.e. columns some script bolted on."""
    core = set(CORE_PLAYERS) | {teacher}
    return [n for n in all_players(records) if n not in core]


def strip_suffix(path, suffix):
    base, ext = os.path.splitext(path)
    if base.endswith(f".{suffix}"):
        base = base[: -len(suffix) - 1]
    return base, ext


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def resolve_transcript(given, teacher, chain=True):
    """(path, records, note). Prefers a sibling .openai transcript when one exists."""
    if not os.path.exists(given):
        raise SystemExit(f"no transcript at {given}")
    records = oa.load_transcript(given)
    if not records:
        raise SystemExit(f"{given} is empty")
    if not chain or extra_players(records, teacher):
        return given, records, None
    base, ext = strip_suffix(given, "modal")
    sibling = f"{base}.openai{ext or '.jsonl'}"
    if os.path.exists(sibling):
        chained = oa.load_transcript(sibling)
        if chained and extra_players(chained, teacher):
            return sibling, chained, f"chained onto the OpenAI pass in {sibling}"
    return given, records, None


def resolve_report(given, chain=True):
    """(path, html, note). Prefers a sibling .openai report when one carries the column."""
    if given and os.path.exists(given):
        html = read_text(given)
        if not chain or 'class="h-o"' in html:
            return given, html, None
    else:
        html = None
    if chain and given:
        base, ext = strip_suffix(given, "modal")
        sibling = f"{base}.openai{ext or '.html'}"
        if os.path.exists(sibling):
            chained = read_text(sibling)
            if 'class="h-o"' in chained:
                return sibling, chained, f"chained onto the OpenAI column in {sibling}"
    return given, html, None


# --------------------------------------------------------------------------------------
# report: inject the Modal column, in its own colour, beside whatever is already there
# --------------------------------------------------------------------------------------

MODAL_CSS = (
    "<style>:root{--modal:#8A2B6B}"
    '@media (prefers-color-scheme:dark){:root:not([data-theme="light"])'
    "{--modal:#E9A2CE}}"
    ':root[data-theme="dark"]{--modal:#E9A2CE}'
    ".c-m{color:var(--modal);font-weight:600}"
    "th.h-m{color:var(--modal)}</style>"
)


# The report's per-player tables are the ones whose rows are metrics and whose columns
# are players. A table that merely mentions the players is not one of them - the
# agreement matrix an earlier pass wrote has "base" and "distilled" in its header too,
# and a column of em dashes down the side of it helps nobody.
PLAYER_TABLE_FIRST_COL = {"metric", "reasoning depth"}


def is_player_table(thead_html):
    """A per-player table is one whose header row names the local players."""
    if "Base student" in thead_html:
        return True
    first = re.search(r"<th[^>]*>(.*?)</th>", thead_html, re.S)
    return bool(
        first
        and oa.norm_label(first.group(1)) in PLAYER_TABLE_FIRST_COL
        and re.search(r"<th[^>]*>\s*distilled\s*</th>", thead_html, re.I)
    )


def inject_column(html, st, records, label, teacher):
    """Append a Modal column to every per-player table in the report."""
    if "</head>" in html and ".c-m{" not in html:
        html = html.replace("</head>", MODAL_CSS + "</head>", 1)

    touched = [0]

    def fix_section(sec_match):
        block = sec_match.group(0)
        tm = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
        title = tm.group(1) if tm else ""

        def fix_table(match):
            table = match.group(0)
            head = re.search(r"<thead>.*?</thead>", table, re.S)
            if not head or not is_player_table(head.group(0)):
                return table  # not a per-player table (e.g. the W4A16 comparison)
            touched[0] += 1
            table = table.replace(
                "</tr></thead>",
                f'<th class="h-m">{html_mod.escape(label)}</th></tr></thead>',
                1,
            )

            def fix_row(rm):
                row = rm.group(0)
                first = re.search(r"<td[^>]*>(.*?)</td>", row, re.S)
                value = oa.cell_value(
                    first.group(1) if first else "", title, st, records, teacher
                )
                return row[: -len("</tr>")] + f'<td class="c-m">{html_mod.escape(value)}</td></tr>'

            body = re.search(r"<tbody>.*?</tbody>", table, re.S)
            if body:
                new_body = re.sub(r"<tr>.*?</tr>", fix_row, body.group(0), flags=re.S)
                table = table.replace(body.group(0), new_body, 1)
            return table

        return re.sub(r"<table>.*?</table>", fix_table, block, flags=re.S)

    html = re.sub(r"<section\b.*?</section>", fix_section, html, flags=re.S)
    return html, touched[0]


def serving_cost(st, meta, rate_per_hour):
    """(total, per question) in USD from measured wall-clock, or None."""
    if rate_per_hour is None or not meta or not meta.get("wall_s"):
        return None
    fetched = meta.get("fetched") or st["n"]
    total = meta["wall_s"] / 3600.0 * rate_per_hour
    return total, total / fetched if fetched else total


def new_sections(records, st, player, label, elo, teacher, others, system,
                 base_url, meta, rate_per_hour, parity_with=None, model_info=None):
    n = st["n"]
    parts = []
    # The player this column is a second copy of, when it is one. An endpoint
    # may be serving the very model in another column - the same weights, packed
    # and served, where the gap between the two is what SERVING cost - or it may
    # be a different model altogether, which is a contender and not a control.
    # Only the person running it knows which, so only --parity-with says so.
    twin = player_stats(records, parity_with, teacher) if parity_with else None
    dist = player_stats(records, "distilled", teacher)
    tea = player_stats(records, teacher, teacher)

    # --- narrative ---------------------------------------------------------------------
    lines = [
        f"<p><strong>{html_mod.escape(label)}</strong>, served from Modal, answered the same "
        f"{n} questions from the same prompt. It scores <strong>{pct(st['acc_all'])}</strong> "
        f"on the answer key and gives the teacher's answer {pct(st['agree_teacher_pct'])} of "
        "the time.</p>"
    ]
    if twin:
        delta = st["acc_all"] - twin["acc_all"]
        verdict = "ahead of" if delta > 0.005 else ("behind" if delta < -0.005 else "level with")
        aw, bw, both, neither = oa.head_to_head(records, player, parity_with)
        same, total = oa.agreement(records, player, parity_with)
        lines.append(
            f"<p>That is {verdict} the locally scored <code>{html_mod.escape(parity_with)}</code> "
            f"({pct(twin['acc_all'])}) by {abs(delta) * 100:.1f} points. Question by question: "
            f"both right on {both}, both wrong on {neither}, the served copy alone right on "
            f"{aw}, the local copy alone right on {bw}, the same letter on {same} of {total} "
            f"({pct(same / total) if total else DASH}).</p>"
        )
        lines.append(
            "<p>These are meant to be the same weights, so the gap between the two columns is "
            "what serving cost them - quantisation, a different kernel, a different sampler - "
            "and not what training bought.</p>"
        )
    else:
        if dist:
            delta = st["acc_all"] - dist["acc_all"]
            verdict = ("ahead of" if delta > 0.005
                       else ("behind" if delta < -0.005 else "level with"))
            lines.append(
                f"<p>That is {verdict} the distilled student ({pct(dist['acc_all'])}) by "
                f"{abs(delta) * 100:.1f} points"
                + (f", and {'ahead of' if st['acc_all'] > tea['acc_all'] else 'behind'} the "
                   f"teacher ({pct(tea['acc_all'])})" if tea else "")
                + ".</p>"
            )
        lines.append(
            "<p>This is a separate model measured on the same answer key, not a copy of "
            "anything else in this table: it was not distilled from this teacher and nothing "
            "here was trained on it. It is a contender, not a control - the tables below say "
            "where it agrees and where it differs, not what any gap cost.</p>"
        )
    parts.append(
        '<section class="verdict"><div><div class="big big-m">'
        f'{pct(st["acc_all"], 0)}<span>the Modal-served model on the same answer key</span>'
        f'</div></div><div>{"".join(lines)}</div></section>'
    )

    # --- the served model itself ---------------------------------------------------------
    rows = [
        [("Questions asked", ""), (n, "c-m")],
        [("Produced a parseable answer", ""), (f"{st['parseable']} / {n}", "c-m")],
        [("Correct, counting all", ""), (pct(st["acc_all"]), "c-m")],
        [("Correct, when it answered", ""), (pct(st["acc_answered"]), "c-m")],
        [("Gave the teacher's answer", ""),
         (f"{st['agree_teacher']} / {n}  ({pct(st['agree_teacher_pct'])})", "c-m")],
        [("Answered in the trained <Answer> format", ""), (f"{st['tagged']} / {n}", "c-m")],
        [("Answered, but wrong", ""), (f"{st['wrong_answered']} / {n}", "c-m")],
        [("Never committed to a letter", ""), (f"{st['no_commit']} / {n}", "c-m")],
        [("Ran out of tokens mid-<think>", ""), (str(st["truncated"]), "c-m")],
        [("Fell into a repetition loop", ""), (f"{st['repetition']} / {n}", "c-m")],
        [("Endpoint errors after retries", ""), (f"{st['errors']} / {n}", "c-m")],
    ]
    if st["mean_latency"] is not None:
        rows.append([("Mean wall-clock per question", ""), (f"{st['mean_latency']:.2f}s", "c-m")])
        rows.append([("95th percentile", ""), (f"{st['p95_latency']:.2f}s", "c-m")])
    if st["prompt_tokens"] is not None:
        rows.append([("Prompt tokens, total", ""), (f"{st['prompt_tokens']:,}", "c-m")])
    if st["completion_tokens"] is not None:
        rows.append([("Completion tokens, total", ""), (f"{st['completion_tokens']:,}", "c-m")])
    if st["mean_think_tokens"] is not None:
        rows.append([("Mean tokens spent thinking", ""),
                     (f"{st['mean_think_tokens']:.0f}", "c-m")])
    if meta and meta.get("wall_s"):
        wall, fetched = meta["wall_s"], meta.get("fetched") or n
        rows.append([("Wall-clock for the whole sweep", ""),
                     (f"{wall:.1f}s for {fetched} question(s), {meta.get('workers', 1)} in "
                      "parallel", "c-m")])
        if wall > 0:
            rows.append([("Throughput", ""), (f"{fetched / wall * 60:.1f} questions/min", "c-m")])
        if st["completion_tokens"] and wall > 0:
            rows.append([("Output tokens per second", ""),
                         (f"{st['completion_tokens'] / wall:.0f} tok/s", "c-m")])
    cost = serving_cost(st, meta, rate_per_hour)
    if cost:
        rows.append([("GPU cost for this run", ""), (f"${cost[0]:.4f}", "c-m")])
        rows.append([("GPU cost per question", ""), (f"${cost[1]:.5f}", "c-m")])
    mix = ", ".join(f"{k} {v}" for k, v in st["how_mix"].items() if v)
    rows.append([("How its answer was read off", ""), (mix or DASH, "c-m")])
    parts.append(section(
        f"The Modal endpoint {MID} {label}",
        tbl("Measured on the same held-out questions, over HTTP, against the model you are "
            "serving yourself. Latency and throughput are of that endpoint under this run's "
            "concurrency - they are not a benchmark of the model in isolation.",
            ["Measure", "Value"], rows),
    ))

    # --- serving parity ------------------------------------------------------------------
    # Only when the caller says the two are the same weights. Against a model
    # trained some other way this table would be measuring a difference it then
    # mislabels as a serving cost.
    if twin:
        aw, bw, both, neither = oa.head_to_head(records, player, parity_with)
        same, total = oa.agreement(records, player, parity_with)
        drift = [
            r for r in records
            if r.get("players", {}).get(player, {}).get("answer")
            != r.get("players", {}).get(parity_with, {}).get("answer")
        ]
        rows = [
            [("Questions compared", ""), (total, "c-x")],
            [("Same letter", ""), (f"{same}  ({pct(same / total) if total else DASH})", "c-m")],
            [("Different letter or a blank on one side", ""), (len(drift), "c-m")],
            [("Both right", ""), (both, "c-x")],
            [("Served copy only", ""), (aw, "c-m")],
            [("Local copy only", ""), (bw, "c-d")],
            [("Both wrong", ""), (neither, "c-x")],
            [("Accuracy, served", ""), (pct(st["acc_all"]), "c-m")],
            [("Accuracy, local", ""), (pct(twin["acc_all"]), "c-d")],
            [("Difference", ""),
             (f"{(st['acc_all'] - twin['acc_all']) * 100:+.1f} points", "c-m")],
        ]
        parts.append(section(
            f"Serving parity with the local {parity_with}",
            tbl("The same weights scored two ways will not always agree: a 4-bit pack, a "
                "different attention kernel and a different sampler each move individual "
                "answers. This is how far they moved here.",
                ["Measure", "Value"], rows),
        ))

    # --- agreement matrix ------------------------------------------------------------------
    pool = others + [player]
    rows = []
    for a in pool:
        row = [(a, "")]
        for b in pool:
            k, total = oa.agreement(records, a, b)
            row.append((pct(k / total) if total else DASH,
                        "c-m" if player in (a, b) else "c-x"))
        rows.append(row)
    parts.append(section(
        "Who answers like whom",
        tbl("Share of questions where the two committed to the same letter. A blank answer "
            "never counts as agreement, so a row need not reach 100% against itself. This "
            "matrix covers every player; an earlier pass may have left a smaller one above.",
            ["Agreement"] + pool, rows),
    ))

    # --- head to head -----------------------------------------------------------------------
    rows = []
    for opp in others:
        aw, bw, both, neither = oa.head_to_head(records, player, opp)
        rows.append([(opp, ""), (both, "c-x"), (aw, "c-m"), (bw, "c-d"), (neither, "c-x")])
    parts.append(section(
        f"{label} against each player, question by question",
        tbl(f"Out of all {n} questions.",
            ["Opponent", "Both right", f"{label} only", "Opponent only", "Both wrong"], rows),
    ))

    # --- Elo ---------------------------------------------------------------------------------
    rows = []
    for name in sorted(elo, key=lambda k: -elo[k][0]):
        rating, err = elo[name]
        cls = "c-m" if name == player else ("c-d" if name == "distilled" else "c-x")
        rows.append([(name, ""), (f"{rating:.0f}  {PM}{err:.0f}", cls)])
    parts.append(section(
        "Elo, recomputed with every player now in the pool",
        tbl("Bradley-Terry fit over per-question pairwise outcomes (correct beats incorrect, "
            "equal outcomes draw), anchored to a mean of 1000, "
            f"{PM} from 24 bootstrap resamples over questions. Recomputed here because adding "
            "a player moves every rating - these are not comparable with the Elo row in the "
            "tables above, nor with an Elo table an earlier pass may have written.",
            ["Player", "Elo"], rows),
    ))

    # --- provenance ---------------------------------------------------------------------------
    served_by = next(
        (r["players"][player].get("served_by") for r in records
         if r.get("players", {}).get(player, {}).get("served_by")), base_url
    )
    model_id = next(
        (r["players"][player].get("model") for r in records
         if r.get("players", {}).get(player, {}).get("model")), label
    )
    # Whatever the endpoint said about itself, so "which model was this column"
    # has an answer in the document rather than in someone's memory.
    described = ""
    for key, title in MODEL_FIELDS:
        value = (model_info or {}).get(key)
        if value in (None, ""):
            continue
        if key == "created":
            try:
                value = datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
            except (TypeError, ValueError, OSError):
                pass
        described += f"{title:<10}{html_mod.escape(str(value))}\n"
    parts.append(section(
        "How the Modal endpoint was asked",
        "<p>The user message is byte-for-byte the <code>prompt</code> field already in the "
        "transcript - the same question, options and formatting every other player saw - sent "
        "to an OpenAI-compatible <code>/chat/completions</code> endpoint:</p>"
        f'<pre class="cmd">endpoint  {html_mod.escape(str(served_by))}\n'
        f"model     {html_mod.escape(str(model_id))}\n"
        f"{described}"
        f'system    {html_mod.escape(system or "(none - the bare prompt, as the local players were asked)")}</pre>'
    ))
    return "".join(parts)


STANDALONE_EXTRA_CSS = (
    ":root{--modal:#8A2B6B}"
    '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--modal:#E9A2CE}}'
    ':root[data-theme="dark"]{--modal:#E9A2CE}'
    ".c-m{color:var(--modal);font-weight:600}th.h-m{color:var(--modal)}"
    ".big-m{color:var(--modal)}"
)


def standalone_report(records, st, player, label, elo, teacher, others, system,
                      base_url, meta, rate_per_hour, parity_with=None,
                      model_info=None):
    """Used when no existing report.html was supplied: everything the transcript can prove."""
    pool = others + [player]
    cls_of = {"base": "c-b", "distilled": "c-d", teacher: "c-t", player: "c-m"}
    stats = {name: player_stats(records, name, teacher) for name in pool}

    def row(title, fn):
        return [(title, "")] + [
            (fn(stats[p]) if stats[p] else DASH, cls_of.get(p, "c-o")) for p in pool
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
                cells.append((DASH, cls_of.get(p, "c-o")))
            else:
                share = slot["correct"] / slot["total"] if slot["total"] else 0
                cells.append((f"{slot['correct']} {MID} {pct(share)}", cls_of.get(p, "c-o")))
        hop_rows.append(cells)

    body = (
        section("The answer key", tbl(f"{len(records)} held-out questions.",
                                      ["Metric"] + pool, core))
        + section("Accuracy by reasoning depth",
                  tbl("Correct out of everything asked.", ["Reasoning depth"] + pool, hop_rows))
        + new_sections(records, st, player, label, elo, teacher, others, system,
                       base_url, meta, rate_per_hour, parity_with, model_info)
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Arena {MID} Modal endpoint</title>"
        f"<style>{oa.STANDALONE_CSS}{STANDALONE_EXTRA_CSS}</style>"
        '</head><body><div class="wrap"><header>'
        '<p class="eyebrow">Knowledge distillation &middot; arena</p>'
        f"<h1>{html_mod.escape(label)} on the held-out set</h1>"
        '<p class="lede">Rebuilt from the arena transcript alone. No existing report.html '
        "was supplied, so figures that need the model weights (perplexity, KL divergence, "
        "throughput on the training box) are not shown.</p></header>"
        f"{body}"
        f"<footer>Generated {datetime.now():%Y-%m-%d %H:%M} by modal_arena.py."
        "</footer></div></body></html>"
    )


def rebuild_report(report_html, report_path, records, player, label, out_path,
                   system, teacher, base_url, meta, rate_per_hour,
                   parity_with=None, model_info=None):
    st = player_stats(records, player, teacher)
    if not st:
        raise SystemExit(f"no player named {player!r} in the transcript")

    others = [n for n in all_players(records) if n != player]
    elo = oa.bradley_terry_elo(records, others + [player])

    if report_html:
        if 'class="h-m"' in report_html:
            raise SystemExit(
                f"{report_path} already carries an injected Modal column. Point --report at "
                "the report without it so the column is added once, not twice."
            )
        html, touched = inject_column(report_html, st, records, label, teacher)
        extra = new_sections(records, st, player, label, elo, teacher, others, system,
                             base_url, meta, rate_per_hour, parity_with, model_info)
        for anchor in ('<section><h2>How it was trained</h2>', "<footer"):
            if anchor in html:
                html = html.replace(anchor, extra + anchor, 1)
                break
        else:
            html = html.replace("</div></body>", extra + "</div></body>", 1)
        if ".big-m{" not in html:
            html = html.replace("</head>", f"<style>{STANDALONE_EXTRA_CSS}</style></head>", 1)
        html = html.replace(
            "</footer>",
            f" The {html_mod.escape(label)} column was added on "
            f"{datetime.now():%Y-%m-%d %H:%M} by modal_arena.py, measured over HTTP against a "
            "model served on Modal.</footer>",
            1,
        )
    else:
        if report_path:
            print(f"  ! {report_path} not found, writing a standalone report instead")
        html = standalone_report(records, st, player, label, elo, teacher, others, system,
                                 base_url, meta, rate_per_hour, parity_with, model_info)
        touched = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    return st, touched


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------


def slug(text):
    return re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-") or "model"


def build_parser():
    ap = argparse.ArgumentParser(
        description="Ask a model served on Modal the arena questions and add it to the report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--transcript", default="arena-transcript.jsonl",
                    help="input arena transcript; a sibling .openai one is preferred when it "
                         "exists, so both columns end up in one report")
    ap.add_argument("--report", default="report.html",
                    help="existing report to extend; a sibling .openai one is preferred when "
                         "it carries the OpenAI column. If neither exists a standalone report "
                         "is written")
    ap.add_argument("--no-chain", action="store_true",
                    help="ignore any OpenAI pass and build from the files named above")
    ap.add_argument("--out-transcript", default=None,
                    help="where to write the transcript with the new player "
                         "(default: <transcript>.modal<ext>)")
    ap.add_argument("--out-report", default=None,
                    help="where to write the new report (default: <report>.modal.html)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the resolved input transcript and report instead")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible root of the Modal endpoint, ending in /v1 "
                         "(default: $MODAL_BASE_URL)")
    ap.add_argument("--api-key", default=None,
                    help="bearer token, if the endpoint wants one (default: $MODAL_API_KEY, "
                         "then $MODAL_TOKEN)")
    ap.add_argument("--modal-key", default=None,
                    help="Modal proxy-auth token id, sent as Modal-Key (default: $MODAL_KEY)")
    ap.add_argument("--modal-secret", default=None,
                    help="Modal proxy-auth secret, sent as Modal-Secret (default: $MODAL_SECRET)")
    ap.add_argument("--header", action="append", default=[], metavar="'Name: value'",
                    help="extra request header; repeatable")
    ap.add_argument("--ca-bundle", default=None, metavar="PEM",
                    help="certificate authorities to verify the endpoint against "
                         "(default: $SSL_CERT_FILE, else certifi, else the system store)")
    ap.add_argument("--model", required=True,
                    help="served model id. Required: an endpoint can serve several, "
                         "and which one a column was measured on is not a detail to "
                         "leave to whatever /v1/models happens to list first")
    ap.add_argument("--player-name", default=None,
                    help="key for the new player in the transcript (default: modal-<model>)")
    ap.add_argument("--label", default=None,
                    help="column heading in the report (default: modal <model>)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM,
                    help="system message; the default is none, matching how the local players "
                         "were asked")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="vLLM sampling seed")
    ap.add_argument("--extra-body", default=None,
                    help='JSON merged into every request body, e.g. '
                         '\'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="only sent to models that accept it")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests; keep it under the endpoint's batch capacity")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-request timeout; the first call may wait out a cold start")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    ap.add_argument("--cache", default=None,
                    help="answer cache (default: <out-transcript>.cache.json)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the endpoint and rebuild the report from a transcript that "
                         "already has the player")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the inputs and the endpoint, print the plan, write nothing")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None,
                    help="USD/hour for the Modal GPU, to show a cost line")
    ap.add_argument("--parity-with", default=None, metavar="PLAYER",
                    help="the player this endpoint is serving a copy of, e.g. "
                         "'distilled-w4a16'. Adds a serving-parity section reading the "
                         "gap between the two as what serving cost. Leave unset for a "
                         "model trained some other way - it is a contender, not a control")
    ap.add_argument("--teacher", default="teacher", help="player key used as the teacher")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    chain = not args.no_chain

    # --- what is on disk -----------------------------------------------------------------
    transcript_path, records, t_note = resolve_transcript(
        args.transcript, args.teacher, chain)
    report_path, report_html, r_note = resolve_report(args.report, chain)
    present = all_players(records)
    extras = extra_players(records, args.teacher)

    # --- the endpoint --------------------------------------------------------------------
    base_url = args.base_url or os.environ.get("MODAL_BASE_URL") or ""
    headers = auth_headers(
        args.api_key or os.environ.get("MODAL_API_KEY") or os.environ.get("MODAL_TOKEN"),
        args.modal_key or os.environ.get("MODAL_KEY"),
        args.modal_secret or os.environ.get("MODAL_SECRET"),
        dict(parse_header(h) for h in args.header),
    )
    ssl_context, trust = build_ssl_context(args.ca_bundle)
    model, served, endpoint_note, model_info = args.model, [], None, None
    if not args.report_only:
        if not base_url:
            raise SystemExit(
                "no endpoint: pass --base-url or set MODAL_BASE_URL to the /v1 root of the "
                "Modal web endpoint (or use --report-only to skip the endpoint)"
            )
        served, err = list_models(base_url, headers, min(args.timeout, 120.0),
                                  ssl_context)
        if err:
            endpoint_note = f"could not list models: {err}"
            if not model:
                hint = ("Pass --model to skip discovery if the endpoint is up but does not "
                        "serve /v1/models, or check the URL and the credentials.")
                if "CERTIFICATE_VERIFY_FAILED" in err:
                    hint = (
                        f"This run verified against {trust}. A stale OS certificate store is "
                        "the usual cause on Windows - the server's own certificate is often "
                        "fine.\n"
                        "  Try:  python -m pip install --upgrade certifi\n"
                        "  or:   --ca-bundle <path to a current .pem>  (a corporate root "
                        "belongs here too)")
                elif "HTTP 401" in err or "HTTP 403" in err:
                    hint = (
                        "The endpoint answered, but rejected the credentials. Set the ones "
                        "your deployment uses:\n"
                        "  proxy auth:   $env:MODAL_KEY / $env:MODAL_SECRET\n"
                        "  bearer token: $env:MODAL_API_KEY\n"
                        "  anything else: --header 'Name: value'")
                raise SystemExit(f"{base_url}/models did not answer ({err}).\n  {hint}")
        else:
            ids = [m.get("id") for m in served]
            endpoint_note = "reachable, serving: " + (", ".join(ids) or "(nothing listed)")
            # /v1/models answered, so its list is authoritative: a model that is
            # not on it will fail on every question, and failing now costs one
            # request instead of a sweep. (When the listing itself fails we do
            # not second-guess the caller - see the branch above.)
            if ids and model not in ids:
                raise SystemExit(
                    f"{base_url} does not serve {model!r}.\n"
                    f"  It serves: {', '.join(ids)}\n"
                    "  Run scripts/probe_modal.py to see which of those your "
                    "credentials can actually call.")
            model_info = next((m for m in served if m.get("id") == model), None)
            described = ", ".join(
                f"{title} {model_info[key]}" for key, title in MODEL_FIELDS
                if (model_info or {}).get(key) not in (None, "") and key != "created")
            if described:
                endpoint_note += f"\n                    {described}"
    player = args.player_name or f"modal-{slug(model)}"
    label = args.label or str(model)

    tbase, text = strip_suffix(transcript_path, "openai")
    tbase, _ = strip_suffix(tbase + (text or ".jsonl"), "modal")
    out_transcript = args.out_transcript or (
        transcript_path if args.in_place else f"{tbase}.modal{text or '.jsonl'}"
    )
    rbase, rext = strip_suffix(args.report, "openai")
    rbase, _ = strip_suffix(rbase + (rext or ".html"), "modal")
    out_report = args.out_report or (
        report_path if (args.in_place and report_path) else f"{rbase}.modal{rext or '.html'}"
    )
    # Adding a SECOND served model chains onto the first one's output, and the
    # default name for that output is the name it already has - so the run would
    # write over the file it was given without being asked to. Both readings are
    # reasonable (accumulate in place, or keep each model's file), so neither is
    # assumed: the run stops and the caller says which.
    def same_file(a, b):
        return a and b and os.path.abspath(a) == os.path.abspath(b)

    if not args.in_place:
        if same_file(out_transcript, transcript_path) and not args.out_transcript:
            raise SystemExit(
                f"the default output is the input itself ({transcript_path}).\n"
                "  That happens when you chain onto a transcript this script already "
                "wrote - adding a second served model, say.\n"
                "  --in-place                     accumulate every column in that one file\n"
                f"  --out-transcript <name>.jsonl  keep this model's answers separately")
        if same_file(out_report, report_path) and not args.out_report:
            raise SystemExit(
                f"the default report output is the input report ({report_path}).\n"
                "  Pass --out-report <name>.html, or --in-place to overwrite it.")

    cache_path = args.cache or f"{os.path.splitext(out_transcript)[0]}.cache.json"
    cache = oa.load_cache(cache_path)
    # Answers held FOR THIS MODEL AND THIS SYSTEM MESSAGE, not entries in the
    # file: the key is a hash of all three, so a cache shared with an earlier
    # model holds nothing this run can use, and saying otherwise would promise a
    # cheap run that is about to be a full one.
    scope_records = records[: args.limit or len(records)]
    cached = sum(
        1 for rec in scope_records
        if not (cache.get(oa.cache_key(model, args.system, rec["prompt"])) or
                {"error": 1}).get("error")
    )

    # --- the plan, always printed before anything is written -------------------------------
    scope = args.limit or len(records)
    print("verification")
    print(f"  transcript        {transcript_path}   ({len(records)} questions)")
    print(f"  players present   {', '.join(present) or '(none)'}")
    if t_note or r_note:
        for note in (t_note, r_note):
            if note:
                print(f"  earlier pass      {note}")
    elif extras and chain:
        print(f"  earlier pass      extra player(s) already in this transcript: "
              f"{', '.join(extras)}")
    else:
        print("  earlier pass      none found, building from the original files"
              + (" (--no-chain)" if not chain else ""))
    print(f"  report base       {report_path or '(none)'}"
          + ("  [has the OpenAI column]" if report_html and 'class="h-o"' in report_html
             else ("  [not found, a standalone report will be written]"
                   if not report_html else "")))
    if args.parity_with:
        if args.parity_with not in present:
            raise SystemExit(
                f"--parity-with {args.parity_with!r} is not in the transcript. "
                f"It has: {', '.join(present)}")
        print(f"  parity with       {args.parity_with} - the gap between the two columns "
              "will be read as what serving cost")
    else:
        print("  parity with       nothing - scored as an independent model, not as a "
              "served copy of another column")
    if player in present:
        print(f"  this player       {player!r} is ALREADY in the transcript and will be "
              "refreshed from the cache/endpoint")
    else:
        print(f"  this player       {player!r} is new")
    if not args.report_only:
        print(f"  endpoint          {base_url}")
        print(f"                    verifying against {trust}")
        print(f"                    {endpoint_note}")
        print(f"  model             {model}")
        print(f"  cache             {cache_path}  ({cached} answer(s) held, "
              f"{max(0, scope - cached)} to fetch at most)")
    print(f"  will write        {out_transcript}")
    print(f"                    {out_report}")
    if args.in_place:
        print("                    (--in-place: the inputs above are the outputs)")

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    # --- ask ------------------------------------------------------------------------------
    if args.report_only:
        if player not in present:
            raise SystemExit(
                f"--report-only, but there is no player {player!r} in {transcript_path}. "
                "Point --transcript at the transcript this script wrote earlier, or pass "
                "--player-name."
            )
    else:
        extra_body = json.loads(args.extra_body) if args.extra_body else {}
        client = ModalClient(
            headers=headers, seed=args.seed, top_p=args.top_p, extra_body=extra_body,
            ssl_context=ssl_context,
            api_key="", model=model, base_url=base_url, max_tokens=args.max_tokens,
            temperature=args.temperature, reasoning_effort=args.reasoning_effort,
            timeout=args.timeout, max_retries=args.max_retries, verbose=args.verbose,
        )
        print(f"\nasking {model} the same {scope} questions")
        run_modal(records, client, args.system, cache, cache_path, args.workers, args.limit)
        attached = oa.attach_player(records, client, args.system, cache, player, args.limit)
        for rec in records:
            entry = rec.get("players", {}).get(player)
            if entry is not None:
                entry["served_by"] = base_url
        print(f"  attached {attached} answers as player {player!r}")
        oa.save_transcript(out_transcript, records)
        print(f"wrote {out_transcript}")

    scored = [r for r in records if player in r.get("players", {})]
    st, touched = rebuild_report(
        report_html, report_path, scored, player, label, out_report, args.system,
        args.teacher, base_url, cache.get("__meta__"), args.gpu_cost_per_hour,
        args.parity_with, model_info,
    )
    print(f"wrote {out_report}  ({touched} existing tables extended)")
    print(
        f"\n{label}: {st['correct']}/{st['n']} correct ({pct(st['acc_all'])}), "
        f"parseable {st['parseable']}/{st['n']}, "
        f"agrees with the teacher {pct(st['agree_teacher_pct'])}"
        + (f", {st['errors']} endpoint error(s)" if st["errors"] else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
