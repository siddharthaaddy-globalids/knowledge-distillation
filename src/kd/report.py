"""
Readable evaluation reports - Markdown and self-contained HTML.

Separated from the measurement code because they answer to different readers.
kd.evaluate produces numbers; this module turns them into something a person who
did not run the job can act on, leading with a plain-English summary so nobody
needs to know what perplexity is to learn whether the training worked.

Every claim in the summary is derived from the measurements rather than asserted.
The HTML is a single file with no external assets, so it can be opened offline or
sent to someone as-is.

THE MAIN SCORE IS CLOSENESS TO THE TEACHER
------------------------------------------
Distillation buys a student that answers like its teacher, so that is what the
page leads with: how often the distilled student gave the teacher's answer on
the held-out set, and how alike its explanations are. Who beat the answer key
comes after, as context - it is as much a fact about the teacher as about the
training. See `_headline` and `_closeness_rows`.

The page also says where the adapter it scored lives - on this machine and in
the bucket - and the one command that re-runs only the evaluation against it,
so the numbers can be reproduced without training anything. See
`_adapter_facts`.
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

    # Closeness first. It is the main score, and the sentence that carries it
    # has to come before anything about the answer key.
    close = _closeness(arena)
    cb, cd = close.get("base") or {}, close.get("distilled") or {}
    same_b, same_d = cb.get("same_answer_pct"), cd.get("same_answer_pct")
    if isinstance(same_d, (int, float)):
        before = (f", up from {same_b * 100:.1f}% before training"
                  if isinstance(same_b, (int, float)) else "")
        lines.append(
            f"On {total} held-out questions the distilled student gave the "
            f"teacher's answer {same_d * 100:.1f}% of the time{before}. That is "
            f"how close it is to the teacher, and it is the number to quote.")
    moved = pct(dist) - pct(base)
    verb = ("is ahead of" if moved > 0 else
            "is level with" if moved == 0 else "is behind")
    lines.append(
        f"On the answer key itself it scores {pct(dist):.1f}%, against "
        f"{pct(base):.1f}% for the same model before training - it {verb} where "
        f"it started, by {abs(moved):.1f} points.")

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

    # What the fine-tune bought the TEACHER, in the same terms. A teacher no
    # better than the stock model it came from had nothing to pass on, and a
    # student that matches it has matched something the stock model already
    # knew - which is worth saying next to the number above.
    teacher_base = players.get("teacher-base")
    if teacher and teacher_base:
        bought = pct(teacher) - pct(teacher_base)
        if bought > 0:
            lines.append(
                f"For scale: the teacher's own fine-tune took it from "
                f"{pct(teacher_base):.1f}% (its stock base) to {pct(teacher):.1f}%, "
                f"a gain of {bought:.1f} points - that is what there was to distil.")
        else:
            lines.append(
                f"For scale: the teacher's stock base already scores "
                f"{pct(teacher_base):.1f}%, at or above the fine-tuned teacher - so "
                f"on this set the fine-tune added nothing for the student to learn.")

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
        return (_arena_summary(payload.get("arena") or {})
                + _quantization_summary(payload))

    fid, cap = payload["fidelity"], payload["capability"]
    lift = fid["agreement_lift_pts"]
    recovered = cap.get("gap_recovered_pct")
    # The answer-level closeness leads even when the token-level numbers are
    # here: it is measured on real questions, and it is the main score.
    lines = _arena_summary(payload.get("arena") or {}) if payload.get("arena") else []

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

    eff = payload["efficiency"]
    if eff["teacher_params"]:
        ratio = eff["student_params"] / eff["teacher_params"]
        speed = ((eff["distilled_tok_per_s"] / eff["teacher_tok_per_s"])
                 if eff.get("teacher_tok_per_s") else None)
        tail = f" and runs {speed:.1f}x faster" if speed else ""
        lines.append(f"It does this at {ratio:.0%} of the teacher's size{tail}.")
    lines += _quantization_summary(payload)
    return lines


def _quantization_summary(payload):
    """What packing to 4 bits cost, in a sentence, or nothing.

    Two sentences rather than one, because the two costs are independent and
    people quote whichever they saw first: a perplexity that barely moves next
    to an accuracy that drops three points is a real and common outcome, and a
    summary naming only the first would be a true sentence used to support a
    false conclusion.
    """
    quant = payload.get("quantization") or {}
    aquant = ((payload.get("arena") or {}).get("quantization")) or {}
    if not (quant or aquant):
        return []

    scheme = quant.get("scheme") or aquant.get("scheme") or "4-bit"
    lines = []

    size = ""
    if quant.get("compression") and quant.get("bytes"):
        size = (f" - {quant['bytes'] / 2 ** 30:.1f} GiB on disk against "
                f"{quant['dense_bytes'] / 2 ** 30:.1f}, {quant['compression']:.1f}x "
                f"smaller")
    change = quant.get("perplexity_change_pct")
    if isinstance(change, (int, float)):
        faster = quant.get("throughput_change_pct")
        speed = (f", and decodes {faster:+.0f}% "
                 f"{'faster' if faster > 0 else 'slower'}"
                 if isinstance(faster, (int, float)) else "")
        lines.append(
            f"Packed to {scheme}{size}, it is {change:+.2f}% worse on perplexity "
            f"over the identical {quant.get('scored_tokens', '?')} tokens{speed}.")
    elif size:
        lines.append(f"Packed to {scheme}{size}.")

    delta = aquant.get("accuracy_delta")
    if isinstance(delta, (int, float)):
        changed = aquant.get("changed_answer")
        moved = (f" It gave a different letter on {changed} of "
                 f"{aquant.get('questions')} questions."
                 if changed is not None else "")
        if abs(delta) < 0.005:
            lines.append(
                f"On the answer key the packed student scores the same as the "
                f"dense one, within half a point.{moved}")
        else:
            lines.append(
                f"On the answer key it scores {delta * 100:+.1f} points against "
                f"the dense student ({aquant['accuracy_packed'] * 100:.1f}% "
                f"against {aquant['accuracy_dense'] * 100:.1f}%).{moved}")
    return lines


def _closeness(arena):
    """{player: {same_answer, of, same_answer_pct}}.

    From the arena's own `closeness` block when it wrote one, else derived here
    from the agreement table - so an arena.json written before the block existed
    still gets the same headline.
    """
    block = (arena.get("closeness") or {}).get("players")
    if block:
        return block
    players = arena.get("players") or {}
    if "teacher" not in players:
        return {}
    agreement = arena.get("agreement") or {}
    pair = lambda table, name: (table.get(f"{name} vs teacher")
                                or table.get(f"teacher vs {name}") or {})
    return {name: {"same_answer": pair(agreement, name).get("same"),
                   "of": pair(agreement, name).get("of"),
                   "same_answer_pct": pair(agreement, name).get("pct")}
            for name in sorted(players) if name != "teacher"}


def _headline(payload):
    """(percentage, caption) for the big number, or (None, caption).

    How close the distilled student is to the teacher. The answer-level figure
    from the arena leads when there is one - it is measured on real questions
    with a real ceiling. kd.evaluate's token-level agreement is the fallback,
    which is the same question asked of the next token instead of the answer.
    """
    close = _closeness(payload.get("arena") or {})
    same = (close.get("distilled") or {}).get("same_answer_pct")
    if isinstance(same, (int, float)) and math.isfinite(same):
        return same * 100, "of the time the distilled student gives the teacher's answer"
    token = (payload.get("closeness_to_teacher") or {}).get(
        "prediction_agreement_distilled_pct")
    if isinstance(token, (int, float)) and math.isfinite(token):
        return token, "of the time the distilled student predicts the teacher's next token"
    return None, "how close the distilled student is to the teacher"


def _adapter_facts(payload):
    """(facts, commands) for the section that says where the adapter is.

    `facts` are (label, value) pairs: the local path, the S3 copy or why there
    is none. `commands` re-run ONLY the evaluation against that adapter - the
    whole point of naming the S3 copy is that anyone with the bucket can
    reproduce these numbers without training anything, and the command they
    need should not have to be assembled from three documents.
    """
    where = payload.get("adapter_locations") or {}
    local = where.get("local") or payload.get("adapter") \
        or (payload.get("arena") or {}).get("adapter")
    if not local and not where.get("s3"):
        return [], []

    facts = [("On this machine", local or "-")]
    if where.get("s3"):
        status = where.get("s3_status")
        facts.append(("On S3", where["s3"] + (f"  — {status}" if status else "")))
    else:
        facts.append(("On S3", f"not there - {where.get('note')}"
                      if where.get("note") else "not recorded"))

    profile = payload.get("profile") or "<profile>.yaml"
    # The S3 copy when there is one: it is the address that works from any
    # machine, which the local path is not.
    target = where.get("s3") or local
    commands = [
        ("everything this report can show (evaluate, arena, report), into the "
         "adapter's own bundle under evaluation/",
         f"./run.sh --config {profile} eval --adapter {target}"),
        ("the answer key and the similarity table alone (no teacher fidelity)",
         f"./run.sh --config {profile} arena --adapter {target}"),
    ]
    return facts, commands


# --------------------------------------------------------------------------- #
# How it was trained
# --------------------------------------------------------------------------- #
def training_settings(config):
    """The training knobs the report explains, lifted from the resolved config.

    Goes into the payload as `training`, so the report is self-contained: the
    page says what the run did without the reader opening config.resolved.yaml
    and knowing which of its ninety keys matter. Returns None when the profile
    turns the section off with `evaluation.report_training: false`.
    """
    if not (config.get("evaluation") or {}).get("report_training", True):
        return None
    gkd = config.get("gkd") or {}
    training = config.get("training") or {}
    lora = config.get("lora") or {}
    return {
        "gkd": {key: gkd.get(key) for key in
                ("beta", "ce_alpha", "lmbda", "temperature", "max_new_tokens",
                 "seq_kd")},
        "training": {key: training.get(key) for key in
                     ("max_steps", "batch_size", "gradient_accumulation_steps",
                      "learning_rate", "lr_scheduler_type", "warmup")},
        "lora": {key: lora.get(key) for key in
                 ("r", "alpha", "dropout", "target_modules")},
    }


def _number(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _training_summary(gkd):
    """One paragraph: what the student actually did during training.

    Built from the two settings that decide it, so the sentence is true of THIS
    run rather than of GKD in general.
    """
    beta, lmbda = _number(gkd.get("beta"), 0.5), _number(gkd.get("lmbda"), 0.0)
    ce_alpha = _number(gkd.get("ce_alpha"), 0.0)
    if lmbda <= 0:
        data = ("The student read the curriculum's own answers; nothing the "
                "student wrote itself was used in training.")
    elif lmbda >= 1:
        data = ("Every training example was written by the student itself, then "
                "corrected by the teacher token by token.")
    else:
        data = (f"In {lmbda:.0%} of batches the student wrote its own answer and "
                f"was corrected on that; in the rest it read the curriculum's "
                f"answers.")
    if beta <= 0.35:
        pull = ("was pulled to cover everything the teacher considered possible "
                "at each token (forward KL: broad, cautious)")
    elif beta >= 0.65:
        pull = ("was pulled to commit to the teacher's most likely tokens "
                "(reverse KL: sharp, decisive)")
    else:
        pull = ("was pulled toward the teacher's token probabilities with a "
                "balanced penalty (symmetric JSD)")
    if ce_alpha <= 0:
        loss = ("The loss is the generalised Jensen-Shannon divergence and "
                "nothing else - no cross-entropy on gold labels, no separate "
                "entropy term.")
    elif ce_alpha >= 1:
        loss = ("The loss is plain cross-entropy on the gold tokens - the "
                "teacher's distribution was computed but carried no weight, "
                "so this run is supervised fine-tuning, not distillation.")
    else:
        loss = (f"The loss is {1 - ce_alpha:.0%} that generalised "
                f"Jensen-Shannon divergence and {ce_alpha:.0%} cross-entropy on "
                f"the gold token, so the curriculum's own answer keeps a pull "
                f"of its own even where the teacher is unsure or wrong. The "
                f"cross-entropy applies only to gold text, never to the "
                f"student's own rollouts.")
    return f"{data} At every token the student {pull}. {loss}"


def _training_knobs(gkd):
    """[(name, value, what it does, what it means at this value)] for the gkd knobs."""
    beta, lmbda = _number(gkd.get("beta"), 0.5), _number(gkd.get("lmbda"), 0.0)
    show = lambda v: "-" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))
    on_policy = lmbda > 0

    if beta <= 0.35:
        beta_now = ("Mostly forward KL: the student is punished for giving no "
                    "weight to a token the teacher likes, so it spreads its bets. "
                    "Safe, but a small student ends up blurry.")
    elif beta >= 0.65:
        beta_now = ("Mostly reverse KL: the student is punished for weight the "
                    "teacher would not give, not for ignoring some of the "
                    "teacher's options. Sharper, more decisive, less variety.")
    else:
        beta_now = ("Balanced: forward and reverse in equal measure, each "
                    "compared to the average of the two. The middle of the "
                    "road; push it up if explanations read vague next to the "
                    "teacher's.")

    if lmbda <= 0:
        lmbda_now = ("Fully off-policy: ordinary distillation. Fast and safe, "
                     "but the student is only ever corrected on good text - "
                     "never on the mistakes it makes on its own.")
    elif lmbda >= 1:
        lmbda_now = ("Fully on-policy: the student writes every answer and is "
                     "corrected on its own mistakes. Slow - a generation every "
                     "step.")
    else:
        lmbda_now = (f"A coin flip per batch: {lmbda:.0%} of batches are on the "
                     f"student's own text, the rest on the curriculum's.")

    inert = "Inert in this run because lmbda is 0: it only applies when the student writes its own text."
    temperature = _number(gkd.get("temperature"))
    temp_now = inert if not on_policy else (
        "Tame, mostly-likely tokens." if temperature is not None and temperature < 0.5
        else "Wilder rollouts that exercise rarer situations." if temperature is not None and temperature > 0.9
        else "Mostly likely tokens, with the occasional less likely one.")
    tokens_now = inert if not on_policy else (
        "The ceiling for one student rollout; a reasoning chain cut off here is "
        "scored as a fragment.")
    seq_kd = gkd.get("seq_kd")
    seq_now = ("The teacher rewrites every curriculum completion before the student "
               "sees it: more teacher-like, at a teacher generation per sample, and "
               "the curriculum's own explanations are thrown away."
               if seq_kd else
               "The curriculum's own completions are used as written - the right "
               "call while the curriculum is good.")

    ce_alpha = _number(gkd.get("ce_alpha"), 0.0)
    if ce_alpha <= 0:
        ce_now = ("Pure distillation: the gold token matters only through the "
                  "weight the teacher gives it.")
    elif ce_alpha >= 1:
        ce_now = ("Pure supervised fine-tuning: the teacher's distribution is "
                  "computed and ignored.")
    elif ce_alpha <= 0.35:
        ce_now = ("Teacher-led, gold-anchored: the divergence does most of the "
                  "work and the gold token keeps the student from following a "
                  "teacher that is soft or wrong on a row.")
    else:
        ce_now = ("Closer to SFT with the teacher as a regulariser than to "
                  "distillation; the teacher's dark knowledge is a minority "
                  "of the signal.")

    return [
        ("beta", show(gkd.get("beta")),
         "Which way the student is pulled toward the teacher. 0 = forward KL "
         "(cover everything the teacher considers possible), 1 = reverse KL "
         "(commit to what the teacher finds most likely), 0.5 = balanced.",
         beta_now),
        ("ce_alpha", show(gkd.get("ce_alpha", 0.0)),
         "How much of the loss is plain cross-entropy on the gold token "
         "(the SFT loss) rather than the divergence from the teacher. 0 = "
         "the teacher alone, 1 = the gold labels alone.",
         ce_now),
        ("lmbda", show(gkd.get("lmbda")),
         "Whose text the lesson is taught on: the fraction of batches where the "
         "student writes its own answer and the teacher corrects THAT. This is "
         "the G in GKD; at 0 it is plain distillation.",
         lmbda_now),
        ("temperature", show(gkd.get("temperature")),
         "How adventurous the student is when it writes its own text for an "
         "on-policy batch. Lower is tamer; higher exercises rarer situations.",
         temp_now),
        ("max_new_tokens", show(gkd.get("max_new_tokens")),
         "How long a student rollout may run in an on-policy batch. A ceiling, "
         "not a target.",
         tokens_now),
        ("seq_kd", show(seq_kd),
         "Who writes the off-policy text: false = the curriculum's real "
         "completions, true = the teacher generates a completion first and the "
         "student trains on that.",
         seq_now),
    ]


def _training_facts(block):
    """(label, value) for the plainer settings: steps, batch, LR, LoRA."""
    training, lora = block.get("training") or {}, block.get("lora") or {}
    facts = []
    steps = training.get("max_steps")
    batch, accum = training.get("batch_size"), training.get("gradient_accumulation_steps")
    if steps is not None:
        facts.append(("Optimizer steps", str(steps)))
    if batch is not None and accum is not None:
        facts.append(("Effective batch", f"{batch} x {accum} = {int(batch) * int(accum)}"))
    if training.get("learning_rate") is not None:
        lr = f"{training['learning_rate']}"
        if training.get("lr_scheduler_type"):
            lr += f", {training['lr_scheduler_type']} schedule"
        if training.get("warmup") is not None:
            lr += f", warmup {training['warmup']}"
        facts.append(("Learning rate", lr))
    if lora.get("r") is not None:
        facts.append(("LoRA", f"r={lora['r']}, alpha={lora.get('alpha')}, "
                              f"dropout={lora.get('dropout')}"))
    if lora.get("target_modules"):
        facts.append(("LoRA targets", ", ".join(str(m) for m in lora["target_modules"])))
    return facts


# The loss, written once, for the section and for the docstring above it.
LOSS_FORMULA = ("L = (1 - ce_alpha) * JSD_beta(P || Q) + ce_alpha * CE\n"
                "JSD_beta(P || Q) = beta * KL(P || M) + (1 - beta) * KL(Q || M),"
                "   M = beta * P + (1 - beta) * Q\n"
                "CE = -log Q(y)   (y = the gold token)")


def loss_note(beta, ce_alpha=0.0):
    """The sentence under the formula, with the ceiling for THIS run's beta.

    Generalised JSD is bounded by the binary entropy of beta - ln 2 = 0.693 at
    0.5, 0.325 at 0.9 - which is the number that tells a reader whether a loss
    curve is high, since the same 0.4 is ordinary at one beta and impossible at
    another. The cross-entropy term is unbounded, so with ce_alpha > 0 the
    ceiling applies to the JSD component alone, which the log prints as jsd=.
    """
    b = _number(beta, 0.5)
    a = _number(ce_alpha, 0.0)
    which = "the JSD part of one token's loss" if a > 0 else "one token's loss"
    if 0 < b < 1:
        bound = -(b * math.log(b) + (1 - b) * math.log(1 - b))
        ceiling = (f"At beta {b:g} {which} is at most "
                   f"{bound:.3f} (the binary entropy of beta), so a running "
                   f"jsd near that is a student that has learned nothing yet.")
    else:
        ceiling = (f"At beta {b:g} the divergence is a plain KL, which is unbounded.")
    if a > 0:
        ceiling += (" The cross-entropy term has no ceiling; it is the student's "
                    "perplexity on the gold text, in nats.")
    return ("P is the teacher's next-token distribution, Q the student's, both "
            "after temperature scaling; averaged over completion tokens only. "
            + ceiling)


# The models a report can have a column for, in the order the columns print,
# and what each column is called. Most sections compare the models; the
# similarity table compares PAIRS of them, and carries its own headers.
#
# The teacher's stock base is a column only when it was scored - a merged
# teacher checkpoint cannot say what it was built from, and a report that
# printed a column of dashes for it would look like a measurement that failed.
COLUMNS = (("teacher-base", "Teacher base"), ("base", "Base student"),
           ("teacher", "Teacher"), ("distilled", "Distilled"),
           ("distilled-w4a16", "Distilled W4A16"))
DEFAULT_HEADERS = ("Teacher base", "Base student", "Teacher", "Distilled")

# What each column IS, which is what colours it. By name rather than by
# position, because the order above is a reading order and not a ranking: the
# two stock checkpoints first, then the teacher they produced, then the students
# distilled from it, then anything external. Colour by position would call
# whatever happens to sit last "the target".
ROLES = {"teacher-base": "c-b", "base": "c-b", "teacher": "c-t",
         "distilled": "c-d", "distilled-w4a16": "c-d"}


def _columns(payload):
    """The model columns this payload can fill: [(player, header), ...].

    A column appears only when something measured it. The packed student is the
    clearest case: most runs never quantise anything, and an always-present
    W4A16 column full of dashes would suggest a measurement that failed rather
    than one that was never asked for.

    A player the pipeline does not produce - an API model, or one served over
    HTTP, added to the transcript afterwards - gets a column too, under its own
    key, after all of them: the pipeline's own columns tell one story in order,
    and an outside reference is not part of it.
    """
    players = (payload.get("arena") or {}).get("players") or {}
    fid = payload.get("fidelity") or {}
    cap = payload.get("capability") or {}
    has_teacher_base = ("teacher-base" in players
                        or "top1_agreement_teacher_base_pct" in fid
                        or "perplexity_teacher_base" in cap)
    has_packed = ("distilled-w4a16" in players or bool(payload.get("quantization")))
    optional = {"teacher-base": has_teacher_base, "distilled-w4a16": has_packed}
    # When the arena says who played, that list is the authority: a report
    # rebuilt from a transcript may be about four of the players, or about ones
    # the pipeline never heard of, and a column nobody measured is worse than a
    # missing one. Without an arena the token-level sections still carry the
    # pipeline's own four, so the shape below is the fallback.
    if players:
        optional = {name: name in players for name, _header in COLUMNS}
        optional["teacher-base"] = has_teacher_base or "teacher-base" in players
    columns = [(name, header) for name, header in COLUMNS
               if optional.get(name, True)]

    known = {name for name, _header in COLUMNS}
    return columns + [(name, name) for name in players if name not in known]


def _headers(payload):
    return ("Metric",) + tuple(header for _name, header in _columns(payload))


def _closeness_rows(arena, total, pct, columns):
    """Rows for the closeness section, one cell per model column.

    The teacher column is the ceiling by definition - it agrees with itself on
    everything - and is printed rather than left blank so the table says what
    100% means.
    """
    close = _closeness(arena)
    if not close:
        return []
    players = arena.get("players") or {}
    names = [name for name, _header in columns]
    entry = lambda name: close.get(name) or {}

    def same(name):
        if name == "teacher":
            return f"{total} / {total}  (100.0%)"
        e = entry(name)
        if e.get("same_answer") is None:
            return "-"
        return f"{e['same_answer']} / {e['of']}  ({pct(e.get('same_answer_pct'))})"

    rows = [("Gave the teacher's answer", *(same(n) for n in names))]
    # Accuracy as a share of the teacher's: the same closeness, asked of the
    # answer key. Skipped when the teacher scored nothing, since a share of
    # zero is not a number.
    tea = (players.get("teacher") or {}).get("accuracy")
    if isinstance(tea, float) and tea > 0:
        def share(name):
            if name == "teacher":
                return "100%"
            value = (players.get(name) or {}).get("accuracy")
            return f"{value / tea * 100:.0f}%" if isinstance(value, float) else "-"
        rows.append(("Accuracy, as a share of the teacher's",
                     *(share(n) for n in names)))
    return rows


def _gib(value):
    """Bytes as GiB, or a dash. Binary, because that is what a disk reports."""
    return f"{value / 2 ** 30:.1f} GiB" if isinstance(value, (int, float)) and value \
        else "-"


def _hop_sections(players, names, columns, pct):
    """Three tables, one per question, split by reasoning depth.

    The eval split is weighted toward depths the curriculum never taught, so a
    single accuracy averages the taught and the untaught together and cannot say
    whether anything GENERALISED. Split by depth, that is the only question.

    Three tables and not one, because they fail apart:
      accuracy               correct out of everything asked at that depth
      answer rate            how often a parseable answer appeared at all
      accuracy when answered correct out of what it committed to

    A model can hold the third flat while the second collapses - that is a
    budget problem, not a knowledge problem - and only reading them side by side
    shows which one moved.
    """
    hops = sorted({hop for name in names if name in players
                   for hop in (players[name].get("by_hop") or {})},
                  key=lambda h: int(h))
    if not hops:
        return []

    headers = ("Reasoning depth",) + tuple(h for _n, h in columns)

    def table(field, counted):
        rows = []
        for hop in hops:
            asked = max((((players[n].get("by_hop") or {}).get(hop) or {}).get("n", 0)
                         for n in names if n in players), default=0)
            cells = []
            for name in names:
                cell = ((players.get(name) or {}).get("by_hop") or {}).get(hop)
                cells.append(counted(cell) if cell else "-")
            rows.append((f"hop {hop}  ({asked} questions)", *cells))
        return rows

    return [
        ("Accuracy by reasoning depth — correct out of everything asked",
         table("accuracy", lambda c: f"{c['correct']} · {pct(c.get('accuracy'))}"),
         headers),
        ("Answer rate by reasoning depth — how often it committed at all",
         table("answer_rate",
               lambda c: f"{c['answered']} · {pct(c.get('answer_rate'))}"),
         headers),
        ("Accuracy when answered, by reasoning depth",
         table("accuracy_when_answered",
               lambda c: (f"{c['correct']}/{c['answered']} · "
                          f"{pct(c.get('accuracy_when_answered'))}"
                          if c.get("answered") else "-")),
         headers),
    ]


def _quantization_section(quant, aquant, arena):
    """What packing the student to 4 bits cost, in one table.

    Both halves of it are differences, and both are measured in the same run on
    the same inputs - the token-level pair on identical completion tokens, the
    answer-key pair on identical questions. That identity is the entire reason
    `distilled` and `distilled-w4a16` both play; see kd.arena.PLAYERS.
    """
    scheme = quant.get("scheme") or aquant.get("scheme") or "W4A16"
    rows = []
    headers = ("Measure", "Dense (bf16)", f"Packed ({scheme})", "Change")

    def num(value, spec=".4f"):
        return (format(value, spec)
                if isinstance(value, (int, float)) and math.isfinite(value) else "-")

    def signed(value, spec="+.2f", suffix=""):
        return (format(value, spec) + suffix
                if isinstance(value, (int, float)) and math.isfinite(value) else "-")

    if quant.get("perplexity") is not None:
        rows.append(("Perplexity", num(quant.get("dense_perplexity")),
                     num(quant.get("perplexity")),
                     signed(quant.get("perplexity_change_pct"), "+.2f", "%")))
        rows.append(("Negative log-likelihood", num(quant.get("dense_nll")),
                     num(quant.get("nll")), signed(quant.get("nll_change"), "+.4f")))
    if quant.get("bytes"):
        ratio = quant.get("compression")
        rows.append(("On disk", _gib(quant.get("dense_bytes")), _gib(quant["bytes"]),
                     f"{ratio:.1f}× smaller" if ratio else "-"))
    if quant.get("tok_per_s"):
        rows.append(("Decode throughput (tokens/sec)",
                     num(quant.get("dense_tok_per_s"), ".2f"),
                     num(quant.get("tok_per_s"), ".2f"),
                     signed(quant.get("throughput_change_pct"), "+.1f", "%")))

    # The answer-key half, from the arena. Independent of everything above: a
    # perplexity that barely moves and an accuracy that drops three points is a
    # real and common outcome, and reporting only the first would miss it.
    if aquant:
        pct = lambda v: (f"{v * 100:.1f}%" if isinstance(v, float) else "-")
        rows.append(("Accuracy on the answer key",
                     pct(aquant.get("accuracy_dense")),
                     pct(aquant.get("accuracy_packed")),
                     signed((aquant.get("accuracy_delta") or 0) * 100
                            if aquant.get("accuracy_delta") is not None else None,
                            "+.1f", " pts")))
        rows.append(("Questions it committed to",
                     str(aquant.get("answered_dense", "-")),
                     str(aquant.get("answered_packed", "-")),
                     signed(aquant.get("answered_delta"), "+.0f")))
        rows.append(("Elo", num(aquant.get("elo_dense"), ".0f"),
                     num(aquant.get("elo_packed"), ".0f"),
                     signed(aquant.get("elo_delta"), "+.1f")))
        if aquant.get("changed_answer") is not None:
            rows.append(("Gave a different letter", "-", "-",
                         f"{aquant['changed_answer']} of "
                         f"{aquant.get('questions', arena.get('questions', 0))}"))

    config = []
    for label, key in (("scheme", "scheme"), ("group size", "group_size"),
                       ("kept at full width", "ignore"),
                       ("calibration sequences", "calibration_samples"),
                       ("format", "format")):
        value = quant.get(key) if quant.get(key) is not None else aquant.get(key)
        if value is not None:
            config.append(f"{label} {value if not isinstance(value, list) else ', '.join(value)}")
    title = f"What {scheme} cost"
    if config:
        title += " — " + " · ".join(config)
    return (title, rows, headers)


def _sections(payload):
    """(title, rows, headers) for every section, headers defaulted."""
    default = _headers(payload)
    return [(s[0], s[1], s[2] if len(s) > 2 else default)
            for s in _report_rows(payload)]


def _report_rows(payload):
    """(section, [(label, cell, cell, ...)]) for both report formats.

    Every row carries one cell per model column (see `_columns`) unless the
    section carries a third element, its own column headers, for a table whose
    columns are not the models.
    """
    fid = payload.get("fidelity")
    cap = payload.get("capability")
    eff = payload.get("efficiency")
    close = payload.get("closeness_to_teacher") or {}
    fmt = lambda v, spec=".4f": (format(v, spec)
                                 if isinstance(v, (int, float)) and math.isfinite(v)
                                 else "-")
    columns = _columns(payload)
    names = [name for name, _header in columns]
    sections = []

    def per_model(values):
        """One cell per column from {player: text}; '-' where nothing was measured."""
        return tuple(values.get(n, "-") for n in names)

    # Closeness to the teacher first: the main score. Then the answer key -
    # who was RIGHT - as context. Fidelity further down says the same thing
    # token by token.
    arena = payload.get("arena") or {}
    players = arena.get("players") or {}
    if players:
        total = arena.get("questions", 0)
        pct = lambda v: (f"{v * 100:.1f}%" if isinstance(v, float) else "-")

        def each(cell):
            return per_model({n: cell(players[n]) for n in names if n in players})

        subset = (f" — a SUBSET of {arena['available']}, not the score"
                  if arena.get("available") and arena.get("limited_to") else "")
        close_rows = _closeness_rows(arena, total, pct, columns)
        if close_rows:
            sections.append(
                (f"How close is it to the teacher? — {total} held-out "
                 f"questions{subset}", close_rows))

        rows = [
            ("Produced a parseable answer",
             *each(lambda e: f"{e['answered']} / {total}")),
            (f"Correct, counting all {total}",
             *each(lambda e: pct(e.get("accuracy")))),
            ("Correct, when it answered",
             *each(lambda e: pct(e.get("accuracy_when_answered")))),
            ("Answered in the trained <Answer> format",
             *each(lambda e: f"{e.get('in_trained_format', 0)} / {total}")),
            ("Elo", *each(lambda e: f"{e['elo']:.0f}  ±{e['elo_spread']:.0f}")),
        ]
        sections.append(
            (f"The answer key — {total} held-out questions{subset} "
             f"(random baseline {arena.get('random_baseline', 0.25) * 100:.0f}%)",
             rows))

        # WHERE THE QUESTIONS WENT. Accuracy counts silence as error, so a model
        # that reasons past the token ceiling and never commits scores the same
        # as one that answers confidently and wrongly. Those are completely
        # different problems, and this is the only table that separates them.
        commitment = [
            ("Correct", *each(lambda e: f"{e.get('correct', 0)} / {total}")),
            ("Answered, but wrong", *each(lambda e: f"{e.get('wrong', 0)} / {total}")),
            ("Never committed to a letter",
             *each(lambda e: f"{e.get('unanswered', 0)} / {total}")),
        ]
        if any((players[n] or {}).get("unterminated_think") for n in names
               if n in players):
            commitment.append(
                ("...of which ran out of tokens mid-<think>",
                 *each(lambda e: str(e.get("unterminated_think", 0)))))
        if any((players[n] or {}).get("mean_think_tokens") for n in names
               if n in players):
            commitment.append(
                ("Mean tokens spent thinking",
                 *each(lambda e: fmt(e.get("mean_think_tokens"), ".0f"))))
        if any((players[n] or {}).get("repetition") for n in names if n in players):
            commitment.append(
                ("Fell into a repetition loop",
                 *each(lambda e: f"{e.get('repetition', 0)} / {total}")))
        sections.append(("Where the questions went", commitment))

        sections += _hop_sections(players, names, columns, pct)

    quant = payload.get("quantization") or {}
    aquant = arena.get("quantization") or {}
    if quant or aquant:
        sections.append(_quantization_section(quant, aquant, arena))

    if not (fid and cap and eff):
        return sections

    # kd.evaluate's columns. The teacher-base cells exist only when the second
    # pass ran (evaluation.players names it), and the column only when
    # _columns says so - the two agree because both read the same keys.
    tb_agree = fid.get("top1_agreement_teacher_base_pct")
    tb_kl = fid.get("kl_teacher_base")
    tb_ppl = cap.get("perplexity_teacher_base")
    quant = payload.get("quantization") or {}
    sections += [
        ("How close is it to the teacher, token by token?", [
            ("Prediction agreement", *per_model({
                "base": fmt(close.get("prediction_agreement_base_pct"), ".2f") + "%",
                "distilled": fmt(close.get("prediction_agreement_distilled_pct"), ".2f") + "%",
                "teacher-base": fmt(tb_agree, ".2f") + "%" if tb_agree is not None else "-",
                "teacher": "100%"})),
        ]),
        ("Fidelity - does it predict what the teacher predicts?", [
            ("Top-1 agreement with teacher", *per_model({
                "base": fmt(fid["top1_agreement_base_pct"], ".2f") + "%",
                "distilled": fmt(fid["top1_agreement_distilled_pct"], ".2f") + "%",
                "teacher-base": fmt(tb_agree, ".2f") + "%" if tb_agree is not None else "-",
                "teacher": "100%"})),
            ("Top-5 overlap with teacher", *per_model({
                "distilled": fmt(fid.get("top5_overlap_distilled"), ".4f"),
                "teacher": "1.0"})),
            ("KL divergence from teacher (lower is better)", *per_model({
                "base": fmt(fid["kl_base"]), "distilled": fmt(fid["kl_distilled"]),
                "teacher-base": fmt(tb_kl), "teacher": "0"})),
        ]),
        ("Capability - is it better at the task?", [
            ("Held-out perplexity (lower is better)", *per_model({
                "base": fmt(cap["perplexity_base"], ".3f"),
                "distilled": fmt(cap["perplexity_distilled"], ".3f"),
                "distilled-w4a16": fmt(quant.get("perplexity"), ".3f"),
                "teacher-base": fmt(tb_ppl, ".3f"),
                "teacher": fmt(cap["perplexity_teacher"], ".3f")})),
        ]),
        ("Cost", [
            ("Parameters", *per_model({
                "distilled": f"{eff['student_params'] / 1e9:.3f}B",
                "teacher-base": (f"{eff['teacher_base_params'] / 1e9:.3f}B"
                                 if eff.get("teacher_base_params") else "-"),
                "teacher": f"{eff['teacher_params'] / 1e9:.3f}B"})),
            ("On disk", *per_model({
                "distilled": _gib(quant.get("dense_bytes")),
                "distilled-w4a16": _gib(quant.get("bytes"))})),
            ("Decode throughput (tokens/sec)", *per_model({
                "distilled": fmt(eff.get("distilled_tok_per_s"), ".1f"),
                "distilled-w4a16": fmt(quant.get("tok_per_s"), ".1f"),
                "teacher": fmt(eff.get("teacher_tok_per_s"), ".1f")})),
            ("Trainable adapter parameters", *per_model({
                "distilled": f"{eff['adapter_params'] / 1e6:.2f}M"})),
        ]),
    ]

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
.wrap{max-width:78rem;margin:0 auto;padding:3.5rem 1.5rem 4rem;
  display:flex;flex-direction:column;gap:2.75rem}
.eyebrow{font:500 .72rem/1 "IBM Plex Mono",ui-monospace,monospace;
  letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 .9rem}
h1{font:600 2.5rem/1.1 Newsreader,Georgia,serif;margin:0;text-wrap:balance;
  letter-spacing:-.01em}
.lede{color:var(--muted);margin:.5rem 0 0;font-size:1.02rem}
h2{font:600 1.15rem/1.3 Newsreader,Georgia,serif;margin:0 0 1rem;text-wrap:balance}
header{border-bottom:1px solid var(--rule);padding-bottom:2rem;position:relative}
.theme{position:absolute;top:0;right:0;appearance:none;cursor:pointer;
  background:var(--card);color:var(--muted);border:1px solid var(--rule);
  border-radius:999px;padding:.35rem .9rem;
  font:500 .72rem/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.08em;
  text-transform:uppercase}
.theme:hover{color:var(--ink);border-color:var(--ink)}
/* Who scored what, ranked. The first thing on the page, and the one block that
   has to read at a glance: name, how far along the track, the number. */
.board{background:var(--card);border-radius:12px;padding:1.5rem 1.6rem 1.7rem}
.board h2{margin:0 0 .35rem}
.board-sub{margin:0 0 1.35rem;color:var(--muted);font-size:.87rem}
.board-grid{display:grid;gap:.85rem;
  grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr))}
.card{position:relative;background:var(--paper);border:1px solid var(--rule);
  border-radius:10px;padding:1rem .95rem .85rem}
.card .rank{position:absolute;top:.6rem;right:.75rem;color:var(--muted);
  font:500 .68rem/1 "IBM Plex Mono",ui-monospace,monospace}
.card .v{font:600 1.85rem/1 "IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.card .v small{font-size:.9rem;font-weight:500;color:var(--muted);margin-left:.08rem}
.card .n{margin:.45rem 0 .7rem;font-size:.8rem;line-height:1.3;color:var(--muted);
  overflow-wrap:anywhere}
.card .t{height:.3rem;background:var(--rule);border-radius:999px;overflow:hidden}
.card .f{height:100%;background:var(--muted);border-radius:999px}
.card .c{margin-top:.5rem;color:var(--muted);
  font:400 .7rem/1 "IBM Plex Mono",ui-monospace,monospace}
.card.ours{border-color:var(--accent);background:var(--card)}
.card.ours .v{color:var(--accent)}
.card.ours .n{color:var(--ink);font-weight:600}
.card.ours .f{background:var(--accent)}
.card.best .rank{color:var(--ink);font-weight:600}

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
.c-t{color:var(--target)} .c-x{color:var(--muted)}
tbody tr:last-child td{border-bottom:none}

dl{display:grid;grid-template-columns:max-content 1fr;gap:.45rem 1.4rem;
  margin:0;font-size:.89rem}
dt{color:var(--muted)}
dd{margin:0;word-break:break-word;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.84rem}
.knob{padding:1rem 0;border-top:1px solid var(--rule)}
.knob:last-of-type{border-bottom:1px solid var(--rule)}
.knob-h{display:flex;justify-content:space-between;align-items:baseline;
  margin-bottom:.35rem}
.knob-h b{font:500 .95rem/1.3 "IBM Plex Mono",ui-monospace,monospace}
.knob-h em{font-style:normal;color:var(--accent);
  font:600 .95rem/1 "IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
.knob p{margin:.3rem 0;font-size:.93rem}
.knob .now{color:var(--muted)}
.knob .now span{color:var(--ink);font-weight:500}
.trained{margin-top:1.25rem}
.cmd-h{margin:1rem 0 .35rem;font-size:.85rem;color:var(--muted)}
.cmd{margin:0;padding:.7rem .9rem;background:var(--card);border-radius:8px;
  font:400 .82rem/1.5 "IBM Plex Mono",ui-monospace,monospace;overflow-x:auto;
  white-space:pre}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85em}
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

    def scoreboard(payload):
        """Who got the most right, ranked, as the first thing on the page.

        Everything else here answers "how close is the student to the teacher",
        which is the question the pipeline exists to answer. This answers the
        question everyone asks first - who scored best - and it should not take
        scrolling and a table to find out. Ranked rather than in column order,
        because a ranking is read in one pass and a row of percentages is not.
        """
        arena = payload.get("arena") or {}
        players = arena.get("players") or {}
        total = arena.get("questions") or 0
        rows = [(name, header, players.get(name) or {})
                for name, header in _columns(payload) if name in players]
        rows = [(n, h, s) for n, h, s in rows
                if isinstance(s.get("accuracy"), (int, float))]
        if not rows or not total:
            return ""
        rows.sort(key=lambda r: -r[2]["accuracy"])
        best = rows[0][2]["accuracy"]
        baseline = arena.get("random_baseline") or 0.25

        out = []
        for rank, (name, header, stats) in enumerate(rows, start=1):
            share = stats["accuracy"]
            correct = stats.get("correct", round(share * total))
            # The accent marks what this pipeline BUILT - the distilled student
            # and its packed copy - wherever they land, because they are what
            # the reader came for. Whoever scored highest is already obvious
            # from the ranking; colouring that too would say the accent means
            # "best", and then a run where the student is not best would read
            # as a run where the accent moved.
            cls = " ours" if ROLES.get(name) == "c-d" else ""
            cls += " best" if share >= best else ""
            # The track runs from the random baseline, not from zero. Every
            # model here is far above guessing, so a zero-based bar draws seven
            # near-identical blocks and says nothing; measured from the point
            # where a coin would sit, the differences are the shape.
            fill = max(0.0, min((share - baseline) / (1 - baseline), 1.0)) * 100
            out.append(
                f'<div class="card{cls}">'
                f'<div class="rank">{rank}</div>'
                f'<div class="v">{share * 100:.1f}<small>%</small></div>'
                f'<div class="n">{esc(header)}</div>'
                f'<div class="t"><div class="f" style="width:{fill:.2f}%"></div></div>'
                f'<div class="c">{correct} of {total}</div></div>')
        return (
            '<section class="board"><h2>Correct on the answer key</h2>'
            f'<p class="board-sub">{total} held-out questions, one answer key, every '
            f'model asked the same way. Bars run from {baseline * 100:.0f}% '
            '(guessing) to 100%.</p>'
            f'<div class="board-grid">{"".join(out)}</div></section>')

    # The closeness bars lead, because they are the main score; the token-level
    # pair from kd.evaluate follow when the evaluation ran.
    answer = _closeness(payload.get("arena") or {})
    ab, ad = answer.get("base") or {}, answer.get("distilled") or {}
    as_pct = lambda v: (v * 100 if isinstance(v, (int, float)) else None)
    bars = "".join([
        bar("Gives the teacher's answer",
            as_pct(ab.get("same_answer_pct")), as_pct(ad.get("same_answer_pct"))),
        bar("Predicts the same next word as the teacher",
            close.get("prediction_agreement_base_pct"),
            close.get("prediction_agreement_distilled_pct")),
    ])

    head = [
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600'
        '&family=Newsreader:opsz,wght@6..72,600&display=swap">',
        f"<style>{_REPORT_CSS}</style>",
    ]

    # rstrip first: an s3:// prefix ends in a slash, and a name split on that
    # is empty - which used to make the lede forget the teacher entirely.
    student = str(payload.get("student") or "the student").rstrip("/").split("/")[-1]
    teacher = str(payload.get("teacher") or "").rstrip("/").split("/")[-1]
    body = ['<div class="wrap"><header>',
            # The page follows the reader's system theme until they say
            # otherwise; the button is for when the room disagrees with the
            # laptop. The choice is remembered per browser, and the stylesheet
            # already keys off data-theme, so this only has to set it.
            '<button id="theme" class="theme" type="button" '
            'aria-label="Switch between light and dark">Light</button>',
            '<p class="eyebrow">Knowledge distillation &middot; evaluation</p>',
            f'<h1>{esc(student)}</h1>',
            (f'<p class="lede">Distilled from <strong>{esc(teacher)}</strong>, '
             f'then measured against it and against its own untrained self.</p>'
             if teacher else
             '<p class="lede">Measured against its own untrained self on a '
             'held-out answer key.</p>'),
            '</header>']

    # Who scored what, first. Then the headline the pipeline cares about.
    body.append(scoreboard(payload))

    # The headline number is how close the distilled student is to the teacher.
    # The caption says which measurement is on the page, because the answer-
    # level and token-level figures would otherwise look identical.
    value, caption = _headline(payload)
    big = f"{value:.0f}%" if value is not None else "&mdash;"
    body.append(f'<section class="verdict"><div><div class="big">{big}'
                f'<span>{esc(caption)}</span>'
                '</div></div><div>'
                + "".join(f"<p>{esc(line)}</p>" for line in summary)
                + '</div></section>')

    if bars:
        body.append(f'<section><h2>How close it got</h2>{bars}</section>')

    # Colour by what each column IS (see ROLES): stock checkpoints muted, the
    # distilled students in the accent, the teacher as the target, anything
    # external neutral. A table whose columns are not the models - the W4A16
    # comparison, with its own headers - falls back to position, where the first
    # cell is the before and the last the after.
    model_headers = tuple(header for _name, header in _columns(payload))
    model_classes = [ROLES.get(name, "c-x") for name, _header in _columns(payload)]

    def styled(cells, headers):
        if tuple(headers[1:]) == model_headers and len(cells) == len(model_classes):
            classes = model_classes
        else:
            classes = ["c-x"] * len(cells)
            if cells:
                classes[0], classes[-1] = "c-b", "c-t"
        return "".join(f'<td class="{cls}">{esc(text)}</td>'
                       for cls, text in zip(classes, cells))

    for title, rows, headers in sections:
        cells = "".join(
            f'<tr><td>{esc(label)}</td>{styled(values, headers)}</tr>'
            for label, *values in rows)
        head_cells = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body.append(
            f'<section><h2>{esc(title)}</h2><div class="tbl"><table><thead><tr>'
            f'{head_cells}'
            f'</tr></thead><tbody>{cells}</tbody></table></div></section>')

    # How it was trained: the knobs, what each does, and what each meant at the
    # value this run used. Here rather than in the tables above because the
    # reader of the numbers is the same person who will ask "so what do I
    # change", and the answer should be on the same page.
    block = payload.get("training")
    if block:
        gkd = block.get("gkd") or {}
        knobs = "".join(
            f'<div class="knob"><div class="knob-h"><b>{esc(name)}</b>'
            f'<em>{esc(value)}</em></div>'
            f'<p>{esc(what)}</p><p class="now"><span>At {esc(value)}:</span> {esc(now)}</p></div>'
            for name, value, what, now in _training_knobs(gkd))
        trained = _training_facts(block)
        body.append(
            '<section><h2>How it was trained</h2>'
            f'<p>{esc(_training_summary(gkd))}</p>'
            f'<pre class="cmd">{esc(LOSS_FORMULA)}</pre>'
            f'<p class="lede">{esc(loss_note(gkd.get("beta"), gkd.get("ce_alpha")))}</p>'
            + knobs
            + ('<dl class="trained">'
               + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in trained)
               + '</dl>' if trained else '')
            + '<p class="lede">Change any of these for one run without editing '
              'the profile: <code>--set gkd.beta=0.9</code>, or the short form '
              '<code>--lmbda 0.25</code>.</p>'
            + '</section>')

    adapter_facts, commands = _adapter_facts(payload)
    if adapter_facts:
        body.append(
            '<section><h2>The adapter</h2><dl>'
            + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in adapter_facts)
            + '</dl>'
            + '<p class="lede">To re-run just the evaluation against it - no '
              'training - from any machine with the bucket:</p>'
            + "".join(f'<p class="cmd-h">{esc(what)}</p><pre class="cmd">{esc(cmd)}</pre>'
                      for what, cmd in commands)
            + '<p class="lede">The first writes into the adapter&#39;s bundle under '
              '<code>evaluation/</code>, here and in the bucket; add '
              '<code>--skip upload</code> to keep the result off S3.</p>'
            + '</section>')

    body.append('<section><h2>This run</h2><dl>'
                + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts)
                + '</dl></section>')
    body.append('<footer>Every figure is also reported for the untrained base '
                'student, because that column is what separates &ldquo;distillation '
                'worked&rdquo; from &ldquo;the small model could already do this&rdquo;. '
                'Read the change, not the absolute value.</footer></div>')

    # Applied before the body renders, so a reader who chose light does not get
    # a flash of dark first. Everything it touches is one attribute, and both
    # the read and the write are guarded: a file:// page with site data blocked
    # throws on localStorage, and a report that will not open is worse than one
    # that forgets a preference.
    script = (
        "<script>(function(){var r=document.documentElement,k='kd-theme';"
        "function paint(t){if(t){r.dataset.theme=t}else{delete r.dataset.theme}"
        "var b=document.getElementById('theme');if(b){var dark=t?t==='dark':"
        "window.matchMedia('(prefers-color-scheme: dark)').matches;"
        "b.textContent=dark?'Light':'Dark'}}"
        "var saved=null;try{saved=localStorage.getItem(k)}catch(e){}paint(saved);"
        "document.addEventListener('DOMContentLoaded',function(){paint(saved);"
        "var b=document.getElementById('theme');if(!b)return;"
        "b.addEventListener('click',function(){var dark=r.dataset.theme?"
        "r.dataset.theme==='dark':window.matchMedia('(prefers-color-scheme: dark)')"
        ".matches;var next=dark?'light':'dark';saved=next;"
        "try{localStorage.setItem(k,next)}catch(e){}paint(next)})})})();</script>")

    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Distillation evaluation</title>"
            + "".join(head) + script + "</head><body>" + "".join(body)
            + "</body></html>")


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
    if payload.get("profile"):
        facts.append(("Profile", payload["profile"]))
    if payload.get("device"):
        hardware = f"{payload['device']} ({payload.get('dtype')})"
        # Named only when it is not the default. Two arenas generated by
        # different engines are not comparable token for token, so a report that
        # used vLLM has to say so; one that used transformers is the baseline
        # and adding "hf" to every report would be noise.
        if arena.get("engine") and arena["engine"] != "hf":
            hardware += f", {arena['engine']}"
        facts.append(("Hardware", hardware))
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
        block = payload.get("training")
        if block:
            gkd = block.get("gkd") or {}
            out += ["", "## How it was trained", "", _training_summary(gkd), "",
                    "```", LOSS_FORMULA, "```", "", loss_note(gkd.get("beta"), gkd.get("ce_alpha")), "",
                    "| Setting | Value | What it does | At this value |",
                    "|---|---|---|---|"]
            out += [f"| `{name}` | **{value}** | {what} | {now} |"
                    for name, value, what, now in _training_knobs(gkd)]
            trained = _training_facts(block)
            if trained:
                out += ["", "| | |", "|---|---|"]
                out += [f"| {k} | {v} |" for k, v in trained]
            out += ["", "Change any of these for one run without editing the "
                    "profile: `--set gkd.beta=0.9`, or the short form `--lmbda 0.25`."]
        adapter_facts, commands = _adapter_facts(payload)
        if adapter_facts:
            out += ["", "## The adapter", "", "| | |", "|---|---|"]
            out += [f"| {k} | {v} |" for k, v in adapter_facts]
            out += ["", "To re-run just the evaluation against it - no training - "
                    "from any machine with the bucket:", ""]
            for what, cmd in commands:
                out += [f"{what}:", "", "```bash", cmd, "```", ""]
            out += ["The first writes into the adapter's bundle under evaluation/, "
                    "here and in the bucket; add `--skip upload` to keep the "
                    "result off S3."]
        out += ["", "## Run", "", "| | |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in facts]
        for title, rows, headers in sections:
            out += ["", f"## {title}", "",
                    "| " + " | ".join(headers) + " |",
                    "|" + "---|" * len(headers)]
            for label, *cells in rows:
                # The distilled column bold, as before: it is the one being read.
                marked = [f"**{c}**" if i == 1 else c for i, c in enumerate(cells)]
                out.append(f"| {label} | " + " | ".join(marked) + " |")
        out += ["", "---", "",
                "Every metric is reported for the untrained base student as well, "
                "because that column is what separates \"distillation worked\" from "
                "\"the small model could already do this\". Read the change, not the "
                "absolute value."]
        target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


