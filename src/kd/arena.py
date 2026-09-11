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

THE HEADLINE IS CLOSENESS TO THE TEACHER
----------------------------------------
Distillation buys a student that answers like its teacher. So the number the
report leads with is not who beat the answer key - that is as much a fact about
the teacher as about the training - but how close the distilled student got to
the teacher: how often it gave the teacher's answer, and how alike its
explanations are. See `closeness`. Accuracy and Elo follow, as context.

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

# The letters an option can be. Deliberately narrow: `[A-Z]` would let a stray
# capital in prose - and English is full of "A" and "I" - be read as an answer.
OPTIONS = "A-F"

# How an answer might be written, most specific first.
#
# ONE of these is the format the curriculum teaches and the student is trained
# to produce. The rest are how a model that was never taught that format still
# manages to answer, and leaving them out is how every player in an arena scores
# zero while answering most questions correctly in plain English.
#
# That is not hypothetical: the first real run scored base, distilled AND
# teacher at 0% with everything unanswered. When every player fails identically,
# the parser is wrong, not the players.
ANSWER_PATTERNS = [
    # The curriculum's own shape: an <Answer> tag with the letter after it.
    # Tried first, so a tagged answer always beats whatever the explanation
    # above it happened to mention.
    ("tagged", re.compile(rf"<Answer>\s*:?\s*\n?\s*([{OPTIONS}])\b", re.I)),
    # "the answer is C", "Answer: D", "answer is **B**"
    ("labelled", re.compile(
        rf"\banswers?\s*(?:is|:|=)\s*\**\(?([{OPTIONS}])\)?\b", re.I)),
    # "option C", "choice B"
    ("named", re.compile(
        rf"\b(?:option|choice)\s+\**\(?([{OPTIONS}])\)?\b", re.I)),
    # A line that is nothing but the letter: "C", "**C**", "(C)", "D."
    # Anchored to the whole line, so "A star forms..." cannot match.
    ("bare", re.compile(
        rf"(?:^|\n)[ \t]*\**\(?([{OPTIONS}])\)?\**[ \t]*[.):]?[ \t]*$", re.M)),
]

# How many unanswered completions to keep per player, and how much of each.
# Enough to see the pattern - they are nearly always the same failure repeated -
# without turning arena.json into a transcript.
UNANSWERED_KEPT = 5
UNANSWERED_CHARS = 800

K_FACTOR = 24
START_RATING = 1000.0
RATING_SCALE = 400.0


def extract_answer_detail(text):
    """(letter, how it was written), or (None, None) if it never committed.

    Patterns are tried in order, and within a pattern the LAST match wins. Both
    rules matter: a model discusses the options before concluding, so the first
    "option B" in a paragraph is usually something being ruled out, and the last
    is the conclusion.

    `how` is worth carrying because "answered" and "answered in the format it
    was trained to produce" are different achievements. A student answering
    `tagged` learned the curriculum's shape; one answering `labelled` is
    answering in spite of it.
    """
    if not text:
        return None, None
    for how, pattern in ANSWER_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].upper(), how
    return None, None


def extract_answer(text):
    """The letter a completion settled on, or None when it never committed."""
    return extract_answer_detail(text)[0]


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
            record = json.loads(line)
            turns = record["messages"]
            gold = extract_answer(turns[-1]["content"])
            if not gold:
                skipped += 1
                continue
            questions.append({
                "prompt": turns[-2]["content"],
                # The curriculum's own answer, in full - the explanation it
                # teaches, not just the letter. Kept so a transcript can show
                # what a model SHOULD have said next to what it did say.
                "reference": turns[-1]["content"],
                "gold": gold,
                # Reasoning depth, carried through by
                # scripts/prepare_curriculum.py. None for a corpus prepared
                # before that, which the hop breakdown then simply omits.
                "hop": record.get("hop_count"),
                "item_id": record.get("item_id"),
            })
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


def correctness(predictions, golds):
    """{player: [was it right?]} - what Elo and the head-to-head record run on."""
    return {name: [p == g for p, g in zip(picks, golds)]
            for name, picks in predictions.items()}


def letter_agreement(predictions):
    """{"a vs b": {same, of, pct}} - how often two players CHOSE THE SAME LETTER.

    Not the same question as the head-to-head record, which asks who was right.
    Two players can agree on every answer and both be wrong, or split every
    question and score identically. Agreement says whether one is tracking the
    other - which for a distilled student and its teacher is the thing being
    bought - and the answer-key columns cannot say it.

    Counted over every question, with an unanswered one never agreeing: two
    models that both failed to produce a letter have not agreed on anything.
    """
    names = sorted(predictions)
    total = len(next(iter(predictions.values()))) if predictions else 0
    table = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = sum(1 for x, y in zip(predictions[a], predictions[b])
                       if x is not None and x == y)
            table[f"{a} vs {b}"] = {
                "same": same, "of": total,
                "pct": (same / total) if total else None}
    return table


# --------------------------------------------------------------------------- #
# Closeness to the teacher
# --------------------------------------------------------------------------- #
# The player the others are measured against. A constant rather than a setting
# because the arena has exactly one ceiling, and a report that let it move would
# be comparing against something the reader has to go and look up.
REFERENCE = "teacher"


def closeness(payload, reference=REFERENCE):
    """How close each student is to the teacher. The headline.

    Two views of one question, both from what the arena already measured, so
    nothing here costs a generation:

      same_answer          how often the player chose the TEACHER's letter,
                           right or wrong. From `agreement`, so it is counted
                           over every question and an unanswered one never
                           agrees.
      explanation_cosine   mean cosine between the player's explanations and
                           the teacher's. From `similarity`, so None when
                           sentence-transformers was absent.

    Returns {"reference": ..., "players": {name: {...}}} for every player that
    is not the reference, or {} when the teacher did not play - there is
    nothing to be close to.

    Read `distilled` against `base`: the rise between them is what training
    bought, and base is what the small model would have said anyway.
    """
    players = payload.get("players") or {}
    if reference not in players:
        return {}
    agreement = payload.get("agreement") or {}
    pairs = (payload.get("similarity") or {}).get("pairs") or {}

    def pair(table, name):
        return table.get(f"{name} vs {reference}") or table.get(f"{reference} vs {name}")

    out = {}
    for name in sorted(players):
        if name == reference:
            continue
        same = pair(agreement, name) or {}
        alike = pair(pairs, name) or {}
        out[name] = {
            "same_answer": same.get("same"),
            "of": same.get("of"),
            "same_answer_pct": same.get("pct"),
            "explanation_cosine": alike.get("overall"),
        }
    return {"reference": reference, "players": out}


def summarise(predictions, golds, formats=None, unanswered=None,
              rounds=25, seed=42):
    """The whole payload: what each player answered, how often it was right, Elo.

    Two accuracies, deliberately, because they answer different questions and a
    single number hides which one is failing:

      over ALL questions      unanswered counts as wrong. What the model is
                              worth in practice, since a response nobody can
                              parse is no use however sound the reasoning was.
      over ANSWERED questions the model's accuracy when it did commit. A model
                              that is right 95% of the time it answers but only
                              answers two thirds of the time has a FORMAT
                              problem, not a knowledge problem - and those have
                              completely different fixes.
    """
    total = len(golds)
    results = correctness(predictions, golds)
    ratings = elo(results, rounds=rounds, seed=seed)
    formats = formats or {}
    unanswered = unanswered or {}

    players = {}
    for name, picks in predictions.items():
        answered = sum(1 for p in picks if p is not None)
        hits = sum(1 for p, g in zip(picks, golds) if p == g)
        # How each answer was written, counted. "tagged" is the curriculum's own
        # shape, so it is the one that says the FORMAT transferred - a student
        # answering correctly in prose has learned the content and not the form,
        # which is a different result and worth being able to see.
        how = {}
        for value in formats.get(name) or []:
            if value:
                how[value] = how.get(value, 0) + 1
        players[name] = {
            "answered": answered,
            "questions": total,
            "correct": hits,
            "accuracy": (hits / total) if total else None,
            "accuracy_when_answered": (hits / answered) if answered else None,
            "unanswered": total - answered,
            "answer_formats": how,
            "in_trained_format": how.get("tagged", 0),
            # The completions that produced no letter at all. The only failure
            # the numbers cannot explain, so the text is kept.
            "unanswered_examples": unanswered.get(name) or [],
            "elo": round(ratings[name]["rating"], 1),
            "elo_spread": round(ratings[name]["spread"], 1),
        }
    payload = {
        "questions": total,
        "random_baseline": 0.25,
        "elo_rounds": rounds,
        "players": players,
        "agreement": letter_agreement(predictions),
        "head_to_head": head_to_head(results),
        "predictions": {name: list(picks) for name, picks in predictions.items()},
        "gold": list(golds),
    }
    # The headline, from the agreement table. Recomputed by the caller once the
    # similarity table exists, which adds the explanation column.
    payload["closeness"] = closeness(payload)
    return payload


def render(payload):
    """The summary table, for a log or a terminal."""
    players = payload["players"]
    order = sorted(players, key=lambda n: -(players[n]["accuracy"] or 0))
    total = payload["questions"]
    width = max(12, *(len(n) for n in order)) if order else 12

    def pct(value):
        return f"{value * 100:.1f}%" if isinstance(value, float) else "-"

    def row(label, cell):
        return ("  " + label.ljust(34)
                + "".join(cell(name).rjust(width + 2) for name in order))

    lines = [
        f"  {total} held-out questions, random baseline "
        f"{payload['random_baseline'] * 100:.0f}%",
    ]
    lines += render_closeness(payload.get("closeness") or closeness(payload))
    lines += [
        "",
        "  " + "measure".ljust(34) + "".join(n.rjust(width + 2) for n in order),
        "  " + "-" * 34 + "".join("-" * (width + 2) for _ in order),
        row("produced a parseable answer",
            lambda n: f"{players[n]['answered']}/{total}"),
        row(f"correct, counting all {total}",
            lambda n: pct(players[n]["accuracy"])),
        row("correct, when it answered",
            lambda n: pct(players[n]["accuracy_when_answered"])),
        row("in the trained <Answer> format",
            lambda n: f"{players[n].get('in_trained_format', 0)}/{total}"),
        row("elo", lambda n: f"{players[n]['elo']:.0f}"),
        row("elo +/-", lambda n: f"{players[n]['elo_spread']:.0f}"),
    ]

    agreement = payload.get("agreement") or {}
    if agreement:
        lines += ["", "  chose the same letter"]
        for pair, entry in sorted(agreement.items()):
            lines.append(f"    {pair:<30} {entry['same']}/{entry['of']}"
                         f"  ({pct(entry['pct'])})")

    lines += ["", "  won on the answer key"]
    for pair, record in sorted((payload.get("head_to_head") or {}).items()):
        lines.append(f"    {pair:<30} {record['win']}W {record['loss']}L "
                     f"{record['draw']}D")

    # Three notes, each answering a question the table above provokes.
    lines += [
        "",
        "  Two accuracy rows because they fail differently. A model far better "
        "on the",
        "  second than the first is not getting questions wrong - it is failing "
        "to say",
        "  an answer in a form anything can read, which is a format problem "
        "with a",
        "  different fix.",
        "",
        f"  +/- is the standard deviation across {payload['elo_rounds']} "
        f"shuffled orderings.",
        "  A gap smaller than it is noise.",
    ]
    return "\n".join(lines)


def render_closeness(close):
    """The headline block: how close each student is to the teacher.

    Separate from the table because it is the one thing to read first, and a
    row lost among eight others is not read first.
    """
    players = (close or {}).get("players") or {}
    if not players:
        return []
    names = sorted(players)

    def same(entry):
        pct = entry.get("same_answer_pct")
        if entry.get("same_answer") is None:
            return "-"
        return (f"{entry['same_answer']}/{entry['of']}"
                + (f" ({pct * 100:.1f}%)" if isinstance(pct, float) else ""))

    def alike(entry):
        value = entry.get("explanation_cosine")
        return f"{value:.3f}" if isinstance(value, float) else "-"

    lines = ["", f"  how close to the {close.get('reference', REFERENCE)} "
                 f"- the headline",
             "    " + "".ljust(32) + "".join(n.rjust(18) for n in names)]
    for label, cell in (("gave the teacher's answer", same),
                        ("explanations alike (cosine)", alike)):
        lines.append("    " + label.ljust(32) + "".join(
            cell(players[n]).rjust(18) for n in names))
    lines.append("    read distilled against base: the rise is what training "
                 "bought")
    return lines


def split_prompt(prompt):
    """(question, options) from the curriculum's tagged prompt.

    Both are already inside `prompt`; pulling them apart is for whoever reads
    the transcript, who wants the options as a list rather than as a substring
    they have to find. Returns the whole prompt as the question and no options
    when the tags are absent, because a transcript that omits a field is worse
    than one that repeats it.
    """
    question = re.search(r"<Question>\s*(.*?)\s*</Question>", prompt, re.S)
    options = re.search(r"<Options>\s*(.*?)\s*</Options>", prompt, re.S)
    return (question.group(1) if question else prompt,
            [line.strip() for line in options.group(1).splitlines() if line.strip()]
            if options else [])


def write_transcript(path, questions, predictions, formats, completions):
    """Everything, per question: what was asked, what each player said, verbatim.

    One line per QUESTION rather than per player, so the three answers to the
    same question sit side by side and "why did they differ" is a matter of
    reading one record instead of joining three files.

    NOTHING IS TRUNCATED OR OMITTED. The whole prompt, the options as a list,
    the reasoning depth, the curriculum's own reference answer, and every word
    each player generated - which for a model that reasons out loud includes its
    <think> block, since that is simply part of what it produced.

    arena.json keeps the numbers; this keeps the evidence. The numbers say what
    happened, and only the evidence says why - which is worth having on disk
    before the machine that produced it is destroyed.

    JSONL because it can be read a line at a time, grepped, and appended to -
    and because 137 questions times three players times a few thousand
    characters is a file to stream rather than load.
    """
    names = sorted(predictions)
    with open(path, "w", encoding="utf-8") as handle:
        for index, question in enumerate(questions):
            text, options = split_prompt(question["prompt"])
            record = {
                "question": index + 1,
                "item_id": question.get("item_id"),
                "hop_count": question.get("hop"),
                "asked": text,
                "options": options,
                "prompt": question["prompt"],
                "gold": question["gold"],
                "reference_answer": question.get("reference"),
                "players": {},
            }
            for name in names:
                completion = (completions.get(name) or [""] * len(questions))[index]
                record["players"][name] = {
                    "answer": predictions[name][index],
                    # How it was written - "tagged" is the curriculum's own
                    # shape, anything else is the model answering in spite of
                    # the format rather than in it.
                    "how": (formats.get(name) or [None] * len(questions))[index],
                    "correct": predictions[name][index] == question["gold"],
                    "completion": completion,
                    "completion_chars": len(completion),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Semantic similarity
# --------------------------------------------------------------------------- #
# The model that turns a completion into a vector. Small (~90 MB), fast on CPU,
# and trained for exactly this: cosine between two of its embeddings is a
# similarity anyone would recognise as one.
#
# Deliberately NOT one of the models being scored. Embedding with a player would
# measure similarity in that player's own representation space, which flatters
# it and makes the three columns incomparable.
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MISSING_SENTENCE_TRANSFORMERS = (
    "sentence-transformers is needed for the similarity table and is not "
    "installed.\n"
    "  uv sync --extra eval        (or: pip install sentence-transformers)\n"
    "It is optional: every other number the arena reports is computed without "
    "it, and the table is simply omitted when it is absent.")


def embed(texts, model_name=SIMILARITY_MODEL, log=None):
    """Unit-normalised embeddings for `texts`, or None if the library is absent.

    Normalised at encode time so a cosine is a dot product, which is what makes
    the pairwise loop below trivial and exact rather than approximately right.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        if log:
            for line in MISSING_SENTENCE_TRANSFORMERS.splitlines():
                log.info(f"      !! {line}")
        return None

    if log:
        log.info(f"      embedding {len(texts)} completions with {model_name}")
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)


def similarity(completions, questions, model_name=SIMILARITY_MODEL, log=None):
    """Cosine similarity between every pair of players, overall and per hop.

    Answers a question the answer key cannot: two models can pick the same
    letter for entirely different reasons, or different letters by nearly
    identical reasoning. Correctness sees neither. This measures how alike the
    EXPLANATIONS are, which for a distilled student and its teacher is close to
    the thing being bought.

    Broken down by hop count because reasoning depth is the axis the curriculum
    is built on. A student that tracks its teacher at one hop and diverges at
    four has a specific, findable weakness; one average over all depths hides
    exactly that.

    Returns None when sentence-transformers is absent - the arena's other
    numbers do not depend on it, so its absence omits a table rather than
    failing a stage.
    """
    names = sorted(completions)
    if len(names) < 2:
        return None

    total = len(questions)
    # One encode call for everything, then slice. Loading the model costs more
    # than embedding a few hundred short texts, so batching across players is
    # most of the saving available here.
    flat = [text for name in names for text in completions[name]]
    vectors = embed(flat, model_name=model_name, log=log)
    if vectors is None:
        return None

    per_player = {name: vectors[i * total:(i + 1) * total]
                  for i, name in enumerate(names)}

    def mean_cosine(a, b, indices):
        if not indices:
            return None
        # Both sides are unit vectors, so the dot product IS the cosine.
        # Clamped because floating point can put it a hair outside [-1, 1], and
        # a similarity of 1.0000000002 reads as a bug.
        values = [max(-1.0, min(1.0, float(per_player[a][i] @ per_player[b][i])))
                  for i in indices]
        return sum(values) / len(values)

    hops = sorted({q.get("hop") for q in questions if q.get("hop") is not None})
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a} vs {b}"
            entry = {"overall": mean_cosine(a, b, list(range(total))),
                     "by_hop": {}}
            for hop in hops:
                indices = [i for i, q in enumerate(questions) if q.get("hop") == hop]
                entry["by_hop"][str(hop)] = {
                    "n": len(indices), "cosine": mean_cosine(a, b, indices)}
            pairs[key] = entry

    return {"model": model_name, "pairs": pairs, "hops": [str(h) for h in hops]}


def render_similarity(sim):
    """The hop-wise similarity table, for a log or a terminal."""
    if not sim or not sim.get("pairs"):
        return ""
    pairs = sorted(sim["pairs"])
    width = max(len(p) for p in pairs)
    hops = sim.get("hops") or []

    lines = ["", f"  explanation similarity (cosine, 0-1) - {sim['model']}",
             "", "  " + "pair".ljust(width + 2)
             + "".join(f"hop {h}".rjust(9) for h in hops) + "overall".rjust(10)
             + "  n"]
    lines.append("  " + "-" * (width + 2) + "-" * (9 * len(hops)) + "-" * 13)
    for pair in pairs:
        entry = sim["pairs"][pair]
        cells = ""
        for hop in hops:
            value = (entry["by_hop"].get(hop) or {}).get("cosine")
            cells += (f"{value:.3f}" if value is not None else "-").rjust(9)
        overall = entry.get("overall")
        cells += (f"{overall:.3f}" if overall is not None else "-").rjust(10)
        total = sum((entry["by_hop"].get(h) or {}).get("n", 0) for h in hops)
        lines.append("  " + pair.ljust(width + 2) + cells + f"  {total}")

    lines += [
        "",
        "  How alike the EXPLANATIONS are, not whether they agree on a letter.",
        "  Two models can pick the same option for different reasons, or differ",
        "  on the letter while reasoning almost identically - the answer key",
        "  sees neither. Read `teacher vs distilled` against `teacher vs base`:",
        "  the rise between them is what distillation moved.",
    ]
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


def _answer_all(model, tokenizer, questions, device, max_new_tokens, label, log,
                show=0):
    """Every question, greedily.

    Returns (predictions, unanswered examples) - a letter or None per question,
    and the first few completions that produced no letter at all.

    KEEPING THE UNANSWERED ONES MATTERS
    -----------------------------------
    An unanswered question is the only failure the numbers cannot explain. A
    wrong letter is a wrong letter; a missing one could be a model that ran out
    of tokens mid-explanation, one that answered in prose the pattern does not
    match, or one that refused - and those have completely different fixes.

    So the completion is kept for a bounded sample of them, and it goes into
    arena.json. Without it, `unanswered: 4` sends you back to re-run the whole
    stage with --show just to see what was said. Bounded because the alternative
    - every completion from every player - is megabytes of text nobody reads
    when the run went fine.
    """
    predictions, formats, unanswered, completions = [], [], [], []
    for index, question in enumerate(questions, start=1):
        text = _generate(model, tokenizer, question["prompt"], device,
                         max_new_tokens)
        predicted, how = extract_answer_detail(text)
        if predicted is None and len(unanswered) < UNANSWERED_KEPT:
            unanswered.append({
                "question": index,
                "gold": question["gold"],
                "prompt": question["prompt"][:400],
                # Both ends: the opening says what shape it started in, and the
                # last line says whether it was still going when the ceiling cut
                # it off - which is the difference between "wrong format" and
                # "needed more tokens".
                "completion_head": text[:UNANSWERED_CHARS],
                "completion_tail": text[-200:] if len(text) > UNANSWERED_CHARS else "",
                "completion_chars": len(text),
            })
        if show and index <= show and log:
            log.info(f"      --- {label} #{index} (gold {question['gold']}, "
                     f"parsed {predicted}) " + "-" * 20)
            for line in text.strip().splitlines()[:24]:
                log.info(f"        {line}")
        predictions.append(predicted)
        formats.append(how)
        completions.append(text)
        if log and (index % 10 == 0 or index == len(questions)):
            hits = sum(1 for p, q in zip(predictions, questions) if p == q["gold"])
            said = sum(1 for p in predictions if p)
            log.info(f"      {label:<10} {index}/{len(questions)}  "
                     f"{hits / index * 100:5.1f}% correct, {said} answered")

    # A player that never answered anything is the case worth interrupting for -
    # it is almost always one thing wrong for every question, not many things.
    if unanswered and log and not any(predictions):
        example = unanswered[0]
        log.info(f"      !! {label} answered NONE of {len(questions)}. "
                 f"What it said to #{example['question']} "
                 f"({example['completion_chars']} chars):")
        for line in example["completion_head"].strip().splitlines()[:8]:
            log.info(f"         {line}")
    return predictions, formats, unanswered, completions


def play(config, hardware, adapter, questions, max_new_tokens=512, log=None,
         players=("base", "distilled", "teacher"), show=0):
    """Load each player in turn, answer every question, return what each said.

    Returns {player: [letter or None, ...]}, one entry per question in order.
    The LETTERS rather than right/wrong, because two of the things worth
    reporting - whether a player answered at all, and whether two players chose
    the same option - cannot be recovered from a list of booleans.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import paths

    device, dtype = hardware["device"], hardware["dtype"]
    # The adapter's own record wins over the config. Both the "base" and
    # "distilled" players load these weights, so getting it wrong would compare
    # the adapter against a control it was never trained on.
    base_id = paths.base_for_adapter(adapter, config["models"]["student"], log=log)

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
        log.info(f"      base     : {base_id}")
        log.info(f"      tokenizer: {source}")

    predictions, formats, unanswered, completions = {}, {}, {}, {}

    def run(label, build):
        if log:
            log.info(f"      loading {label}")
        model = build()
        model.eval()
        try:
            (predictions[label], formats[label], unanswered[label],
             completions[label]) = _answer_all(
                model, tokenizer, questions, device, max_new_tokens, label, log,
                show=show)
        finally:
            _free(model)

    if "base" in players:
        run("base", lambda: AutoModelForCausalLM.from_pretrained(
            base_id, dtype=dtype, low_cpu_mem_usage=True).to(device))

    if "distilled" in players and adapter:
        def build_distilled():
            model = AutoModelForCausalLM.from_pretrained(
                base_id, dtype=dtype, low_cpu_mem_usage=True)
            # The same trim training applied. Reproduced from the two configs
            # rather than carried in the adapter, which would mean shipping a
            # gigabyte of untrained embedding with every run.
            # What the adapter recorded, first: that travels with it, so an
            # arena on another machine needs no teacher to load the student.
            # Falling back to deriving it from the two configs, which needs the
            # teacher present and is why the recorded value exists.
            target = paths.adapter_meta(adapter).get("vocab_size")
            if not target:
                target = paths.vocab_target(
                    base_id, config["models"]["teacher"], len(tokenizer))
            paths.fit_vocab(model, target, label="student")
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

    return predictions, formats, unanswered, completions


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
                            help="Adapter directory, or an s3:// URI which is "
                                 "fetched into the shared cache. Default: the "
                                 "newest under project.runs_dir.")
        parser.add_argument("--file", default=None, metavar="PATH",
                            help="Held-out .jsonl, local or s3://. Default: "
                                 "evaluation.arena_file.")
        parser.add_argument("--limit", type=int, default=None, metavar="N",
                            help="Score only the first N questions")
        parser.add_argument("--show", type=int, default=0, metavar="N",
                            help="Print each player's first N completions. Use "
                                 "this when a player scores 0%% with everything "
                                 "unanswered - it distinguishes a wrong answer "
                                 "from an answer the <Answer> pattern missed.")
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
                            help="Where the payload goes (default: ./arena.json). "
                                 "The transcript and the report are written "
                                 "beside it.")
        parser.add_argument("--report", default=None, metavar="PATH",
                            help="Write an HTML report too (default: "
                                 "<json>-report.html). --report= (empty) skips it.")
        parser.add_argument("--no-save", action="store_true",
                            help="Print the tables and write nothing. For a "
                                 "--limit smoke check, where the output is not "
                                 "worth keeping.")
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

    # s3:// anywhere a path is taken. The adapter a run uploaded is the obvious
    # thing to score from another machine, and making the caller download it
    # first is a step with no judgement in it.
    from . import paths

    try:
        # Tolerant of a path that names a file inside the adapter, which is
        # what copying out of a bucket listing gives you.
        adapter = (paths.localise(paths.adapter_dir_of(args.adapter), config,
                                  log=log, label="adapter")
                   if args.adapter else None)
        path = paths.localise(path, config, log=log, label="held-out set")
    except RuntimeError as exc:
        log.error(f"xx  {exc}")
        return 1

    if not adapter and "distilled" not in (args.skip or []):
        found = discover_adapters(config["project"].get("runs_dir") or "./runs")
        adapter = found[0] if found else None
        if adapter:
            log.info(f"==> newest adapter: {adapter}")
    if adapter and not is_adapter(adapter):
        log.error(f"xx  {adapter} has no adapter_config.json")
        return 1

    players = tuple(p for p in ("base", "distilled", "teacher")
                    if p not in (args.skip or []))

    # The teacher is the only input that may live in object storage, and the
    # only one worth several gigabytes - so it is fetched when it is a player
    # and left alone when `--skip teacher` means it will never be loaded. Which
    # is why `players` is decided before this, not after.
    if "teacher" in players and paths.is_remote(config["models"].get("teacher")):
        try:
            config["models"]["teacher"] = paths.localise(
                config["models"]["teacher"], config, log=log, label="teacher")
        except RuntimeError as exc:
            log.error(f"xx  {exc}")
            return 1

    questions, skipped = load_questions(path)
    available = len(questions)
    limit = args.limit or settings.get("arena_limit")
    if limit:
        questions = questions[:int(limit)]
    log.info(f"==> {len(questions)} questions from {path}"
             + (f" ({skipped} ungradeable rows skipped)" if skipped else ""))
    # Said loudly, and recorded in the payload below, because the arena now
    # saves by default: a five-question run writes an arena.json and a report
    # that look exactly like a real score. Whoever opens that file next week
    # has to be able to tell without remembering which flags were typed.
    if limit and len(questions) < available:
        log.info(f"    !! --limit {limit} of {available}: a SUBSET, not the "
                 f"score. Drop --limit for the real number.")
    predictions, formats, unanswered, completions = play(
        config, hardware, adapter, questions,
        max_new_tokens=int(args.max_new_tokens
                           or settings.get("arena_max_new_tokens") or 512),
        log=log, players=players, show=int(getattr(args, "show", 0) or 0))

    payload = summarise(predictions, [q["gold"] for q in questions],
                        formats=formats, unanswered=unanswered,
                        rounds=int(settings.get("arena_elo_rounds") or 25),
                        seed=int(config["project"]["seed"]))
    payload["arena_file"] = str(path)
    payload["adapter"] = str(adapter) if adapter else None
    if limit and len(questions) < available:
        payload["limited_to"] = len(questions)
        payload["available"] = available
    # Present from the first write, so the file always SAYS whether there is a
    # similarity table rather than leaving a reader to infer it from a missing
    # key - which reads the same as an older payload that never had one.
    payload["similarity"] = None

    # ----------------------------------------------------------------------- #
    # ORDER MATTERS HERE, and it is the opposite of the obvious one.
    #
    # Generation is the expensive, unrepeatable part: three models over a held-out
    # set, hours of it. Everything below - the similarity table, the terminal
    # summary, the report - is cheap and derived. So the derived work happens
    # AFTER the raw result is on disk, not before it.
    #
    # The specific accident this avoids: similarity() downloads an embedding
    # model. It handles the library being absent, but not a network that drops
    # or a cache that is corrupt, and it used to run before the first write - so
    # a failed download three hours in ended the process with nothing saved.
    # ----------------------------------------------------------------------- #
    saving = not getattr(args, "no_save", False)
    target = getattr(args, "json", None) or "arena.json"
    stem = target[:-5] if target.endswith(".json") else target
    transcript = None

    def write_payload():
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    if saving:
        write_payload()
        try:
            # The only record of what each model actually SAID, question by
            # question. Its own try because a transcript that fails to write
            # must not take the report down with it.
            transcript = write_transcript(f"{stem}-transcript.jsonl", questions,
                                          predictions, formats, completions)
        except Exception as exc:  # noqa: BLE001 - nothing here is worth dying for
            log.warning(f"  !! could not write the transcript: {exc}")

    # Optional, and reported as absent rather than fatal: it needs
    # sentence-transformers, which the pod install deliberately skips.
    try:
        payload["similarity"] = similarity(completions, questions, log=log)
    except Exception as exc:  # noqa: BLE001 - a download, an encode, a disk
        payload["similarity"] = None
        log.warning(f"      !! similarity skipped: {exc}")
    if payload.get("similarity"):
        payload["closeness"] = closeness(payload)   # now with explanations
        if saving:
            write_payload()          # now with the table in it

    log.info("")
    log.info(render(payload))
    if payload.get("similarity"):
        log.info(render_similarity(payload["similarity"]))

    if not saving:
        log.info("")
        log.info("  --no-save: nothing written")
        return 0

    # An arena-only report: the answer key and the similarity table, without the
    # token-level sections kd.evaluate produces. Smaller than the pipeline's
    # report, and honest about it rather than padded with blanks.
    report = getattr(args, "report", None)
    if report is None:
        report = f"{stem}-report.html"
    if report:
        try:
            from .report import write_report

            written = write_report({
                "arena": payload,
                "student": config["models"].get("student"),
                "teacher": config["models"].get("teacher"),
                "teacher_adapter": config["models"].get("teacher_adapter"),
                "adapter": str(adapter) if adapter else None,
                "adapter_locations": paths.adapter_locations(
                    adapter, config, source=args.adapter) if adapter else None,
                "profile": config["_meta"].get("source"),
                "device": hardware["device"],
                "dtype": hardware.get("dtype_name"),
            }, report)
            report = str(written)
        except Exception as exc:  # noqa: BLE001 - the numbers are already safe
            log.warning(f"  !! could not write the report: {exc}")
            log.warning(f"     nothing measured is lost - the numbers are in "
                        f"{target} and the transcript beside it")

    # Label first, path second: paths vary in length, so a trailing description
    # column does not line up on anyone's machine.
    log.info("")
    log.info("  saved")
    log.info(f"    the numbers                        {target}")
    if transcript:
        log.info(f"    every question and every answer    {transcript}")
    if report:
        log.info(f"    the readable summary               {report}")
    return 0
