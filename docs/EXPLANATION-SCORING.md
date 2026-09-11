# Scoring the explanations

**Status:** design, not implemented. Nothing in `src/kd/` does any of this yet.

The arena scores three players on a held-out multiple-choice set and reports
accuracy, Elo, and one similarity table. This document is about that table — why
it cannot be read as it stands, and what to replace it with.

For the arena itself see [`src/kd/arena.py`](../src/kd/arena.py); for how the
stage is wired, `stage_arena` in [`src/kd/pipeline.py`](../src/kd/pipeline.py).

---

## The problem

`arena.similarity()` embeds every completion with
`sentence-transformers/all-MiniLM-L6-v2` and reports the mean cosine between
each pair of players, broken down by hop count. The intent is right: two models
can pick the same letter for entirely different reasons, and the answer key sees
none of that.

The number it produces does not support the claim.

**It has no zero and no one.** MiniLM cosines on answers to the *same* question
sit between roughly 0.75 and 0.90 whether or not the reasoning agrees, because
both completions restate the entities in the prompt. There is no floor in the
table saying what "unrelated" scores and no ceiling saying what "identical"
scores, so 0.84 is a number without units.

This repository already knows this argument. `evaluate.py` passes
`rescale_with_baseline=True` to BERTScore, and the comment there makes the case
exactly:

> RAW BERTScore is not a 0-1 similarity: two entirely unrelated English
> sentences score ~0.86, so the whole meaningful range is compressed into
> roughly 0.85-0.96 and a genuine improvement reads as a rounding error.

Every other number in either stage ships with its own baseline — accuracy has
`random_baseline: 0.25`, Elo reports `elo_spread` across 25 seeded shuffles,
BERTScore is rescaled. The explanation cosine is the only metric in the
codebase printed raw.

**Anisotropy.** Transformer embeddings occupy a narrow cone: every vector shares
a large mean component, which inflates all cosines by roughly a constant and
compresses the range where the signal lives. `embed()` L2-normalises and stops
there.

**It measures topic, not logic.** The signal is dominated by lexical overlap
with the question. A flipped relation or a negation — precisely the multi-hop
failure this curriculum is built to expose — moves the cosine by almost
nothing. "Proxima Centauri is a star" and "Proxima Centauri is not a star"
score above 0.95.

**Truncation.** MiniLM caps at 256 word pieces and truncates silently.
`arena_max_new_tokens` defaults to 2048. Long completions are compared
partially, and completions are longest at hop 4-5 — the column that matters
most is the most corrupted.

**Mean of cosines.** `mean_cosine` averages a bounded non-linear quantity over
a hop bucket that may hold twenty items. One outlier moves it visibly, and the
mean is not the cosine of anything.

**It is pairwise, not anchored.** The table reports `teacher vs distilled` and
`teacher vs base`, which is why `render_similarity` has to instruct the reader
to subtract one from the other. There is no per-player number.

One thing is worth stating plainly: the "cosine is not a metric" objection —
that `1 - cos` violates the triangle inequality and identity of indiscernibles
— is true but inert here. Nothing downstream triangulates. The problem is
calibration, not the axioms.

---

## The borrowed design

Kansal & Jha, *Knowledge Graphs are Implicit Reward Models: Path-Derived
Signals Enable Compositional Reasoning* (arXiv 2601.15160,
[code](https://github.com/scient-lab/kg-implicit-reward-compositional-rl))
score reasoning traces against knowledge-graph paths using set overlap on
tokens, with no embeddings anywhere:

```
coverage(r,P) = |T(r) INTERSECT T(P)| / |T(P)|

R_path  = min(g1 * coverage(r,P) + g2 * [ |T(r) INTERSECT T(P)| >= 2 ], R_max) * phi_rep
          g1 = 1.2,  g2 = 0.3,  R_max = 1.5

R_sim   = |T_model INTERSECT T_target| / |T_model UNION T_target| * phi_rep     (Jaccard)

R_bin   = +0.1 if correct else -1.0

R_think = (0.5 * structure + 0.3 * step-keywords + 0.2 * enumeration) * phi_rep

R_total = R_bin + R_path
```

`phi_rep` is in [0,1] — a repetition penalty from token dominance ratio
(threshold 0.35) and consecutive-run length.

**This is an RL training signal under GRPO, not an evaluation metric.** The
asymmetric `R_bin` and the whole of `R_think` are gradient-shaping devices with
no meaning as measurements — `R_think` rewards a model for containing the
string "step" and for emitting numbered lines. Neither belongs in an eval.

What does transfer is `coverage`, and the reason it transfers is that it is
anchored to a fixed reference and bounded by something real: 0 is disjoint
vocabulary, 1 is full recall of the reference's terms. No anisotropy, no model
download, no truncation, deterministic and exactly reproducible.

### Choose coverage, not Jaccard

Jaccard puts the union in the denominator, so it punishes length mismatch by
construction: a terse but complete student scores low against a verbose
teacher. The three arena players differ substantially in verbosity, so Jaccard
would largely measure that. Coverage — intersection over the reference only,
i.e. recall against the gold — does not have this property. The paper itself
uses coverage for the reward that matters and relegates Jaccard to an optional
one.

---

## What this repository already has

| Piece | Status |
|---|---|
| Gold explanation per question | **Available now.** `arena.load_questions` puts the curriculum's full `<Explanation>` in `question["reference"]`. It is used for exactly one thing: written into the transcript as `reference_answer`. Nothing scores against it. |
| Correctness | **Available now**, and better designed than `R_bin` — accuracy, accuracy-when-answered, and Elo, none of which need an asymmetric penalty. |
| KG paths | **Recoverable.** Not in `data/enlibra-curriculum/eval-1to5hop.jsonl`, which carries only `hop_count`, `item_id`, `messages`, `split`. The upstream export has them. |
| Token normalisation | **Missing.** Must be written and pinned. |
| Repetition measure | **Missing.** |

The upstream export (`curriculum_verified.json`, 1217 rows) carries per row:

```
paths = [{'start': 'naked-eye observers', 'relation': 'observes', 'end': 'supernova',
          'triple_id': '8794a28006cbcb48'},
         {'start': 'star', 'relation': 'undergoes', 'end': 'supernova', ...},
         {'start': 'proxima centauri', 'relation': 'type_of', 'end': 'star', ...}]
triple_ids, source_concept, target_concept, expansion_edge_count, hop_count
```

`scripts/prepare_curriculum.py` drops `paths` because its carry-through list is
`("hop_count", "item_id", "split")`. The comment directly above that line
already anticipates the cost: *"Recovering it later would mean re-joining
against the export by item_id."*

---

## Design

Three additions, in dependency order. Each is independently shippable and each
is readable on its own.

### Tier 1 — reference coverage

No data change, no new dependency, no model download.

For every player and every question, tokenise the completion's `<Explanation>`
body and the gold `question["reference"]`, and report

```
reference_coverage = |T(completion) INTERSECT T(reference)| / |T(reference)|
```

broken down by hop, exactly as the cosine table is. This is a per-player
number, so it needs no "read the rise between two columns" instruction — base,
distilled and teacher each get a column and they are directly comparable.

Report the **median and IQR**, not the mean. The distribution over a hop bucket
is skewed and the mean hides it.

### Tier 2 — controls for the cosine table

Keep the cosine. It sees paraphrase, which coverage cannot. Give it the floor
and ceiling it is missing, both computed from embeddings already in memory:

- **Floor:** cosine between player A's answer to question *i* and player B's
  answer to question *j != i* (shuffled pairing, seeded). This is what
  "unrelated answers from these two models on this corpus" scores.
- **Ceiling:** the teacher against itself at a second sample of the same
  temperature. This is what "as similar as two correct answers ever get"
  scores.

Print both as rows in the same table. A pair score is then legible as a
position between them rather than as an absolute.

Additionally, **centre the embeddings** — subtract the mean vector over the
pooled corpus before the dot product — which removes most of the anisotropic
component and typically spreads a 0.75-0.90 band across a usable range. This
changes the numbers in existing `arena.json` files, so it is a versioned change
(see *Compatibility*).

### Tier 3 — path coverage

Requires carrying `paths` through data preparation and regenerating the
curriculum.

For each question, the reference term set becomes the KG path's own entities
and relations rather than the prose explanation:

```
T(P) = union of {tokens(start), tokens(relation), tokens(end)} over the triples
path_coverage = |T(completion) INTERSECT T(P)| / |T(P)|
```

This is the version worth having. The prose explanation is one phrasing of the
reasoning; the path *is* the reasoning, with one triple per hop. Two things
follow that no current metric gives:

- **Per-triple hit/miss**, not just a scalar. A student that recalls
  `proxima centauri -> type_of -> star` but drops
  `star -> undergoes -> supernova` has a located failure at a specific hop, and
  it is the same located failure across every question using that triple.
- **A per-triple miss rate across the corpus**, which points at the curriculum
  rather than the student: a triple that every player misses is more likely to
  be a bad row than a shared blind spot.

### Degeneracy flag

Report a `phi_rep`-style repetition ratio per player per hop — but as a
diagnostic column, never as a multiplier. As a training penalty it shapes a
policy away from looping; as an eval it is a measurement, and folding it into
the score would conflate "repeated itself" with "missed the reasoning".

It matters here specifically: a base model with no adapter loops, and
`arena_max_new_tokens: 2048` lets it loop for a long time. A long looping
completion has more tokens and therefore inflates *any* overlap measure.
Without this column, degeneracy reads as fidelity.

---

## Token normalisation

Unspecified normalisation makes coverage meaningless — without stopword removal
it becomes a test of whether the model said "the". This must be pinned in one
function and never varied per call site:

1. Casefold, strip punctuation to word boundaries.
2. Drop a **fixed, in-repo stopword list**. Not NLTK's, not a downloaded one:
   the list must be versioned with the code or the metric is not reproducible
   across machines.
3. Light morphology — plural `-s`/`-es` stripping only. Full stemming merges
   distinct KG entities and is not worth the risk.
4. Numerals and units kept verbatim. `1 keV` is reasoning content in this
   curriculum, not noise.
5. Multi-word KG entities (`naked-eye observers`) matched as **phrases first**,
   then their residual tokens contributed individually. A path entity should
   not be credited to a model that happened to say "observers".

Point 5 is the one that will be got wrong if left implicit. Coverage over a bag
of unigrams credits `star` from any sentence mentioning stars; the question is
whether the model traversed `proxima centauri -> type_of -> star`. Phrase-first
matching over triples is what keeps the measure about traversal.

---

## Changes by file

| File | Change |
|---|---|
| `src/kd/arena.py` | New `normalize_tokens()`, `coverage()`, `repetition_ratio()`. `similarity()` gains floor/ceiling rows and mean-centring. New `explanation_scores()` returning per-player, per-hop coverage. New `render_coverage()`. |
| `src/kd/pipeline.py` | `stage_arena` calls `explanation_scores()` and writes it into the payload — **after** the first `write_payload()`, matching how similarity is already ordered. Generation is the hours-long unrepeatable part and nothing added here may risk it. |
| `src/kd/report.py` | A coverage row beside the existing hop-wise cosine row. |
| `scripts/prepare_curriculum.py` | Tier 3 only: add `"paths"` to the carry-through tuple. |
| `configs/_base.yaml` | `arena_coverage: true`, and a Tier-3 `arena_path_coverage` defaulting off until curricula are regenerated. |
| `tests/` | Extend `test_arena_output.py` and `test_report.py`, which already stub the `by_hop`/`cosine` payload shape. |

### Payload shape

`similarity` keeps its shape and gains two keys:

```jsonc
"similarity": {
  "model": "...", "pairs": {...}, "hops": ["1","2"],
  "centered": true,
  "controls": {
    "floor":   {"overall": 0.31, "by_hop": {}},   // shuffled pairing
    "ceiling": {"overall": 0.88, "by_hop": {}}    // teacher vs itself
  }
}
```

Coverage is a sibling, not a child, because it does not depend on
`sentence-transformers` and must survive that library's absence:

```jsonc
"coverage": {
  "reference": {
    "teacher":   {"overall": {"median": 0.62, "iqr": [0.51, 0.74]},
                  "by_hop": {"1": {}, "2": {}}},
    "distilled": {}, "base": {}
  },
  "path": null,                 // tier 3; null when the curriculum carries no paths
  "repetition": {"teacher": {}, "distilled": {}, "base": {}}
}
```

---

## What this does not measure

Stated so nobody reads more into the column than is there.

- **Negation.** Token overlap is *more* negation-blind than cosine, not less.
  "X undergoes Y" and "X does not undergo Y" differ by one token in forty. A
  student that has learned the entities and inverted a relation scores near
  1.0.
- **Order.** Coverage is a bag of tokens. A correct hop chain and a scrambled
  one are indistinguishable, which for a curriculum whose whole axis is
  reasoning depth is a genuine loss.
- **Correctness of anything.** A completion can recall every path entity and
  reach the wrong letter. That is what accuracy and Elo are for; these numbers
  complement them and never substitute for them.

The honest fix for the first two is an entailment check — a small NLI
cross-encoder over (reference, completion) pairs, where bidirectional
entailment catches exactly the negation case that neither cosine nor coverage
can see. That is out of scope here: it adds a model download and an inference
pass to a stage that currently degrades gracefully to "table omitted". It is
the right next thing after Tier 3.

---

## Validating the metric

A new number is worthless until it is shown to move for the right reason.
Before this is trusted in a report:

1. **Floor.** Coverage of a completion against a *different* question's
   reference. Should be low and flat across hops. If it is not, the stopword
   list is too permissive.
2. **Ceiling.** The gold reference scored against itself is 1.0 by
   construction; more usefully, the teacher against its own second sample. That
   is the realistic maximum.
3. **Monotonicity.** Base < distilled <= teacher on a run already known good.
   If the ordering disagrees with Elo, find out why before shipping the column,
   not after.
4. **Hop sensitivity.** Coverage should decline with hop count for every player
   and decline fastest for base. A flat line across hops means the measure is
   reading topic, which is the defect it exists to avoid.
5. **Degeneracy.** A deliberately looping completion must show a high
   repetition ratio and must not show high coverage.

---

## Compatibility

`report.py` already reads similarity defensively — `entry.get("by_hop", {})`,
null-tolerant formatting — so old `arena.json` files without the new keys render
with the new sections absent rather than failing. Keep that property: every
addition here is read with `.get`, and `stage_arena` already wraps similarity in
a try/except that records `None` rather than failing the stage. Coverage must be
wrapped the same way.

Mean-centring changes existing numbers. Runs before and after are not
comparable, which is why `"centered": true` is written into the payload — a
report reading an old file must be able to tell that its cosines are on a
different scale, rather than silently plotting the two together.
