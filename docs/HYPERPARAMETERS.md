# The four distillation knobs

What `ce_alpha`, `beta`, `lmbda` and `temperature` do, why they sit where they
do in `configs/enlibra/enlibraQ3-14B.yaml`, and why two commonly suggested
values (`temperature: 2.5`, `alpha: 0.5`) are the wrong values for this
pipeline.

Part 1 is the plain-language version. Part 2 is the same argument with the
measurements and citations behind it. Part 3 answers the two suggestions.

Sources are the PDFs in `docs/papers/`, summarised knob by knob in
`docs/papers/README.md`.

---

## Part 1 - the plain version

### The picture

There is a **teacher** (Qwen3-14B, fine-tuned on the neuroscience curriculum)
and a **student** (Qwen3-8B) that should learn from it.

The ordinary way to teach is an answer key: *"Question 1, the answer is B."*

Distillation does something richer. It shows the student the teacher's **whole
opinion**:

> "I think B - about 70% sure. C is possible, maybe 20%. A is 9%. D is
> basically nonsense, 1%."

That is far more information than "the answer is B". The student learns not just
the right answer but how the teacher thinks: what is a near-miss, what is
absurd. That richness is the entire reason to distil instead of fine-tune.

Each of the four knobs answers one simple question.

### alpha = 0.2 - "learn from the answer key, or from the teacher's opinion?"

**0.2 means 20% from the answer key, 80% from the teacher's opinion.**

Mostly the teacher, because the answer key gives one word - "B" - while the
teacher's opinion gives a ranking over every possible word. Far more teaching
per question.

Not *all* teacher, because the teacher is a fine-tuned model, not an oracle. At
100% the student would faithfully copy its mistakes too. The 20% answer key is
a safety rope.

### beta = 0.9 - "should the student commit, or hedge?"

The teacher said "70% B, 20% C, 9% A, 1% D". What should the student copy?

- **Low beta - hedge.** Keep some belief in C, A and D. Never rule out anything
  the teacher did not rule out.
- **High beta - commit.** Go with B, do not spend belief on the rest.

**0.9 means commit**, which fits multiple-choice questions that have one correct
answer. Not 1.0, because at exactly 1.0 the maths becomes unstable - the value
jumps from 0.20 to 2.99 (measured; table in Part 2).

### lmbda = 0.25 - "who writes the practice answers?"

- **0.0** - the student only ever reads worked examples from the textbook.
- **0.25** - one time in four the student writes its own answer first, and
  *then* the teacher marks it.

Think about learning to drive. A hundred videos of good drivers will not teach
you to recover from drifting toward a kerb, because good drivers never do it.
You need an instructor beside you while *you* drive, correcting *your* mistakes.

Same here. In the real exam the student writes its own answer one word at a
time. If it slips early, everything after is built on the slip - and it has
never been shown how to recover, because textbook examples contain no mistakes.

**This profile is currently at 0.0.** That is the real weakness, and it is
blocked on a code change rather than on this value. See "What is actually
blocking lmbda" below.

### temperature = 1.0 - and the big misunderstanding

**The word means two completely different things.**

**Meaning A - "blur the teacher's opinion."** If the teacher is 99% sure of B,
everything else rounds to zero and you cannot see whether C was a close second
or as silly as D. Blurring spreads it out so the ranking underneath becomes
visible. This is Hinton's classic idea, and 2 to 2.5 is completely normal *in
that setting*.

**Meaning B - "how randomly does the student write its practice answers."**

**This codebase only has Meaning B.** The temperature setting is passed to the
part that generates practice answers and is *never* passed to the part that
compares teacher and student. So `temperature: 2.5` would make practice answers
more random. It would blur nothing. And with `lmbda: 0.0` there are no practice
answers, so right now it does nothing at all.

---

## Part 2 - the same argument, with the evidence

### The objective

```
L  =  (1 - ce_alpha) * D_beta(P_teacher || Q_student)
        +  ce_alpha  * CE(gold_token, Q_student)

D_beta(P || Q)  =  beta * KL(P || M) + (1 - beta) * KL(Q || M),
                   M = beta*P + (1 - beta)*Q
```

`ce_alpha` and `beta` shape the **loss**. `lmbda` and `temperature` decide what
**text** the loss is evaluated on. Moving one from each pair in the same run
makes the result unattributable.

`HybridGKDTrainer.compute_loss` (`src/kd/train.py:86`) drops the CE term on
student-written batches, so the effective CE weight is `ce_alpha * (1 - lmbda)`
- 0.15 at `lmbda: 0.25`. That is intended; do not compensate for it.

### alpha - why small, and why not zero

`CE(y, Q) = KL(delta_y || Q) + const`, where `delta_y` is a one-hot point mass.
So alpha is not "distillation vs supervision" - it interpolates between two
*targets*: the teacher's distribution, and a zero-entropy point mass asserting
that every non-gold token has probability exactly zero. For natural language
that assertion is false, which is why the weight on it stays low.

- **Small:** Hinton et al. 2015 report best results with a considerably lower
  weight on the hard-label term.
- **Not zero:** GKD, DistiLLM and MiniLLM all train at `ce_alpha` 0, but their
  teachers are not LoRA-SFT checkpoints on the same 1384 rows. The gold token
  anchors the student when the teacher is diffuse or wrong.
- **0.2 is a practitioner's value, not a paper's.** No ablation in these three
  papers sweeps it. 0.1 vs 0.2 vs 0.3 is untested here and not worth defending.

### beta - the endpoints are discontinuous

In TRL's implementation (`gkd_trainer.py:295-310`), beta approaching 0 gives
forward KL (mass-covering) and beta approaching 1 gives reverse KL
(mode-seeking). **State this convention explicitly** - papers differ on the
orientation and it is the most common source of confusion in the room.

Measured on this pair, one token, teacher peaked and student diffuse:

| beta | 0.0 | 0.05 | 0.1 | 0.5 | 0.9 | 0.95 | 0.999 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| D_beta | **1.639** | 0.077 | 0.143 | 0.407 | 0.197 | 0.116 | 0.003 | **2.986** |

The endpoints are special-cased to the pure, *unbounded* KLs while the interior
is the bounded mixture form, so the function is discontinuous at 0 and 1. The
interior is bounded by the binary entropy of beta (H(0.9) = 0.325; the observed
0.197 sits under it).

Two consequences:

- **0.9 rather than 1.0.** 0.9 buys reverse-KL behaviour with the bound still
  on. Expanded at 0.9 the minor term is `KL(Q || 0.1Q + 0.9P)` - exactly
  DistiLLM's skew reverse KL at their best skew of 0.1 (Fig. 8: *"both SKL and
  SRKL achieve the best performance on the alpha value of 0.1"*).
- **Not 0.5.** Both papers rank the symmetric middle last (DistiLLM Fig. 6;
  GKD's GSM8K row). The table shows the mechanism: 0.5 maximises the bounded
  objective, committing to neither geometry.

**Concede this knob first.** At 14.8B to 8.19B the capacity gap is only 1.8x,
far too small for MiniLLM's small-student argument. The arena decodes greedily
(`src/kd/arena.py:768`), and under greedy evaluation GKD Fig. 4 has the
divergences within 0.8 ROUGE-2 of each other. **beta is the least load-bearing
knob here.**

### lmbda - the actual thesis

The fix for exposure bias: under pure off-policy KD the student only ever sees
teacher-forced prefixes from the curriculum, but at inference it conditions on
its own prefixes, including its own errors - a distribution it was never trained
on, so errors compound.

- GKD Fig. 8, verbatim: *"As we increase fraction of student-generated data
  beyond 25%, performance typically improves."*
- DistiLLM's hand-tuned fixed mixes independently land at 0.3-0.4.
- Not 1.0: GKD A.2 puts on-policy at 1.8-2.2x off-policy compute, and DistiLLM's
  central caution is that an untrained student's early rollouts are too poor to
  be informative - which is why their adaptive schedule *starts* at 0 and ramps.

0.25 is the threshold where gains are predicted at all, so it is the cheapest
point with any support behind it.

#### What is actually blocking lmbda

Qwen3's chat template renders a prompt ending at `<|im_start|>assistant`, so the
student generates from a position where Qwen3 opens a real reasoning chain, is
cut off at `max_new_tokens` mid-thought, and the teacher scores a fragment.

Rendering the prompt with `enable_thinking=False` ends it after the closed
`<think>` block - which is where the student should start. Verified:

- TRL's collator uses a pre-rendered `prompt` column when the row carries one,
  and only falls back to `apply_chat_template` when it does not
  (`trl/experimental/utils.py:298`).
- `GKDTrainer` forces `remove_unused_columns = False` (`gkd_trainer.py:136`), so
  such a column survives to the collator.
- On Qwen3-8B, the `enable_thinking=False` rendering is still an exact prefix of
  the full rendered exchange, so label masking is unaffected.

So this is a small change in `kd.data`, not a TRL fork. **Do not raise `lmbda`
before it lands** - without it the run gets worse, not better.

### temperature - two parameters, one name

- **Hinton's T** divides logits *inside the loss*, softening both distributions.
- **GKD's gamma** is the *sampling* temperature for rollouts.

`GKDConfig.temperature` is gamma. It reaches `GenerationConfig`
(`gkd_trainer.py:236`). `compute_loss` calls `generalized_jsd_loss` **without a
temperature argument** (`gkd_trainer.py:440-446`), so the divergence runs at
T = 1 unconditionally. The parameter exists in the function signature but is
never reached from the training path.

1.0 per GKD 3.1 - *"we use a temperature of gamma = 1 to encourage diversity in
student generated sequences"* - and the Thinking Machines Qwen3 recipe. Lower
values visit only the student's confident paths, the ones least in need of
correction.

### max_new_tokens = 640

Sized from the corpus, not guessed. Completion lengths under the Qwen3
tokenizer:

| File | n | mean | p90 | p99 | max |
|---|---|---|---|---|---|
| `sft-1to3hop.jsonl` | 1369 | 295 | 331 | 366 | **412** |
| `eval-1to5hop.jsonl` | 142 | 292 | 324 | 354 | 390 |
| `identity.jsonl` | 15 | 19 | - | - | 28 |

640 is 1.55x the longest training completion and 1.75x p99, so a rollout that
answers properly is never cut.

A tight cap is safe because **the pressure to stop is applied per token, not at
the end**: around position 300 the teacher's distribution puts its mass on
`<|im_end|>`, and a student that does not is penalised by the divergence at that
position whether or not the sequence is later truncated. Letting it ramble to
8192 does not teach it to stop, it pays for the rambling.

Truncation also costs far less in a rollout than in the arena - a cut rollout
still carries real answer tokens from position 1 and scores normally over them,
where a cut arena answer loses the `<Answer>` tag and reads as never answered.
That is why `arena_max_new_tokens` stays high while this drops.

The 12.8x that 8192/640 suggests is a **worst case, not a per-run multiplier**:
generation stops at end-of-turn, so a well-behaved student costs ~300 tokens
either way, and `batch_size: 1` means no sibling sequence can hold a batch open.
What the old ceiling bought was an unbounded tail - at `lmbda: 0.25` this
profile makes ~700 rollouts, and each one that fails to stop costs 8192
sequential passes instead of 640.

---

## Part 3 - the two suggested values

### Why `temperature: 2.5` is not ideal

Three independent reasons; any one is sufficient.

**1. It is not wired to the loss.** In TRL 1.12 the parameter is the sampling
temperature. Setting 2.5 makes rollouts wilder, not targets softer - and with
`lmbda: 0.0` it does nothing whatsoever. Demonstrating that it is a no-op is
itself the point: it shows the implementation was read, not just the paper.

**2. If it were wired, it would silently rescale the learning rate.** Hinton
multiplies the soft loss by T^2 because the gradient scales as 1/T^2. TRL
applies no such correction. Measured at beta 0.9:

| T | 1.0 | 2.0 | 2.5 | 4.0 |
|---|---|---|---|---|
| D_beta | 0.197 | 0.073 | **0.046** | 0.016 |
| shrink vs T=1 | - | 2.69x | **4.30x** | 12.3x |

T = 2.5 cuts the learning signal about 4.3x. Without the T^2 rescale that is a
learning-rate change and a target-shape change confounded in one knob - the
result could not be attributed to either.

**3. It erases the geometry beta encodes.** Raising T flattens both
distributions toward uniform, which pushes any divergence toward mass-covering
and smooths away the modes. `temperature: 2.5` and `beta: 0.9` pull against each
other: choosing mode-seeking, then destroying the modes.

**Where the objection is right, and should be conceded.** T in [2, 20] is
well-founded in classical KD; DistilBERT used T = 2. Hinton's dark-knowledge
argument is real: with 10-1000 classes and one target per example, relative
probabilities among wrong classes carry similarity structure a confident softmax
suppresses. The claim is not that temperature is wrong in general - it is that
**the argument does not transfer to autoregressive full-distribution KD**, where
there are 151,936 classes at every one of ~300 positions and the entire
distribution is already being matched. The tail information Hinton had to
manufacture entropy to expose is already in the objective, weighted by the
teacher's own probability mass.

One legitimate pro-T argument in this setup: the teacher is SFT'd on this exact
curriculum and may be sharply peaked, carrying little more than the gold label.
If that is raised, answer that the peakedness is *signal* - a confident correct
teacher is what is worth transferring, and flattening it destroys the
distinction between a confident answer and a hedge. To settle it empirically,
measure the teacher's mean per-token entropy on the curriculum. If it is near
zero, distillation is buying little over SFT, and the honest conclusion is a
different one than temperature.

### Why `alpha: 0.5` is not ideal

**At 0.5 the convention ambiguity vanishes.** Hinton's alpha conventionally
weights the *soft* term while `ce_alpha` weights the *hard* term, so the two
readings normally disagree - which is why 0.25 was ambiguous (25% CE under one
convention, 75% under the other). At 0.5 both mean the same thing: **50%
distillation, 50% supervised cross-entropy.**

**The setup-specific argument, which is the strongest one.** The teacher here
*is* Qwen3-14B fine-tuned on this same curriculum. The gold tokens the CE term
trains against are the data the teacher is made of. At 0.5, half of every step
is plain SFT against that curriculum - which already exists as its own profile
(`configs/enlibra/enlibraQ25-3B-sftonly.yaml`) and is the baseline this run
exists to beat. **It dilutes the treatment with the control.**

**The general argument.** It contradicts every source: Hinton puts a
considerably lower weight on the hard term; GKD, DistiLLM and MiniLLM use 0
outright. Nobody in this literature runs 50/50.

**The honest caveat.** No ablation in these papers sweeps alpha, because they all
fix it at 0. So 0.2 cannot be claimed as empirically optimal - only as small,
which is what the literature supports, and 0.5 is outside that range in the
direction that makes distillation pointless.

---

## Questions to expect

| Question | Answer |
|---|---|
| "Why is lmbda zero if on-policy is the paper's whole point?" | Implementation constraint, not a design choice. The fix is identified and small. Do not defend 0.0 on the merits. |
| "Your compression ratio is only 1.8x - why mode-seeking?" | Concede. beta is the least load-bearing knob; under greedy eval the divergences barely separate. |
| "What is your baseline?" | Four players: stock Qwen3-8B, distilled student, stock Qwen3-14B, teacher. The `teacher-base` player bounds how much there was to distil. |
| "How would you test any of this?" | 142 held-out questions, accuracy plus Elo over 25 seeded shuffles. Say up front that 142 is small - differences under ~5 points are not resolvable. |
| "Isn't `ce_alpha` non-standard?" | Yes. Stock `GKDTrainer` has no CE term; `HybridGKDTrainer` (`src/kd/train.py:43`) adds it. Own it as a local modification. |

## Summary

| Knob | Value | One-line reason |
|---|---|---|
| `ce_alpha` | 0.2 | Mostly learn from the teacher's full distribution; a small gold-token anchor against teacher error. |
| `beta` | 0.9 | Commit rather than hedge, and bounded - 1.0 is unbounded and unstable. Least important knob. |
| `lmbda` | 0.0 to **0.25** | Let the student write some answers and have the teacher mark them. Blocked on a `kd.data` change. |
| `temperature` | 1.0 | Sampling temperature, not Hinton's. 2.5 would only make rollouts wilder. |
| `max_new_tokens` | 640 | 1.55x the longest real completion (412 tokens, measured). |

## Sources

- Agarwal et al., *On-Policy Distillation of Language Models*, ICLR 2024 -
  `docs/papers/gkd-agarwal-2024-on-policy-distillation.pdf`
- Ko et al., *DistiLLM*, ICML 2024 -
  `docs/papers/distillm-ko-2024-skew-kl.pdf`
- Gu et al., *MiniLLM*, ICLR 2024 -
  `docs/papers/minillm-gu-2024-reverse-kl.pdf`
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, 2015
  (arXiv 1503.02531) - not in this folder; the origin of the alpha/T formulation.
- Thinking Machines Lab, *On-Policy Distillation*, 2025 -
  <https://thinkingmachines.ai/blog/on-policy-distillation/>
