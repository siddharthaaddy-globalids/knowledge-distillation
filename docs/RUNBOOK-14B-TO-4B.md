# Runbook: the 14B → 4B pair, driven by two scripts

Qwen3-14B (SFT LoRA) → **Qwen3-4B**, on the neuroscience curriculum, with the
distilled student packed to W4A16 and scored against four other players.

The same six steps as [RUNBOOK-14B.md](RUNBOOK-14B.md), in the same order:

```
train  →  upload  →  quantize  →  upload  →  eval  →  upload
```

The difference is that this pair needs **no hand-installing**. Two scripts cover
the whole cycle — `./run.sh` on the training pod, `./scripts/score-pod.sh` on
the scoring pod — and the reason the 8B profile cannot do that is
[at the bottom](#why-quantization-is-off-in-the-training-profile).

| | |
| --- | --- |
| Training profile | `configs/enlibra/enlibraQ3-14B-to-4B.yaml` |
| Scoring profile | `configs/enlibra/enlibraQ3-14B-to-4B-score.yaml` |
| Teacher | unchanged — the same SFT LoRA the 8B run distilled |
| Held-out set | unchanged — the same 142 rows |

Everything that goes wrong along the way, and what it actually means, is in
[POD-FAILURES.md](POD-FAILURES.md).

---

## What this run is, and what it is expected to cost

14B → 8B is a 1.85× compression. This is **3.7×** — still an ordinary ratio for
a *narrow* student: one curriculum, one fixed `<Explanation>`/`<Answer>` shape.
The artifact it produces is roughly half the size to serve: **~2.5 GB** packed
to W4A16 against 5.7 GB for the 8B.

Only the student changes. Same teacher, same corpus, same `r=128`, same 350
steps, same arena, same `lmbda: 0.0`. That is deliberate — it makes the two
bundles comparable on the same 142 questions, so the difference between them is
the student and nothing else.

**Where the difference is expected to show.** The held-out set runs to 5 hops
where training tops out at 3, and 80 of the 142 rows are 4- or 5-hop. Depth of
reasoning chain is the capability most sensitive to parameter count, so a gap
against the 8B student should concentrate in those rows and be small or absent
on the 1- and 2-hop ones. **Read the report by hop depth, not by the headline
accuracy** — a lower total with an intact 1–3 hop column is a different result
from a model that has stopped answering.

**What `lmbda: 0.0` costs here.** Off-policy KD scores the teacher's
distribution over *fixed* completions; the on-policy correction that makes GKD
better than plain KD scores it over the student's own. That correction matters
more as the student shrinks, and this is the smallest student the neuroscience
profiles have trained. If the 4B falls short, this is the first knob to reach
for — and the Qwen3 chat-template problem documented in `enlibraQ3-14B.yaml` has
to be solved before it can be. It stays `0.0` for the first run so that this run
differs from the 8B run in one thing.

---

## Before you rent anything: the headroom check

Whether a 4B can hold these chains is not answerable from first principles, and
the arena is the instrument that answers it. Score the **stock** student on the
same file — no training, no teacher, one player:

```bash
python3 -m kd arena --config configs/enlibra/enlibraQ3-14B-to-4B.yaml \
    --skip distilled --skip distilled-w4a16 --skip teacher --skip teacher-base
```

`--skip distilled` is what keeps it cheap: with that player out, `kd arena`
never goes looking for an adapter (`src/kd/arena.py:1253`), so the only weights
that load are stock Qwen3-4B. Twenty minutes on a 48 GB card.

Compare its per-hop accuracy against stock Qwen3-8B's column in the 8B report.

| Stock Qwen3-4B on the held-out set | What it means |
| --- | --- |
| near stock Qwen3-8B | the headroom is the same — make the run |
| collapses on the 4- and 5-hop rows | 1384 training rows will not rescue it; the answer is a bigger student or a bigger corpus, not a longer run |

---

## Provisioning

| | Pod A — training | Pod B — packing and scoring |
| --- | --- | --- |
| GPU | 80 GB — H100 80GB PCIe, H200, A100 80GB | 48 GB — L40S, A6000 |
| Volume at `/workspace` | 160 GB | 200 GB |
| Container disk | 40 GB | 40 GB |
| Template | any stock **PyTorch** image | same |

**Why still 80 GB for training.** The teacher dominates, and the teacher has not
changed:

```
teacher (14.8B, frozen)      29.6 GB
student (4.02B, training)     8.0 GB
                             --------
weights at bfloat16          37.6 GB      (the 8B profile: 46.0)

LoRA adapter (264M params)    0.5 GB
gradients                     0.5 GB
AdamW moments (fp32)          2.1 GB
activations, 36 layers @ ~800 tokens, gradient_checkpointing off
two sets of vocab-wide logits for the JSD
                             --------
working set                ~45-48 GB      (the 8B profile: ~55-60)
```

A 48 GB L40S is therefore *arguable* rather than hopeless — but arguable at the
very top of the card, and the allocator does not politely refuse: it fails
partway through the first forward pass, after both models have downloaded. To
try it anyway, buy the headroom explicitly:

```bash
./run.sh --config configs/enlibra/enlibraQ3-14B-to-4B.yaml \
    --set training.gradient_checkpointing=true --skip upload
```

Recomputing activations in the backward pass returns ~3–4 GB for roughly 30%
slower steps. Passing it as `--set` rather than editing the profile puts it in
the run manifest, so the bundle records that the run was squeezed onto a smaller
card. The `smoke` gate settles the question in two real steps.

48 GB is ample for Pod B either way: players load one at a time, and the 29.5 GB
dense teacher is the ceiling, not the student.

---

## The shell, on every pod

Unchanged from [RUNBOOK-14B.md](RUNBOOK-14B.md) — run it in every new SSH
session and every new tmux pane:

```bash
tmux new -s kd

cd /workspace
git clone <repo> knowledge-distillation
cd knowledge-distillation

export HF_HOME=/workspace/hf-cache
export KD_RUNS_DIR=/workspace/runs
export HF_HUB_DISABLE_XET=1          # download once instead of stage-then-reconstruct
export KD_PRICE_PER_HOUR=2.79        # the rate you agreed to; the cost cap is inert without it
export HF_TOKEN=hf_...               # ~38 GB from the Hub, rate-limited without it

export AWS_ACCESS_KEY_ID=...         # re-export before EVERY step marked "fresh token"
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

Then check the machine before spending anything on it:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace /
```

`/` is the 40 GB container disk that must never receive a model.

---

## Pod A — train

### 1. Train — *fresh token*

Name the profile once, as on pod B — `run.sh` reads `KD_CONFIG` (`run.sh:73`),
and this pod runs the **training** profile, not the scoring one:

```bash
export KD_CONFIG=configs/enlibra/enlibraQ3-14B-to-4B.yaml
```

```bash
./run.sh --skip upload --set s3.cache_dir=/workspace/kd-cache
```

One command. `run.sh` installs (keeping the template's CUDA torch), then runs
`doctor` → `check` → the gated pipeline: `preflight → teacher-check → smoke →
train`. `quantize` is skipped because the training profile sets
`quantization.enabled: false`; `evaluation` because `after_training` is false;
`upload` because you asked.

**Read `doctor`'s "remote inputs" section.** A FAIL there is a run that aborts in
preflight, and on a pod that means paying to find out.

**Write down the run id from the banner.** Every step below names it.

Hours. Somewhat faster per step than the 8B run — but the teacher forward
dominates and has not changed, so do not expect half. `save_steps: 50` means a
hard stop costs you at most fifty steps.

### 2. Upload the adapter — *fresh token*

```bash
./run.sh upload <run-id>
```

Ships `final_adapter/`, `checkpoints/`, logs and metrics — under a gigabyte of
adapter plus ~5 GB of optimizer state, a few minutes. **Do this before anything
else**: until it finishes, the only copy of hours of GPU time is on a pod that
can be reclaimed.

Always name the run. With no argument `kd upload` takes "the latest run", which
falls back to the alphabetically largest directory name, and run ids lead with
the profile rather than the timestamp.

**Then terminate pod A.** Everything that matters is in the bucket at:

```
s3://enlibra/dss/dev/runs/20260818_215840_neuroscience_8f33eb7cf609/outputs/gkd/runs/<run-id>/
```

---

## Pod B — pack, score, and the last two uploads

### 3. Point the scoring profile at the run you trained

The run id appears **twice** — in `evaluation.adapter` and in
`quantization.output_dir` — and both must move together. If they disagree the
packing succeeds and the arena reports `distilled-w4a16 skipped: no packed
checkpoint`, hours later.

```bash
sed -i 's/RUN-ID-FROM-THE-TRAINING-BANNER/<run-id>/g' \
    configs/enlibra/enlibraQ3-14B-to-4B-score.yaml

grep -n RUN-ID-FROM-THE-TRAINING-BANNER \
    configs/enlibra/enlibraQ3-14B-to-4B-score.yaml     # must print nothing
```

### 4. Everything else — *fresh token at every pause*

**Name the profile once.** Both shell scripts read `KD_CONFIG`
(`score-pod.sh:85`, `run.sh:73`), so nothing in this session can drift onto
another profile through a mistyped path:

```bash
export KD_CONFIG=configs/enlibra/enlibraQ3-14B-to-4B-score.yaml
```

```bash
./scripts/score-pod.sh all
```

`kd` itself does **not** read `KD_CONFIG` — only these two scripts do — so a
direct `python3 -m kd ...` call still spells `--config` out.

It runs `setup → quantize → upload → eval → upload`, pausing before each
billable step, and encodes every trap in [POD-FAILURES.md](POD-FAILURES.md):
ordered installs, the torchvision repair, the cache-on-the-volume refusal, and
an `--out` read from the config so it cannot drift from where the arena looks.

**Re-export AWS credentials immediately before each pause** — `quantize` and
`eval` both reach the bucket in their first seconds, and both uploads obviously
do.

It is idempotent: re-run the same line after any failure and it skips the
install, skips the packing if the checkpoint is already there, and carries on.

### The five steps, one at a time

`all` is the five below in order. Any one of them runs alone when something
needs redoing by itself — with `KD_CONFIG` exported above, each is one word:

```bash
./scripts/score-pod.sh setup       # 1
./scripts/score-pod.sh quantize    # 2   fresh token
./scripts/score-pod.sh upload      # 3   fresh token
./scripts/score-pod.sh eval        # 4   fresh token
./scripts/score-pod.sh upload      # 5   fresh token - the evaluation
```

| | Step | What it does | Re-running it |
| --- | --- | --- | --- |
| 1 | `setup` | Environment, disk and GPU checks, then llm-compressor with `transformers` deliberately unpinned and **no** vLLM. | Free — skips when the imports already work. |
| 2 | `quantize` | Merges the adapter and packs it to W4A16 into `quantization.output_dir`. ~20 min. Fetches the adapter from the bucket in its first seconds. | Free — skips when the packed checkpoint is already there. |
| 3 | `upload` | Ships the bundle: adapter, packed student, checkpoints, logs, metrics. | Re-sends everything; safe, just slow. |
| 4 | `eval` | Installs vLLM — which **replaces torch** with the build it pins — then scores five players with `--skip upload`. Hours. | Starts the scoring over; steps 1–3 are untouched. |
| 5 | `upload` | The same command again. This is [the evaluation upload](#where-the-evaluation-upload-happens). | As step 3. |

`upload` is the same word twice on purpose — one command, run at two moments,
shipping whatever the bundle holds at the time. Step 3 exists so that hours of
packing are in the bucket before the scoring starts; step 5 adds the scores.

Two options worth knowing: `--run ID` overrides the run id the script otherwise
reads from `evaluation.adapter`, and `--yes` removes the pause before each
billable step — **don't**, unless you are certain the credentials in this shell
are fresh, because the pauses are where you re-export them.

### Check the banner before walking away

```
players : base, distilled, distilled-w4a16, teacher-base, teacher
```

Five names. Four means the packed student was not found, and the run is worth
killing rather than paying for.

Hours. Expect a lower number than the training distribution would give — see
[what this run is expected to cost](#what-this-run-is-and-what-it-is-expected-to-cost).

### Where the evaluation upload happens

`all`'s **second** `upload` is the evaluation upload. There is no separate
command for it: the eval writes into `runs/<train-run-id>/evaluation/<eval-id>/`
inside the training bundle, this profile inherits `evaluation` in `s3.upload`,
and `kd upload <run-id>` therefore carries `evaluation/**` with it
(`score-pod.sh:320`).

**It re-sends the rest of the bundle too.** `kd upload` has no skip-if-exists —
`s3.py:404-409` ships every file the named groups match, every time — so that
second upload re-transmits `final_adapter/`, `quantized/` and `checkpoints/`,
roughly 8 GB you already sent in step 2. On a slow pod uplink with a 15-minute
token that is the difference between an upload that fits in the window and one
that does not.

To send only what is new, narrow the groups — *fresh token*:

```bash
PYTHONPATH=/workspace/knowledge-distillation/src python3 -m kd upload <run-id> \
    --config configs/enlibra/enlibraQ3-14B-to-4B-score.yaml \
    --set 's3.upload=[evaluation,metrics,report,logs]'
```

The run id is the **training** run id, not an eval id: the evaluation lives
inside that adapter's bundle, and it lands at
`<bundle>/evaluation/w4a16-<date>/`.

**Do not use `./run.sh` for this on pod B.** The scoring profile has
`quantization.enabled: true`, so `run.sh`'s mandatory `setup` would try the
llm-compressor install again — and by this point `do_eval` has moved
`transformers` to the version vLLM pins, so the probe may not be satisfied and
the whole conflict comes back. `python3 -m kd upload` directly, or
`./scripts/score-pod.sh upload`, which bypasses `run.sh` for exactly this
reason.

Then terminate pod B in the console. Nothing stops one you started by hand.

---

## Why quantization is off in the training profile

`configs/enlibra/enlibraQ3-14B-to-4B.yaml` sets `quantization.enabled: false`,
and `enlibraQ3-14B-to-4B-score.yaml` flips it back to `true`. That one line is
what lets `./run.sh` drive the training pod at all.

`run.sh` calls `setup` before **every** command (`run.sh:256`). On a pod that
delegates to `scripts/runpod.sh`, which reads the config to decide which
optional groups to install (`runpod.sh:214-232`):

| config says | group added | brings in |
| --- | --- | --- |
| `evaluation.engine: vllm` | `serve` | vLLM |
| `quantization.enabled: true` | `quantize` | llm-compressor, compressed-tensors |

It then builds **one** pip call out of all of them (`runpod.sh:296`) — including
`transformers>=5.16.1` from `[project.dependencies]`. llm-compressor caps
transformers at `<=5.14.1`, so that resolve has no solution. Because `setup`
runs first for every command, the failure stops `check` and `doctor` as surely
as it stops the pipeline. This is the conflict
[RUNBOOK-14B.md](RUNBOOK-14B.md) hand-installs around for the 8B profile.

Nothing is lost by moving it. The packing belongs on the scoring pod regardless,
and `scripts/score-pod.sh` does the ordered install itself
(`score-pod.sh:265` — llm-compressor with transformers deliberately unpinned,
then vLLM afterwards, for the step that needs it). The 80 GB card does nothing
but train, which is the split [RUNBOOK-14B.md](RUNBOOK-14B.md) already
recommends on cost grounds.

Every other packing setting — `scheme`, `group_size`, `ignore`,
`calibration_file` — stays in the training profile and is inherited by the
scoring one, so the packed student is described in exactly one place.

### What is still unverified

`evaluation.engine: vllm` is inherited from `_base.yaml`, which states that no
profile overrides it, so the `serve` group is still added on the **training**
pod. vLLM is useless there — nothing on that pod scores — and installing it
replaces the template's torch with vLLM's own CUDA build. Training works either
way; it costs a download.

**Whether vLLM's own transformers pin also collides with `transformers>=5.16.1`
has not been tested.** The install says so in its first minute, and

```bash
./run.sh --config configs/enlibra/enlibraQ3-14B-to-4B.yaml doctor
```

surfaces it before any GPU time goes into training. Run that first. If it does
collide, the fix is a `--no-extras` path through `run.sh`, which does not exist
yet.

---

## What Qwen3-4B changes, and what it does not

Verified against the Hub while writing the profile:

| | Qwen3-8B | Qwen3-4B |
| --- | --- | --- |
| `vocab_size` | 151936 | 151936 |
| `num_hidden_layers` | 36 | 36 |
| `hidden_size` | 4096 | 2560 |
| `intermediate_size` | 12288 | 9728 |
| `tie_word_embeddings` | false | **true** |
| LoRA r=128 over seven projections | 349M trainable | **264M** trainable |

- **GKD is unaffected.** Identical vocabulary and the same Qwen3 chat template,
  so the shared-vocabulary requirement standard GKD depends on holds exactly as
  it does for the 8B student.
- **`group_size: 128` is still valid.** It must divide the widths it groups:
  2560 = 20 groups, 9728 = 76 groups.
- **`ignore: [lm_head]` protects a different matrix.** Qwen3-4B ties its
  embeddings, so what stays at full width is the shared 151936 × 2560 embedding
  — 0.78 GB at bf16 rather than the 8B's standalone 1.2 GiB projection. Same
  reasoning (its rounding error lands straight on the logits with no later layer
  to absorb it), and a larger share of a smaller model — so expect the packed
  checkpoint nearer 2.4× smaller than the 2.7× the 8B saw.
- **`r=128` is deliberately unchanged.** Raising it to compensate for the
  narrower base is the obvious thing to try and the wrong thing to try *first*:
  it would confound the one variable this profile exists to isolate. Make it a
  third run, once this one has a number.

---

## What ends up in the bucket

```
<prefix>/runs/<run-id>/
    manifest.json
    config.resolved.yaml
    final_adapter/            ~0.5 GB   the result
    quantized/                ~2.5 GB   what actually gets deployed
    checkpoints/                ~5 GB   optimizer state, for a resumed run
    run.log  events.jsonl
    metrics.json  arena.json  arena-transcript.jsonl
    evaluation/w4a16-<date>/            scores, transcript, report.html
```

Bundles land beside the 8B ones under the same `s3.prefix`. They never collide:
the run id leads with the profile name, and this profile's is
`enlibra-neuroscience-q3-14b-to-4b`.

---

## If a step fails

| Failed at | Redo | Keep |
| --- | --- | --- |
| any upload | that upload alone, with fresh credentials | everything |
| eval, partway | `./scripts/score-pod.sh eval` | the adapter and packed student, already in the bucket |
| quantize | `./scripts/score-pod.sh quantize` | training |
| training | **the whole training run** | only what `save_steps: 50` wrote |

**There is no resume.** Nothing in the pipeline passes
`resume_from_checkpoint`, so a training run that is killed — by a limit, a spot
reclaim, a dropped SSH session — cannot be continued. What `save_steps: 50` buys
is a partial adapter you can inspect, score or ship, not a run you can restart
where it stopped. Upload it before starting over — the ordinary upload, since
this profile inherits `checkpoints` in `s3.upload` and ships them anyway:

```bash
./run.sh --config configs/enlibra/enlibraQ3-14B-to-4B.yaml upload <run-id>
```

The rule the three uploads exist to enforce: **nothing expensive should be on
the pod alone for longer than it has to be.**

---

## Cost, roughly

| | |
| --- | --- |
| install + downloads, pod A | ~30 min @ 80 GB |
| training | hours @ 80 GB |
| install, pod B | ~10 min @ 48 GB |
| packing | ~20 min @ 48 GB |
| scoring | 1.5–3 h @ 48 GB |

At roughly $2.79/hour for an 80 GB card against ~$1.00 for a 48 GB one. The
profile caps a run at `limits.max_cost_usd: 30.00`, which is **inert unless
`KD_PRICE_PER_HOUR` is exported**.

---

## See also

- [RUNBOOK-14B.md](RUNBOOK-14B.md) — the 8B student, and the hand-installed
  variant of this same cycle
- [POD-FAILURES.md](POD-FAILURES.md) — every error seen on these pods, and what
  it really meant
- [CONFIG.md](CONFIG.md) — what every key in the profile does
- [EXPLANATION-SCORING.md](EXPLANATION-SCORING.md) — how to read the report
