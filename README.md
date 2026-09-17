# knowledge-distillation

Distil a large teacher model into a small LoRA student, with on-policy knowledge
distillation (GKD). Config-driven: point it at a YAML file and it runs.

Everything about a run — models, dataset, LoRA shape, schedule, hardware, spend
limits — comes from that file. Retargeting to a different teacher, student or
dataset never requires a code edit.

```bash
./distill.sh --config configs/qwen/finance.yaml          # Linux, macOS, containers
.\distill.ps1 -Config configs\qwen\finance.yaml          # Windows
```

```
  run   finance-2026-09-08-1412               device cuda (bfloat16)
  cfg   configs/qwen/finance.yaml  <- _base.yaml   overrides: training.max_steps=500 (--set)

[1/7] preflight            OK       2s
[2/7] teacher-check        OK    1m48s
[3/7] smoke                OK    1m12s
      3.40 s/step measured -> 500 steps is about 28m
[4/7] train                ...
```

---

## What it does

Two pipelines, because training and evaluation have different lives.

**Training** — `kd pipeline` — runs every level of verification, gated,
stopping at the first failure, and ends with the adapter in the bucket:

| Stage | Gate | |
|---|---|---|
| `preflight` | ✓ | Resolve config and hardware, fetch remote inputs, split a teacher that is a LoRA adapter into base + adapter. Seconds. |
| `teacher-check` | ✓ | Load **only** the teacher: missing weights, NaN scan, coherence. |
| `smoke` | ✓ | Two real steps. Measures s/step and projects the full run against your limits. |
| `train` | ✓ | |
| `evaluation` | | The whole evaluation pipeline below, inside the run. **Off by default**; `evaluation.after_training: true` turns it on. |
| `publish` | | To the Hugging Face Hub. Off by default. |
| `quantize` | | Pack the student to W4A16 for deployment. Off by default; `quantization.enabled: true` turns it on. |
| `upload` | | To S3. Off by default; also runs after a failure, so logs survive. |

**Evaluation** — `kd eval` — scores an adapter that already exists: this
checkout's newest, or one named by path or `s3://` URI. Run it as many times
as there are questions to ask of one adapter, on whatever machine is to hand:

| Stage | Gate | |
|---|---|---|
| `preflight` | ✓ | Fetch the teacher and the adapter; say what will be scored and where it goes. |
| `evaluate` | | Fidelity and capability, token by token, against the teacher. |
| `arena` | | Accuracy and Elo on a held-out answer key. Off unless `evaluation.arena_file` is set. |
| `report` | | A readable `report.html`, led by how close the distilled student is to the teacher. |
| `upload` | | To S3, **beside the adapter it scored**. Off unless `s3.enabled`. |

Four players: the **base** student, the **distilled** student, the
**teacher-base** (the stock model the teacher was fine-tuned from) and the
**teacher**. Every evaluation is a directory of its own inside the adapter's
bundle, so a bucket listing shows every time an adapter was scored:

```
runs/enlibraQ25-3B-2026-09-10-1416/
  final_adapter/
  evaluation/
    quick-2026-09-15-1030/     evaluation.json  arena.json  report.html  ...
    full-2026-09-15-1412/
```

Narrow either with `--only train`, `--from arena`, `--skip smoke`.

The `teacher-check` and `smoke` gates exist because of how this fails in
practice. GKD trains the student to match the teacher's *output distribution*, so
a teacher that loads without raising but is partly randomly initialised produces
a broken student after a full, apparently successful run — and nothing says so
until you read the final samples. The check loads only the teacher and takes a
couple of minutes.

## Getting started

**New here?** Clone and run one script. It installs everything and runs the
config you name — `--config` is required, and nothing is ever substituted for
it:

```bash
git clone <this repo> && cd knowledge-distillation
./run.sh --config configs/enlibra/enlibraQ3-8B-smoke.yaml
```

Then, once a real run has put an adapter in the bucket, score it:

```bash
./run.sh --config configs/enlibra/enlibraQ25-3B.yaml eval \
    --adapter s3://enlibra/.../runs/enlibraQ25-3B-2026-09-10-1416/final_adapter
```

Walkthrough: **[docs/START-HERE.md](docs/START-HERE.md)**. The rest of this page
is the underlying `kd` command, which `run.sh` drives.

You need [uv](https://docs.astral.sh/uv/) and Python 3.13. `run.sh` installs uv
if it is missing.

```bash
git clone <this repo> && cd knowledge-distillation
uv sync
source .venv/bin/activate            # Windows: .venv\Scripts\activate

kd doctor                                    # what can this machine do?
kd check --config configs/smollm/smoke.yaml         # resolve everything, run nothing
kd pipeline --config configs/smollm/smoke.yaml      # ~2 min end to end
```

`uv sync` installs a real `kd` executable into `.venv`. Without activating, prefix
with `uv run`; `python -m kd` works too and needs nothing on `PATH`.

`configs/smollm/smoke.yaml` trains SmolLM2-360M into SmolLM2-135M for two steps. It
proves the machine works before you commit a real budget.

Then pick a real profile, or write one:

```bash
kd pipeline --config configs/qwen/finance.yaml
```

### Or download a runner

No clone, no uv, no Python setup. CI publishes `distill.sh` and `distill.ps1`
pinned to the commit they were built from; each installs uv if it is missing,
fetches that exact source, and hands over to the pipeline. Grab one from the
**Build distillation runner** workflow artifacts or a release. This is the way to
put it on someone else's machine.

```bash
chmod +x distill.sh
./distill.sh --config configs/qwen/finance.yaml
```

## Commands

```
kd pipeline    Train: every stage, gated             kd check       Resolve and print, run nothing
kd eval        Score an adapter, into its bundle     kd doctor      Environment and credentials
kd train       Just training                         kd ui          Browser control panel
kd evaluate    Fidelity vs the teacher, standalone   kd runpod      Rent a GPU (optional)
kd arena       Accuracy + Elo on an answer key       kd upload      Re-ship a run or an evaluation to S3
kd quantize    Pack the student to W4A16 for vLLM    kd publish     To the Hugging Face Hub

kd check-teacher      Is this teacher fit to distil from?
kd fix-teacher        Repair a checkpoint with mislabelled tensors
kd convert-adapter    MLX / unsloth LoRA -> PEFT format
```

`kd <command> --help` for anything. `python -m kd` works identically and needs
nothing on PATH.

## Configuring a run

Profiles inherit from [`configs/_base.yaml`](configs/_base.yaml), which holds
every default, and list only what they change. They are grouped by what they
distil — `configs/enlibra/`, `configs/smollm/`, `configs/qwen/`. One profile
describes both the training and the evaluation of a pair: `kd pipeline` reads
the training half, `kd eval` the `evaluation:` section.

```yaml
extends: _base.yaml

models:
  teacher: Qwen/Qwen3.5-2B
  student: Qwen/Qwen3.5-0.8B
training:
  max_steps: 300
```

Override anything from the command line without editing a file:

```bash
kd pipeline --config configs/qwen/finance.yaml \
  --set training.max_steps=500 \
  --set hardware.dtype=bfloat16 \
  --set limits.max_cost_usd=2.0
```

Precedence is `_base.yaml < profile < KD_* < --set < flag`, and every value that
differs from the base is echoed at startup with the layer that set it. A mistyped
key is a hard error with a suggestion, not a silent no-op.

Full key-by-key reference: **[docs/CONFIG.md](docs/CONFIG.md)**.

### Shipped profiles

| | |
|---|---|
| `configs/smollm/default.yaml` | SmolLM2 360M → 135M. Runs anywhere. |
| `configs/smollm/mac.yaml` | Same pair, Apple Silicon (MPS) with real batching. |
| `configs/smollm/smoke.yaml` | Two steps, tiny pools, evaluation included. CI and first-run validation. |
| `configs/qwen/finance.yaml` | A finance-tuned Qwen3.5-2B (base + LoRA) → Qwen3.5-0.8B. |
| `configs/qwen/qwen-poc.yaml` | Stock Qwen3.5-2B → 0.8B. A known-good pairing for proving the pipeline. |
| `configs/enlibra/enlibraQ25-3B.yaml` | The SFT teacher (Qwen2.5-3B) → Qwen2.5-1.5B. ~8.6 GB of weights, so a 16 GB GPU is enough. Start here. |
| `configs/enlibra/enlibraQ25-3B-sftonly.yaml` | The same run without the rl rows: what those rows are worth is the difference between the two scores. |
| `configs/enlibra/enlibraQ25-3B-smoke.yaml` | The same **real** pair on a laptop, training on a small slice of the corpus, evaluation included. |
| `configs/enlibra/enlibraQ25-3B-lora-smoke.yaml` | The smoke run again, with the teacher named as the SFT run's **LoRA adapter** in S3 rather than a merged checkpoint. Proves the adapter-as-teacher route: fetch, find the base, merge, check, train, score four players. |
| `configs/enlibra/enlibraQ25-flow.yaml` | Qwen2.5-1.5B → 0.5B, both from the Hub. 3.8 GB, minutes, no S3 download — proves every stage, evaluation included. |
| `configs/enlibra/enlibraQ3-8B.yaml` | An RL-tuned Qwen3-8B → Qwen3-1.7B on the enLibra space curriculum. Needs a 48 GB GPU. |
| `configs/enlibra/enlibraQ3-8B-smoke.yaml` | The same run with stand-in models, small enough for a 16 GB laptop. Two steps — proves the plumbing. |
| `configs/enlibra/enlibraQ3-8B-mac.yaml` | Stand-in models again, but the **full** schedule and the whole evaluation. Hours, free, and it answers whether distillation works on this data. |
| `configs/enlibra/enlibraQ3-14B.yaml` | The SFT-tuned Qwen3-14B → Qwen3-8B, on the neuroscience curriculum. The largest pair here; needs an 80 GB GPU. |
| `configs/enlibra/enlibraQ3-14B-smoke.yaml` | The same pair with stand-in models, small enough to prove the plumbing on a laptop. |
| `configs/enlibra/enlibraQ3-14B-score.yaml` | Pack and score an adapter **already in the bucket**, on a pod that never trains. Names the adapter, pins where the packed copy goes, and keeps the S3 cache on the volume. Driven by `scripts/score-pod.sh`. |

The training profiles end with the adapter in the bucket and do not evaluate;
the smoke, flow and mac profiles set `evaluation.after_training: true` because
proving every stage is what they are for. Any profile can be flipped the same
way with `--set evaluation.after_training=true`.

Both models are resident at once, so their **sum** is what has to fit: Qwen3-8B
plus Qwen3-1.7B is 19.0 GB of bf16 weights before activations, which is why the
pod profile asks for a 48 GB card and why the laptop profiles shrink both halves.
[docs/START-HERE.md](docs/START-HERE.md) has the table.

The enlibra profiles read `data/enlibra-curriculum/`, which **is committed** — 3.4 MB, so
a fresh clone on a rented pod has the corpus already and needs no credentials for
it. Regenerate it when the curriculum exports change:

```bash
python scripts/prepare_curriculum.py --out data/enlibra-curriculum \
    --stats-tokenizer Qwen/Qwen3-8B \
    curriculum_verified.json curriculum_sft.json
```

That converts the bespoke multiple-choice export into chat JSONL, one file per
`split`, and prints the token budgets the result needs.
`curriculum_verified.json` is the superset and supplies sft, rl and eval;
`curriculum_sft.json` is passed only for the persona rows the superset does not
carry — everything else in it deduplicates away against what the first file
already wrote.

| | | |
|---|---|---|
| `sft-1to3hop.jsonl` | 937 | trains |
| `rl-1to2hop.jsonl` | 143 | trains |
| `identity.jsonl` | 15 | trains |
| `eval-1to5hop.jsonl` | 137 | **held out** — what the `arena` stage scores |

The converter is deterministic, so an unchanged export leaves `git status`
clean, and `manifest.json` records the sha1 of every export the corpus was built
from. The held-out file is reported as such and left out of `dataset.domains`;
pointing a domain at it takes a deliberate edit.

### Using what it produced

A run produces a LoRA adapter, not a model. `./infer.sh` merges it into the base
and lets you talk to the result:

```bash
./infer.sh --ask "Who are you?"      # newest adapter, merged in memory
./infer.sh --chat                    # keep asking
./infer.sh --out ./merged            # write a standalone checkpoint
```

Needs no config — the adapter records its own base — so it works on one pulled
out of S3 or sent by someone else. Full guide: **[docs/INFERENCE.md](docs/INFERENCE.md)**.

## What a run leaves behind

```
runs/finance-2026-09-08-1412/
  config.resolved.yaml   every value after every override - re-runnable as-is
  manifest.json          git sha, package versions, per-stage timings, exit code
  run.log                everything the terminal showed, including library output
  events.jsonl           {stage, step, loss, elapsed, spend_usd} - one per line
  metrics.json           the final numbers
  final_adapter/
  checkpoints/
  evaluation/            one directory per scoring of the adapter, by `kd eval`
    full-2026-09-15-1412/  or by the run itself (evaluation.after_training)
      config.resolved.yaml  manifest.json  run.log  events.jsonl
      evaluation.json  arena.json  arena-transcript.jsonl  report.html
```

One directory per run, never overwritten, and the same unit that gets uploaded to
S3 — so a run is either entirely recoverable or entirely absent. An evaluation
run elsewhere uploads its own directory to the same place in the bucket, so the
bundle there always holds every evaluation whichever machine produced it.

## Spend and time limits

```yaml
limits:
  max_runtime_minutes: null   # no clock by default — see below
  max_cost_usd: 2.00          # only meaningful on a rented GPU
  confirm_above_usd: 1.00
```

**There is no wall-clock ceiling by default.** One does not pause a run, it
*kills* it — mid-epoch, keeping only the last `training.save_steps` checkpoint —
and a run that is merely slower than someone guessed is not a run that has gone
wrong. Set one deliberately, per run, when you want that guard.

Checked twice. The `smoke` stage measures s/step, so a run that cannot finish
inside its limits is **refused before it starts**:

```
This run would cost about $0.12, over the $0.05 limits.max_cost_usd.
  measured 4.30 s/step over 300 steps at $0.34/hr
  raise the ceiling:  --set limits.max_cost_usd=0.15
  or shorten the run: --set training.max_steps=110
```

In flight, a breach is a hard stop. The artifact you keep is the last checkpoint,
so tighten `training.save_steps` when limits are tight.

## Measuring what you got

```bash
kd eval --config configs/enlibra/enlibraQ25-3B.yaml --adapter s3://.../final_adapter
kd eval --config configs/qwen/finance.yaml            # this checkout's newest adapter
kd eval --config configs/qwen/finance.yaml --name after-parser-fix --from arena
kd eval --config configs/enlibra/enlibraQ25-3B.yaml --name quick \
    --set evaluation.arena_limit=8 --set evaluation.samples=8   # a look, in minutes
```

The same profile that trained the pair evaluates it: `kd eval` reads its
`evaluation:` section (players, samples, the held-out file, the ceilings) and
the models, tokenizer and bucket from the rest.

`kd eval` is the evaluation pipeline: fidelity, the arena, the report and the
upload, written into `<bundle>/evaluation/<name>-<date>-<time>/` beside the
adapter — here and in the bucket. `--adapter` names what to score (a
directory, a file inside one, or an `s3://` URI); `evaluation.adapter` in the
config does the same; with neither, it scores the newest adapter under `runs/`.
`--name` (or `evaluation.name`) says what the scoring was for and leads the
directory name, so a bundle's `evaluation/` listing reads as a history.

Two questions, measured separately, because the literature is explicit that they
do not track each other:

- **Fidelity** — does the student reproduce the *teacher's* predictions? Top-1
  agreement, KL divergence.
- **Capability** — is the student actually better at the task? Held-out
  perplexity.

Five models are scored, each answering a question the others cannot:

| Player | | What it tells you |
|---|---|---|
| `base` | the stock student, no adapter | The control. "The small model could already do this" lives here. |
| `distilled` | the student plus the adapter | The result. |
| `distilled-w4a16` | the same student, packed to 4 bits | What ships. Scored beside `distilled` because what quantization cost is a **difference**, and one column cannot carry one. Skipped when nothing has been packed. |
| `teacher-base` | the stock model the teacher was fine-tuned from | What the fine-tune bought the **teacher** — the most there was to distil. Needs `models.teacher_base`, or a teacher given as base + adapter. |
| `teacher` | the fine-tuned teacher | The ceiling, and the reference every closeness figure is measured against. |

`evaluation.players` picks the set; drop `teacher` and `teacher-base` to compare
base against distilled without loading eight gigabytes of weights.

`kd evaluate` and `kd arena` remain as standalone tools for one measurement at
a time. `kd evaluate --quantized DIR` scores a packed checkpoint on the same
tokens as the dense one, which is what isolates the cost of quantization.

### The teacher can be a LoRA adapter

`models.teacher` may point at a merged checkpoint, at a base model with
`models.teacher_adapter` beside it, or straight at a **LoRA adapter** — a
directory, an `s3://` prefix or a Hub repo holding `adapter_config.json`.
Preflight notices, reads the base the adapter records (or `models.teacher_base`
when set), and rewrites the config into the base + adapter form before anything
loads. The merge happens once, from the canonical base, which sidesteps the
key-layout problems of merged exports written by other frameworks.

### Multiple-choice accuracy, and asking the student directly

For a corpus with a known correct answer per row — the enLibra curriculum — the
number that decides whether the run worked is accuracy, not fidelity. A student
that mirrors a mediocre teacher perfectly scores well above and badly here.

```bash
# score the held-out split: the rows training never saw
python scripts/ask.py --config configs/enlibra/enlibraQ3-8B.yaml --accuracy

# ask it one thing
python scripts/ask.py --config configs/enlibra/enlibraQ3-8B.yaml \
    --question "What are stars formed from?" \
    --option "A. Iron cores" --option "B. Molecular clouds" \
    --option "C. Accretion disks" --option "D. Supernova shockwaves"
```

It finds the newest adapter under `runs/` on its own, loads the tokenizer from
the adapter directory so the chat template matches training exactly, wraps a
free-text question in the `<Question>`/`<Options>` shape the student was trained
on, and decodes greedily so the score is the same number twice. `--accuracy`
rebuilds the held-out split from the config and `project.seed`, the same way
`kd evaluate` does.

## Optional: S3 and rented GPUs

Both off by default; nothing in the pipeline imports their SDKs.

Any path in a config may be an `s3://` URI — teacher, student, adapter, dataset —
fetched in `preflight` and cached. Run bundles sync back up.

```bash
kd runpod launch --config configs/qwen/finance-pod.yaml
```

Rents the GPU you named and **no other**, enforces your spend cap from your own
machine, and always terminates the pod. See
**[docs/RUNPOD.md](docs/RUNPOD.md)**.

## Repository map

```
configs/          _base.yaml holds every default; profiles extend it
  enlibra/        the enLibra space curriculum (Qwen2.5 and Qwen3 pairs)
  smollm/         SmolLM2 - the starter pair that runs anywhere
  qwen/           stock Qwen3.5 pairs, finance
  eval/           evaluation-only profiles, for `kd eval`
src/kd/
  arena.py        accuracy and Elo on a held-out multiple-choice set
  cli.py          the `kd` command
  config.py       extends, merge, strict validation, device resolution
  runlog.py       run directories, logs, events, manifest
  pipeline.py     the gated stages: training, and evaluation
  limits.py       time / step / cost ceilings
  paths.py        s3:// -> local, cached; a teacher that is an adapter
  data.py         dataset assembly
  train.py        GKD training
  evaluate.py     fidelity and capability
  merge.py        adapter + base -> one dense checkpoint, vocabulary trap and all
  quantize.py     W4A16 packing, for the checkpoint that actually ships
  vllm_runner.py  the arena's generation batched through vLLM, in a subprocess
  report.py       Markdown and self-contained HTML
  teacher.py      loading and verifying a teacher
  teacher_fix.py  repairing a mislabelled checkpoint
  adapters.py     MLX -> PEFT conversion
  publish.py      Hugging Face Hub
  remote/         s3.py, runpod.py - optional
  ui/             control.py, compare.py - Gradio
scripts/          distill.{sh,ps1}.template - bootstrappers, rendered by CI
docker/           Dockerfile.cuda - the image a rented GPU runs
tests/            plain asserts, no framework, no downloads
docs/             CONFIG.md, RUNPOD.md, HOW_IT_WORKS.md, DEMO_PROMPTS.md
  papers/         the GKD, DistiLLM and MiniLLM papers, and what they say about the gkd knobs
```

## Tests

No framework, no model downloads, about a second each:

```bash
for t in tests/*.py; do uv run python "$t"; done
```

They cover config precedence and validation, run bundles and their failure paths,
pipeline gating and limits, the answer parser, the merge helper's vocabulary
routes, the vLLM engine's plumbing (without vLLM installed), the runner
templates, S3, and — with fake SDKs — GPU selection and the terminate guarantee.

## Further reading

- **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** — CI build to a usable model, one path start to finish
- **[docs/RUNBOOK-SOURCE.md](docs/RUNBOOK-SOURCE.md)** — the full runbook, from a clone with `uv run`
- **[docs/RUNBOOK-RUNNER.md](docs/RUNBOOK-RUNNER.md)** — the same runbook, using only the downloaded `distill.sh`
- **[docs/CONFIG.md](docs/CONFIG.md)** — every configuration key
- **[docs/RUNPOD.md](docs/RUNPOD.md)** — renting a GPU, and keeping it cheap
- **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** — what GKD is doing and why
- **[docs/DEMO_PROMPTS.md](docs/DEMO_PROMPTS.md)** — prompts where the difference shows
