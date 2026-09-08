# Runbook — from a source checkout

Everything from a fresh clone to a trained adapter — and how to switch on
evaluation extras, the Hugging Face Hub, S3, and rented GPUs when you need them.

**This document assumes you have cloned the repository** and drives everything
through `uv run`, so nothing needs to be activated or put on your `PATH`. It is
the right document for working *on* the pipeline, or for running it on a machine
you already develop on.

> Deploying to a machine that has no clone and no Python setup? Use
> [RUNBOOK-RUNNER.md](RUNBOOK-RUNNER.md) instead — the same pipeline, driven by a
> single downloaded script.

If you would rather type `kd` than `uv run kd`, activate the environment once
(`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) and drop
the prefix from every command below.

- [1. Install and first run](#1-install-and-first-run)
- [2. The eight stages](#2-the-eight-stages)
- [3. Running a real job](#3-running-a-real-job)
- [4. What you get back](#4-what-you-get-back)
- [5. Turning on the optional parts](#5-turning-on-the-optional-parts)
- [6. Deploying to other machines](#6-deploying-to-other-machines)
- [7. Keeping the cost down](#7-keeping-the-cost-down)
- [8. When something goes wrong](#8-when-something-goes-wrong)

---

## 1. Install and first run

Three commands to prove the machine works before you commit a real training
budget. You need [uv](https://docs.astral.sh/uv/) and Python 3.13. Nothing else —
no CUDA, no accounts, no credentials.

```bash
# 1. Install dependencies (first run pulls torch, a few minutes)
uv sync

# 2. What can this machine actually do?
uv run kd doctor

# 3. Resolve the config and hardware without running anything
uv run kd check --config configs/smoke.yaml

# 4. The whole pipeline, end to end (~2 minutes)
uv run kd pipeline --config configs/smoke.yaml
```

`configs/smoke.yaml` distils SmolLM2-360M into SmolLM2-135M for two steps. It is
deliberately too short to learn anything — its job is to prove every stage runs
on your hardware.

A successful run looks like this:

```
[1/8] preflight            OK       0s
[2/8] teacher-check        OK      22s
[3/8] smoke                OK      51s
      5.13 s/step measured -> 2 steps is about 10s
[4/8] train                OK      36s
[5/8] evaluate             OK      22s
[6/8] report               OK       0s
[7/8] publish              skipped - publish.enabled is false
[8/8] upload               skipped - s3.enabled is false

  run bundle : runs/20260908T152208Z-smoke-25561e8
  adapter    : runs/20260908T152208Z-smoke-25561e8/final_adapter
  report     : runs/20260908T152208Z-smoke-25561e8/report.html
```

Open that `report.html` in a browser. It leads with a plain-English summary, so
it is readable by someone who does not know what perplexity is.

### The commands

| Command | What it does |
|---|---|
| `kd pipeline` | Every stage, gated. The one you will use most. |
| `kd check` | Resolve config and hardware, print, run nothing. Instant. |
| `kd doctor` | Device, package versions, which optional features and credentials are present. |
| `kd train` | Just training, no gates. |
| `kd evaluate` | Score an adapter against the teacher. |
| `kd check-teacher` | Is this teacher fit to distil from? Loads only the teacher. |
| `kd fix-teacher` | Repair a checkpoint whose tensor names do not match its architecture. |
| `kd convert-adapter` | MLX / unsloth LoRA into PEFT format. |
| `kd publish` | Push adapter and merged model to the Hugging Face Hub. |
| `kd runpod` | Rent a GPU and run there. |
| `kd ui` | Browser control panel; `--compare` for the three-way generation view. |

`python -m kd` is identical and needs nothing on `PATH` at all — that is the
form the generated runners and the container use.

---

## 2. The eight stages

One command runs every level of verification in order and stops at the first
failed gate.

| # | Stage | Gate | |
|---|---|---|---|
| 01 | `preflight` | ✓ | Resolve config and hardware; fetch any `s3://` inputs. Seconds, so a bad bucket or a typo fails immediately. |
| 02 | `teacher-check` | ✓ | Loads *only* the teacher: randomly-initialised weights, NaN scan, is the output actually language. |
| 03 | `smoke` | ✓ | Two real steps. Measures seconds-per-step and projects the full run against your limits. |
| 04 | `train` | ✓ | The run itself, under the time and cost ceilings. |
| 05 | `evaluate` | | Fidelity and capability, measured against the untrained base student. |
| 06 | `report` | | A readable `report.html` in the run bundle. |
| 07 | `publish` | | To the Hugging Face Hub. Skipped unless switched on. |
| 08 | `upload` | | To S3. Skipped unless switched on — and runs even after a failure, so the logs survive. |

A **gate** stage that fails aborts the run. A non-gate stage that fails is
reported and the run continues, so a broken report never destroys a good adapter.

### Why the first two gates exist

GKD trains the student to match the teacher's *output distribution*. A teacher
that loads without raising but is partly randomly initialised produces a broken
student after a full, apparently successful run — and nothing tells you until you
read the final samples. The teacher check loads only the teacher and takes a
couple of minutes. It is the cheapest check that catches the most expensive
mistake.

### Running part of it

```bash
uv run kd pipeline --config configs/finance.yaml --only train
uv run kd pipeline --config configs/finance.yaml --from evaluate
uv run kd pipeline --config configs/finance.yaml --skip smoke
```

`--from` and `--only` open a new run directory but pick up the newest previous
run's adapter or evaluation, so you can re-report without retraining.

### Exit codes

| | |
|---|---|
| `0` | Everything that ran, worked. |
| `1` | A gate failed. |
| `2` | The teacher is not fit to distil from. |
| `3` | The run completed, but the adapter did not improve on the base student. |
| `4` | A limit was hit and the run was stopped. |

---

## 3. Running a real job

| Profile | Pair | For |
|---|---|---|
| `default` | SmolLM2 360M → 135M | Runs anywhere. |
| `mac` | Same pair | Apple Silicon (MPS), with real batching instead of accumulation. |
| `smoke` | Same pair, 2 steps | CI and first-run validation. |
| `finance` | Qwen3.5-2B → 0.8B | A finance-tuned teacher on finance-alpaca. |
| `qwen-poc` | Qwen3.5-2B → 0.8B | Stock models. A known-good pairing for proving the pipeline. |

```bash
uv run kd pipeline --config configs/finance.yaml
```

### Overriding without editing anything

`--set` reaches *any* key by dotted path and is repeatable:

```bash
uv run kd pipeline --config configs/finance.yaml \
  --set training.max_steps=500 \
  --set hardware.dtype=bfloat16 \
  --set gkd.lmbda=0.25 \
  --set limits.max_runtime_minutes=120
```

Short flags exist for the common ones — `--teacher`, `--student`, `--steps`,
`--device`, `--dtype`, `--lr`, `--lmbda` — and beat `--set`. Precedence runs
`_base.yaml → profile → KD_* env → --set → flag`, and every value that differs
from the base is echoed at startup with the layer that won it:

```
 overrides     :
   gkd.lmbda = 0.25  <- --set  (base: 0.5)
   training.max_steps = 500  <- flag  (base: 300)
```

### Writing your own profile

Every default lives in [`configs/_base.yaml`](../configs/_base.yaml). A profile
inherits from it and lists only what it changes:

```yaml
extends: _base.yaml

project:
  name: my-distillation
models:
  teacher: Qwen/Qwen3.5-2B
  student: Qwen/Qwen3.5-0.8B
  tokenizer: student
training:
  max_steps: 600
```

> **Two things that will save you a run.** Teacher and student **must share a
> tokenizer vocabulary** — that is what standard GKD requires. And **check
> `lora.target_modules` against your architecture**: Qwen3.5 is hybrid, so a
> Llama-style target list silently misses the attention in 18 of its 24 layers.

A mistyped key is a hard error with a suggestion, not a silent no-op:

```
Configuration is not valid:
  - unknown key 'gkd.lmbdaa'  did you mean 'gkd.lmbda'?
  - unknown key 'gkd.temprature'  did you mean 'gkd.temperature'?
```

Every key is documented in [CONFIG.md](CONFIG.md).

---

## 4. What you get back

One directory per run, never overwritten — and the same unit that gets uploaded
to S3.

```
runs/20260908T1412Z-finance-bb0c874/
  config.resolved.yaml   # every value after every override; re-runnable as-is
  manifest.json          # git sha, package versions, per-stage timings, exit code
  run.log                # everything the terminal showed, library output included
  events.jsonl           # {stage, step, loss, elapsed, spend_usd} - one per line
  metrics.json           # the final numbers
  report.html
  final_adapter/         # what you actually wanted
  checkpoints/
```

The manifest is written even when a run crashes or you interrupt it, so a failed
run is still diagnosable. `events.jsonl` is the machine-readable stream — feed it
to a dashboard or a CI check rather than parsing prose out of the log.

### Choosing where it goes

Two settings, and they do different things:

```bash
# A parent directory. Each run still gets its own <timestamp>-<profile>-<sha>
# subdirectory inside it, so runs never overwrite each other.
uv run kd pipeline --config configs/finance.yaml   --set project.runs_dir=/data/kd-runs

# The run directory itself, pinned. The adapter lands at
# /data/finance-adapter/final_adapter, at a path you can predict and script.
uv run kd pipeline --config configs/finance.yaml --output /data/finance-adapter
```

`--output` is shorthand for `--set project.output_dir`. Because it pins the exact
directory, a second run with the same value **overwrites the first** - which is
right for a nightly job that always publishes to the same place, and wrong for
comparing two experiments. `runs_dir` is the safer default for the latter.

`KD_RUNS_DIR` and `KD_OUTPUT_DIR` set the same two from the environment, which is
useful in CI where the path comes from the job rather than the command.

### Using the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
model = PeftModel.from_pretrained(base, "./runs/<run-id>/final_adapter")
tok = AutoTokenizer.from_pretrained("./runs/<run-id>/final_adapter")
```

### Looking at it in a browser

```bash
uv run kd ui              # control panel: train, evaluate, compare
uv run kd ui --compare    # base vs distilled vs teacher, side by side
```

Every screen prints the exact command it is about to run, so the panel teaches
the CLI rather than hiding it.

---

## 5. Turning on the optional parts

Four features ship switched off. Nothing in the core pipeline imports their
dependencies, so a machine that never uses them never needs them installed.

### Benchmark evaluation — *off: extra not installed*

Score base, distilled and teacher on lm-evaluation-harness tasks, and compare
free-running generations with BERTScore.

```bash
uv sync --extra eval

uv run kd evaluate --config configs/finance.yaml \
  --tasks ifeval --limit 100 \
  --gen-similarity 20
```

Slow — it scores three models. Start with `--limit`.

### Hugging Face Hub — *off: `publish.enabled: false`*

Publishes the adapter to `<repo>-lora` and a merged, ready-to-run model to
`<repo>`. The merge is verified to generate language before anything uploads.

```bash
export HF_TOKEN=...

uv run kd pipeline --config configs/finance.yaml \
  --set publish.enabled=true \
  --set publish.repo=my-org/qwen-finance
```

Needs `HF_TOKEN`, or `hf auth login`.

### S3 storage — *off: `s3.enabled: false`*

Run bundles sync up when a run ends. Inputs can come *down*: teacher, student,
adapter and dataset may each be an `s3://` URI, fetched in preflight and cached.

```bash
uv sync --extra remote
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

uv run kd pipeline --config configs/finance.yaml \
  --set s3.enabled=true \
  --set s3.bucket=my-bucket
```

Set `s3.endpoint_url` for MinIO, R2 or RunPod volumes.

### Rented GPUs — *off: `runpod.enabled: false`*

Rents the GPU you named, runs the pipeline there, enforces your spend cap from
your own machine, and always terminates the pod.

```bash
uv sync --extra remote
export RUNPOD_API_KEY=...

# rents nothing - just checks auth and reads the catalogue
uv run kd runpod gpus --config configs/finance.yaml \
  --set runpod.enabled=true
```

Needs a published image — see [section 6](#6-deploying-to-other-machines) and
[RUNPOD.md](RUNPOD.md).

### Checking what is on

`uv run kd doctor` tells you which of these are installed and which credentials are
present — without printing any of the values:

```
 optional features
   evaluation extra (--tasks, --gen-similarity)  available
   S3 (boto3)                                    available
   RunPod                                        available

 credentials (presence only, never values)
   HF_TOKEN                 set   Hugging Face publish/private models
   AWS_ACCESS_KEY_ID        set   S3
   RUNPOD_API_KEY           -     RunPod
```

### Making the switches permanent

Rather than repeating `--set` flags, write a profile that extends your training
config:

```yaml
# configs/finance-pod.yaml
extends: finance.yaml

runpod:
  enabled: true
  gpu_type: RTX A4000
  max_price_per_hour: 0.60
  image: ghcr.io/<org>/kd:abc1234
s3:
  enabled: true
  bucket: my-bucket
limits:
  max_cost_usd: 2.00
  max_runtime_minutes: 90
```

```bash
uv run kd runpod launch --config configs/finance-pod.yaml
```

> **Secrets never go in a config file.** The YAML names the *variable* —
> `api_key_env: RUNPOD_API_KEY`, `token_env: HF_TOKEN` — and the environment
> holds the value. Config files get committed; environments do not.

---

## 6. Deploying to other machines

Everything above assumes a checkout. To put the pipeline on a machine that has
none - a colleague's laptop, a build agent, a GPU box - you do not clone it there
and repeat this setup. You hand over one file.

CI's **Build distillation runner** workflow produces `distill.sh` and
`distill.ps1` on every push, each pinned to the commit it was built from. They
install uv if it is missing, fetch that exact source, and run. Download either
from the workflow's artifacts or from a release.

**[RUNBOOK-RUNNER.md](RUNBOOK-RUNNER.md) is the same guide as this one, written
for that path.** Send it to whoever is running the job.

### The container image, for a GPU

Renting a GPU needs a published image. Run the **Build CUDA image** workflow
manually (*Actions -> Build CUDA image -> Run workflow*); it publishes
`ghcr.io/<org>/kd:<sha>` with CUDA 12.8, torch and the source already baked in.

It is deliberately not on every push: it is a multi-gigabyte build that only
matters when someone is about to rent a GPU. Baking the dependencies in saves
four to six minutes of *paid* GPU time on every run, and pins the exact
environment a result came from.

```bash
uv run kd runpod launch --config configs/finance.yaml   --set runpod.enabled=true   --set runpod.image=ghcr.io/<org>/kd:abc1234   --set s3.enabled=true --set s3.bucket=my-bucket   --set limits.max_cost_usd=2.00
```

### What happens when a GPU is unavailable

The launcher rents the card you named and **no other** - silently taking the next
one up is how a $0.34/hr run becomes a $2.80/hr run. If it cannot be had, it
stops before renting anything and asks:

```
==> requested  RTX A4000 (spot, cap $0.60/hr)
!!  RTX A4000: no spot capacity right now
    available now, under your $0.60/hr cap:
      1) RTX A5000            24GB  $0.28/hr  spot
      2) RTX 4090             24GB  $0.44/hr  spot
    pick 1-2, or 'q' to abort (nothing has been rented):
```

In a non-interactive session it refuses instead and tells you how to name one
explicitly. `--yes` accepts the *cost estimate*; it never accepts a different GPU.

### CI on your own changes

The runner workflow byte-compiles and imports every module, runs all seven test
files, resolves every profile, checks override precedence, then renders both
runners, shellchecks the bash, parses the PowerShell and proves the two dispatch
identically. Locally:

```bash
for t in tests/*.py; do uv run python "$t"; done
```

No framework, no model downloads, about a second each.

---

## 7. Keeping the cost down

Limits are checked twice: once before the run starts, and again while it is
going.

```yaml
limits:
  max_runtime_minutes: 180
  max_cost_usd: 2.00        # only meaningful on a rented GPU
  confirm_above_usd: 1.00
```

The smoke stage measures seconds-per-step, which turns your step count into a
projected duration and cost. A run that cannot finish inside its limits is
**refused before it starts**, with the fix spelled out:

```
This run would cost about $0.12, over the $0.05 limits.max_cost_usd.
  measured 4.30 s/step over 300 steps at $0.34/hr
  raise the ceiling:  --set limits.max_cost_usd=0.15
  or shorten the run: --set training.max_steps=110
```

> **A breach in flight is a hard stop.** No evaluation, no report. What survives
> is the last checkpoint written by `training.save_steps` — so tighten that when
> limits are tight. On a rented machine the bundle *and* its checkpoints are
> synced to S3 before the pod is terminated, so a stopped run still leaves you
> something.

### The levers that actually matter

| Setting | Effect |
|---|---|
| `gkd.lmbda` | The dominant cost. It is the fraction of batches where the student generates its own completion before the teacher scores it — `max_new_tokens` sequential forward passes versus one. `0.5` to `0.25` removes half of them. `0.0` is plain off-policy KD: several times faster, but it loses the on-policy correction that makes GKD better. |
| `gkd.max_new_tokens` | Generation cost is linear in it. |
| `runpod.spot` | Roughly half price, interruptible. |
| `runpod.volume_gb` | The model cache lives on the volume, so weights download once instead of once per run. |
| `--only train` | Skip stages you have already passed on this config. |

Dataset size does *not* drive wall clock. `max_steps × batch_size ×
gradient_accumulation_steps` sets how many samples are consumed; a larger pool
only changes how often they repeat.

---

## 8. When something goes wrong

### The teacher fails its check

Most often the checkpoint's tensor names do not match the architecture
transformers built for it — common in merged exports from MLX or unsloth.
`from_pretrained` does not raise; it randomly initialises what it could not map.

```bash
uv run kd fix-teacher --config configs/finance.yaml   # rename, no retraining
uv run kd check-teacher --teacher ./teacher-fixed
```

If the teacher was trained as a LoRA adapter, point at the base model and let it
be merged instead of using a merged export:

```bash
uv run kd check-teacher --teacher Qwen/Qwen3.5-2B --teacher-adapter ./lora
```

### Exit 3 — no improvement over the base student

Not a malfunction: the numbers are real and the report is still written. Check
whether the teacher passed its check, whether the step budget was large enough,
whether `lora.target_modules` covers this architecture, and whether `gkd.lmbda`
is above zero.

### A pod is still running

```bash
uv run kd runpod stop <pod-id>
```

If termination itself failed you will have seen a loud message with the console
link — that is the one failure that keeps costing money.

### Out of memory

Lower `hardware.dtype` to `bfloat16` (halves the weights, needs CUDA or macOS
14+), then `dataset.max_total_tokens`, then `lora.r`. On Apple Silicon raise
`training.batch_size` rather than accumulation — unified memory makes real
batching cheap.

### Nothing looks wrong but the numbers are flat

Read `events.jsonl`. Every step, loss, checkpoint and budget tick is there as a
structured record, and it survives whatever happened to the terminal.

---

## Where to read more

- [CONFIG.md](CONFIG.md) — every configuration key, with defaults and why each matters
- [RUNPOD.md](RUNPOD.md) — renting a GPU in full, and keeping it cheap
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — what GKD is actually doing, and the mechanics
- [`configs/_base.yaml`](../configs/_base.yaml) — the single source of truth for every default
- [README.md](../README.md) — the short version
