# Runbook — with `distill.sh`

Everything from a downloaded script to a trained adapter — and how to switch on
evaluation extras, the Hugging Face Hub, S3, and rented GPUs when you need them.

**This document assumes nothing but a machine.** No clone, no Python setup, no
uv. `distill.sh` installs uv if it is missing, fetches the source pinned to the
commit it was built from, and runs. One file, about 7 KB.

> Working *on* the pipeline from a clone? Use
> [RUNBOOK-SOURCE.md](RUNBOOK-SOURCE.md) instead — the same guide, driven through
> `uv run` in a checkout.

Windows users: everywhere this document writes `./distill.sh`, write
`.\distill.ps1`, and write `-Config` in place of `--config`. Every other flag is
spelled identically. There is a full Windows example in
[section 6](#6-running-it-elsewhere).

- [1. Get it and run it](#1-get-it-and-run-it)
- [2. The eight stages](#2-the-eight-stages)
- [3. Running a real job](#3-running-a-real-job)
- [4. What you get back](#4-what-you-get-back)
- [5. Turning on the optional parts](#5-turning-on-the-optional-parts)
- [6. Running it elsewhere](#6-running-it-elsewhere)
- [7. Keeping the cost down](#7-keeping-the-cost-down)
- [8. When something goes wrong](#8-when-something-goes-wrong)

---

## 1. Get it and run it

Download `distill.sh` from the **Build distillation runner** workflow's artifacts
or from a GitHub release, then:

```bash
chmod +x distill.sh

# What can this machine actually do?
./distill.sh doctor

# Resolve the config and hardware without running anything
./distill.sh check --config configs/smoke.yaml

# The whole pipeline, end to end (~2 minutes plus first-run downloads)
./distill.sh --config configs/smoke.yaml
```

The first invocation installs uv, clones the pinned source into
`~/.cache/kd-runner`, and downloads torch. That takes a few minutes once;
everything after it is fast.

`configs/smoke.yaml` distils SmolLM2-360M into SmolLM2-135M for two steps. It is
deliberately too short to learn anything — its job is to prove every stage runs
on this machine.

A successful run looks like this:

```
==> Platform: Linux/x86_64
==> Cloning https://github.com/<org>/<repo>.git @ 053cee7f9a2b
==> Syncing dependencies (the first run downloads torch - this takes a while)

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

### How the runner reads your arguments

With no leading subcommand — or when the first argument starts with `-` — it runs
the full gated pipeline. Otherwise the first argument is the subcommand:

```bash
./distill.sh --config configs/finance.yaml      # -> the full pipeline
./distill.sh evaluate --config configs/x.yaml   # -> just evaluation
./distill.sh doctor                             # -> just the environment report
```

Everything it does not recognise is passed straight through, so every flag in
this document reaches the pipeline unchanged.

| Subcommand | What it does |
|---|---|
| *(omitted)* | Every stage, gated. The one you will use most. |
| `check` | Resolve config and hardware, print, run nothing. |
| `doctor` | Device, package versions, which optional features and credentials are present. |
| `train` | Just training, no gates. |
| `evaluate` | Score an adapter against the teacher. |
| `check-teacher` | Is this teacher fit to distil from? Loads only the teacher. |
| `fix-teacher` | Repair a checkpoint whose tensor names do not match its architecture. |
| `convert-adapter` | MLX / unsloth LoRA into PEFT format. |
| `publish` | Push adapter and merged model to the Hugging Face Hub. |
| `runpod` | Rent a GPU and run there. |
| `ui` | Browser control panel; `--compare` for the three-way generation view. |

### The runner's own flags

These few are consumed by `distill.sh` itself rather than passed on:

| Flag | |
|---|---|
| `--runner-help` | The bootstrapper's own help. `--help` reaches the pipeline instead. |
| `--runner-version` | The commit this runner was built from. |
| `--ref BRANCH\|TAG\|SHA` | Run a different revision than the one this runner is pinned to. Manual only — see [Trying another branch](#trying-another-branch). |
| `--extra NAME` | Install an optional dependency group — `eval` or `remote`. **Repeatable, and name every group you want each time**: `uv` removes extras it was not asked for, so `--extra remote` alone would uninstall `eval`. See [section 5](#5-turning-on-the-optional-parts). |
| `--workdir DIR` | Where to keep the fetched source. Default `~/.cache/kd-runner`. |
| `--local` | Use the checkout the script sits in instead of fetching. For working on the pipeline itself. |

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
./distill.sh --config configs/finance.yaml --only train
./distill.sh --config configs/finance.yaml --from evaluate
./distill.sh --config configs/finance.yaml --skip smoke
```

`--from` and `--only` open a new run directory but pick up the newest previous
run's adapter or evaluation, so you can re-report without retraining.

### Exit codes

The runner exits with whatever the pipeline returned, so a shell script can gate
on it:

| | |
|---|---|
| `0` | Everything that ran, worked. |
| `1` | A gate failed. |
| `2` | The teacher is not fit to distil from. |
| `3` | The run completed, but the adapter did not improve on the base student. |
| `4` | A limit stopped the run - or refused to start it. Not a malfunction; the message says what to change. |

---

## 3. Running a real job

The configs live in the source the runner fetched, so you can name any shipped
profile without having anything locally:

| Profile | Pair | For |
|---|---|---|
| `configs/default.yaml` | SmolLM2 360M → 135M | Runs anywhere. |
| `configs/mac.yaml` | Same pair | Apple Silicon (MPS), with real batching instead of accumulation. |
| `configs/smoke.yaml` | Same pair, 2 steps | First-run validation. |
| `configs/finance.yaml` | Qwen3.5-2B → 0.8B | A finance-tuned teacher on finance-alpaca. |
| `configs/qwen-poc.yaml` | Qwen3.5-2B → 0.8B | Stock models. A known-good pairing for proving the pipeline. |

```bash
./distill.sh --config configs/finance.yaml
```

### Overriding without editing anything

This matters more here than in a checkout: you have no working copy to edit, so
`--set` is how you change a run. It reaches *any* key by dotted path and is
repeatable:

```bash
./distill.sh --config configs/finance.yaml \
  --set models.teacher=Qwen/Qwen3.5-2B \
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

A mistyped key is a hard error with a suggestion, not a silent no-op:

```
Configuration is not valid:
  - unknown key 'gkd.lmbdaa'  did you mean 'gkd.lmbda'?
```

Every key is documented in [CONFIG.md](CONFIG.md).

### Using a config file of your own

You do not need to download anything to see what a profile contains. `--full`
writes the effective configuration to stdout and the banner to stderr, so it
redirects cleanly into a file you can edit:

```bash
./distill.sh check --config configs/finance.yaml --full > my-run.yaml
```

That gives you all 133 lines with every value already resolved - a complete,
runnable starting point rather than a blank page. Edit what you want and run it.

Or write one from scratch; `extends` pulls in everything you did not mention:

```bash
cat > my-run.yaml <<'YAML'
extends: _base.yaml

project:
  name: my-distillation
models:
  teacher: Qwen/Qwen3.5-2B
  student: Qwen/Qwen3.5-0.8B
  tokenizer: student
training:
  max_steps: 600
YAML

./distill.sh --config "$PWD/my-run.yaml"
```

Two things to know. `extends: _base.yaml` finds the one that shipped with the
runner's source even though your file lives elsewhere, so you do not need a path
into the cache directory. But **pass `--config` as an absolute path**: the runner
changes directory into the fetched source before running, so a relative path
would be looked for in the checkout and not found.

> **Two things that will save you a run.** Teacher and student **must share a
> tokenizer vocabulary** — that is what standard GKD requires. And **check
> `lora.target_modules` against your architecture**: Qwen3.5 is hybrid, so a
> Llama-style target list silently misses the attention in 18 of its 24 layers.

---

## 4. What you get back

The runner does its work inside the fetched source, so run bundles land under
`~/.cache/kd-runner/src/runs/` by default. Point them somewhere you will find
them:

```bash
./distill.sh --config configs/finance.yaml --set project.runs_dir="$PWD/runs"
```

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

### Using the adapter

The adapter is a standard PEFT directory. Load it anywhere:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
model = PeftModel.from_pretrained(base, "./runs/<run-id>/final_adapter")
tok = AutoTokenizer.from_pretrained("./runs/<run-id>/final_adapter")
```

### Looking at it in a browser

```bash
./distill.sh ui                        # control panel: train, evaluate, compare
./distill.sh ui --compare              # base vs distilled vs teacher, side by side
./distill.sh ui --host 0.0.0.0 --port 8080   # serve it on the LAN
```

Every screen prints the exact command it is about to run, so the panel teaches
the CLI rather than hiding it.

---

## 5. Turning on the optional parts

Four features ship switched off. Two of them need dependencies the core install
does not carry, and `--extra` is how the runner installs them.

### Benchmark evaluation — *needs `--extra eval`*

Score base, distilled and teacher on lm-evaluation-harness tasks, and compare
free-running generations with BERTScore.

```bash
./distill.sh --extra eval evaluate --config configs/finance.yaml \
  --tasks ifeval --limit 100 \
  --gen-similarity 20
```

Slow — it scores three models. Start with `--limit`. The extra installs once and
stays in the runner's cached environment — but name every group you want on each
invocation, because `uv` removes extras it was not asked for:

```bash
./distill.sh --extra eval --extra remote --config configs/finance.yaml
```

### Hugging Face Hub — *`publish.enabled: false`*

Publishes the adapter to `<repo>-lora` and a merged, ready-to-run model to
`<repo>`. The merge is verified to generate language before anything uploads.

```bash
export HF_TOKEN=...

./distill.sh --config configs/finance.yaml \
  --set publish.enabled=true \
  --set publish.repo=my-org/qwen-finance
```

### S3 storage — *needs `--extra remote`, `s3.enabled: false`*

Run bundles sync up when a run ends. Inputs can come *down*: teacher, student,
adapter and dataset may each be an `s3://` URI, fetched in preflight and cached.
On a throwaway machine this is how you get results off it.

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

./distill.sh --extra remote --config configs/finance.yaml \
  --set s3.enabled=true \
  --set s3.bucket=my-bucket
```

Set `s3.endpoint_url` for MinIO, R2 or RunPod volumes.

### Rented GPUs — *needs `--extra remote`, `runpod.enabled: false`*

Rents the GPU you named, runs the pipeline there, enforces your spend cap from
the machine you launched from, and always terminates the pod.

```bash
export RUNPOD_API_KEY=...

# rents nothing - just checks auth and reads the catalogue
./distill.sh --extra remote runpod gpus \
  --config configs/finance.yaml \
  --set runpod.enabled=true
```

### Checking what is on

```bash
./distill.sh doctor
```

tells you which of these are installed and which credentials are present —
without printing any of the values:

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

Repeating a wall of `--set` flags gets old. Two ways to avoid it.

**A config file of your own**, as in [section 3](#3-running-a-real-job):

```yaml
# finance-pod.yaml
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
./distill.sh --extra remote runpod launch --config "$PWD/finance-pod.yaml"
```

**Or environment variables**, which the runner passes through:

```bash
export KD_EXTRAS="remote eval"          # installed on every invocation
export KD_MAX_STEPS=500
export KD_DEVICE=cuda
export KD_WORKDIR=/mnt/scratch/kd       # keep the checkout off the boot disk

./distill.sh --config configs/finance.yaml
```

`KD_*` variables sit below `--set` and explicit flags, so a flag on the command
line still wins.

> **Secrets never go in a config file.** The YAML names the *variable* —
> `api_key_env: RUNPOD_API_KEY`, `token_env: HF_TOKEN` — and the environment
> holds the value. Config files get committed; environments do not.

---

## 6. Running it elsewhere

### Windows

`distill.ps1` is the same bootstrapper in PowerShell, built by the same CI job
from the same stage definition, so the two cannot drift.

```powershell
.\distill.ps1 -Config configs\smoke.yaml
.\distill.ps1 doctor
.\distill.ps1 -Extra eval evaluate -Config configs\finance.yaml --tasks ifeval
.\distill.ps1 -Config configs\finance.yaml --set training.max_steps=500
```

`-Config`, `-Extra`, `-WorkDir`, `-Local`, `-RunnerHelp` and `-RunnerVersion` use
PowerShell's parameter style; everything else — `--set`, `--only`, `--tasks` — is
spelled exactly as it is above and passed through untouched.

### Inside a container or a CI job

Nothing is interactive, so it drops straight into a step:

```bash
curl -fsSL -o distill.sh "$RUNNER_URL"
chmod +x distill.sh
./distill.sh --config configs/finance.yaml --set training.max_steps=200
```

Pin `--workdir` to a cached path if the job has one; the fetched source and the
model cache both live under it, so a warm cache turns a five-minute start into
seconds.

### On a rented GPU

`runpod launch` does the whole lifecycle from wherever you run it. It needs a
published image — CI's **Build CUDA image** workflow (run manually from *Actions*)
publishes `ghcr.io/<org>/kd:<sha>` with CUDA 12.8, torch and the source baked in.

```bash
./distill.sh --extra remote runpod launch --config configs/finance.yaml \
  --set runpod.enabled=true \
  --set runpod.image=ghcr.io/<org>/kd:abc1234 \
  --set s3.enabled=true --set s3.bucket=my-bucket \
  --set limits.max_cost_usd=2.00
```

The launcher rents the card you named and **no other** — silently taking the next
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

### Which commit am I running?

```bash
./distill.sh --runner-version
```

Every runner is pinned. To move to a newer build, download a newer runner — the
one you have will keep fetching its own commit forever, which is what makes a
result reproducible months later.

### Trying another branch

Sometimes you want to run a feature branch without waiting for CI to build a
runner for it. `--ref` points the same runner at a different revision:

```bash
./distill.sh --ref feature/new-scheduler --config configs/finance.yaml
./distill.sh --ref v1.2.0 --config configs/finance.yaml       # a tag
./distill.sh --ref 9f3a1c2 --config configs/finance.yaml      # any commit
```

It is manual and it announces itself, every time:

```
!!  running ref 'feature/new-scheduler', NOT the pinned <org>/<repo>@053cee7
!!  this run is only reproducible if that ref is a commit or an immutable tag
```

That warning is the point. A pinned runner guarantees a result can be traced back
to an exact revision; a branch moves, so a run made against one cannot be
reproduced from the runner alone. `KD_REF` does the same thing from the
environment, for CI jobs that build a matrix of branches.

The checkout is shared, so switching back and forth is cheap — the runner fetches
the ref and checks out `FETCH_HEAD` rather than a local branch name, which means
a branch that has moved since you last used it is picked up rather than silently
re-run from stale code.

> **Manual builds, not automatic ones.** CI builds runners automatically only on
> `main`/`master`. To get a *pinned* runner for another branch, run the **Build
> distillation runner** workflow manually from *Actions* and select that branch —
> `workflow_dispatch` works from any branch. Use `--ref` for a quick look; use a
> manually built runner when you want the result to stay traceable.

---

## 7. Keeping the cost down

Limits are checked twice: once before the run starts, and again while it is
going.

```bash
./distill.sh --config configs/finance.yaml \
  --set limits.max_runtime_minutes=180 \
  --set limits.max_cost_usd=2.00
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
| `--workdir` on a cached disk | Avoids re-downloading torch and the models on every job. |

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
./distill.sh fix-teacher --config configs/finance.yaml   # rename, no retraining
./distill.sh check-teacher --teacher ./teacher-fixed
```

If the teacher was trained as a LoRA adapter, point at the base model and let it
be merged instead of using a merged export:

```bash
./distill.sh check-teacher --teacher Qwen/Qwen3.5-2B --teacher-adapter ./lora
```

### Exit 3 — no improvement over the base student

Not a malfunction: the numbers are real and the report is still written. Check
whether the teacher passed its check, whether the step budget was large enough,
whether `lora.target_modules` covers this architecture, and whether `gkd.lmbda`
is above zero.

### A missing dependency

```
boto3 is needed for s3 support and is not installed.
  uv sync --extra remote
```

From the runner, that is `--extra remote` — it does the same `uv sync` inside the
fetched checkout:

```bash
./distill.sh --extra remote --config configs/finance.yaml --set s3.enabled=true
```

### A pod is still running

```bash
./distill.sh --extra remote runpod stop <pod-id>
```

If termination itself failed you will have seen a loud message with the console
link — that is the one failure that keeps costing money.

### Out of memory

Lower `hardware.dtype` to `bfloat16` (halves the weights, needs CUDA or macOS
14+), then `dataset.max_total_tokens`, then `lora.r`. On Apple Silicon raise
`training.batch_size` rather than accumulation — unified memory makes real
batching cheap.

```bash
./distill.sh --config configs/finance.yaml --set hardware.dtype=bfloat16
```

### The runner itself misbehaves

```bash
./distill.sh --runner-help       # the bootstrapper's own flags
./distill.sh --runner-version    # which commit it fetches
rm -rf ~/.cache/kd-runner        # start the checkout and environment over
```

Nothing you care about lives in the cache unless you left `project.runs_dir` at
its default — see [section 4](#4-what-you-get-back).

### Nothing looks wrong but the numbers are flat

Read `events.jsonl` in the run bundle. Every step, loss, checkpoint and budget
tick is there as a structured record, and it survives whatever happened to the
terminal.

---

## Where to read more

- [RUNBOOK-SOURCE.md](RUNBOOK-SOURCE.md) — the same guide, from a clone with `uv run`
- [CONFIG.md](CONFIG.md) — every configuration key, with defaults and why each matters
- [RUNPOD.md](RUNPOD.md) — renting a GPU in full, and keeping it cheap
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — what GKD is actually doing, and the mechanics
- [README.md](../README.md) — the short version
