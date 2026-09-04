# knowledge-distillation

On-policy **knowledge distillation (GKD)** of a large "teacher" LLM into a small
"student" LLM, driven entirely by YAML config and shipped as a single
self-contained shell runner: **`distill.sh`**.

One command trains a LoRA adapter that makes a small model behave more like a big
one on your domain, on whatever hardware you have — Apple Silicon (MPS), CUDA, or
plain CPU.

```bash
./distill.sh --check          # what would run, and on what hardware
./distill.sh --check-teacher  # is the teacher fit to distil from?
./distill.sh --smoke          # 2 steps end-to-end (~1 min)
./distill.sh                  # the real run
./distill.sh --evaluate       # how much of the teacher actually transferred?
./distill.sh --ui             # compare base vs distilled vs teacher in a browser
```

If you want to understand *what* is happening rather than *how to run it*, read
**[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** — the high-level story and the technical
mechanics of the training loop.

---

## Table of contents

1. [What `distill.sh` is](#1-what-distillsh-is)
2. [Downloading `distill.sh`](#2-downloading-distillsh)
3. [Requirements](#3-requirements)
4. [Your first run](#4-your-first-run-recommended-order)
5. [What the runner does, step by step](#5-what-the-runner-does-step-by-step)
6. [Full command reference](#6-full-command-reference)
7. [Config profiles](#7-config-profiles)
8. [Settings precedence and `KD_*` environment variables](#8-settings-precedence-and-kd_-environment-variables)
9. [Where files land](#9-where-files-land)
10. [Recipes](#10-recipes)
11. [Running without `distill.sh`](#11-running-without-distillsh)
12. [Repository map](#12-repository-map)
13. [Troubleshooting](#13-troubleshooting)
14. [Suggestions and known rough edges](#14-suggestions-and-known-rough-edges)
15. [Measuring the result](#15-measuring-the-result)

---

## 1. What `distill.sh` is

`distill.sh` is a **generated artifact**, not a file you will find committed in this
repository. It is rendered by GitHub Actions
(`.github/workflows/build-runner.yml`) from the tracked template
`scripts/distill.sh.template`, with build metadata substituted in:

| Placeholder | Becomes |
|---|---|
| `__REPO_URL__` | `https://github.com/<owner>/<repo>.git` |
| `__REPO__` | `<owner>/<repo>` |
| `__SHA__` | the full commit the runner is pinned to |
| `__SHA_SHORT__` | short commit, shown by `--version` |
| `__BUILT_AT__` | UTC build timestamp |

Because the commit is baked in, a copy of `distill.sh` always reproduces the exact
source it was built from — it clones that commit, never `HEAD`. `.gitignore`
deliberately ignores `/distill.sh` so the rendered file never lands in git history.

The runner is a single bash script that:

* bootstraps [`uv`](https://docs.astral.sh/uv/) if it is missing,
* clones the pinned source into a cache directory,
* installs the locked Python environment (`uv sync`),
* detects Apple Silicon / CUDA / CPU and sets the right allocator flags,
* translates its own CLI flags into `KD_*` environment variables,
* and runs one of ten modes (train, smoke, check, check-teacher, fix-teacher,
  convert-adapter, evaluate, publish, ui, shell).

It never installs anything into your system Python and never writes into the
directory you run it from.

---

## 2. Downloading `distill.sh`

Three ways to get it. Pick by how you consume the project.

### Option A — from a GitHub Release (easiest, if a release exists)

The workflow attaches `distill.sh` to every published release.

```bash
# latest release
curl -fsSL -o distill.sh \
  https://github.com/siddharthaaddy-globalids/knowledge-distillation/releases/latest/download/distill.sh
chmod +x distill.sh
./distill.sh --version
```

A specific tag:

```bash
curl -fsSL -o distill.sh \
  https://github.com/siddharthaaddy-globalids/knowledge-distillation/releases/download/v0.1.0/distill.sh
chmod +x distill.sh
```

> If this 404s, no release has been published yet — use Option B or C. See
> [suggestions](#14-suggestions-and-known-rough-edges): cutting a release is the
> single easiest improvement to distribution.

### Option B — from a GitHub Actions artifact (always available)

Every push to `main`/`master`, every PR, and every manual run uploads an artifact
named `distill-runner-<sha_short>` containing `distill.sh` **and** a copy of
`configs/`. Artifacts are retained for **90 days** and require you to be signed in
to GitHub.

**Web UI:** repo → **Actions** → *Build distillation runner* → newest green run →
**Artifacts** → download `distill-runner-<sha>` → unzip.

**With the GitHub CLI:**

```bash
# newest successful run of the workflow
gh run download -R siddharthaaddy-globalids/knowledge-distillation \
  "$(gh run list -R siddharthaaddy-globalids/knowledge-distillation \
       -w 'Build distillation runner' -s success -L 1 --json databaseId -q '.[0].databaseId')" \
  -D ./runner

chmod +x ./runner/distill.sh
./runner/distill.sh --version
```

Or, if you already know the run id:

```bash
gh run download <run-id> -R siddharthaaddy-globalids/knowledge-distillation -D ./runner
chmod +x ./runner/distill.sh
```

> The `configs/` folder inside the artifact is a convenience copy for reading. The
> runner does **not** read it: at runtime it `cd`s into its own pinned checkout and
> uses the `configs/` from there. Editing the downloaded copy has no effect — see
> [§9](#9-where-files-land).

### Option C — render it yourself from the template

Useful for a private fork, an air-gapped mirror, or when you want the runner
pinned to your own local commit.

```bash
git clone https://github.com/siddharthaaddy-globalids/knowledge-distillation.git
cd knowledge-distillation

REPO_URL=$(git remote get-url origin)
REPO=$(printf '%s' "$REPO_URL" | sed -E 's#.*github\.com[:/](.+)$#\1#; s#\.git$##')
SHA=$(git rev-parse HEAD)

sed -e "s|__REPO_URL__|$REPO_URL|g" \
    -e "s|__REPO__|$REPO|g" \
    -e "s|__SHA__|$SHA|g" \
    -e "s|__SHA_SHORT__|$(git rev-parse --short HEAD)|g" \
    -e "s|__BUILT_AT__|$(date -u +%Y-%m-%dT%H:%M:%SZ)|g" \
    scripts/distill.sh.template > distill.sh

chmod +x distill.sh
bash -n distill.sh          # syntax check
./distill.sh --version
```

The runner clones `__REPO_URL__` at `__SHA__`, so **the commit must be pushed**
before the rendered script will work on another machine.

### Sanity-check the download

```bash
chmod +x distill.sh
./distill.sh --version   # prints  distill.sh pinned to <repo>@<sha> (built <ts>)
./distill.sh --help      # full option list
./distill.sh --check     # resolves config + hardware; installs the env on first use
```

`--version` and `--help` do no network work and touch nothing. `--check` is the
first command that bootstraps `uv`, clones, and runs `uv sync` (a multi-GB torch
download on first use).

---

## 3. Requirements

| Requirement | Notes |
|---|---|
| **bash** | the runner is bash, not POSIX sh (uses `PIPESTATUS`, `${VAR:0:12}`) |
| **git** | used to clone the pinned source |
| **curl** | only to bootstrap `uv` when it is absent |
| **~15 GB free disk** | torch wheels + HF model cache + checkpoints |
| **Python** | *not* required up front — `uv` installs the pinned 3.13 toolchain |
| **RAM** | 8 GB is fine for the SmolLM2 profiles; a 2B teacher + 0.8B student is ~11 GB in float32, ~6 GB in bfloat16 |

**Platform support**

* **macOS (Apple Silicon)** — first-class. MPS is auto-detected, allocator limits
  lifted, `configs/mac.yaml` auto-selected. `bfloat16` requires macOS ≥ 14.0.
* **Linux** — CPU by default; CUDA auto-detected when present. `pyproject.toml`
  pins CPU-only torch wheels on Linux/Windows to avoid a multi-GB CUDA runtime
  download, so **for CUDA training you must install a CUDA torch build yourself**.
* **Windows** — `distill.sh` needs a POSIX shell. Use **Git Bash** or **WSL2**.
  Alternatively skip the runner and call the Python entrypoints directly (see
  [§11](#11-running-without-distillsh)) — those are fully cross-platform and are
  how this repo is developed on Windows.

**Hugging Face access** — public models need nothing. Gated or private repos need
`hf auth login` or `HF_TOKEN` exported before you run; the runner inherits your
environment, so both work.

---

## 4. Your first run (recommended order)

Each step is cheap and rules out a class of failure before the next one costs you
time.

```bash
# 1. Does the config resolve, and what hardware did it find?   (seconds after setup)
./distill.sh --check

# 2. Is the teacher actually a usable distillation target?     (a few minutes)
./distill.sh --check-teacher

# 3. Does the whole pipeline run end to end?                   (~1 minute)
./distill.sh --smoke

# 4. Train.                                                    (minutes to hours)
./distill.sh

# 5. Measure what you got.                                    (a few minutes)
./distill.sh --evaluate

# 6. Look at what you got.
./distill.sh --ui
```

**Step 2 is not optional in spirit.** GKD trains the student to match the
teacher's output distribution, so a broken teacher produces a broken student while
the run reports success. `transformers` does *not* raise when a checkpoint's tensor
keys don't match the architecture — it randomly initialises what it couldn't map
and prints a warning that scrolls past. `--check-teacher` catches exactly that:
randomly-initialised weights, NaN/Inf parameters, and near-uniform output
distributions. It exits `2` on failure, so it can gate a script.

If step 2 fails, the runner prints the repair to try:

```bash
./distill.sh --fix-teacher                                    # tensor keys mislabelled
./distill.sh --convert-adapter --teacher-adapter <mlx-repo>   # MLX LoRA -> PEFT
```

---

## 5. What the runner does, step by step

1. **Platform detection.** `uname`; on Apple Silicon it also reads `sw_vers` and
   `hw.memsize`, warns if you asked for `bfloat16` on macOS < 14, and exports
   `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (use the whole unified pool) and
   `PYTORCH_ENABLE_MPS_FALLBACK=1` (unsupported ops run on CPU instead of raising).
2. **Profile selection.** No `--profile` → `mac` on Apple Silicon, otherwise
   `default`. `--smoke` switches to the `smoke` profile *only* if you didn't ask
   for a profile explicitly — otherwise the projected run time would describe a
   different model pair than the one you're about to train.
3. **`uv` bootstrap.** Installed to `~/.local/bin` via the official installer if
   missing.
4. **Pinned checkout.** Clones `__REPO_URL__` into `$KD_WORKDIR/src` (default
   `~/.cache/kd-runner/src`) and checks out the baked-in SHA. Later runs fetch and
   re-checkout rather than re-cloning.
5. **`uv sync`.** Creates the locked virtualenv. First run downloads torch — this
   is the slow part.
6. **Export overrides.** Only values you explicitly set become `KD_*` variables.
   Anything you left alone stays unset so the profile's YAML wins. (This is
   deliberate and regression-tested in CI: non-empty defaults here would silently
   clobber the profile.)
7. **Run the selected mode**, then print an `ALL OK` / `FAILED` banner with the
   next command to run.

---

## 6. Full command reference

```
./distill.sh [OPTIONS]
```

### Modes (mutually exclusive; default is `train`)

| Flag | What it does |
|---|---|
| *(none)* | Train. Runs `train_scaled.py` with the resolved profile. |
| `--check` | Resolve config + hardware, print the effective YAML, exit. No model downloads. |
| `--check-teacher` | Load **only** the teacher and verify it is fit to distil from: missing/randomly-initialised weights, NaN scan, entropy + generation probe. Exits `2` if unfit. |
| `--fix-teacher` | Repair a merged checkpoint whose tensor keys don't match its architecture — renames tensors to what the model asks for and drops components a text-only model never builds (e.g. a vision tower). No GPU, no retraining. Writes `$KD_WORKDIR/teacher-fixed`. |
| `--convert-adapter` | Convert an MLX-LM / unsloth-on-Apple-Silicon LoRA adapter into PEFT format. Requires `--teacher-adapter`. Writes `$KD_WORKDIR/peft-adapter`. |
| `--publish REPO` | Publish the trained adapter to `REPO-lora` and a `transformers`-merged model to `REPO` on the Hugging Face Hub. Verifies the merged model generates language *before* uploading anything. Needs `hf auth login` or `HF_TOKEN`. |
| `--evaluate` | Measure how much of the teacher transferred. Reports **fidelity** (top-1 agreement with the teacher, KL divergence) and **capability** (held-out perplexity) for the base student *and* the distilled student against the same teacher, plus efficiency. Optionally runs lm-evaluation-harness benchmarks with `--tasks`. Exits `3` if the adapter did not improve on the base. See [§15](#15-measuring-the-result). |
| `--smoke` | 2-step end-to-end validation (~1 min). Prints trainable params, held-out loss, s/step, and projects the full run time. |
| `--ui` | Launch the Gradio comparison UI (original student vs distilled student vs teacher) instead of training. |
| `--shell` | Drop into a shell with the environment activated. |
| `-h, --help` | Full help text. |
| `-V, --version` | Pinned build revision. |

### Models and data

| Flag | Meaning |
|---|---|
| `-t, --teacher MODEL` | Teacher model id or local path. |
| `-s, --student MODEL` | Student model id or local path. |
| `--teacher-adapter PATH` | LoRA adapter merged into the teacher at load time. Use when the teacher is *base + adapter* rather than a merged checkpoint — merged exports from other frameworks often keep that framework's key layout, which plain `transformers` may not map back. HF repo id or local dir. |
| `-d, --dataset NAME` | Training dataset (HF id or local). |

> **Teacher and student must share a tokenizer vocabulary.** Standard GKD compares
> per-token distributions, so mismatched vocabularies are meaningless. Every
> profile in `configs/` documents the vocab check done for its pairing.

### Hardware

| Flag | Values | Meaning |
|---|---|---|
| `--device DEV` | `auto` \| `cpu` \| `mps` \| `cuda` | `auto` prefers CUDA, then MPS, then CPU. Requesting an unavailable device falls back to CPU with a warning. |
| `--dtype TYPE` | `auto` \| `float32` \| `bfloat16` \| `float16` | `auto` = float32 on CPU/MPS, bfloat16 (or float16) on CUDA. `bfloat16` on MPS needs macOS ≥ 14.0; anything non-float32 on CPU is downgraded to float32. |

### Training

| Flag | Meaning |
|---|---|
| `--steps N` | Optimizer steps. |
| `--batch-size N` | Per-device batch size. |
| `--grad-accum N` | Gradient accumulation steps (effective batch = batch × accum). |
| `--lr RATE` | Learning rate, e.g. `3e-4`. |
| `--lora-r N` | LoRA rank. |
| `--lora-alpha N` | LoRA alpha. |
| `--lmbda F` | On-policy fraction, 0.0–1.0. The **dominant cost knob** — see [§8](#the-two-cost-knobs-that-matter). |
| `-o, --output DIR` | Output directory. |
| `-p, --profile NAME` | `default` \| `mac` \| `smoke` \| `finance` \| `qwen-poc`. |

### Evaluation options (with `--evaluate`)

| Flag | Default | Meaning |
|---|---|---|
| `--tasks LIST` | *(none)* | Also score base/distilled/teacher on lm-evaluation-harness benchmarks (`ifeval,hellaswag,arc_easy`) and print the retention table. Installs the `eval` extra on demand. Slow — it runs three models. |
| `--eval-samples N` | `50` | Held-out samples to score. |
| `--eval-limit N` | *(none)* | Per-task example cap for `--tasks`. |
| `--eval-json PATH` | *(none)* | Write every metric to JSON. |
| `--adapter DIR` | `<output>/final_adapter` | Which adapter to evaluate. |

### UI options (with `--ui`)

| Flag | Default | Meaning |
|---|---|---|
| `--port N` | `7860` | Port to serve on. |
| `--host ADDR` | `127.0.0.1` | Bind address; `0.0.0.0` exposes it on the LAN. |
| `--share` | off | Create a public Gradio share link. |
| `--adapter DIR` | newest found | Compare a specific adapter directory. |

All flags accept both `--flag value` and `--flag=value`. Unknown flags and stray
positional arguments are rejected rather than ignored.

---

## 7. Config profiles

Profiles live in `configs/*.yaml` and are the source of truth for anything you
don't override.

| Profile | Teacher → Student | Dataset | Steps | Notes |
|---|---|---|---|---|
| `default` | SmolLM2-360M-Instruct → SmolLM2-135M-Instruct | `HuggingFaceTB/smoltalk`, 5 domains + synthetic | 300 | Windows/Linux CPU baseline. Batch 1 × accum 4. |
| `mac` | same as `default` | same | 300 | `device: mps`, batch **4** × accum 1 — unified memory makes real batching cheaper than accumulation. |
| `smoke` | same as `default` | tiny slices of 2 smoltalk domains | 2 | LoRA r=8 on `q_proj`/`v_proj` only. Used by CI and `--smoke`. |
| `qwen-poc` | Qwen/Qwen3.5-2B → Qwen/Qwen3.5-0.8B | `gbharti/finance-alpaca` | 100 | Known-good pairing: identical tokenizer and canonical key layout, both verified. `lmbda 0.25`, `max_new_tokens 24`, LoRA r=16. Pair with `--dtype bfloat16` on 16 GB machines. |
| `finance` | `siddhartha-addy-globalids-labs/qwen3.5-2b-finance-alpaca` → Qwen/Qwen3.5-0.8B | `gbharti/finance-alpaca` | 300 | Real domain run. **That teacher checkpoint uses MLX/unsloth key naming** — run `--fix-teacher` first, or use base + `--teacher-adapter`. |

Two structural things the profiles encode that are easy to miss:

* **`target_modules` is architecture-specific.** Qwen3.5 is a hybrid: 18 of its 24
  layers use linear attention (`in_proj_qkv`, `in_proj_z`, `out_proj`) and only 6
  use standard attention (`q/k/v/o_proj`). A Llama-style target list would leave
  attention untrained in 18 layers, so the Qwen profiles list both sets.
* **`pool` vs `quota`.** `quota` is how many samples to keep; `pool` is how many
  rows to scan to find them. The prompt-length filter is strict — only ~2% of
  `smol-summarize` prompts fit under 128 tokens — so long-document domains need a
  large pool.

---

## 8. Settings precedence and `KD_*` environment variables

```
built-in DEFAULTS  <  --config YAML  <  KD_* environment  <  CLI flag
```

`distill.sh` exports a `KD_*` variable **only** for values you explicitly passed.
Anything you leave alone stays unset, so the profile's YAML value wins.

Every runner flag has a `KD_*` equivalent:

| Env var | Flag | Config path |
|---|---|---|
| `KD_TEACHER_MODEL` | `--teacher` | `models.teacher` |
| `KD_STUDENT_MODEL` | `--student` | `models.student` |
| `KD_TEACHER_ADAPTER` | `--teacher-adapter` | `models.teacher_adapter` |
| `KD_DATASET` | `--dataset` | `dataset.source` |
| `KD_DEVICE` | `--device` | `hardware.device` |
| `KD_DTYPE` | `--dtype` | `hardware.dtype` |
| `KD_MAX_STEPS` | `--steps` | `training.max_steps` |
| `KD_BATCH_SIZE` | `--batch-size` | `training.batch_size` |
| `KD_GRAD_ACCUM` | `--grad-accum` | `training.gradient_accumulation_steps` |
| `KD_LEARNING_RATE` | `--lr` | `training.learning_rate` |
| `KD_LORA_R` | `--lora-r` | `lora.r` |
| `KD_LORA_ALPHA` | `--lora-alpha` | `lora.alpha` |
| `KD_LMBDA` | `--lmbda` | `gkd.lmbda` |
| `KD_OUTPUT_DIR` | `--output` | `project.output_dir` |

**Env-only knobs** (honoured by `kd_config.py`, no runner flag):

| Env var | Config path | Meaning |
|---|---|---|
| `KD_TOKENIZER` | `models.tokenizer` | `teacher` \| `student` \| an explicit id |
| `KD_THREADS` | `hardware.threads` | CPU thread count (`0` = all cores) |
| `KD_BETA` | `gkd.beta` | Divergence interpolation: `0` = forward KL, `1` = reverse KL, `0.5` = symmetric JSD |
| `KD_MAX_NEW_TOKENS` | `gkd.max_new_tokens` | Length of on-policy rollouts — generation cost is linear in this |
| `KD_SEED` | `project.seed` | Seed for dataset shuffle and trainer |

**Runner-only variables:**

| Env var | Default | Meaning |
|---|---|---|
| `KD_WORKDIR` | `$HOME/.cache/kd-runner` | Checkout + artifact cache |
| `KD_UI_HOST` / `KD_UI_PORT` | `127.0.0.1` / `7860` | UI bind address/port |
| `KD_ADAPTER_PATH` | *(auto-discovered)* | Adapter the UI and `--evaluate` use |
| `KD_EVAL_TASKS` / `KD_EVAL_SAMPLES` | *(none)* / `50` | Preset the `--evaluate` options |

Example:

```bash
KD_LMBDA=0.25 KD_MAX_NEW_TOKENS=24 KD_BETA=0.0 ./distill.sh --steps 600
```

### The two cost knobs that matter

`lmbda` is the fraction of steps for which the student **generates its own
completion** before the teacher scores it. A supervised batch is one forward pass;
an on-policy batch is `max_new_tokens` *sequential* forward passes plus the loss
pass. So:

* `lmbda 0.5 → 0.25` removes half the generation passes.
* `lmbda 0.0` is pure off-policy distillation: several times faster, but it loses
  the on-policy correction that makes GKD better than plain KD.
* `max_new_tokens` scales generation cost linearly.

---

## 9. Where files land

| Thing | Path |
|---|---|
| Pinned source checkout | `$KD_WORKDIR/src` (default `~/.cache/kd-runner/src`) |
| Python environment | `$KD_WORKDIR/src/.venv` |
| Repaired teacher (`--fix-teacher`) | `$KD_WORKDIR/teacher-fixed` |
| Converted adapter (`--convert-adapter`) | `$KD_WORKDIR/peft-adapter` |
| **Trained adapter** | `$KD_WORKDIR/src/<output_dir>/final_adapter` |
| Intermediate checkpoints | `$KD_WORKDIR/src/<output_dir>/checkpoint-<step>/` |
| Hugging Face model cache | `~/.cache/huggingface` (override with `HF_HOME`) |

`<output_dir>` comes from the profile — `./distilled_smollm_scaled` for `default`,
`./distilled_qwen_poc` for `qwen-poc`, and so on. Because it is a *relative* path
and the runner `cd`s into its checkout, **your adapter is written inside the cache
directory, not into your current directory.** The success banner prints the
absolute path. To put results somewhere you control:

```bash
./distill.sh --output "$PWD/my-run"
```

Checkpoints carry full optimizer state (~100 MB+ each); `save_total_limit` keeps
only the newest 2 by default.

---

## 10. Recipes

**See what would run, without running it**

```bash
./distill.sh --check
```

**Bigger teacher — the single highest-leverage change**

```bash
./distill.sh --teacher HuggingFaceTB/SmolLM2-1.7B-Instruct
```

**Known-good proof of concept, sized for a 16 GB machine**

```bash
./distill.sh --profile qwen-poc --check-teacher
./distill.sh --profile qwen-poc --smoke
./distill.sh --profile qwen-poc --dtype bfloat16
```

**Retarget everything**

```bash
./distill.sh -t Qwen/Qwen2.5-1.5B-Instruct \
             -s Qwen/Qwen2.5-0.5B-Instruct \
             -d HuggingFaceTB/smoltalk --steps 600
```

**Teacher is a base model plus a LoRA adapter (preferred over a merged export)**

```bash
./distill.sh --teacher Qwen/Qwen3.5-2B --teacher-adapter ./lora_model --check-teacher
./distill.sh --teacher Qwen/Qwen3.5-2B --teacher-adapter ./lora_model
```

**Teacher was trained with unsloth on a Mac (MLX adapter)**

```bash
./distill.sh --teacher-adapter <org>/<mlx-lora> --convert-adapter
# -> writes ~/.cache/kd-runner/peft-adapter
./distill.sh --teacher Qwen/Qwen3.5-2B \
             --teacher-adapter ~/.cache/kd-runner/peft-adapter --check-teacher
```

**Merged teacher fails `--check-teacher` because its keys are mislabelled**

```bash
./distill.sh --profile finance --fix-teacher
./distill.sh --profile finance --teacher ~/.cache/kd-runner/teacher-fixed --check-teacher
./distill.sh --profile finance --teacher ~/.cache/kd-runner/teacher-fixed
```

**More on-policy correction, longer run, explicit output**

```bash
./distill.sh --lmbda 0.9 --steps 600 -o "$PWD/run-lmbda09"
```

**Cheap run: pure off-policy distillation**

```bash
./distill.sh --lmbda 0.0 --steps 600
```

**Force CPU on a Mac (slow, for comparison)**

```bash
./distill.sh --device cpu --steps 50
```

**Inspect the result, then share it on the LAN**

```bash
./distill.sh --ui
./distill.sh --ui --host 0.0.0.0 --port 8080
./distill.sh --ui --adapter "$PWD/my-run/final_adapter"
```

**Measure how much of the teacher transferred**

```bash
./distill.sh --profile finance --evaluate
./distill.sh --profile finance --evaluate --eval-samples 100 --eval-json ./run.json

# ...with the standard task benchmarks too (slow: three models scored)
./distill.sh --profile finance --evaluate --tasks ifeval --eval-limit 100
```

**Publish to the Hugging Face Hub**

```bash
hf auth login          # or: export HF_TOKEN=hf_...
./distill.sh --profile qwen-poc --publish my-org/qwen3.5-0.8b-finance
# -> my-org/qwen3.5-0.8b-finance-lora   (adapter)
# -> my-org/qwen3.5-0.8b-finance        (merged, verified before upload)
```

---

## 11. Running without `distill.sh`

The runner is a convenience wrapper. Everything it does is available directly, and
on Windows this is the path of least resistance.

```bash
git clone https://github.com/siddharthaaddy-globalids/knowledge-distillation.git
cd knowledge-distillation
uv sync
```

| Runner mode | Direct equivalent |
|---|---|
| `--check` | `uv run python train_scaled.py --config configs/mac.yaml --print-config` |
| `--smoke` | `uv run python train_scaled.py --config configs/smoke.yaml --dry-run` |
| *(train)* | `uv run python train_scaled.py --config configs/default.yaml` |
| `--check-teacher` | `uv run python check_teacher.py --config configs/finance.yaml` |
| `--fix-teacher` | `uv run python fix_teacher.py --config configs/finance.yaml --out ./teacher-fixed` |
| `--convert-adapter` | `uv run python convert_mlx_adapter.py --adapter <mlx> --out ./peft-adapter` |
| `--publish` | `uv run python publish_model.py --repo my-org/name --config configs/qwen-poc.yaml` |
| `--evaluate` | `uv run python evaluate.py --config configs/finance.yaml` |
| `--ui` | `uv run python app.py --config configs/default.yaml` |

`train_scaled.py` has flags the runner does not expose:

| Flag | Meaning |
|---|---|
| `--print-config` | Resolve everything, print the effective YAML, exit. |
| `--dry-run` | Build the dataset and run 2 steps; skips the final save. |
| `--allow-bad-teacher` | Train even if the teacher pre-flight fails. Not recommended. |
| `--no-eval` | Disable held-out validation loss (saves time). |
| `--eval-every N` | Override `training.benchmark_every`. |
| `--output-dir DIR` | Override `project.output_dir`. |

Useful extras elsewhere:

* `python app.py --list-adapters` — show every discoverable adapter.
* `python check_teacher.py --teacher some-org/model --dtype bfloat16` — check any
  checkpoint, no config needed.
* `python fix_teacher.py --dry-run` — print the key mapping, write nothing.
* `python publish_model.py --repo ... --dry-run` — build, verify and print the
  model cards without uploading.

---

## 12. Repository map

| File | Role |
|---|---|
| `scripts/distill.sh.template` | Source of the `distill.sh` runner. **Edit this**, not the rendered file. |
| `.github/workflows/build-runner.yml` | Validates every config, byte-compiles the sources, runs static and behavioural checks, renders + shellchecks `distill.sh`, uploads it as an artifact and attaches it to releases. |
| `kd_config.py` | Config loading (defaults → YAML → `KD_*` → CLI) and hardware/dtype resolution. The only place platform logic lives. |
| `train_scaled.py` | The training pipeline: multi-domain dataset build, teacher pre-flight, LoRA injection, GKD trainer, telemetry callbacks, adapter save. |
| `configs/*.yaml` | The five profiles. Heavily commented with the measurements behind each number. |
| `check_teacher.py` | Standalone teacher verification. Exit `0` healthy, `2` unfit. |
| `fix_teacher.py` | Rename mislabelled tensors in a merged checkpoint to match the architecture; drop unused components. No GPU. |
| `convert_mlx_adapter.py` | MLX-LM/unsloth LoRA → PEFT LoRA (renames + transposes; `lora_alpha = round(scale × r)`). |
| `publish_model.py` | Publish adapter + `transformers`-merged model to the Hub, with a pre-upload soundness check. |
| `evaluate.py` | Fidelity (top-1 agreement, KL) and capability (held-out perplexity) of the distilled student vs the base student vs the teacher, plus an optional lm-evaluation-harness sweep. |
| `app.py` | Gradio side-by-side UI: original student vs distilled student vs teacher. |
| `main.py` | The original 50-step PoC script. Superseded by `train_scaled.py`; kept for reference. |
| `test_inference.py` | Console three-way comparison against the hardcoded SmolLM2 PoC adapter. |
| `DEMO_PROMPTS.md` | Prompts where the distilled 135M beats the base in 3/3 runs — **and the ones that regressed**, with reasons. Read before demoing. |
| `HOW_IT_WORKS.md` | How the distillation actually works, high level and technical. |

---

## 13. Troubleshooting

**`uv sync` takes forever / downloads gigabytes**
Expected on first use — that's torch. It is cached in `$KD_WORKDIR/src/.venv` and
reused. On Linux/Windows the CPU-only wheel index is used deliberately to avoid
pulling the CUDA runtime.

**"Running on CPU despite Apple Silicon"**
MPS was not detected — usually a torch build without MPS support. Check inside the
env: `./distill.sh --shell`, then
`python -c "import torch; print(torch.backends.mps.is_available(), torch.backends.mps.is_built())"`.

**`bfloat16` silently became `float32`**
On MPS it needs macOS ≥ 14.0; on CPU non-float32 is always downgraded because it
is slow and numerically unstable there. Both cases print a `!!` note in the banner.

**TEACHER PRE-FLIGHT FAILED / "near-uniform output"**
The teacher isn't predicting language. Almost always a key-layout mismatch in a
merged checkpoint (`language_model.model.*` vs `model.language_model.*`). Fix with
`--fix-teacher`, or switch to base + `--teacher-adapter`, which sidesteps the
problem entirely. `--allow-bad-teacher` (direct Python only) forces the run — the
student will faithfully learn the noise.

**A dataset domain prints `SKIPPED`**
Either the config/split doesn't exist, or the expected column is missing. The line
tells you which. For Alpaca-style sources set `format: alpaca` plus the column
names.

**`kept 40/240` — far fewer samples than the quota**
The prompt-length filter is doing its job. Raise `pool`, or raise
`dataset.max_prompt_tokens` / `max_total_tokens`.

**Out of memory**
In order of effect: `--dtype bfloat16` (≈ halves weights), lower `--batch-size` and
raise `--grad-accum`, lower `--lmbda` and `KD_MAX_NEW_TOKENS` (shrinks the
generation KV cache), lower `--lora-r`, or use a smaller teacher.

**Training is far slower than expected**
Check `lmbda` first. At `lmbda 0.5` roughly half your steps pay `max_new_tokens`
sequential forward passes. `--smoke` prints s/step and projects the full run.

**401/403 from Hugging Face**
Gated or private repo. `hf auth login`, or export `HF_TOKEN` before invoking the
runner.

**The UI is slow**
`app.py` runs all three models on **CPU** regardless of the training device, and
loads all three at once. Fine for SmolLM2, painful for a 2B teacher. See
[suggestions](#14-suggestions-and-known-rough-edges).

**Port 7860 already in use**
`./distill.sh --ui --port 8081`.

**"No adapter at ..." in the UI**
Nothing has been trained into a discoverable directory. Pass one explicitly:
`./distill.sh --ui --adapter <path>/final_adapter`, or run
`python app.py --list-adapters`.

**Windows: `./distill.sh` does nothing / bad interpreter**
Run it from Git Bash or WSL2, or use the direct Python entrypoints in
[§11](#11-running-without-distillsh).

---

## 14. Suggestions and known rough edges

Ordered roughly by value-for-effort. These are gaps worth closing, not defects
against a stated requirement.

### Distribution

1. **Cut a GitHub Release.** Actions artifacts expire after 90 days and require a
   logged-in GitHub session, which makes the "just download and run" story worse
   than it needs to be. One published release makes
   `curl -fsSL .../releases/latest/download/distill.sh` work for anyone, forever.
2. **Ship a checksum next to the runner.** The workflow already knows the SHA it
   pinned; emitting `distill.sh.sha256` as a second release asset gives users a way
   to verify a downloaded script before executing it.
3. **Add a `make runner` / `scripts/render.sh` target.** The template→`distill.sh`
   substitution only exists inside the workflow YAML today, so rendering locally
   means reading CI and reimplementing it by hand — [§2 Option
   C](#option-c--render-it-yourself-from-the-template) is exactly that
   reimplementation, and it will drift.

### Runner ergonomics

4. **Default the output directory to the caller's directory.** A relative
   `output_dir` resolves inside `~/.cache/kd-runner/src`, so a successful run
   leaves its adapter somewhere users don't think to look — and a cache wipe
   deletes it. Defaulting `KD_OUTPUT_DIR` to something like
   `$PWD/kd-runs/<profile>-<timestamp>` when `--output` is absent fixes both.
5. **Add `--resume`.** The trainer writes real checkpoints with optimizer state,
   but nothing wires `resume_from_checkpoint`, so an interrupted 600-step run
   restarts from zero.
6. **Add a passthrough escape.** The parser handles `--` and then discards
   everything after it. Forwarding the remainder to `train_scaled.py` would expose
   `--no-eval`, `--allow-bad-teacher`, `--eval-every` and future flags without
   growing the runner's own surface.
7. **Expose the env-only knobs as flags.** `KD_BETA`, `KD_MAX_NEW_TOKENS`,
   `KD_SEED`, `KD_THREADS` and `KD_TOKENIZER` are all honoured but undiscoverable
   from `--help`. `--beta`, `--max-new-tokens`, `--seed` are one line each.
8. **Forward more publish options.** `--publish` hardcodes the happy path;
   `publish_model.py` also supports `--dry-run`, `--private`, `--adapter-only` and
   `--merged-only` — exactly the flags someone wants before their first upload.
9. **A `distill.ps1`, or a documented WSL path.** This repo is developed on Windows
   but its headline entrypoint requires a POSIX shell. Even a thin PowerShell
   wrapper around `uv run python train_scaled.py` would remove the mismatch.

### Correctness and quality

10. **Let the UI use the training device.** `app.py` pins `DEVICE = "cpu"` and
    loads three models simultaneously at import-adjacent time. Reusing
    `kd_config.resolve_device()` would make the Qwen profiles usable in the UI, and
    lazy-loading each column would cut startup time and memory.
11. **Make `check_teacher.py` probe with the config's prompts.** Its `PROBES` are
    hardcoded finance questions, so checking a SmolLM2 teacher asks it about Roth
    IRAs. `config["benchmark_prompts"]` is right there.
12. **Move the CI assertions into `tests/`.** `build-runner.yml` contains genuinely
    good tests — AST checks that `apply_config` binds its globals, that `app.py`
    has no shadowing local imports, shape-reconciliation cases for
    `fix_teacher.py` — written as inline heredocs. As `pytest` files they'd be
    runnable locally and reviewable in diffs, with the workflow reduced to
    `uv run pytest`.
13. **Declare `huggingface_hub` explicitly.** `publish_model.py` imports it, and it
    currently arrives only as a transitive dependency of `transformers`.
14. **Resolve the fate of `main.py` and `test_inference.py`.** Both hardcode the
    SmolLM2 PoC paths and are superseded by `train_scaled.py` / `app.py`. Either
    move them to `legacy/` with a one-line note, or make `test_inference.py`
    config-driven so it works with any profile.
15. ~~**Add an evaluation harness.**~~ Done — `evaluate.py` and
    `./distill.sh --evaluate`, see [§15](#15-measuring-the-result). Still worth
    doing: turn `DEMO_PROMPTS.md`'s manual win/loss list into a scripted per-prompt
    regression check, so the specific behaviours it documents are asserted rather
    than remembered.
16. **Fill in the `pyproject.toml` description.** It still reads "Add your
    description here".

### Bigger swings

17. **Gradient checkpointing as a config option** — the cheapest remaining memory
    lever for larger teachers.
18. **Cache or batch the on-policy rollouts.** At `lmbda > 0` most wall-clock is
    student generation. Reusing rollouts across gradient-accumulation micro-steps,
    or batching generation, would cut run time meaningfully.
19. **Assert the tokenizer match at training startup.** `evaluate.py` now fails
    loudly on a vocabulary mismatch, but `train_scaled.py` still does not — so the
    error surfaces after a full training run rather than before it. The same check
    (`len(tokenizer)` plus a sample of token ids) belongs in the teacher
    pre-flight.

---

## 15. Measuring the result

`./distill.sh --evaluate` answers "how much of the teacher actually transferred?"
It uses the metrics the distillation literature uses; nothing here is invented.

The important framing, from Stanton et al., *Does Knowledge Distillation Really
Work?* (NeurIPS 2021): **fidelity and capability are different measurements and
they do not track each other.** A student can score well on a task while
disagreeing with its teacher on a large fraction of individual predictions. Both
are reported.

### What it measures

| Section | Metric | Standard? |
|---|---|---|
| **Fidelity** | **Top-1 agreement** — % of positions where `argmax(student) == argmax(teacher)` | Yes — the standard predictive-agreement / fidelity metric. |
| | **KL(teacher ‖ student)** per token | Yes — distribution-level fidelity. |
| **Capability** | **Held-out perplexity** for teacher, base student, distilled student | Yes — the standard LM quality metric. |
| | **Gap recovered** — `(base − distilled) / (base − teacher)` | The recovery-rate framing used in distillation papers. |
| **Efficiency** | params, decode throughput, size and speed ratios | Yes — retention is meaningless without the cost it bought. |
| **Tasks** (`--tasks`) | lm-evaluation-harness benchmarks + **retention %** = `distilled / teacher` | Yes — the DistilBERT-style headline ("retains 97% of BERT on GLUE"). |

Every metric is reported for **both the base student and the distilled student**
against the same teacher. That base column is the whole point: without it there is
no way to separate "distillation worked" from "the small model could already do
this". The number to read is the **lift**, not the distilled value.

Scoring uses the held-out split rebuilt with the training seed — the exact split
the student never trained on — and greedy decoding, because sampled output is not
evidence.

### Running it

```bash
# fidelity + perplexity + efficiency; no extra dependencies
./distill.sh --profile finance --evaluate

# more samples, machine-readable output
./distill.sh --profile finance --evaluate --eval-samples 100 --eval-json ./run.json

# also score the standard task benchmarks (slow: three models)
./distill.sh --profile finance --evaluate --tasks ifeval --eval-limit 100
```

`--evaluate` exits `3` when the distilled student did not improve on the base, so
it can gate a pipeline.

### Which benchmark to pass to `--tasks`

`--tasks` shells out to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
the de facto standard (it is what the HF Open LLM Leaderboard runs), so the numbers
are comparable with published ones rather than only with each other. It is an
optional extra — the runner installs it on demand via `uv sync --extra eval`.

* **`ifeval`** — the right default here. Verifiable instruction following ("exactly
  three bullet points", "no commas"), checked programmatically. This is precisely
  the capability `DEMO_PROMPTS.md` records the distilled model gaining, so it is
  the benchmark that turns that manual list into a defensible number.
* **`hellaswag`, `arc_easy`** — cheap general-capability guards. Their job is to
  show you did not *break* anything while specialising. Watch for regression, not
  improvement.
* **Skip `mmlu` / `gsm8k` on a small student** — the scores sit at chance and tell
  you nothing.
* For domain answers that are not programmatically checkable (the finance
  profiles), the accepted standard is **LLM-as-judge pairwise win rate**
  (MT-Bench, AlpacaEval 2.0 length-controlled). That is not wired in; run it
  separately against the published model.

### Reading the output

| Observation | What it means |
|---|---|
| Agreement lift ≈ 0 | Distillation changed nothing, however good the retention % looks. |
| Retention > 100% | Real, and does happen — a student can beat its teacher on a narrow task. Check for eval contamination before celebrating. |
| Task score up, HellaSwag/ARC down | You traded general capability for task compliance. Often an acceptable trade — state it rather than hide it. |
| High agreement, no capability lift | The student is mimicking the teacher's distribution including its mistakes. This is the fidelity/generalization split; the lever is raising `lmbda`. |
| Perplexity up, agreement up | The student moved toward the teacher and away from the reference text. Expected when the teacher is domain-tuned and the held-out text is not. |

### Memory note

`--evaluate` holds the teacher and the student in memory at the same time —
agreement and KL need both distributions for the same position. One student
instance covers both columns (PEFT's `disable_adapter()` turns the LoRA branches
off, which *is* the base student), so peak memory is teacher + one student. For a
2B teacher and a 0.8B student, pass `--dtype bfloat16`.
