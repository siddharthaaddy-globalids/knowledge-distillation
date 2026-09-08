# Walkthrough — CI build to a usable model

A single path, start to finish: build a runner from GitHub Actions on your
branch, download it, train, evaluate, and use what comes out.

Written against this repository and the `feature/runpod-s3-support` branch.
Substitute your own branch name where it appears.

- [0. Before you start](#0-before-you-start)
- [1. Push the branch](#1-push-the-branch)
- [2. Build the runner in Actions](#2-build-the-runner-in-actions)
- [3. Download the artifact](#3-download-the-artifact)
- [4. Make it runnable](#4-make-it-runnable)
- [5. Prove it works](#5-prove-it-works)
- [6. Decide what to train](#6-decide-what-to-train)
- [7. Train](#7-train)
- [8. Evaluate](#8-evaluate)
- [9. Use the model](#9-use-the-model)
- [10. Where everything ended up](#10-where-everything-ended-up)
- [11. If something goes wrong](#11-if-something-goes-wrong)

---

## 0. Before you start

What you need, and nothing else:

| | |
|---|---|
| Push access to the repository | to trigger the workflow on your branch |
| `git` on the machine that will run it | the runner clones with it |
| An internet connection | models come from the Hugging Face Hub |

You do **not** need Python, uv, CUDA, or a clone on the machine that runs the
training. The runner installs uv and fetches the source itself.

This repository is **public**, so the runner clones it with no credentials. If it
is ever made private, the machine running it will need a git credential helper or
a token with read access, or the clone step fails.

---

## 1. Push the branch

The runner is built from a commit. Anything not pushed is not in it — including,
right now, the `--ref` and `--extra` flags this guide uses.

```bash
git status --short          # expect a clean tree before building
git push origin feature/runpod-s3-support
```

Note the commit you pushed. The artifact is named after it:

```bash
git rev-parse --short=7 HEAD
```

---

## 2. Build the runner in Actions

The workflow builds automatically only on `main` and `master`. On a feature
branch you start it by hand — which is exactly what `workflow_dispatch` is for.

1. Open the repository on GitHub → **Actions**.
2. In the left sidebar, choose **Build distillation runner**.
3. Click **Run workflow** (top right of the run list).
4. In the branch dropdown, select **`feature/runpod-s3-support`**.
5. Click the green **Run workflow**.

It takes roughly three to five minutes and has two jobs:

| Job | What it proves |
|---|---|
| **Validate** | every module imports, all seven test files pass, every profile resolves, override precedence holds, the docs still describe the code |
| **Build runners** | renders `distill.sh` and `distill.ps1`, shellchecks the bash, parses the PowerShell, and proves the two dispatch identically |

If **Validate** fails, no runner is produced — deliberately. A runner that fetches
source which does not import is worse than no runner.

### With the `gh` CLI instead

```bash
gh workflow run "Build distillation runner" --ref feature/runpod-s3-support
gh run watch
```

---

## 3. Download the artifact

Once the run is green:

1. Open the completed workflow run.
2. Scroll to **Artifacts** at the bottom of the summary page.
3. Download **`distill-runner-<sha>`** — where `<sha>` is the seven-character
   commit you pushed.

GitHub serves artifacts as a **zip**, so you get `distill-runner-<sha>.zip`
containing two files:

```
distill.sh      # macOS, Linux, containers
distill.ps1     # Windows
```

Take whichever suits the machine that will do the training. They are the same
bootstrapper; CI builds both from the same stage definition so they cannot drift.

### With the `gh` CLI instead

```bash
gh run download --name "distill-runner-$(git rev-parse --short=7 HEAD)"
```

---

## 4. Make it runnable

### macOS, Linux, containers

Zip archives do not carry the executable bit, so it has to be restored:

```bash
unzip distill-runner-*.zip
chmod +x distill.sh
./distill.sh --runner-version
```

That last command should print the commit you built from:

```
distill.sh pinned to <org>/knowledge-distillation@045a061 (built 2026-09-08T16:12:04Z)
```

If it prints a different commit, you downloaded an older artifact.

### Windows

Windows marks downloaded files as untrusted, and PowerShell refuses to run
unsigned scripts by default. Both are one-liners:

```powershell
Expand-Archive distill-runner-*.zip -DestinationPath .
Unblock-File .\distill.ps1
.\distill.ps1 -RunnerVersion
```

If the last command is blocked by execution policy, run it for this session only —
this does not change the machine's setting:

```powershell
powershell -ExecutionPolicy Bypass -File .\distill.ps1 -RunnerVersion
```

---

## 5. Prove it works

Three commands, in increasing cost. Do not skip them — each one catches a class
of problem the next would waste more time discovering.

```bash
# 1. What can this machine do? Seconds.
./distill.sh doctor

# 2. Does the config resolve, and onto what hardware? Seconds.
./distill.sh check --config configs/smoke.yaml

# 3. Does the whole pipeline run end to end? A few minutes.
./distill.sh --config configs/smoke.yaml
```

The first invocation installs uv, clones the pinned source into
`~/.cache/kd-runner`, and downloads torch. Expect a few minutes once; everything
after it is fast.

`configs/smoke.yaml` trains SmolLM2-360M into SmolLM2-135M for two steps. Too
short to learn anything — its job is to prove every stage runs here. You are
looking for eight green stages:

```
[1/8] preflight            OK       0s
[2/8] teacher-check        OK      22s
[3/8] smoke                OK      51s
[4/8] train                OK      36s
[5/8] evaluate             OK      22s
[6/8] report               OK       0s
[7/8] publish              skipped - publish.enabled is false
[8/8] upload               skipped - s3.enabled is false
```

`doctor` also tells you the device that matters for everything below:

```
 device     : cpu (float32)
```

or `cuda (bfloat16)` if there is a GPU.

---

## 6. Decide what to train

This is the decision that determines whether your run takes forty minutes or
overnight. It comes down to what `doctor` printed.

| Your device | Sensible first real run | Why |
|---|---|---|
| **CPU** | `configs/default.yaml` — SmolLM2 360M → 135M, 300 steps | Roughly one to two hours. Small enough to actually finish. |
| **CUDA GPU** | `configs/qwen-poc.yaml` — Qwen3.5-2B → 0.8B, 100 steps | A known-good pairing, verified identical tokenizers. Add `--set hardware.dtype=bfloat16`. |
| **Apple Silicon** | `configs/default.yaml`, or `qwen-poc` on 32 GB+ | Unified memory fits a larger teacher than the nominal size suggests. |

**Do not start `qwen-poc` or `finance` on a CPU box.** A 2B teacher plus a 0.8B
student is about 11 GB of float32 weights before activations, and `bfloat16` is
ignored on CPU because it is slow and unstable there. It will not fail cleanly —
it will simply take a very long time.

You do not have to guess. The `smoke` stage measures seconds-per-step on your
actual hardware and projects the full run *before* training starts:

```
[3/8] smoke                OK      51s
      5.13 s/step measured -> 300 steps is about 26m
```

And a run that cannot finish inside its limits is refused rather than started:

```
This run would take about 220 min, over the 180 min limits.max_runtime_minutes.
  measured 44.10 s/step over 300 steps
  raise the ceiling:  --set limits.max_runtime_minutes=265
  or shorten the run: --set training.max_steps=244
```

That is the machinery doing its job. Take the suggestion, or pick a smaller pair.

---

## 7. Train

Point `project.runs_dir` somewhere you will find it. By default the runner writes
inside its own cache directory, which is not where you want a model you care
about:

```bash
./distill.sh --config configs/default.yaml \
  --set project.runs_dir="$PWD/runs"
```

On Windows:

```powershell
.\distill.ps1 -Config configs\default.yaml `
  --set project.runs_dir="$PWD\runs"
```

That runs all eight stages. While it goes you get a step line roughly once a
second:

```
 [step  147/300] jsd=1.8342 run20=1.9011 grad=0.4127 lr=2.31e-04  15.2s/step  eta=38m
```

and benchmark generations every `training.benchmark_every` steps, so quality
drift is visible while it happens rather than only at the end.

### Common adjustments

```bash
# Shorter run
--set training.max_steps=150

# Cheaper: fewer on-policy rollouts. lmbda is the dominant cost.
--set gkd.lmbda=0.25

# A different teacher entirely
--set models.teacher=HuggingFaceTB/SmolLM2-1.7B-Instruct

# Put a ceiling on it
--set limits.max_runtime_minutes=90
```

### If you only want the training stage

You have already passed the gates once, so skip re-running them:

```bash
./distill.sh --config configs/default.yaml --only train \
  --set project.runs_dir="$PWD/runs"
```

### Stopping and resuming

Ctrl+C is safe. The manifest is written, and the last checkpoint under
`checkpoints/` survives — `training.save_steps` controls how often that is
written.

---

## 8. Evaluate

The pipeline already evaluated and wrote a report. Open it:

```
runs/<run-id>/report.html
```

It leads with a plain-English summary, so it is readable by someone who does not
know what perplexity is.

To evaluate again — a different adapter, more samples, or benchmark tasks:

```bash
./distill.sh evaluate --config configs/default.yaml \
  --adapter "$PWD/runs/<run-id>/final_adapter" \
  --samples 100 \
  --report "$PWD/report.html"
```

With no `--adapter`, it uses the newest one it can find.

### Reading the numbers

Two questions, measured separately, because they do not track each other:

- **Fidelity** — does the student reproduce the *teacher's* predictions? Top-1
  agreement, KL divergence.
- **Capability** — is the student actually better at the task? Held-out
  perplexity.

Both are reported for the **untrained base student** as well. That column is what
separates "distillation worked" from "the small model could already do this".
**Read the change, not the absolute value** — the base student already scores most
of the absolute number, because it shares an architecture and a tokenizer with
the teacher.

The figure to look at is `gap recovered`: how much of the base-to-teacher
distance the training closed.

### Benchmark tasks (optional, slow)

Needs the `eval` extra, which the runner installs on request:

```bash
./distill.sh --extra eval evaluate --config configs/default.yaml \
  --tasks ifeval --limit 100
```

It scores three models, so budget accordingly.

### Exit codes

`0` means the adapter improved on the base student. **`3` means it did not** —
that is a result, not a crash. The numbers are real and the report is still
written. If you get a 3, check that the teacher passed its check, that the step
budget was not tiny, and that `lora.target_modules` covers your architecture.

**`4` means a limit stopped it** — or refused to start it. Also not a
malfunction: the smoke stage measured your hardware, projected the full run, and
declined because it would not fit. The message names the two ways out, and
nothing was trained or spent:

```
[3/8] smoke                REFUSED  10m04s
      This run would take about 624 min, over the 180 min limits.max_runtime_minutes.
        measured 122.09 s/step over 300 steps
        raise the ceiling:  --set limits.max_runtime_minutes=749
        or shorten the run: --set training.max_steps=73
```

---

## 9. Use the model

### What you have

```
runs/<run-id>/final_adapter/
  adapter_config.json          which base model, r, alpha, target_modules
  adapter_model.safetensors    the trained weights - about 20-40 MB
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
```

It is a **LoRA adapter, not a full model**. The base weights are not in there;
`adapter_config.json` names the base it belongs to.

### Load it in Python

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = PeftModel.from_pretrained(base, "./runs/<run-id>/final_adapter")
tok = AutoTokenizer.from_pretrained("./runs/<run-id>/final_adapter")

messages = [{"role": "user", "content": "List three states of matter."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=80)
print(tok.decode(out[0], skip_special_tokens=True))
```

For faster inference, collapse the adapter into the weights once:

```python
merged = model.merge_and_unload()
merged.save_pretrained("./my-model")
tok.save_pretrained("./my-model")
```

### Compare it against the teacher in a browser

```bash
./distill.sh ui --compare --adapter "$PWD/runs/<run-id>/final_adapter"
```

Three columns — base student, your distilled student, teacher — answering the
same prompt. This is the quickest way to see whether the training did what you
wanted, and the honest counterweight to a good-looking metric.

### Publish it

```bash
export HF_TOKEN=...

./distill.sh publish --repo my-org/my-distilled-model \
  --adapter "$PWD/runs/<run-id>/final_adapter"
```

That pushes two things: the adapter to `my-org/my-distilled-model-lora`, and a
**merged, ready-to-run model** to `my-org/my-distilled-model`. The merge is done
by transformers, so the checkpoint uses canonical tensor names rather than a
framework-specific layout — and it is verified to generate language before
anything uploads.

Anyone can then use it with no adapter handling at all:

```python
model = AutoModelForCausalLM.from_pretrained("my-org/my-distilled-model")
```

---

## 10. Where everything ended up

```
runs/<utc-timestamp>-<profile>-<sha7>/
  final_adapter/         the model you wanted
  checkpoints/           resumable optimizer state; large, safe to delete
  report.html            the readable result
  metrics.json           the same numbers, machine-readable
  evaluation.json        the full evaluation payload
  config.resolved.yaml   every value after every override - re-runnable as-is
  manifest.json          git sha, package versions, per-stage timings, exit code
  run.log                everything the terminal showed
  events.jsonl           one JSON object per step, eval and checkpoint
```

`runs/latest.txt` points at the newest run.

Two files are worth knowing about beyond the adapter. **`config.resolved.yaml`**
is a complete, runnable config — the exact one that produced this result, so you
can reproduce or tweak it without reconstructing your flags:

```bash
./distill.sh --config "$PWD/runs/<run-id>/config.resolved.yaml"
```

**`manifest.json`** records the commit, the package versions and the per-stage
timings, which is what lets you explain a result months later.

---

## 11. If something goes wrong

**The workflow has no "Run workflow" button.** The workflow file must exist on the
branch you want to build. Confirm `.github/workflows/build-runner.yml` is
committed on your branch and pushed.

**Validate failed, so no artifact.** Open the failed step — it names what broke.
Run the same checks locally from a clone: `for t in tests/*.py; do uv run python
"$t"; done`.

**`./distill.sh: Permission denied`.** The zip lost the executable bit:
`chmod +x distill.sh`.

**`.\distill.ps1 cannot be loaded`.** `Unblock-File .\distill.ps1`, or run it
with `powershell -ExecutionPolicy Bypass -File .\distill.ps1 ...`.

**`--runner-version` shows the wrong commit.** You downloaded an artifact from an
older run. Each artifact is pinned to the commit it was built from — that is the
point — so download the one from your latest run.

**The clone step fails with an authentication prompt.** The repository is private.
Configure a git credential helper on that machine, or use a token.

**`Config file not found`** for a config of your own. The runner changes directory
into the fetched source before running, so pass **absolute** paths:
`--config "$PWD/my-run.yaml"`.

**You cannot find the adapter.** You did not set `project.runs_dir`, so it is
under `~/.cache/kd-runner/src/runs/`. Move it, and pass
`--set project.runs_dir="$PWD/runs"` next time.

**The teacher fails its check.** Usually a merged checkpoint whose tensor names do
not match its architecture — common in exports from MLX or unsloth.
`./distill.sh fix-teacher --config <config>` renames them without retraining.

**A run stopped on a limit.** That is the ceiling working. The last checkpoint
survives; raise the limit or shorten the run, both spelled out in the message.

**You want to try a different branch without rebuilding.** `--ref` points the same
runner at another revision, and says so loudly every time:

```bash
./distill.sh --ref another-branch --config configs/default.yaml
```

Use a manually built runner instead when you want the result to stay traceable.

---

## Where to read more

- [RUNBOOK-RUNNER.md](RUNBOOK-RUNNER.md) — the full reference for this path
- [RUNBOOK-SOURCE.md](RUNBOOK-SOURCE.md) — the same, from a clone with `uv run`
- [CONFIG.md](CONFIG.md) — every configuration key
- [RUNPOD.md](RUNPOD.md) — renting a GPU when CPU is not enough
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — what GKD is actually doing
