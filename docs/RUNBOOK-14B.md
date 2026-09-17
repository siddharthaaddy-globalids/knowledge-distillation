# Runbook: the 14B pair, from a bare pod to a scored bundle

Qwen3-14B (SFT LoRA) → Qwen3-8B, on the neuroscience curriculum, with the
distilled student packed to W4A16 and scored against four other players.

Six steps, always in this order:

```
train  →  upload  →  quantize  →  upload  →  eval  →  upload
```

Two ways to arrange them: **[one pod](#one-pod)**, or **[two pods](#two-pods)** —
an 80 GB card for the training and a cheaper 48 GB card for everything after.
Two pods is the better trade; the reasons are at the bottom.

Everything that goes wrong along the way, and what it actually means, is in
[POD-FAILURES.md](POD-FAILURES.md).

---

## Why there are three uploads and not one

**The credential expires every 15 minutes.** Training and scoring both take far
longer, so the upload stage — which runs last — is guaranteed to fail with
`ExpiredToken`. Every session therefore runs with `--skip upload` and ships by
hand, with freshly exported credentials, as its own command. Three upload steps
is not caution; it is the only arrangement that works.

**llm-compressor and vLLM cannot be installed together.** llm-compressor caps
`transformers` at `<=5.14.1`, `pyproject.toml` wants `>=5.16.1`, and one
environment cannot satisfy both. They are never needed at the same moment, so
the installs are ordered: the packing set first, vLLM afterwards, for the step
that needs it. This is why the runbook below installs by hand instead of letting
`./run.sh` do it — that script installs everything a config implies in one pip
call, which for this profile has no solution.

---

## Provisioning

| | one pod | two pods |
| --- | --- | --- |
| GPU | 80 GB — H100 80GB PCIe, H200, A100 80GB | **A:** 80 GB · **B:** 48 GB (L40S, A6000) |
| Volume at `/workspace` | 200 GB | 200 GB each |
| Container disk | 40 GB | 40 GB |
| Template | any stock **PyTorch** image | same |

A 48 GB card cannot train this pair — 46 GB of weights are resident at once and
the allocator fails partway through the first forward pass, after both models
have downloaded. It is ample for packing and scoring, where players load one at
a time and the 29.5 GB teacher is the ceiling.

---

## The shell, on every pod

Run this first, in every new SSH session and every new tmux pane. Nothing below
works without it, and an empty variable reaches argparse as a flag with nothing
after it.

```bash
tmux new -s kd

cd /workspace
git clone <repo> knowledge-distillation
cd knowledge-distillation

export PYTHONPATH=/workspace/knowledge-distillation/src
export HF_HOME=/workspace/hf-cache
export KD_RUNS_DIR=/workspace/runs
export HF_HUB_DISABLE_XET=1          # download once instead of stage-then-reconstruct
export KD_PRICE_PER_HOUR=2.79        # the rate you agreed to; the cost cap is inert without it
export HF_TOKEN=hf_...               # ~46 GB from the Hub, rate-limited without it

export AWS_ACCESS_KEY_ID=...         # re-export before EVERY step marked "fresh token"
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

Then check the machine before spending anything on it:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace /
```

80 GB (or 48) on the card, 200 GB free on `/workspace`, and `/` is the 40 GB
container disk that must never receive a model.

---

## One pod

Everything on a single 80 GB card. Simplest to follow, most expensive to run:
the packing and the arena are billed at the training card's rate.

### 1. Install the packing set — no vLLM yet

```bash
python3 -m pip install -q uv
python3 -m uv pip install --system --break-system-packages \
    transformers trl peft accelerate datasets pyyaml boto3 \
    llmcompressor compressed-tensors
```

`transformers` is deliberately unpinned: left free, the resolver settles on a
version llm-compressor accepts, which is all that matters until the eval.

Then prove the environment before it costs anything:

```bash
python3 -c "import transformers.modeling_utils" || python3 -m pip uninstall -y torchvision torchaudio
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` on the second line, or stop — a CPU wheel trains about a hundred times
slower at the GPU's hourly rate. About 3 minutes.

### 2. Train, and pack in the same run — *fresh token*

```bash
python3 -m kd doctor --config configs/enlibra/enlibraQ3-14B.yaml
python3 -m kd check  --config configs/enlibra/enlibraQ3-14B.yaml

python3 -m kd pipeline --config configs/enlibra/enlibraQ3-14B.yaml \
    --skip upload --set s3.cache_dir=/workspace/kd-cache
```

Read `doctor`'s "remote inputs" section: a FAIL there is a run that aborts in
preflight, and on a pod that means paying to find out.

The pipeline runs `preflight → teacher-check → smoke → train → quantize`. The
`evaluation` stage is skipped because the profile sets
`evaluation.after_training: false`, and `upload` because you asked. **The
packing happens here**, inside the training run, writing to
`runs/<run-id>/quantized` — there is no separate quantize command in this flow.

Watch for the run id in the banner; every step below names it.

Hours. `save_steps: 50` means a hard stop costs you at most fifty steps.

### 3. Upload the adapter and the packed student — *fresh token*

```bash
export RUN=<run-id>
python3 -m kd upload $RUN --config configs/enlibra/enlibraQ3-14B.yaml
```

Ships `final_adapter/`, `quantized/`, `checkpoints/`, logs and metrics. ~14 GB,
a few minutes. **Do this before the eval, not after** — until it finishes, the
only copy of hours of GPU time is on a pod that can be reclaimed.

Always name the run. With no argument `kd upload` takes "the latest run", which
falls back to the alphabetically largest directory name, and run ids lead with
the profile rather than the timestamp.

### 4. Install vLLM

```bash
python3 -m uv pip install --system --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple vllm

python3 -c "import vllm, torch, transformers.modeling_utils; print(torch.__version__, torch.cuda.is_available())"
```

This replaces torch with the build vLLM pins and moves `transformers` again —
both fine now that the packing is done. If the import fails here, **do not**
uninstall torchvision this time; vLLM needs it. Reinstall the matching pair:

```bash
python3 -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

5 minutes with uv, 15 with pip.

### 5. Score — *fresh token*

```bash
python3 -m kd eval --config configs/enlibra/enlibraQ3-14B.yaml --skip upload \
    --set s3.cache_dir=/workspace/kd-cache \
    --adapter /workspace/runs/$RUN/final_adapter
```

`--set s3.cache_dir` is not optional. The arena materialises the teacher as a
dense checkpoint for vLLM — 29.5 GB — beside the S3 cache, and the default cache
is on the 40 GB container disk.

**Check the banner before walking away:**

```
players : base, distilled, distilled-w4a16, teacher-base, teacher
```

Five names. Four means the packed student was not found, and the run is worth
killing rather than paying for.

Hours. Expect lower accuracy than training suggests: 80 of the 142 held-out
questions run to 4 or 5 hops where training tops out at 3.

### 6. Upload the evaluation — *fresh token*

```bash
python3 -m kd upload $RUN --config configs/enlibra/enlibraQ3-14B.yaml \
    --set 's3.upload=[evaluation,metrics,report,logs]'
```

The `--set` narrows it to what is new, so you do not re-send the 14 GB from
step 3.

Then terminate the pod in the console. Nothing stops one you started by hand.

---

## Two pods

The recommended arrangement. Training needs 80 GB; packing and scoring do not,
and there is no reason to pay a training card's rate for them.

### Pod A — training only, 80 GB

Steps 1 and 2 above, with one change: skip the packing as well, so the 80 GB
card does nothing but train.

```bash
python3 -m uv pip install --system --break-system-packages \
    transformers trl peft accelerate datasets pyyaml boto3
python3 -c "import transformers.modeling_utils" || python3 -m pip uninstall -y torchvision torchaudio

python3 -m kd doctor --config configs/enlibra/enlibraQ3-14B.yaml

python3 -m kd pipeline --config configs/enlibra/enlibraQ3-14B.yaml \
    --skip quantize --skip upload --set s3.cache_dir=/workspace/kd-cache
```

Then, *fresh token*:

```bash
python3 -m kd upload <run-id> --config configs/enlibra/enlibraQ3-14B.yaml
```

Write the run id down. **Terminate pod A.** Everything that matters is in the
bucket at:

```
s3://enlibra/dss/dev/runs/20260818_215840_neuroscience_8f33eb7cf609/outputs/gkd/runs/<run-id>/
```

### Pod B — packing and scoring, 48 GB

One script covers the whole session, because it encodes every trap in
[POD-FAILURES.md](POD-FAILURES.md): ordered installs, the torchvision repair,
the cache-on-the-volume refusal, and an `--out` it reads from the config so it
cannot drift from where the arena looks.

First point the profile at the run you just trained. It names the run id twice —
in `evaluation.adapter` and in `quantization.output_dir` — and both must move:

```bash
sed -i 's/enlibraQ3-14B-train-2026-09-17-0558/<run-id>/g' \
    configs/enlibra/enlibraQ3-14B-score.yaml
grep -n '<run-id>' configs/enlibra/enlibraQ3-14B-score.yaml
```

Then, with credentials exported:

```bash
./scripts/score-pod.sh all --config configs/enlibra/enlibraQ3-14B-score.yaml
```

It runs `setup → quantize → upload → eval → upload`, pausing before each
billable step. It is idempotent: re-run it after any failure and it skips the
install, skips the packing if the checkpoint is already there, and carries on.

Individual steps, when something needs redoing on its own:

```bash
./scripts/score-pod.sh setup    --config configs/enlibra/enlibraQ3-14B-score.yaml
./scripts/score-pod.sh quantize --config configs/enlibra/enlibraQ3-14B-score.yaml
./scripts/score-pod.sh upload   --config configs/enlibra/enlibraQ3-14B-score.yaml
./scripts/score-pod.sh eval     --config configs/enlibra/enlibraQ3-14B-score.yaml
```

Re-export AWS credentials immediately before each `upload`, and before
`quantize` and `eval` — both fetch from the bucket in their first seconds.

---

## What ends up in the bucket

```
<prefix>/runs/<run-id>/
    manifest.json
    config.resolved.yaml
    final_adapter/            0.7 GB   the result
    quantized/                5.7 GB   what actually gets deployed
    checkpoints/                7 GB   optimizer state, for a resumed run
    run.log  events.jsonl
    metrics.json  arena.json  arena-transcript.jsonl
    evaluation/<eval-id>/              scores, transcript, report.html
```

---

## If a step fails

| Failed at | Redo | Keep |
| --- | --- | --- |
| any upload | that upload alone, with fresh credentials | everything |
| eval, partway | `eval` from the start | the adapter and packed student, already in the bucket |
| quantize | `quantize` alone | training |
| training | **the whole training run** | only what `save_steps: 50` wrote — see below |

**There is no resume.** Nothing in the pipeline passes `resume_from_checkpoint`, so a training
run that is killed — by a limit, a spot reclaim, a dropped SSH session — cannot
be continued. What `training.save_steps: 50` buys is a partial adapter you can
inspect, score or ship, not a run you can restart where it stopped. Upload it
before starting over:

```bash
python3 -m kd upload <run-id> --config configs/enlibra/enlibraQ3-14B.yaml --with-checkpoints
```

The rule the three uploads exist to enforce: **nothing expensive should be on
the pod alone for longer than it has to be.**

---

## Cost, roughly

| | one pod | two pods |
| --- | --- | --- |
| install + downloads | ~40 min @ 80 GB | ~30 min @ 80 GB + ~10 min @ 48 GB |
| training | hours @ 80 GB | same |
| packing | ~20 min @ 80 GB | ~20 min @ 48 GB |
| scoring | 1.5–3 h @ 80 GB | 1.5–3 h @ 48 GB |

At roughly $2.79/hour for an 80 GB card against ~$1.00 for a 48 GB one, the
split saves several dollars per cycle and costs one extra `git clone`. The
profile caps a run at `limits.max_cost_usd: 40`, which is inert unless
`KD_PRICE_PER_HOUR` is exported.

---

## See also

- [POD-FAILURES.md](POD-FAILURES.md) — every error seen on these pods, and what it really meant
- [RUNPOD.md](RUNPOD.md) — renting the pod, and the `kd runpod` launcher
- [CONFIG.md](CONFIG.md) — what every key in the profile does
- [EXPLANATION-SCORING.md](EXPLANATION-SCORING.md) — how to read the report
