# knowledge-distillation

Distil a large teacher model into a small LoRA student, with on-policy knowledge
distillation (GKD). Config-driven: point it at a YAML file and it runs.

Everything about a run — models, dataset, LoRA shape, schedule, hardware, spend
limits — comes from that file. Retargeting to a different teacher, student or
dataset never requires a code edit.

```bash
./distill.sh --config configs/finance.yaml          # Linux, macOS, containers
.\distill.ps1 -Config configs\finance.yaml          # Windows
```

```
  run   20260908T1412Z-finance-bb0c874        device cuda (bfloat16)
  cfg   configs/finance.yaml  <- _base.yaml   overrides: training.max_steps=500 (--set)

[1/8] preflight            OK       2s
[2/8] teacher-check        OK    1m48s
[3/8] smoke                OK    1m12s
      3.40 s/step measured -> 500 steps is about 28m
[4/8] train                ...
```

---

## What it does

One command runs every level of verification, gated, stopping at the first
failure:

| Stage | Gate | |
|---|---|---|
| `preflight` | ✓ | Resolve config and hardware, fetch remote inputs. Seconds. |
| `teacher-check` | ✓ | Load **only** the teacher: missing weights, NaN scan, coherence. |
| `smoke` | ✓ | Two real steps. Measures s/step and projects the full run against your limits. |
| `train` | ✓ | |
| `evaluate` | | Fidelity and capability, against the untrained base student. |
| `report` | | A readable `report.html`. |
| `publish` | | To the Hugging Face Hub. Off by default. |
| `upload` | | To S3. Off by default; also runs after a failure, so logs survive. |

Narrow a run with `--only train`, `--from evaluate`, `--skip smoke`.

The `teacher-check` and `smoke` gates exist because of how this fails in
practice. GKD trains the student to match the teacher's *output distribution*, so
a teacher that loads without raising but is partly randomly initialised produces
a broken student after a full, apparently successful run — and nothing says so
until you read the final samples. The check loads only the teacher and takes a
couple of minutes.

## Getting started

You need [uv](https://docs.astral.sh/uv/) and Python 3.13. The runner installs uv
if it is missing.

```bash
git clone <this repo> && cd knowledge-distillation
uv sync

uv run kd doctor                                    # what can this machine do?
uv run kd check --config configs/smoke.yaml         # resolve everything, run nothing
uv run kd pipeline --config configs/smoke.yaml      # ~2 min end to end
```

`configs/smoke.yaml` trains SmolLM2-360M into SmolLM2-135M for two steps. It
proves the machine works before you commit a real budget.

Then pick a real profile, or write one:

```bash
uv run kd pipeline --config configs/finance.yaml
```

### Or download a runner

CI publishes `distill.sh` and `distill.ps1` pinned to the commit they were built
from. Each is a bootstrapper: it installs uv, fetches that exact source, and hands
over to the pipeline. Grab one from the **Build distillation runner** workflow
artifacts or a release.

```bash
chmod +x distill.sh
./distill.sh --config configs/finance.yaml
```

## Commands

```
kd pipeline    Every stage, gated                    kd check       Resolve and print, run nothing
kd train       Just training                         kd doctor      Environment and credentials
kd evaluate    Score an adapter vs the teacher       kd ui          Browser control panel
kd publish     To the Hugging Face Hub               kd runpod      Rent a GPU (optional)

kd check-teacher      Is this teacher fit to distil from?
kd fix-teacher        Repair a checkpoint with mislabelled tensors
kd convert-adapter    MLX / unsloth LoRA -> PEFT format
```

`kd <command> --help` for anything. `python -m kd` works identically and needs
nothing on PATH.

## Configuring a run

Profiles inherit from [`configs/_base.yaml`](configs/_base.yaml), which holds
every default, and list only what they change:

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
kd pipeline --config configs/finance.yaml \
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
| `default` | SmolLM2 360M → 135M. Runs anywhere. |
| `mac` | Same pair, Apple Silicon (MPS) with real batching. |
| `smoke` | Two steps, tiny pools. CI and first-run validation. |
| `finance` | A finance-tuned Qwen3.5-2B → Qwen3.5-0.8B. |
| `qwen-poc` | Stock Qwen3.5-2B → 0.8B. A known-good pairing for proving the pipeline. |

## What a run leaves behind

```
runs/20260908T1412Z-finance-bb0c874/
  config.resolved.yaml   every value after every override - re-runnable as-is
  manifest.json          git sha, package versions, per-stage timings, exit code
  run.log                everything the terminal showed, including library output
  events.jsonl           {stage, step, loss, elapsed, spend_usd} - one per line
  metrics.json           the final numbers
  report.html
  final_adapter/
  checkpoints/
```

One directory per run, never overwritten, and the same unit that gets uploaded to
S3 — so a run is either entirely recoverable or entirely absent.

## Spend and time limits

```yaml
limits:
  max_runtime_minutes: 180
  max_cost_usd: 2.00        # only meaningful on a rented GPU
  confirm_above_usd: 1.00
```

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
kd evaluate --config configs/finance.yaml
```

Two questions, measured separately, because the literature is explicit that they
do not track each other:

- **Fidelity** — does the student reproduce the *teacher's* predictions? Top-1
  agreement, KL divergence.
- **Capability** — is the student actually better at the task? Held-out
  perplexity, optionally lm-eval benchmarks.

Both are reported for the **base** student as well as the distilled one. That
column is what separates "distillation worked" from "the small model could
already do this" — read the change, not the absolute value.

`--tasks ifeval` and `--gen-similarity 20` need `uv sync --extra eval`.

## Optional: S3 and rented GPUs

Both off by default; nothing in the pipeline imports their SDKs.

Any path in a config may be an `s3://` URI — teacher, student, adapter, dataset —
fetched in `preflight` and cached. Run bundles sync back up.

```bash
kd runpod launch --config configs/finance-pod.yaml
```

Rents the GPU you named and **no other**, enforces your spend cap from your own
machine, and always terminates the pod. See
**[docs/RUNPOD.md](docs/RUNPOD.md)**.

## Repository map

```
configs/          _base.yaml holds every default; profiles extend it
src/kd/
  cli.py          the `kd` command
  config.py       extends, merge, strict validation, device resolution
  runlog.py       run directories, logs, events, manifest
  pipeline.py     the gated stages
  limits.py       time / step / cost ceilings
  paths.py        s3:// -> local, cached
  data.py         dataset assembly
  train.py        GKD training
  evaluate.py     fidelity and capability
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
```

## Tests

No framework, no model downloads, about a second each:

```bash
for t in tests/*.py; do uv run python "$t"; done
```

They cover config precedence and validation, run bundles and their failure paths,
pipeline gating and limits, the runner templates, S3, and — with fake SDKs — GPU
selection and the terminate guarantee.

## Further reading

- **[docs/CONFIG.md](docs/CONFIG.md)** — every configuration key
- **[docs/RUNPOD.md](docs/RUNPOD.md)** — renting a GPU, and keeping it cheap
- **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** — what GKD is doing and why
- **[docs/DEMO_PROMPTS.md](docs/DEMO_PROMPTS.md)** — prompts where the difference shows
