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

[1/9] preflight            OK       2s
[2/9] teacher-check        OK    1m48s
[3/9] smoke                OK    1m12s
      3.40 s/step measured -> 500 steps is about 28m
[4/9] train                ...
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
| `arena` | | Accuracy and Elo on a held-out answer key. Off unless `evaluation.arena_file` is set. |
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

**New here?** Clone and run one script — it installs everything, works out
whether it is on a laptop or a GPU, and does the right thing for each:

```bash
git clone <this repo> && cd knowledge-distillation
./run.sh
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
kd check --config configs/smoke.yaml         # resolve everything, run nothing
kd pipeline --config configs/smoke.yaml      # ~2 min end to end
```

`uv sync` installs a real `kd` executable into `.venv`. Without activating, prefix
with `uv run`; `python -m kd` works too and needs nothing on `PATH`.

`configs/smoke.yaml` trains SmolLM2-360M into SmolLM2-135M for two steps. It
proves the machine works before you commit a real budget.

Then pick a real profile, or write one:

```bash
kd pipeline --config configs/finance.yaml
```

### Or download a runner

No clone, no uv, no Python setup. CI publishes `distill.sh` and `distill.ps1`
pinned to the commit they were built from; each installs uv if it is missing,
fetches that exact source, and hands over to the pipeline. Grab one from the
**Build distillation runner** workflow artifacts or a release. This is the way to
put it on someone else's machine.

```bash
chmod +x distill.sh
./distill.sh --config configs/finance.yaml
```

## Commands

```
kd pipeline    Every stage, gated                    kd check       Resolve and print, run nothing
kd train       Just training                         kd doctor      Environment and credentials
kd evaluate    Score an adapter vs the teacher       kd ui          Browser control panel
kd arena       Accuracy + Elo on an answer key        kd runpod      Rent a GPU (optional)
kd publish     To the Hugging Face Hub

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
| `enlibraQ3-8B` | An RL-tuned Qwen3-8B → Qwen3-1.7B on the enLibra space curriculum. Needs a 48 GB GPU. |
| `enlibraQ3-8B-smoke` | The same run with stand-in models, small enough for a 16 GB laptop. Two steps — proves the plumbing. |
| `enlibraQ3-8B-mac` | Stand-in models again, but the **full** schedule and the whole evaluation. Hours, free, and it answers whether distillation works on this data. |

Both models are resident at once, so their **sum** is what has to fit: Qwen3-8B
plus Qwen3-1.7B is 19.0 GB of bf16 weights before activations, which is why the
pod profile asks for a 48 GB card and why the laptop profiles shrink both halves.
[docs/START-HERE.md](docs/START-HERE.md) has the table.

Those three read `data/enlibra-curriculum/`, which **is committed** — 3.4 MB, so
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

### Multiple-choice accuracy, and asking the student directly

For a corpus with a known correct answer per row — the enLibra curriculum — the
number that decides whether the run worked is accuracy, not fidelity. A student
that mirrors a mediocre teacher perfectly scores well above and badly here.

```bash
# score the held-out split: the rows training never saw
python scripts/ask.py --config configs/enlibraQ3-8B.yaml --accuracy

# ask it one thing
python scripts/ask.py --config configs/enlibraQ3-8B.yaml \
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
kd runpod launch --config configs/finance-pod.yaml
```

Rents the GPU you named and **no other**, enforces your spend cap from your own
machine, and always terminates the pod. See
**[docs/RUNPOD.md](docs/RUNPOD.md)**.

## Repository map

```
configs/          _base.yaml holds every default; profiles extend it
src/kd/
  arena.py       accuracy and Elo on a held-out multiple-choice set
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

- **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** — CI build to a usable model, one path start to finish
- **[docs/RUNBOOK-SOURCE.md](docs/RUNBOOK-SOURCE.md)** — the full runbook, from a clone with `uv run`
- **[docs/RUNBOOK-RUNNER.md](docs/RUNBOOK-RUNNER.md)** — the same runbook, using only the downloaded `distill.sh`
- **[docs/CONFIG.md](docs/CONFIG.md)** — every configuration key
- **[docs/RUNPOD.md](docs/RUNPOD.md)** — renting a GPU, and keeping it cheap
- **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** — what GKD is doing and why
- **[docs/DEMO_PROMPTS.md](docs/DEMO_PROMPTS.md)** — prompts where the difference shows
