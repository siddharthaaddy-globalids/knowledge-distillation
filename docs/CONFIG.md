# Configuration reference

Every default lives in [`configs/_base.yaml`](../configs/_base.yaml). Nothing in
`src/kd/` carries a training value, so the YAML you point at is the whole truth
about a run.

A profile opts into the base layer and lists only what it changes:

```yaml
extends: _base.yaml

project:
  name: qwen35-finance-distillation
models:
  teacher: <org>/qwen3.5-2b-finance
  student: Qwen/Qwen3.5-0.8B
```

## Precedence

Lowest to highest:

```
configs/_base.yaml  <  profile  <  KD_* env vars  <  --set  <  explicit flag
```

Every value that differs from the base is echoed at startup with the layer that
won it:

```
 overrides     :
   gkd.lmbda = 0.9  <- --set  (base: 0.5)
   training.max_steps = 500  <- flag  (base: 300)
```

## Overriding

`--set` reaches any key below by dotted path, and is repeatable:

```bash
kd pipeline --config configs/finance.yaml \
  --set training.max_steps=500 \
  --set lora.target_modules='[q_proj, v_proj]' \
  --set limits.max_cost_usd=2.0
```

Values are parsed as YAML, so `500`, `true`, `null`, `1.5` and `[a, b]` all mean
what they would in the file. Scientific notation is accepted in the form people
actually type: `--set training.learning_rate=1e-5` works, even though YAML 1.1
itself requires `1.0e-05`.

Short flags exist for the common ones — `--teacher`, `--student`, `--steps`,
`--device`, `--dtype`, `--lr`, `--lora-r`, `--lora-alpha`, `--lmbda`, `--dataset`,
`--seed`, `--output` — and beat `--set`.

## Typos are errors

An unknown key fails immediately, with a suggestion, and every problem is
reported at once rather than one per run:

```
Configuration is not valid:
  - unknown key 'gkd.lmbdaa'  did you mean 'gkd.lmbda'?
  - unknown key 'gkd.temprature'  did you mean 'gkd.temperature'?
```

A run that silently ignores `traning: {max_steps: 500}` wastes the whole training
budget before anyone notices, which is why this is fatal rather than a warning.

---

## `project`

| Key | Default | Meaning |
|---|---|---|
| `name` | `smollm2-distillation` | Shown in the banner and written into the manifest. |
| `seed` | `42` | Seeds dataset assembly and training. The held-out split is derived from it, which is what lets `kd evaluate` rebuild the exact rows the student never trained on. |
| `runs_dir` | `./runs` | Parent for run bundles: `<runs_dir>/<utc>-<profile>-<sha7>/`. |
| `output_dir` | `null` | Pin the run directory instead of generating a name. Two runs with the same value overwrite each other, so set it only when that is what you want. |

## `models`

| Key | Default | Meaning |
|---|---|---|
| `student` | `HuggingFaceTB/SmolLM2-135M-Instruct` | Hub id, local path, or `s3://` URI. |
| `teacher` | `HuggingFaceTB/SmolLM2-360M-Instruct` | Same. **Must share a tokenizer vocabulary with the student** for standard GKD. |
| `teacher_adapter` | `null` | LoRA adapter merged into the teacher at load time. Prefer this over a merged checkpoint from another framework — those keep that framework's key layout, which plain transformers may not map back. |
| `tokenizer` | `teacher` | `teacher`, `student`, or an explicit id/path. |

## `hardware`

| Key | Default | Meaning |
|---|---|---|
| `device` | `auto` | `auto` \| `cpu` \| `mps` \| `cuda`. `auto` prefers CUDA, then MPS, then CPU. |
| `dtype` | `auto` | `auto` \| `float32` \| `bfloat16` \| `float16`. `auto` is bfloat16 on CUDA, float32 elsewhere. bfloat16 on MPS needs macOS 14+; on CPU it is ignored as slow and often unstable. |
| `threads` | `0` | CPU threads; `0` means all cores. Ignored off CPU. |
| `mps_high_watermark_ratio` | `0.0` | `0.0` lifts the Apple allocator ceiling so a large graph can use the whole unified memory pool. |
| `mps_fallback` | `true` | Route MPS-unsupported ops to CPU instead of raising. |

## `dataset`

| Key | Default | Meaning |
|---|---|---|
| `source` | `HuggingFaceTB/smoltalk` | Hub id, local path, or `s3://` URI. |
| `max_prompt_tokens` | `128` | Prompts at or above this are dropped. Short prompts are what keep on-policy rollouts affordable. |
| `max_total_tokens` | `384` | Prompt + completion ceiling. |
| `validation_size` | `50` | Held-out rows. Each eval pass is a full forward over all of them. |
| `include_synthetic` | `true` | Blend in the generated reasoning / list / constraint prompts. Turn off for a domain-specific run, where they dilute the signal. |
| `domains` | 5 smoltalk domains | One block per domain, below. |

Each entry in `domains`:

| Key | Meaning |
|---|---|
| `name` | Label used in the balance table. |
| `config` | Dataset config name, or `null` for a single-config dataset. |
| `quota` | How many samples to keep. |
| `pool` | How many rows to scan to find them. Long-document domains need a large pool: only ~2% of `smol-summarize` prompts fit under 128 tokens. |
| `format` | `messages` (default) or `alpaca` for instruction/input/output datasets. |
| `instruction_column`, `input_column`, `output_column` | Column names, for `format: alpaca`. |
| `messages_column` | Column holding the chat turns, for `format: messages`. Defaults to `messages`. |
| `data_files` | One filename, or a list, inside a local corpus directory. Use this instead of `config` when `source` is a directory rather than a Hub id — see below. |
| `split` | Overrides the default `train[:pool]` slice. |

`data_files` is what lets several domains be drawn from one prepared corpus:

```yaml
dataset:
  source: ./data/enlibra-curriculum
  domains:
    - {name: curriculum-sft, data_files: sft-1to3hop.jsonl, quota: 937, pool: 1000}
    - {name: curriculum-rl,  data_files: rl-1to2hop.jsonl,  quota: 143, pool: 180}
    - {name: identity,       data_files: identity.jsonl,    quota: 15,  pool: 20}
```

Without it, a directory of files is not addressable per-domain: `load_dataset`
will not accept a path to a single local file, and pointing it at the directory
merges everything into one split — so the per-domain quotas that keep the
calibration set balanced would have nothing to act on. Relative names resolve
against `source`, and a name that does not exist is an error rather than a
skipped domain, because a run that silently trains on two domains out of three
still reports success.

### The arena

A second, different measurement, for datasets that have a **correct answer**.
`kd evaluate` scores fidelity to the *teacher*; the arena scores who is *right*.

| Key | Default | Meaning |
|---|---|---|
| `arena_file` | `null` | Held-out `.jsonl`, one `{"messages": [...]}` per line, assistant turn ending in an `<Answer>` tag. `null` skips the stage. |
| `arena_max_new_tokens` | `2048` | The answer sits *after* the explanation, so too small a value scores as "never answered" rather than as wrong. |
| `arena_elo_rounds` | `25` | Shuffled orderings to average Elo over. Sequential Elo depends on match order; averaging removes that, and the reported spread is the noise floor. |
| `arena_limit` | `null` | Score only the first N questions. For proving the stage runs, not for a real score. |
| `teacher_check_max_new_tokens` | `2048` | How much of the teacher's answer the `teacher-check` stage prints. Not a quality setting — the check reads the first token's distribution — but the text is what a person looks at, and an answer cut off mid-sentence tells them nothing. |

Three players are rated against each other question by question: the **base**
student (no adapter, the control), the **distilled** student, and the
**teacher** (the ceiling). Distilled below base means training hurt; distilled
level with base means the format transferred but the capability did not.

Whichever way it runs — as a pipeline stage or as `kd arena` — it writes
`arena.json` (the numbers, including the hop-wise cosine between the players'
explanations), `arena-transcript.jsonl` (every question and every word each
player said about it) and an HTML report. The standalone command writes them
beside its `--json` path, defaulting to the working directory; `--no-save`
suppresses all three, for a `--limit` smoke check.

## `lora`

| Key | Default | Meaning |
|---|---|---|
| `r` | `32` | Rank. Lower is faster and smaller, at some cost in capacity. |
| `alpha` | `64` | Scaling, conventionally `2 * r`. |
| `dropout` | `0.05` | |
| `target_modules` | 7 Llama-style projections | A list of suffixes, or a single regex string. **Check this against your architecture** — Qwen3.5 is hybrid, and a Llama-style list misses the attention in 18 of its 24 layers. |
| `exclude_modules` | *(unset)* | Optional; same forms. |

## `training`

| Key | Default | Meaning |
|---|---|---|
| `max_steps` | `300` | Optimizer steps. With `batch_size × gradient_accumulation_steps`, this sets how many samples are consumed — dataset size does not drive wall clock. |
| `batch_size` | `1` | Per-device. On Apple unified memory, raise this before reaching for accumulation. |
| `gradient_accumulation_steps` | `4` | |
| `learning_rate` | `3.0e-4` | |
| `lr_scheduler_type` | `cosine` | |
| `warmup` | `0.05` | Float in [0,1) is a ratio of total steps; an int ≥ 1 is an exact count. |
| `max_grad_norm` | `1.0` | |
| `logging_steps` | `1` | |
| `save_steps` | `100` | Checkpoint interval. **This is a cost-control setting**: when a run is stopped by a limit, the last checkpoint is what survives. |
| `save_total_limit` | `2` | |
| `eval_enabled` | `true` | Held-out loss during training. |
| `benchmark_every` | `100` | Generate sample answers this often, so quality drift is visible. |

## `gkd`

| Key | Default | Meaning |
|---|---|---|
| `lmbda` | `0.5` | **The dominant cost.** The fraction of batches where the student generates its own completion before the teacher scores it — that is `max_new_tokens` sequential forward passes versus one. `0.0` is plain off-policy KD: several times faster, but it loses the on-policy correction that makes GKD better. |
| `beta` | `0.5` | Generalized JSD interpolation. |
| `temperature` | `0.7` | |
| `max_new_tokens` | `40` | Second-biggest lever: generation cost is linear in this. |
| `seq_kd` | `false` | |

## `benchmark_prompts`

A list of prompts the student answers at `training.benchmark_every` steps, and
once before and after training. They are not scored - they are there so quality
drift is visible while a run is happening, rather than only in the final report.

Make them representative of the target domain: the finance profiles ask about
IRAs and P/E ratios, because a run whose samples still read like general
chit-chat has not transferred what it was meant to.

## `evaluation`

| Key | Default | Meaning |
|---|---|---|
| `samples` | `50` | Held-out samples scored for fidelity and perplexity. |
| `tasks` | `null` | lm-eval task list, e.g. `"ifeval,arc_easy"`. Needs `uv sync --extra eval`. Slow: three models are scored. |
| `limit` | `null` | Per-task example cap, for a quick look. |
| `gen_similarity` | `0` | Free-running BERTScore prompts. Unlike agreement and KL, this is not teacher-forced, so it sees the student's own drift. Slow. |
| `similarity_model` | `roberta-large` | BERTScore encoder (~1.4 GB on first use). |
| `report_format` | `html` | `html` or `md`. |

## `pipeline`

`stages` is an ordered list of `{name, gate}`. A **gate** stage that fails aborts
the run; a non-gate stage that fails is reported and the run continues, so a
broken report never destroys a good adapter.

| Stage | Gate | What it does |
|---|---|---|
| `preflight` | yes | Resolve config and hardware; fetch any `s3://` inputs. Seconds. |
| `teacher-check` | yes | Loads **only** the teacher: missing weights, NaN scan, coherence. |
| `smoke` | yes | Two real steps. Measures s/step and projects the full run against the limits. |
| `train` | yes | |
| `evaluate` | no | Fidelity and capability against the base student. |
| `report` | no | `report.html` in the run bundle. |
| `publish` | no | Skipped unless `publish.enabled`. |
| `upload` | no | Skipped unless `s3.enabled`. Also runs after a failure, so logs survive. |

Narrow a single run with `--only STAGE`, `--from STAGE` or `--skip STAGE` rather
than editing this list — editing it changes what every run of the profile means.

## `limits`

| Key | Default | Meaning |
|---|---|---|
| `max_runtime_minutes` | `180` | Hard stop. |
| `max_steps` | `null` | Ceiling above `training.max_steps`. The tighter of the two wins; a ceiling above the request can never bind. |
| `max_cost_usd` | `null` | Only meaningful on a rented pod. Inert locally, because nothing is being rented. |
| `confirm_above_usd` | `1.00` | Print the estimate and ask before renting above this. |

Checked twice. The `smoke` stage measures s/step, which turns `max_steps` into a
projected duration and cost — a run that cannot finish inside its limits is
**refused before it starts**, with the fix spelled out:

```
This run would cost about $0.12, over the $0.05 limits.max_cost_usd.
  measured 4.30 s/step over 300 steps at $0.34/hr
  raise the ceiling:  --set limits.max_cost_usd=0.15
  or shorten the run: --set training.max_steps=110
```

In flight, a breach is a **hard stop**: the process ends, a rented pod is
terminated, and no evaluation or report is produced. The artifact you keep is the
last checkpoint — so tighten `training.save_steps` when limits are tight. On a
rented machine the bundle and its checkpoints are synced before the pod dies.

## `publish`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | |
| `repo` | `null` | Publishes `<repo>-lora` (adapter) and `<repo>` (merged). |
| `private` | `true` | |
| `token_env` | `HF_TOKEN` | Name of the variable holding the token — never the token. |

## `s3`

Optional. See [RUNPOD.md](RUNPOD.md) for how it fits with a rented GPU.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | |
| `bucket` | `null` | |
| `prefix` | `kd` | Bundles land at `<prefix>/runs/<run-id>/`. |
| `endpoint_url` | `null` | Set for MinIO, R2, or RunPod volumes. |
| `region` | `null` | |
| `cache_dir` | `~/.cache/kd/s3` | Fetched inputs are cached here, so a second run costs a listing rather than gigabytes. |
| `upload` | `[adapter, logs, report, metrics]` | Also available: `checkpoints`. `manifest.json` and `config.resolved.yaml` always go. |

When enabled, `models.teacher`, `models.student`, `models.teacher_adapter` and
`dataset.source` may be `s3://` URIs. They are fetched in `preflight`, before
anything tries to load them, so a bad bucket fails in seconds rather than after
the dataset build.

Credentials come from the environment (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_PROFILE`, or an instance role) — never from this
file, which is committed.

## `runpod`

Optional. See [RUNPOD.md](RUNPOD.md).

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | |
| `gpu_type` | `RTX A4000` | **Rented exactly, never substituted.** If unavailable, the launcher stops and asks. |
| `gpu_count` | `1` | |
| `cloud_type` | `COMMUNITY` | `COMMUNITY` or `SECURE`. |
| `spot` | `true` | Interruptible, roughly half price. |
| `max_price_per_hour` | `0.60` | Never rent above this, whatever is available. |
| `image` | `null` | `ghcr.io/<org>/kd:<sha>`. |
| `volume_gb` | `40` | Mounted at `/workspace`; the HF cache lives here and survives the pod. |
| `container_disk_gb` | `20` | |
| `terminate_on_exit` | `true` | Leave this on. |
| `api_key_env` | `RUNPOD_API_KEY` | Name of the variable, never the key. |
| `extra_env` | `null` | Extra variables for the pod. `HF_TOKEN` and `AWS_*` are forwarded automatically when set locally. |

## `KD_*` environment variables

For retargeting a downloaded runner without editing YAML. They sit below `--set`
and explicit flags.

`KD_TEACHER_MODEL` `KD_STUDENT_MODEL` `KD_TEACHER_ADAPTER` `KD_TOKENIZER`
`KD_DATASET` `KD_OUTPUT_DIR` `KD_RUNS_DIR` `KD_DEVICE` `KD_DTYPE` `KD_THREADS`
`KD_MAX_STEPS` `KD_BATCH_SIZE` `KD_GRAD_ACCUM` `KD_LEARNING_RATE` `KD_LORA_R`
`KD_LORA_ALPHA` `KD_LMBDA` `KD_BETA` `KD_MAX_NEW_TOKENS` `KD_SEED`
`KD_EVAL_TASKS` `KD_EVAL_SAMPLES` `KD_MAX_RUNTIME_MINUTES` `KD_MAX_COST_USD`

An invalid value is an error, not a silently ignored setting:

```
KD_MAX_STEPS='not-a-number' is not a valid int (it sets training.max_steps)
```
