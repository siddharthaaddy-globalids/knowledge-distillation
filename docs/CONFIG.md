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

Profiles are grouped one directory deep by what they distil — `configs/enlibra/`,
`configs/smollm/`, `configs/qwen/`. One profile describes both the training and
the evaluation of a pair. `extends` resolves relative to the profile first and
then to `configs/`, so `extends: _base.yaml` works from any group and
`extends: ../enlibra/enlibraQ25-3B.yaml` reaches across them.

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
kd pipeline --config configs/qwen/finance.yaml \
  --set training.max_steps=500 \
  --set lora.target_modules='[q_proj, v_proj]' \
  --set limits.max_cost_usd=2.0
```

Values are parsed as YAML, so `500`, `true`, `null`, `1.5` and `[a, b]` all mean
what they would in the file. Scientific notation is accepted in the form people
actually type: `--set training.learning_rate=1e-5` works, even though YAML 1.1
itself requires `1.0e-05`.

Short flags exist for the common ones — `--teacher`, `--teacher-adapter`,
`--teacher-base`, `--student`, `--steps`, `--device`, `--dtype`, `--lr`,
`--lora-r`, `--lora-alpha`, `--lmbda`, `--ce-alpha`, `--dataset`, `--seed`,
`--output` — and beat `--set`.

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
| `teacher` | `HuggingFaceTB/SmolLM2-360M-Instruct` | A merged checkpoint, a base model (with `teacher_adapter` beside it), **or a LoRA adapter alone** — a directory, `s3://` prefix or Hub repo holding `adapter_config.json`. Preflight recognises an adapter, reads the base it records, and rewrites this into base + adapter before anything loads. **Must share a tokenizer vocabulary with the student** for standard GKD. |
| `teacher_adapter` | `null` | LoRA adapter merged into the teacher at load time. Prefer base + adapter over a merged checkpoint from another framework — those keep that framework's key layout, which plain transformers may not map back. |
| `teacher_base` | `null` | The stock model the teacher was fine-tuned from, scored by the evaluation as the `teacher-base` player. Implied when the teacher is base + adapter (it is the base) or was given as an adapter (it is what the adapter records — this key overrides that record). A merged checkpoint cannot say what it was built from, so name it here, or the player is skipped with a note. |
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
| `engine` | `vllm` | How completions are generated: `vllm` (the whole set batched, CUDA and Linux only) or `hf` (transformers, one at a time, runs anywhere). **No profile overrides this** — see below. |
| `vllm` | see below | Engine settings, used only when `engine: vllm`. |
| `teacher_check_max_new_tokens` | `2048` | How much of the teacher's answer the `teacher-check` stage prints. Not a quality setting — the check reads the first token's distribution — but the text is what a person looks at, and an answer cut off mid-sentence tells them nothing. |

#### `engine: vllm`

The arena is the longest thing the pipeline does. Five players over ~140
questions at `arena_max_new_tokens: 8192` is 700 sequential generations under
`hf`, and one batch per player under `vllm` — which is the workload continuous
batching exists for, since every prompt is known before the first token.

```yaml
evaluation:
  engine: vllm
  vllm:
    gpu_memory_utilization: 0.85   # lower it if a player is killed during load
    max_model_len: null            # null sizes it to the longest prompt + the ceiling
    enforce_eager: true            # skip CUDA graph capture: less memory, less variance
    tensor_parallel_size: 1        # cards per player
```

**Every profile uses it. None overrides it.** That uniformity is the point: the
two engines are not bit-identical, so if the laptop profiles scored under `hf`
and the real ones under `vllm`, a smoke run would stop being a rehearsal of the
real run and become a rehearsal of a different measurement.

Three things that follow from it:

- **CUDA and Linux only.** vLLM publishes no Windows wheel, and it is an
  optional extra (`serve`) rather than a dependency — it needs a CUDA build of
  torch, which the project otherwise pins to the CPU index. `docker/Dockerfile.cuda`
  installs it; build with `--build-arg WITH_VLLM=0` to leave it out. A machine
  without it fails at **preflight**, before any weights are fetched, rather than
  falling back — the same rule `run.sh` states for profiles.
- **To score on a Mac or a Windows box, ask on the command line**, where it is a
  visible choice rather than a hidden default:

  ```bash
  kd arena --engine hf --config configs/enlibra/enlibraQ3-8B-smoke.yaml
  kd eval --set evaluation.engine=hf --config configs/smollm/smoke.yaml
  ```

  The engine is recorded in `arena.json` and shown in the report, so a run
  scored that way says so. Do not compare it against one scored under `vllm`:
  both decode greedily from identical token ids — prompts are rendered and
  tokenised by the arena's own tokenizer and the *ids* are what reach vLLM, so a
  chat template cannot drift between them — but not through the same kernels,
  and two floating-point paths through an 8B model do not agree on every token.
- **The distilled player gets merged first.** vLLM cannot be handed base +
  adapter, so the merge lands in `~/.cache/kd/merged/` — outside the run bundle,
  which is uploaded wholesale — and is reused by later runs against the same
  adapter.

The players (`evaluation.players`) are rated against each other question by
question: the **base** student (no adapter, the control), the **distilled**
student, the **teacher-base** (the stock model the teacher was fine-tuned from,
when known) and the **teacher** (the ceiling). Distilled below base means
training hurt; distilled level with base means the format transferred but the
capability did not; teacher level with teacher-base means the fine-tune gave the
teacher nothing to pass on.

The headline is **closeness to the teacher**, not accuracy: how often each
student gave the teacher's answer. It is written to `arena.json` under
`closeness` and to `metrics.json`, and it is the number the report leads with.
Accuracy and Elo are reported beneath it as context, and beneath those the
breakdown by reasoning depth — three tables, because accuracy, answer rate and
accuracy-when-answered fail apart and one table cannot show which moved.

Whichever way it runs — as a stage of `kd eval` or as `kd arena` — it writes
`arena.json` (the numbers, including the hop-wise cosine between the players'
explanations), `arena-transcript.jsonl` (every question and every word each
player said about it) and an HTML report. The standalone command writes them
beside its `--json` path, defaulting to the working directory; `--no-save`
suppresses all three, for a `--limit` smoke check.

## `lora`

| Key | Default | Meaning |
|---|---|---|
| `r` | `128` | Rank. The one LoRA-specific distillation recipe (Thinking Machines, 2025) used 128. Lower is faster and smaller, at some cost in capacity. |
| `alpha` | `256` | Scaling, conventionally `2 * r`. |
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
| `beta` | `0.9` | Generalized JSD interpolation: `0` is forward KL (cover everything the teacher considers possible), `1` is reverse KL (commit to the teacher's modes). The papers favour reverse-leaning values (`0.9`) for a student much smaller than its teacher and for instruction-shaped data. |
| `ce_alpha` | `0.2` | Cross-entropy weight: the loss is `(1 - ce_alpha) * JSD + ce_alpha * CE`, with CE the ordinary SFT term on the gold token. `0` is the pure divergence the papers train with; `1` is SFT with the teacher ignored. Applied only to gold text (curriculum or `seq_kd` teacher completions), never to the student's own on-policy rollouts. |
| `temperature` | `1.0` | Sampling temperature for the student's own completions; inert at `lmbda: 0`. The on-policy recipes all use `1.0`. |
| `max_new_tokens` | `40` | Second-biggest lever: generation cost is linear in this. Inert at `lmbda: 0`. |
| `seq_kd` | `false` | `true` has the teacher rewrite each completion before the student trains on it. Never beats on-policy data in the literature and costs a teacher generation per sample. |

What each value does, and what the papers found: [papers/README.md](papers/README.md).
The report's **How it was trained** section (`evaluation.report_training`) repeats
the explanation next to the values a run actually used.

## `benchmark_prompts`

A list of prompts the student answers at `training.benchmark_every` steps, and
once before and after training. They are not scored - they are there so quality
drift is visible while a run is happening, rather than only in the final report.

Make them representative of the target domain: the finance profiles ask about
IRAs and P/E ratios, because a run whose samples still read like general
chit-chat has not transferred what it was meant to.

## `evaluation`

Evaluation is its own pipeline, `kd eval`, walked against an adapter that
already exists. It writes into a directory of its own **inside the adapter's
bundle** — `runs/<train-run>/evaluation/<name>-<YYYY-MM-DD>-<HHMM>/` — and to
the same path beside the adapter in the bucket, so one adapter accumulates any
number of evaluations and nothing is ever overwritten.

| Key | Default | Meaning |
|---|---|---|
| `adapter` | `null` | What `kd eval` scores: an adapter directory, a file inside one, or an `s3://` URI. `null` means `--adapter` on the command line, else the newest adapter under `project.runs_dir`. |
| `name` | `null` | What the evaluation is for — `full`, `quick`, `after-parser-fix`. Leads the directory name. `null` uses the profile name. |
| `after_training` | `false` | `true` runs the evaluation **inside** `kd pipeline`, as its `evaluation` stage, right after training — the pod already has the teacher resident. Same output layout either way. |
| `players` | `[base, distilled, distilled-w4a16, teacher-base, teacher]` | Who is scored. `distilled-w4a16` is skipped when nothing has been packed and `teacher-base` when the base is unknown; drop `teacher` and `teacher-base` to compare the students without loading the teacher. |
| `stages` | preflight, evaluate, arena, report, upload | What `kd eval` walks; the same `{name, gate}` shape as `pipeline.stages`. |
| `samples` | `50` | Held-out samples scored for fidelity and perplexity. |
| `report_format` | `html` | `html` or `md`. |
| `report_training` | `true` | Include **How it was trained** in the report: each `gkd` knob with its value, what it does, and what it meant at that value, plus the loss formula, steps, batch, LR and LoRA shape. `false` hides the section. |

## `pipeline`

`stages` is an ordered list of `{name, gate}`: what `kd pipeline` walks. A
**gate** stage that fails aborts the run; a non-gate stage that fails is
reported and the run continues, so a broken report never destroys a good
adapter.

| Stage | Gate | What it does |
|---|---|---|
| `preflight` | yes | Resolve config and hardware; fetch any `s3://` inputs; split a teacher given as a LoRA adapter into base + adapter. Seconds. |
| `teacher-check` | yes | Loads **only** the teacher: missing weights, NaN scan, coherence. |
| `smoke` | yes | Two real steps. Measures s/step and projects the full run against the limits. |
| `train` | yes | |
| `evaluation` | no | The evaluation pipeline (`evaluation.stages`), inside the run. Skipped unless `evaluation.after_training`. |
| `publish` | no | Skipped unless `publish.enabled`. |
| `upload` | no | Skipped unless `s3.enabled`. Also runs after a failure, so logs survive. |

The evaluation pipeline that `kd eval` walks:

| Stage | Gate | What it does |
|---|---|---|
| `preflight` | yes | Fetch the teacher and the adapter; validate the players; say where the results go. |
| `evaluate` | no | Fidelity and capability, token by token, against the teacher. |
| `arena` | no | Accuracy and Elo on the answer key. Skipped unless `evaluation.arena_file`. |
| `report` | no | `report.html` in the evaluation directory. |
| `upload` | no | To `<bundle>/evaluation/<id>/` in the bucket. Skipped unless `s3.enabled`. |

Narrow a single run with `--only STAGE`, `--from STAGE` or `--skip STAGE` rather
than editing these lists — editing them changes what every run of the profile means.

## `limits`

| Key | Default | Meaning |
|---|---|---|
| `max_runtime_minutes` | `null` | Hard stop. Off by default — see below. |
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

## `quantization`

Packs the distilled student's weights to 4 bits and writes a
**compressed-tensors** checkpoint that vLLM loads natively — the artifact that
actually gets deployed. Runs as the `quantize` stage, after `train` and before
`evaluation`.

```yaml
quantization:
  enabled: true
  scheme: W4A16
  group_size: 128
  ignore: [lm_head]
  calibration_samples: 128
  max_seq_length: 2048
  dampening: 0.01
  calibration_file: ./data/enlibra-neuroscience/sft-1to3hop.jsonl
  output_dir: null
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Off, and the whole stage reports "not configured". |
| `scheme` | `W4A16` | INT4 weights, FP16 activations. |
| `group_size` | `128` | Weights quantised in groups of this many, each with its own scale. Must divide the hidden size and the FFN width. |
| `ignore` | `[lm_head]` | Modules left at full width. See below. |
| `calibration_samples` | `128` | Sequences GPTQ measures activation statistics against. |
| `max_seq_length` | `2048` | Calibration sequences are truncated here. Lower it if the Hessian pass runs out of memory. |
| `dampening` | `0.01` | Added to the Hessian diagonal before inverting. Raise it if the pass fails on a singular matrix. |
| `calibration_file` | `null` | **The training `.jsonl`.** `null` falls back to `evaluation.arena_file`. |
| `output_dir` | `null` | `null` uses `~/.cache/kd/quantized/`, keyed by adapter and scheme. |

Three things worth knowing:

- **Why W4A16 and not W8A8.** Single-stream decode is memory-bandwidth bound —
  the time goes into moving weights, not multiplying them — so at 8B the bytes
  saved by 4-bit weights outweigh the cost of unpacking, and W4A16 decodes
  *faster* than bf16. W8A8 wins at large batch, where the arithmetic dominates.
  The arena is the large-batch case and would generate faster under W8A8; that
  is the wrong thing to optimise, because what ships answers one person at a time.
- **Why `lm_head` is left alone.** The output projection is vocabulary × hidden
  — 151,669 × 4,096, about 1.2 GiB — and it is the one matrix whose error lands
  straight on the logits with no later layer to absorb it. Keeping it at full
  width is why the checkpoint comes out around **2.7× smaller** rather than the
  ~3.5× four-bit weights would suggest.
- **Calibrate on the training rows, not on generic text.** GPTQ tunes its
  rounding for the distribution it is shown. Web text optimises the rounding for
  a distribution this model will never be asked to produce.

The packed student then plays the arena as `distilled-w4a16`, beside the dense
`distilled`. Both play because **what quantization cost is a difference**, and
one column cannot carry a difference: with only the packed student scored,
"three points short of the teacher" cannot be told apart from "distillation fell
three short, quantization cost nothing". The report's *What W4A16 cost* table is
the subtraction, measured on identical tokens and identical questions in one run.

Needs `llmcompressor`, a CUDA-only optional extra for the same reason vLLM is.

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
| `prefix` | `kd` | Bundles land at `<prefix>/runs/<run-id>/`, where the run id is `<profile>-<YYYY-MM-DD>-<HHMM>` in UTC, e.g. `enlibraQ25-3B-2026-09-10-1416` — the same name as the directory under `runs/`. |
| `endpoint_url` | `null` | Set for MinIO, R2, or RunPod volumes. |
| `region` | `null` | |
| `cache_dir` | `~/.cache/kd/s3` | Fetched inputs are cached here, so a second run costs a listing rather than gigabytes. |
| `upload` | `[adapter, logs, report, metrics, evaluation]` | Also available: `checkpoints`. `evaluation` is every scoring done inside the run; one done later with `kd eval` uploads its own directory to the same place. `manifest.json` and `config.resolved.yaml` always go. |

When enabled, `models.teacher`, `models.teacher_adapter`, `models.teacher_base`,
`models.student`, `dataset.source` and `evaluation.adapter` may be `s3://` URIs. They are fetched in `preflight`, before
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

`KD_TEACHER_MODEL` `KD_STUDENT_MODEL` `KD_TEACHER_ADAPTER` `KD_TEACHER_BASE`
`KD_TOKENIZER` `KD_DATASET` `KD_OUTPUT_DIR` `KD_RUNS_DIR` `KD_DEVICE` `KD_DTYPE`
`KD_THREADS` `KD_MAX_STEPS` `KD_BATCH_SIZE` `KD_GRAD_ACCUM` `KD_LEARNING_RATE`
`KD_LORA_R` `KD_LORA_ALPHA` `KD_LMBDA` `KD_BETA` `KD_CE_ALPHA` `KD_MAX_NEW_TOKENS`
`KD_SEED` `KD_EVAL_TASKS` `KD_EVAL_SAMPLES` `KD_EVAL_ADAPTER` `KD_EVAL_NAME`
`KD_MAX_RUNTIME_MINUTES` `KD_MAX_COST_USD`

An invalid value is an error, not a silently ignored setting:

```
KD_MAX_STEPS='not-a-number' is not a valid int (it sets training.max_steps)
```
