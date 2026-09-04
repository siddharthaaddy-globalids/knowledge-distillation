# How this knowledge distillation works

Two passes over the same subject. **Part 1** is the plain-language version: what
distillation is, why on-policy distillation is different, and what this repository
actually produces. **Part 2** is the technical version: the exact loss, the exact
training step, the data pipeline, the memory and time costs, and the failure modes
this codebase defends against.

For *running* it, see [README.md](README.md).

---

# Part 1 — The high-level story

## The problem

A 2-billion-parameter model that answers finance questions well is expensive to
serve: it needs more memory, more compute per token, and more latency budget than a
0.8B model. A 0.8B model is cheap to serve but answers worse.

Knowledge distillation asks: **can the small model be taught to imitate the big
one closely enough that you can serve the small one instead?**

Not on everything — a 135M model will never hold the world knowledge of a 1.7B
model. But on a *bounded* task ("answer finance questions in this house style",
"follow list-formatting instructions exactly"), the gap is often mostly behavioural,
and behaviour transfers.

## Three vocabulary words

| Term | Meaning here |
|---|---|
| **Teacher** | The big, capable model. Frozen — never trained. Its job is to be the target. |
| **Student** | The small model being trained. |
| **Adapter (LoRA)** | The only thing that actually gets trained. A small set of extra weights bolted onto the student. The student's original weights are untouched. |

The output of a run is *not* a new model file. It is an adapter: roughly 20–40 MB,
loaded on top of the stock student at inference time. You can keep the stock model
and swap adapters per domain, or merge the adapter in permanently for deployment
(`--publish` does both).

## Why not just fine-tune on the teacher's answers?

That is the obvious approach, and it is called **sequence-level KD**: have the
teacher write answers, then train the student on them as if they were ground truth.
It works, and it is what `seq_kd: true` does in this repo.

It has a specific weakness. You are training the student on text the *teacher*
would produce, but at inference time the student produces its own text. Once the
student's output drifts even slightly from what the teacher would have said, it is
in a situation it was never trained on — and errors compound token by token. This
is *exposure bias*: the training distribution and the deployment distribution don't
match.

## What on-policy distillation does differently

The core idea of **GKD (Generalized Knowledge Distillation)** — the method this
repo implements, from [arXiv:2306.13649](https://huggingface.co/papers/2306.13649)
— is to close that gap directly:

> Let the **student** write the answer. Then ask the **teacher**, token by token,
> *"what would you have said here?"* Train the student to close the difference.

Because the student generated the text, the training data is exactly the
distribution the student actually visits at inference time. The teacher acts as a
live corrector on the student's own trajectory, not as an author of text the
student would never have written.

The `lmbda` setting controls the mix:

| `lmbda` | Behaviour |
|---|---|
| `0.0` | Pure off-policy. Only dataset text is used. Fast; this is classic KD. |
| `0.5` | Half the steps use student-generated text (this repo's default). |
| `1.0` | Fully on-policy. Every step trains on the student's own rollouts. |

Higher `lmbda` gives more of the on-policy correction, and costs proportionally
more time — generating a completion is many sequential forward passes, whereas
scoring existing text is one.

## What "close the difference" means

At each position in the text, both models produce a probability distribution over
the entire vocabulary — "the next token is 32% `the`, 11% `a`, 4% `interest`, …".

The training signal is the **divergence** between the teacher's distribution and
the student's, at every position. Minimise it, and the student's next-token
predictions come to resemble the teacher's.

This is much richer than training on the teacher's chosen token alone. The teacher
picking `interest` says one thing; the teacher putting 40% on `interest`, 30% on
`returns`, 25% on `yield` and nearly nothing on the other 248,000 tokens says a
great deal more. The distribution encodes the teacher's uncertainty and its sense
of which alternatives are reasonable. Those "dark knowledge" gradients are why
distillation beats plain fine-tuning on the same amount of text.

## Why only a small adapter is trained

Training all 800 million student parameters would need optimizer state for every
one of them — roughly 3× the model size in extra memory — and would risk
catastrophically overwriting what the student already knows.

**LoRA** instead freezes the student and injects a pair of small low-rank matrices
beside each targeted projection. With rank 32, that is on the order of 1% of the
parameters. Memory drops by an order of magnitude, the original capabilities are
structurally preserved, and the result is a portable file rather than a new
multi-gigabyte checkpoint.

## Does it work?

Honestly: partially, and that is worth stating plainly. From
[DEMO_PROMPTS.md](DEMO_PROMPTS.md), measured on the 360M → 135M SmolLM2 run:

* **Where it wins** — instruction-shape compliance. The base 135M cannot suppress a
  "Here is the list of…" preamble or honour an exact item count; asked for three
  items it announces five. The distilled 135M starts at `1.` and gives three.
  Reproducible 3/3 across runs.
* **Where it does not** — factual accuracy and reasoning. The distilled model is
  fluent and wrong on science questions; algebra is inside sampling noise; one
  "answer in a single word" prompt actively regressed.

That is the expected shape of the result. A 300-step run on ~1,800 samples
transfers *style and format compliance*, which is behavioural, and does not
transfer *knowledge*, which lives in the weights it isn't training. Knowing which
of those you need is the difference between distillation being the right tool and
the wrong one.

## The five things that go wrong

The codebase is largely shaped by defences against these:

1. **The teacher is silently broken.** `transformers` does not raise when a
   checkpoint's tensor names don't match the architecture it built — it randomly
   initialises what it couldn't map and loads anyway. A partly-random teacher emits
   near-uniform token salad, and the student faithfully learns to reproduce it.
   The run "succeeds". → `check_teacher.py`, `--check-teacher`.
2. **The tensor names are wrong but the weights are fine.** Merged exports from
   MLX/unsloth keep that framework's key layout. → `fix_teacher.py`.
3. **The adapter is in the wrong format.** unsloth on Apple Silicon runs on MLX and
   writes MLX-format adapters that PEFT cannot load. →
   `convert_mlx_adapter.py`.
4. **Teacher and student don't share a vocabulary.** Token-level divergence between
   two different vocabularies is meaningless. → documented per profile, and
   asserted at evaluation time by `evaluate.py`; training itself still does not
   check it (see README §14).
5. **The LoRA targets miss most of the model.** Qwen3.5 is a hybrid architecture
   where 18 of 24 layers use linear attention with entirely different module names.
   A Llama-style target list leaves attention untrained in three quarters of the
   network. → per-profile `target_modules`.

---

# Part 2 — The technical mechanics

## Component map

```
distill.sh                    thin runner: env bootstrap, pinned checkout, KD_* export
  └── train_scaled.py         the pipeline
        ├── kd_config.py      DEFAULTS -> YAML -> KD_* env -> CLI  +  device/dtype resolution
        ├── datasets          multi-domain build, length filter, dedup, train/val split
        ├── transformers      AutoModelForCausalLM x2 (teacher frozen, student trainable)
        ├── peft              LoRA injection into the student
        └── trl.experimental.gkd
              ├── GKDConfig    training args + lmbda/beta/temperature/max_new_tokens
              └── GKDTrainer   the on-policy training step and the JSD loss
```

## The loss: generalized Jensen–Shannon divergence

Let `p_T` be the teacher's next-token distribution and `p_S` the student's, both
after temperature scaling by `τ` (`gkd.temperature`, default 0.7). Define the
mixture

```
M = β · p_T  +  (1 − β) · p_S
```

The loss at each unmasked position is

```
L = β · KL(p_T ‖ M)  +  (1 − β) · KL(p_S ‖ M)
```

`beta` selects the divergence family, and the endpoints are special-cased:

| `beta` | Loss | Character |
|---|---|---|
| `0.0` | `KL(p_T ‖ p_S)` — forward KL | **Mode-covering.** The student must put mass wherever the teacher does. Classic KD. |
| `0.5` | symmetric JSD | Balanced. This repo's default. |
| `1.0` | `KL(p_S ‖ p_T)` — reverse KL | **Mode-seeking.** The student may cover one mode sharply and ignore the rest. |

Implementation notes that matter if you read the TRL source:

* The mixture is computed in log-space via `logsumexp` for numerical stability.
* PyTorch's `F.kl_div(input, target)` takes its arguments in the opposite order to
  the mathematical convention, so the calls look swapped relative to the paper.
* Positions where `labels == -100` (prompt tokens and padding) are masked out. Only
  completion tokens contribute.
* Reduction normalises by the *global* count of valid tokens in the batch, so
  gradient accumulation produces the same loss scale as a single large batch.

Temperature is applied to both sets of logits before the softmax. Below 1.0 it
sharpens both distributions, concentrating the gradient signal on the tokens the
teacher is actually confident about.

## The training step

Per optimizer step, `GKDTrainer.training_step` does:

```
r = random()                            # once per step
if r <= lmbda:                          # ON-POLICY branch
    student generates a completion from the prompt (no grad)
    inputs are replaced by that rollout; prompt positions labelled -100
elif seq_kd:                            # SEQUENCE-KD branch
    teacher generates the completion instead
# else: use the dataset's reference completion as-is

student forward   -> student_logits     (with grad)
teacher forward   -> teacher_logits     (no grad, frozen, eval mode)
loss = generalized_jsd(student_logits, teacher_logits, labels, beta, temperature)
backward -> LoRA parameters only
```

Three consequences worth internalising:

* **`lmbda` is sampled per step, not per sample.** The whole batch is on-policy or
  it isn't.
* **The rollout is generated fresh every on-policy step, and thrown away.** This is
  where the wall-clock goes.
* **Only the completion is supervised.** The prompt is masked, so the student is
  never trained to predict the user's own text back.

### The cost model

| Branch | Forward passes per step |
|---|---|
| Off-policy | 1 student + 1 teacher |
| On-policy | `max_new_tokens` sequential student passes (generation) + 1 student + 1 teacher |

At `max_new_tokens = 40`, an on-policy step is roughly 20× the compute of an
off-policy one. Expected cost per step is therefore approximately

```
(1 − lmbda) · C_fwd   +   lmbda · (max_new_tokens · C_gen + C_fwd)
```

which is why `configs/qwen-poc.yaml` sets `lmbda: 0.25` and `max_new_tokens: 24`:
those two numbers, not the model size and not the dataset size, dominate the
runtime of a small run.

## The data pipeline

`build_datasets()` in `train_scaled.py` assembles a **balanced multi-domain**
calibration set rather than reading one dataset end to end.

**Per-domain collection** (`collect_domain`): each entry in `dataset.domains`
names a source config, a `quota` (samples to keep) and a `pool` (rows to scan).
Two input layouts are supported:

* `format: messages` (default) — conversational, e.g. `HuggingFaceTB/smoltalk`.
* `format: alpaca` — `instruction` / `input` / `output` columns, e.g.
  `gbharti/finance-alpaca`. A non-empty `input` is appended to the instruction,
  matching the convention those datasets were written against.

**Exchange selection** (`select_exchange`): for multi-turn conversations, the
*longest* prefix ending in an assistant turn that fits the token budget is chosen —
not the first. This matters: every `everyday-conversations` dialogue opens with
"Hi"/"Hi there", so first-turn extraction produces near-identical prompts that then
collapse under deduplication.

**Length budget** (`within_budget`): both limits are checked against the exact
strings the collator will see, by rendering the chat template twice — once for
`messages[:-1]` with a generation prompt (the prompt), once for the full turn list
(prompt + completion). `max_prompt_tokens` is what keeps on-policy rollouts
affordable, since generation always starts from the prompt.

**Deduplication:** keyed on the first 200 characters of the final user turn, shared
across all domains.

**Synthetic samples:** when `include_synthetic: true`, generated
reasoning/instruction-following prompts are blended in for format retention. The
finance profiles disable this — general science and maths prompts would dilute a
domain-specific run.

**Split:** shuffle with `project.seed`, take `validation_size` (capped at 10% of
the corpus) as held-out eval, the rest as train. The bookkeeping `domain` column is
dropped before the trainer sees the data.

The console prints per-domain pass rates (`kept 240/240 (scanned 14000, 1.7% pass)`)
and the final domain balance, which is how you diagnose a `pool` that is too small.

## LoRA injection

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
    target_modules=lora.target_modules,   # list of suffixes, or a regex string
)
student = get_peft_model(student, lora_config)
```

Each targeted `Linear` of shape `(out, in)` gains `A ∈ R^{r×in}` and
`B ∈ R^{out×r}`, and its output becomes `Wx + (α/r)·BAx`. Only `A` and `B` carry
gradients. At r=32 across all attention and MLP projections this is on the order of
1% of the model's parameters; the run prints the exact count via
`print_trainable_parameters()`.

`target_modules` accepts a **regex string** as well as a suffix list. That matters
for checkpoints like Qwen3.5 that contain a vision tower and an MTP head sharing
module names with the language model — a bare suffix list would inject adapters
into those too. (In practice `AutoModelForCausalLM` drops those components, but the
option is there, and `exclude_modules` is also plumbed through.)

## Teacher pre-flight (`verify_teacher` / `check_teacher.py`)

Three independent checks, all of which must pass:

1. **Coverage.** `from_pretrained(..., output_loading_info=True)` returns
   `missing_keys`. After filtering legitimately-absent entries
   (`rotary_emb.inv_freq`, `.attn_bias`, a tied `lm_head.weight`), anything left is
   a real parameter that was *fabricated* rather than loaded. Any such key is
   fatal.
2. **Numerical health.** Every parameter tensor is scanned for NaN/Inf — the
   signature of a bad merge or an overflowed save.
3. **Behaviour.** A probe prompt is run through the chat template and one forward
   pass. From the final-position logits:

   ```
   top_p    = max(softmax(logits))
   entropy  = −Σ p log p
   uniform  = log(vocab_size)          # ≈ 12.4 for a 248k vocab
   ```

   A healthy instruct model is confident: high `top_p`, entropy far below uniform.
   The failure condition is `entropy > 0.80 · uniform` **or** `top_p < 0.02` — a
   near-uniform distribution over ~248,000 tokens, which is precisely what a
   randomly-initialised head produces and what shows up downstream as multilingual
   token salad.

`check_teacher.py` exits `2` on failure so shell scripts can gate on it.
`train_scaled.py` runs the same logic inline and raises `SystemExit(2)` unless
`--allow-bad-teacher` is passed.

## Checkpoint repair (`fix_teacher.py`)

For the case where the weights are correct but the *names* aren't.

1. Build the architecture from the checkpoint's own `config.json`, and ask
   `transformers` which keys it reported as **missing** — that is the ground-truth
   "wanted" set.
2. Read the actual tensor names from the safetensors file — the "have" set.
3. `build_mapping(have, want)` searches candidate prefix rewrites (e.g.
   `language_model.model.` → `model.`) and picks the one that resolves the most
   wanted keys.
4. `reconcile_shape` handles axis-order differences that are *lossless*: a
   depthwise conv exported as `(channels, kernel, 1)` where the architecture wants
   `(channels, 1, kernel)` is a reshape, not a transpose, and is applied silently.
   A genuine transpose is flagged rather than applied quietly, and irreconcilable
   shapes are refused.
5. Components a text-only causal LM never instantiates (`vision_tower.`, `visual.`,
   `model.visual.`) are dropped — dead payload otherwise, and a large file saving.
6. `config.json` is copied **verbatim**, so `transformers` builds exactly the same
   architecture it built when it reported those missing keys.

No GPU, no retraining. `--dry-run` prints the mapping and writes nothing.

## MLX → PEFT adapter conversion (`convert_mlx_adapter.py`)

unsloth on Apple Silicon runs on MLX, so the adapters it writes are MLX-LM
artifacts throughout:

| MLX-LM | PEFT |
|---|---|
| `adapters.safetensors` | `adapter_model.safetensors` |
| `{rank, scale, dropout, keys}` | `{r, lora_alpha, lora_dropout, target_modules, task_type, peft_type}` |
| `<module>.lora_a` — `(in, r)` | `<module>.lora_A.weight` — `(r, in)` |
| `<module>.lora_b` — `(r, out)` | `<module>.lora_B.weight` — `(out, r)` |
| `language_model.model.layers.N…` | `model.layers.N…` (the *live* module tree) |

Both LoRA factors are transposed, module paths are rewritten using the prefix
mapping that actually resolves against the loaded architecture (derived, not
assumed), and the scaling convention is reconciled: MLX applies `scale` directly to
the LoRA branch while PEFT applies `α/r`, so the converter writes
`lora_alpha = round(scale × r)`. The adapter computes the same function afterwards
— nothing is retrained or approximated.

## Hardware resolution (`kd_config.py`)

One function, `resolve_device()`, is the only platform-aware code in the project.

**Device:** `auto` → CUDA if available, else MPS if available *and built*, else
CPU. An explicit request for an unavailable device warns and falls back rather than
crashing.

**Apple Silicon:** sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (unsupported ops run on CPU
instead of raising) and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (lifts the
allocator's working-set ceiling so a large graph can use the whole unified memory
pool). Unified memory is why a Mac can host a much larger teacher than a discrete
GPU of nominally equivalent size, and why `configs/mac.yaml` prefers real batching
(`batch_size: 4`) over gradient accumulation — batching is nearly free in transfer
terms there, accumulation always costs wall clock.

**dtype:** `auto` → bfloat16 on CUDA where supported (float16 otherwise), float32
everywhere else. Two guarded downgrades: bfloat16 on MPS requires macOS ≥ 14.0
(below that transformers doesn't permit it; and on M1/M2 Metal emulates it over
fp32, so it saves memory without being faster), and any non-float32 request on CPU
is downgraded because it is slow and numerically unstable there. Each downgrade
prints a `!!` note in the startup banner.

**Trainer flags:** `bf16`/`fp16` are set only on CUDA. On MPS the autocast and
grad-scaler path differs from CUDA's, so mixed precision is carried by the model
weights' dtype instead of by the Trainer. `dataloader_pin_memory` is CUDA-only.

## Telemetry

`ScaledDistillationCallback` prints one line per optimizer step:

```
 [step  142/300] jsd=1.8241 run20=1.9033 grad=0.4417 lr=2.31e-04  12.4s/step  eta=32m18s
```

* `jsd` — this step's loss; `run20` — mean over the last 20 steps, which is what to
  watch, since single-step JSD is noisy under on-policy sampling.
* `grad` — gradient norm before clipping at `max_grad_norm`.
* Held-out loss is printed at every eval (`eval_steps = save_steps`); it is the
  generalization signal, as training loss alone can improve while the model
  overfits the calibration set.
* `BenchmarkCallback` generates against `benchmark_prompts` every
  `benchmark_every` steps, so quality drift is visible during the run rather than
  only at the end. Baseline generations are printed once *before* training for
  comparison.

## Outputs

```
<output_dir>/
  checkpoint-100/        full trainer state (optimizer, scheduler, RNG) — resumable, large
  checkpoint-200/
  final_adapter/
    adapter_config.json
    adapter_model.safetensors     <- the trained LoRA weights
    tokenizer.json / tokenizer_config.json / chat_template.jinja
```

Load it:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = PeftModel.from_pretrained(base, "./distilled_smollm_scaled/final_adapter")
tok = AutoTokenizer.from_pretrained("./distilled_smollm_scaled/final_adapter")
```

`publish_model.py` produces both distribution forms: the adapter alone (small,
needs the base at load time) and a merged checkpoint (`merge_and_unload()`,
standalone). The merge is done by `transformers` itself, so the merged file carries
that architecture's canonical tensor names — the whole point, given that a merged
file written by another framework is exactly what causes failure mode #1 above. The
merged model is loaded back with plain `AutoModelForCausalLM` and probed for
missing keys and near-uniform output **before** anything is uploaded; publishing is
refused if it doesn't generate language.

## Measuring transfer

`evaluate.py` (`./distill.sh --evaluate`) reports the two standard families of
distillation metric separately, because they are not the same question:

**Fidelity — did the student absorb the teacher?**

* **Top-1 agreement rate** — the fraction of completion positions where
  `argmax(student_logits) == argmax(teacher_logits)`, teacher-forced on held-out
  text. The standard predictive-agreement metric.
* **KL(p_T ‖ p_S)** per token — the distribution-level version, and the `beta = 0`
  case of the training loss. Note that the training objective at `beta = 0.5` is
  JSD, not KL, so this is a related but distinct measurement.

**Capability — is the student better at the task?**

* **Held-out perplexity**, `exp(mean NLL)` over completion tokens, for teacher,
  base student and distilled student.
* **Gap recovered**, `(ppl_base − ppl_distilled) / (ppl_base − ppl_teacher)` — the
  fraction of the base→teacher gap the adapter closed.
* Optionally, **task benchmarks** through lm-evaluation-harness, reported as
  **retention = distilled / teacher**, the DistilBERT-style headline.

Every metric is computed for the base student as well as the distilled one, and
the base column is what makes the numbers interpretable. Stanton et al.
(*Does Knowledge Distillation Really Work?*, NeurIPS 2021) is the reason these are
kept apart: students that match their teacher on task metrics routinely disagree
with it on a large share of individual predictions, so a good capability number is
not evidence of good fidelity, or the reverse.

Two implementation details worth knowing:

* **One student instance serves both columns.** PEFT's `disable_adapter()` context
  turns the LoRA branches off in place, which is the base student exactly. Loading
  a second copy would double peak memory for no extra information. Teacher and
  student *are* both resident, because agreement and KL need both distributions
  for the same position simultaneously.
* **KL is computed in chunks.** A full `[positions, 248320]` float32 tensor is
  hundreds of megabytes at these vocabulary sizes. Chunking changes float summation
  order, so the KL sum is reproducible to about 1e-3, not bit-exactly — the CI
  assertion is written with that tolerance.

The vocabulary-match requirement is also asserted here rather than only documented:
if the teacher and student logits have different final dimensions the run fails
with a clear message, because a token-level divergence between two different
vocabularies cannot mean anything.

## Reproducibility

`project.seed` drives the dataset shuffle, the train/val split and
`TrainingArguments.seed`. What is *not* deterministic: on-policy rollouts use
sampling (`do_sample=True`, `temperature=0.3`, `top_p=0.9`,
`repetition_penalty=1.15` in `generate_sample`), and `lmbda` is a per-step coin
flip. Two runs with the same seed follow the same data order but will not produce
bit-identical adapters. The runner's pinned commit and `uv.lock` fix the *code and
dependency* side of reproducibility, which is the part that usually causes trouble.

## Reading order for the source

1. `configs/qwen-poc.yaml` — every number is commented with the measurement behind
   it. The fastest way to understand the knobs.
2. `kd_config.py` — precedence chain and hardware resolution; ~320 readable lines.
3. `train_scaled.py::run()` — the seven phases end to end.
4. `train_scaled.py::build_datasets()` and `collect_domain()` — the data pipeline.
5. `.venv/.../trl/experimental/gkd/gkd_trainer.py` — `training_step` and
   `generalized_jsd_loss` are about 80 lines together and are the actual algorithm.
6. `evaluate.py` — the metrics, and the shortest description of what "it worked"
   means for this pipeline.
