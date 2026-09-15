# Design notes: post-training quantization and early S3 upload

Status: research only for part 1. Written 2026-09-14.

Update 2026-09-15: the separation part 2 argues for is done, by a different
route. Evaluation is now its own pipeline (`kd eval`, `evaluation.stages`),
off the training pipeline by default, and every evaluation lands inside the
adapter's bundle under `evaluation/<name>-<date>/` on disk and in the bucket.
The training pipeline ends at `upload`, so the adapter leaves the machine as
soon as the run does; the "early upload" below is now only about shipping it
before a same-run evaluation (`evaluation.after_training: true`) and is still
open. The `--from evaluate` spelling below is `kd eval` now.

Two related changes to the pipeline, both optional and both switched by config:

1. **Optional quantization after training**, with evaluation and reports
   produced for the LoRA student *and* the quantized student.
2. **Upload the adapter (and quantized model) to S3 the moment it exists**,
   before the long evaluation stages run.

They are described separately, but the second is what makes the first
comfortable on a rented machine.

---

## 1. Optional quantization stage

### Where it slots in

The pipeline is an ordered dict of stages with per-stage "enabled" gates
(`src/kd/pipeline.py`, `STAGES` and `CONDITIONAL`). A `quantize` stage sits
between `train` and `evaluate`, gated on `quantization.enabled` in the YAML
exactly the way `publish` and `upload` are gated today:

```
preflight → teacher-check → smoke → train → quantize → evaluate → arena → report → publish → upload
```

When `quantization.enabled` is false the stage reports "not configured" and
everything downstream behaves exactly as it does now. `--only quantize`,
`--from quantize`, `--skip quantize` work for free through `planned_stages`.

### What the stage does

1. Merge the LoRA adapter into the base student (`merge_and_unload`). This is
   already done in `publish.py` and `arena.py`; the code should be lifted into a
   shared helper rather than copied a third time.
2. Quantize the merged model and save it to `runs/<id>/quantized/` as a
   self-contained checkpoint.
3. Return `{"quantized_model": <path>}` into `ctx.results` so downstream stages
   can find it.

### Flow when enabled

```
train
  └─ final_adapter/                        (as today)
quantize
  └─ quantized/                            (merged + quantized, self-contained)
evaluate   ×2  → evaluation.json           (LoRA:  base + adapter)
               → evaluation-quantized.json (quantized dir)
arena      ×2  → arena.json
               → arena-quantized.json
report         → report.html               (LoRA)
               → report-quantized.html     (quantized)
               → optional: one comparison page
publish / upload                           (unchanged; upload ships everything in the run dir)
```

When disabled only the left-hand column runs and nothing in the output changes.

### Why this fits the existing structure

- Stages return results into `ctx.results`, so `stage_evaluate` and
  `stage_arena` only need to check "is there a second student variant? then
  loop". The stage list itself does not change beyond inserting `quantize`.
- `report.py` already takes a payload (evaluation + arena JSON). Running it
  twice with two payloads and two output names is a small loop.
- The `_newest(...)` file-discovery helpers in `pipeline.py` key on filename, so
  a `-quantized` suffix keeps the two variants from being confused on a
  `--from report` rerun.

### The one real code change: a second student load path

`evaluate.py` and `arena.py` currently only know how to load the student as
**base + PEFT adapter** (`PeftModel.from_pretrained(student, adapter_dir)`).
They need a second mode: "this directory is a complete model, load it with plain
`AutoModelForCausalLM`". `publish.py` already does exactly this to verify the
merged checkpoint before uploading, so the pattern exists.

Everything after model loading (fidelity scoring, capability tasks, Elo, JSON
output) is model-agnostic and stays as is.

`run_lm_eval` builds `pretrained=...,peft=...` model-arg strings for lm-eval; it
needs a `pretrained=<quantized dir>` variant with no `peft=` part.

### The "untrained base student" column

The LoRA evaluation gets its base-student column for free by toggling the
adapter off on the *same* model instance (`disable_adapter()`), which avoids
loading a second copy. The quantized model has no adapter to toggle.

Options:

- Load the base separately for the quantized pass. Costs memory, keeps the
  report layout identical.
- **Reuse the base numbers from the LoRA evaluation run.** Same model, same
  prompts, so this is both cheaper and correct. Recommended.

### Two reports vs. one

- **Two separate reports** — simplest. Each is exactly today's report, one per
  variant. Zero template changes.
- **One combined report** — teacher / student-LoRA / student-quantized in
  adjacent columns with the quantization delta called out. More useful for the
  actual question ("what did I lose by shrinking it?") but needs a new template
  section.

Both can coexist: always generate the two per-variant reports, add the
comparison page on top. The combined page is roughly half a day extra.

### Choice of quantization method

This is the main decision, and it affects how much work the evaluator change is.

| Method | Loads back via | Effort | Notes |
|---|---|---|---|
| bitsandbytes 4/8-bit | `transformers` + `BitsAndBytesConfig` | Low | Not really a saved artifact — quantized on load. The "quantize stage" becomes a loader flag. GPU only. Tells you least about a deployed artifact. |
| GPTQ / AWQ | `transformers` (needs `auto-gptq` / `autoawq`) | Medium | Real saved checkpoint. Needs a calibration set — the training data is right there. Existing transformers-based evaluators load it with small changes. |
| GGUF (llama.cpp) | Not `transformers` | High | Separate evaluator path (llama-cpp-python); lm-eval uses a different backend. Most useful for actual deployment. |

Recommendation: GPTQ or AWQ as the honest middle ground — a real artifact that
the existing evaluators can load with minimal changes. bitsandbytes is cheapest
to add but answers a weaker question. GGUF can be a later addition once the
two-variant evaluation loop exists.

### Config sketch

```yaml
quantization:
  enabled: false
  method: awq                # awq | gptq | bnb
  bits: 4
  calibration_samples: 128   # drawn from the training set
```

### Things still to decide

- Which method(s) to support first.
- Should `publish` push the quantized model too (`<repo>-int4` alongside
  `<repo>` and `<repo>-lora`)? Easy once the artifact exists.
- Whether `smoke` should project quantization time against `limits`. Probably
  not — quantization is minutes, not hours.

---

## 2. Upload the adapter to S3 as soon as training finishes

### Problem

Evaluation and arena take far longer than training on a small student. Today the
adapter only leaves the machine at the final `upload` stage, or via
`_rescue_upload` when a limit stops the run. A hard pod kill during the hour of
evaluation loses the adapter — the one artifact that actually cost money.

### What already exists

- `stage_upload(ctx, groups=...)` is already callable with an explicit subset of
  upload groups. That is how `_rescue_upload` adds `checkpoints` after a crash.
- Uploads are keyed by path relative to the run directory under a per-run S3
  prefix, so uploading the adapter early and the rest later lands everything in
  the same bundle.
- `evaluate.py` / `arena.py` already accept an `s3://` adapter via
  `paths.localise`, so an adapter in the bucket is immediately usable elsewhere.

### Flow

```
train         → final_adapter/ written
                └─ immediately: upload groups=[adapter]      (seconds; adapter is small)
quantize      → quantized/ written
                └─ immediately: upload groups=[quantized]    (if enabled)
evaluate      (long)
arena         (long)
report
upload        → ships logs / metrics / report; skips files already in the bucket
```

The early sync sits at the end of `stage_train` (and `stage_quantize`), gated on
`s3.enabled` plus a new opt-in flag, e.g. `s3.upload_on_train: true`. A failure
of the early upload must be a **warning, not a gate failure**: the adapter is
still on disk and the final `upload` stage will retry it.

### Two things to get right

1. **Do not re-upload what is already there.** `upload_bundle` currently pushes
   every selected file. The final stage should skip files whose size / ETag
   already match in the bucket, or the adapter is sent twice. Small change in
   `_selected_files` / `upload_bundle`.
2. **The bundle is incomplete between the two uploads.** `manifest.json` is
   written at upload time and describes the bundle. The early upload should
   write a manifest marking status `training-complete, evaluation pending`, and
   the final upload overwrites it. Otherwise a bucket reader (the compare UI, a
   `--from evaluate` resume on another machine) cannot tell a finished bundle
   from a half-finished one.

### What this unlocks

Once the adapter is in S3 the moment training ends, evaluation no longer has to
run on the same machine:

- Stop the training pod and run `kd pipeline --from evaluate --adapter s3://...`
  on a cheaper GPU.
- Run the LoRA evaluation and the quantized evaluation on two pods in parallel.

Not required for either change above, but the early upload is the piece that
makes it possible.

---

## Suggested order of work

1. Early adapter upload (small, independently useful, de-risks everything else).
2. Second student load path in `evaluate.py` / `arena.py` (merged model dir).
3. `quantize` stage with one method (AWQ or GPTQ), returning the artifact path.
4. Loop evaluate / arena / report over both variants; suffix the outputs.
5. Early upload of the quantized artifact (reuses step 1).
6. Combined comparison report.
7. Later: `publish` pushes the quantized repo; GGUF as a second method.
