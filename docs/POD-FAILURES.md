# What goes wrong on a rented pod, and why

Every entry here cost real money to find. They are written down in the order a
run hits them, with the symptom first — because the symptom is what you have at
2am, and none of these name their own cause.

The short version: `scripts/score-pod.sh` already checks for all of them. This
file is for when you are doing something the script does not cover, or when you
want to know why one of its refusals exists.

---

## 1. The install takes twenty minutes, not ninety seconds

**Symptom.** `scripts/runpod.sh` prints its dependency list and then nothing at
all, for a quarter of an hour, on a card billing by the second.

```
==> Installing dependencies (the template's torch build is kept)
      accelerate>=1.14.0
      ...
      vllm>=0.8
```

**Why.** Two things compound. `pip install --quiet --no-cache-dir` prints no
progress, so silence is the normal state rather than a hang. And the list is
long because `config_extras` reads the *config file* to decide which optional
groups to install: `evaluation.engine: vllm` adds `serve`, and
`quantization.enabled` adds `quantize`. vLLM pins its own exact torch, so the
install replaces the template's 2.5 GB CUDA build and pulls a fresh set of
`nvidia-*` libraries with it.

**What to do.** Nothing, usually — it is under a dollar, and the model download
that follows is bigger. When it matters:

- Install by hand, in the order the steps actually need. Packing wants
  llm-compressor and no vLLM; scoring wants vLLM and no llm-compressor. Doing
  them separately means neither install waits for the other, and the first
  billable step starts in two minutes rather than fifteen. This is what
  `scripts/score-pod.sh` does.
- Use `uv`, which resolves in seconds rather than backtracking for minutes:

  ```
  python3 -m pip install -q uv
  python3 -m uv pip install --system --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple vllm llmcompressor compressed-tensors
  ```

  `satisfied()` in `scripts/runpod.sh` probes every requested extra by import,
  so a hand-install makes the script skip its own slow path entirely.

**What does NOT help.** Nothing survives the pod, because packages land in the
container disk's site-packages. To keep them, install into the volume —
`PIP_USER=1` with `PYTHONUSERBASE=/workspace/pyenv` — or bake
`docker/Dockerfile.cuda` into a custom template and stop paying this at all.

---

## 2. `externally-managed-environment`

**Symptom.** `uv` refuses to install anything:

```
error: The interpreter at /usr is externally managed
hint: Virtual environments were not considered due to the `--system` flag
```

**Why.** Debian marks the system interpreter under PEP 668. The image's `pip`
usually has that suppressed through `/etc/pip.conf`, which `uv` does not read —
so pip works and uv does not, on the same machine, which reads like a uv bug and
is not.

**What to do.** `--break-system-packages`, or `UV_BREAK_SYSTEM_PACKAGES=1`. On a
pod that exists to run one job and then be destroyed, this is the right trade.

---

## 3. `No space left on device` at 27.9 GB of a 29.5 GB file

**Symptom.**

```
[1/4] Loading weights (this is the slow part)...
Task error: File reconstruction error: IO Error: No space left on device (os error 28)
```

**Why.** Two separate mistakes, and they look identical.

*The volume is too small.* Qwen3-14B is 29.5 GB, Qwen3-8B 16.4 GB, and the xet
download backend stages chunks and then **reconstructs** the file, needing room
for both at once. `HF_HUB_DISABLE_XET=1` writes the file once instead, and is
worth setting whenever the volume is tight.

*The container disk is full instead.* This is the subtle one. `s3.cache_dir`
defaults to `~/.cache/kd/s3`, which on a pod is the 40 GB container disk, not
the volume — and `paths.derived_dir` writes a merge **next to the cache** for
any adapter that is not sitting in a run bundle. The teacher is such an adapter,
and the arena has to materialise it as a dense checkpoint for vLLM, so a
scoring run tries to write 29.5 GB of Qwen3-14B onto the container disk, three
hours in.

**What to do.** Put the cache on the volume, in the config or on the command
line:

```
--set s3.cache_dir=/workspace/kd-cache
```

`enlibraQ3-14B-score.yaml` sets it, and `scripts/score-pod.sh` refuses to start
if it is anywhere else.

**How much room.** A full pack-and-score session peaks near 100 GB: 46 GB of
base weights, a 15 GB dense merge of the student, 5.7 GB packed, and 29.5 GB of
merged teacher. The profiles ask for a 200 GB volume and mean it.

---

## 4. `Could not import module 'PreTrainedModel'`

**Symptom.** A traceback that ends in `transformers/utils/import_utils.py` and
names an object that plainly exists:

```
ModuleNotFoundError: Could not import module 'PreTrainedModel'.
Are this object's requirements defined correctly?
```

**Why.** That message is a mask. `transformers` imports lazily and rewrites any
failure inside `modeling_utils` into this one sentence, which names neither the
real module nor the real error. Underneath, it is almost always:

```
RuntimeError: operator torchvision::nms does not exist
OSError: Could not load this library: .../torchaudio/lib/libtorchaudio.so
```

`torchvision` and `torchaudio` carry compiled extensions built against one exact
torch build. Any install that moves torch — vLLM and llm-compressor both pin
their own — leaves them mismatched. transformers 5.x imports both from
`modeling_utils` unconditionally, through the object-detection and RNNT losses,
so a text-only pipeline still trips over them.

**What to do.** Ask directly, so the real error prints:

```
python3 -c "import transformers.modeling_utils"
```

Then, depending on what the step needs:

- **Packing.** Nothing here reads images or audio, and llm-compressor does not
  want torchvision. Remove them: `pip uninstall -y torchvision torchaudio`.
- **Scoring.** vLLM *does* want torchvision, so removing it breaks the engine.
  Reinstall a matching pair instead:

  ```
  pip install --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  ```

---

## 5. `llmcompressor requires transformers<=5.14.1, but you have 5.17.0`

**Symptom.** pip installs anyway and warns; imports then fail in unrelated
places.

**Why.** A genuine conflict, not a stale pin. `pyproject.toml` requires
`transformers>=5.16.1`; llm-compressor caps it below that. One environment
cannot satisfy both.

**What to do.** Order, not arbitration. The two libraries are never needed at
the same moment:

1. Install the packing set with transformers **unpinned** — the resolver settles
   on a version llm-compressor accepts — and pack.
2. Install vLLM, which moves transformers wherever it likes, and score.

Nothing in step 2 imports llm-compressor, and nothing in step 1 imports vLLM.
`scripts/score-pod.sh` does exactly this, in that order.

A `venv --system-site-packages` holding only the older transformers works too,
and inherits torch from the system rather than downloading it again:

```
python3 -m venv --system-site-packages /workspace/quantenv
/workspace/quantenv/bin/pip install "transformers==5.14.1" llmcompressor compressed-tensors
```

---

## 6. `'list' object has no attribute 'column_names'`

**Symptom.** The packer dies seconds in, and the parent blames the card:

```
AttributeError: 'list' object has no attribute 'column_names'
xx  the quantization worker exited 1 ... usually the card running out of memory
```

**Why.** `calibration_rows` returns a plain list of tokenised rows; newer
llm-compressor reads `column_names` off the calibration set before building its
dataloader, which only a `datasets.Dataset` has.

**And the second sentence is wrong.** `kd.quantize` prints its out-of-memory
guess for *any* non-zero exit from the worker. A worker that dies in the first
seconds has not touched a weight yet — that is a library mismatch, never the
card. Read the traceback above it, not the guess.

**Fixed** in `src/kd/quantize.py`: the worker wraps the rows in
`Dataset.from_list` before handing them over. `calibration_rows` itself still
returns a list, which is what the tests assert against.

---

## 7. The arena scores four players instead of five

**Symptom.** Hours of scoring, no error, and a report with no `distilled-w4a16`
column. Somewhere in the log:

```
distilled-w4a16 skipped: no packed checkpoint. Run `kd quantize`, or set quantization.enabled
```

**Why.** One of three path mismatches, all silent:

- **Nothing was packed.** Training ran under a profile with
  `quantization.enabled: false`, so the stage never ran.
- **`--out` disagrees with where the arena looks.** Standalone `kd quantize`
  defaults to `<merged>-w4a16`, while `arena.resolve_quantized` checks
  `quantization.output_dir` and then `paths.quantized_dir`, which is
  `<bundle>/quantized`. Pack to one, look in the other, and the column vanishes.
- **A typo in `--out`.** `quantize` for `quantized` is one letter, and produces
  a 5.7 GB directory that no upload group covers and no arena reads.

**What to do.** Never type the output path twice. `enlibraQ3-14B-score.yaml`
sets `quantization.output_dir`, and `scripts/score-pod.sh` reads `--out` from
it, so they cannot drift.

Then read the banner before walking away. It prints the players it is about to
score:

```
players : base, distilled, distilled-w4a16, teacher-base, teacher
```

Four names there is a run worth killing.

---

## 8. `nothing to upload: matched no files`

**Symptom.**

```
!! upload failed: RuntimeError: nothing to upload: s3.upload=[...] matched no files
```

**Why.** The groups are globs against the run directory — `quantized/**`,
`evaluation/**`, `final_adapter/**`. A directory named anything else matches
nothing, and a bundle rehydrated from S3 into a differently-named directory
matches nothing either.

**What to do.** Look at what is actually there. If the packed model landed in
`quantize/` rather than `quantized/`, rename it; the upload then finds it, and
so does the arena.

---

## 9. `ExpiredToken`, hours into a run

**Symptom.** Training or scoring succeeds and the upload does not.

**Why.** The AWS session token used here expires every 15 minutes. The upload
stage runs **last**, which on any real run is hours after the credentials were
exported. It is not a race — it is a certainty.

**What to do.** Skip the stage and ship by hand, which is what every command in
these runbooks does:

```
./run.sh --config <profile> --skip upload
# export fresh AWS_* credentials
./run.sh --config <profile> upload <run-id>
```

**Name the run.** With no argument, `kd upload` takes "the latest run", which
resolves through `runs/latest.txt` and then falls back to the alphabetically
largest directory name — and run ids lead with the profile, not the timestamp,
so that fallback is not chronological.

Only preflight needs a live token during a run, to fetch the teacher adapter —
a few hundred megabytes, comfortably inside the window. The base weights come
from the Hub and want `HF_TOKEN`, not AWS credentials.

---

## 10. `-c/--config: expected one argument`

**Symptom.**

```
kd upload: error: argument -c/--config: expected one argument
```

**Why.** A shell variable that is empty. A new SSH session, or a fresh tmux
pane, does not inherit exports — and `--config $CFG` with an unset `CFG`
reaches argparse as a flag with nothing after it.

**What to do.** `echo "RUN=[$RUN] CFG=[$CFG]"` before blaming the tool.
`scripts/runpod.sh` writes `/workspace/kd-env.sh` for this; `source` it in every
new shell. The safest habit is to type the paths out for one-off commands, and
let a script read them from the config for everything else.

---

## 11. `Ctrl+C` during the install triggers a full reinstall

**Symptom.** You interrupt pip, and the script immediately starts over:

```
!! pip refused to replace a package the base image owns; retrying with
!!   --ignore-installed, which installs alongside rather than over it
```

**Why.** `scripts/runpod.sh` cannot tell "the user cancelled" from "pip hit an
apt-owned package with no RECORD file", because both are a non-zero exit. The
fallback is correct for the second and wasteful for the first.

**What to do.** Let it finish, or stop it and install by hand with `uv`. Either
way, verify afterwards — an interrupted install and `--ignore-installed` can
both leave two versions' files mixed in one package directory:

```
python3 -c "import torch, transformers, peft, accelerate, datasets, yaml, boto3; print('ok')"
ls -d /usr/local/lib/python3.12/dist-packages/transformers-*.dist-info
```

Two `dist-info` directories means orphaned files. `pip uninstall` will not clear
them — it only removes what the RECORD of the version it knows about lists — so
delete the package directory before reinstalling.

---

## What to provision

| | |
| --- | --- |
| GPU, training the 14B → 8B pair | 80 GB — both models resident, ~55–60 GB working set |
| GPU, packing and scoring only | 48 GB — players load one at a time, so the 29.5 GB teacher is the ceiling |
| Volume | 200 GB, mounted at `/workspace` |
| Container disk | 40 GB, and only with `s3.cache_dir` moved to the volume |

## See also

- [RUNPOD.md](RUNPOD.md) — renting the pod in the first place
- [RUNBOOK-SOURCE.md](RUNBOOK-SOURCE.md) — the training session end to end
- [DESIGN-QUANTIZATION-AND-EARLY-UPLOAD.md](DESIGN-QUANTIZATION-AND-EARLY-UPLOAD.md) — why the packing and the upload are separate steps
