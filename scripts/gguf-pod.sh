#!/usr/bin/env bash
# ===========================================================================
#  Convert, upload, quantize, upload - a GGUF of an adapter already in the bucket.
#
#      git clone <repo> && cd knowledge-distillation
#      export KD_CONFIG=configs/enlibra/enlibraQ3-14B-to-4B-gguf.yaml
#      ./scripts/gguf-pod.sh all
#
#  BY HAND, ONE STAGE AT A TIME, which is the way to run this when the
#  credentials last fifteen minutes. Name the profile once in the environment
#  and every command below reads it:
#
#      export KD_CONFIG=configs/enlibra/enlibraQ3-14B-to-4B-gguf.yaml
#
#      ./scripts/gguf-pod.sh setup          # no AWS
#      <export fresh credentials>
#      ./scripts/gguf-pod.sh convert        # AWS in its first seconds only
#      ./scripts/gguf-pod.sh imatrix        # no AWS
#      <export fresh credentials>
#      ./scripts/gguf-pod.sh upload-f16     # AWS throughout - 8 GB
#      ./scripts/gguf-pod.sh quantize       # no AWS
#      <export fresh credentials>
#      ./scripts/gguf-pod.sh upload         # AWS throughout - ~2.4 GB
#
#  THREE of the seven commands touch the bucket, and only those three care
#  whether the token is alive. `convert` reaches it once, in its first seconds,
#  to fetch the adapter - and the adapter is CACHED, so a convert that failed
#  after that point reruns with no credentials at all.
#
#  EVERY STAGE IS IDEMPOTENT. A finished f16, a built matrix and an existing
#  quant are all detected and skipped, so a command that died halfway is rerun
#  rather than worked around, and nothing already paid for is rebuilt.
#
#  THE 8 GB UPLOAD IS THE ONE TO WATCH. At 10 MB/s it takes thirteen minutes
#  against a fifteen-minute token. Export credentials IMMEDIATELY before
#  `upload-f16`, and if it still expires mid-flight, rerun it - or skip it
#  entirely when converting and quantizing on the same machine, since it exists
#  only so the two halves can run in different places.
#

#  The sibling of scripts/score-pod.sh, for the other deployment target.
#  score-pod.sh packs an adapter to compressed-tensors W4A16 and scores it,
#  which runs on CUDA and nowhere else. This packs the same adapter to GGUF,
#  which runs on a laptop, a Mac, a CPU or a phone. Neither artifact is built
#  from the other - both come from the dense merge - and a run needs this one
#  only when the model is going somewhere vLLM will not follow.
#
#  WHY THE COMMANDS SPLIT WHERE THEY DO
#  ------------------------------------
#  Building a GGUF is two passes, and they want different machines:
#
#      convert    merges the LoRA and rewrites it into GGUF's container.
#                 Needs torch, transformers, peft. NOT quantized - the file
#                 comes out BIGGER than the safetensors it read (~8 GB at 4B).
#      quantize   reads that f16 and writes the 4-bit one. CPU ONLY. No torch,
#                 no transformers, no adapter, no card.
#
#  So the f16 gets uploaded BETWEEN them. Convert on the pod, ship the f16,
#  destroy the pod, quantize days later on a laptop - and because a K-quant can
#  only be produced by llama-quantize (the converter emits f32/f16/bf16/q8_0 and
#  nothing else), that second pass is unavoidable and worth placing well.
#
#  THE SCARS, the same way score-pod.sh records them:
#
#    * llama.cpp has to be BUILT, not just cloned. convert_hf_to_gguf.py is a
#      script and ships with the checkout; llama-quantize and llama-imatrix are
#      compiled and do not. A pod template has a compiler and no llama.cpp, so
#      `setup` clones and builds it - which is minutes, once, on the volume.
#
#    * The converter has its own requirements. It imports `gguf`, numpy,
#      sentencepiece and transformers from llama.cpp's requirements.txt, and
#      fails with a bare ModuleNotFoundError naming none of that.
#
#    * /workspace or bust, exactly as in score-pod.sh. The merge is ~8 GB and
#      the f16 is another ~8 GB and they exist AT THE SAME TIME, on top of the
#      base weights the merge downloads. A 40 GB container disk runs out.
#
#    * gguf.keep_f16 should be true on a pod. False deletes the f16 after
#      quantizing - which is right on a laptop and wrong the moment the pod is
#      about to be destroyed, because rebuilding it means re-merging on a card
#      you no longer have. This script warns when it is false.
#
#    * The 15-minute credential. Every S3 step is its own command, as in
#      score-pod.sh: an upload that starts hours after the token was exported
#      is an upload that fails. Export fresh credentials before each step that
#      asks for them - `convert` reaches the bucket in its first seconds to
#      fetch the adapter.
#
#    * NO GPU IS REQUIRED, and that is a real difference from score-pod.sh.
#      Nothing here is scored, so the teacher never loads; the merge is the
#      largest thing in memory and a 4B merge runs on CPU. A card makes
#      `convert` faster and `imatrix` much faster. `quantize` ignores it
#      entirely. So a missing GPU is a warning here, never a refusal.
#
#  COMMANDS
#
#      setup        environment and disk checks, llama.cpp, conversion deps
#      convert      merge the adapter and convert it to an f16 GGUF
#      imatrix      build the importance matrix from the training rows
#      upload-f16   ship gguf-f16/ - the f16, the matrix and the chat template
#      quantize     quantize the f16 to the profile's gguf.quants
#      upload       ship gguf/ - the quantized builds
#      all          every one of the above, in order
#
#  OPTIONS
#
#      --config PATH   required; a profile with a gguf block. $KD_CONFIG sets
#                      it once for a whole session, which is what the by-hand
#                      sequence above relies on.
#      --run ID        default: read from the config's evaluation.adapter
#      --quants LIST   quantize only. Comma-separated, overriding gguf.quants:
#                          --quants Q4_K_M,Q6_K
#      --yes           do not pause between billable steps
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BOLD=""; DIM=""; OFF=""
if [ -t 1 ]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'; fi
step() { printf '\n%s== %s%s\n' "$BOLD" "$*" "$OFF"; }
note() { printf '%s   %s%s\n' "$DIM" "$*" "$OFF"; }
warn() { printf '!! %s\n' "$*" >&2; }
die()  { printf '\nxx %s\n' "$*" >&2; exit 1; }

CONFIG="${KD_CONFIG:-}"
RUN=""
QUANTS=""
ASSUME_YES=0
COMMAND=""

while [ $# -gt 0 ]; do
  case "$1" in
    setup|convert|imatrix|upload-f16|quantize|upload|all) COMMAND="$1"; shift ;;
    --config)    [ $# -ge 2 ] || die "--config needs a path"; CONFIG="$2"; shift 2 ;;
    --config=*)  CONFIG="${1#*=}"; shift ;;
    --run)       [ $# -ge 2 ] || die "--run needs an id"; RUN="$2"; shift 2 ;;
    --run=*)     RUN="${1#*=}"; shift ;;
    --quants)    [ $# -ge 2 ] || die "--quants needs a list"; QUANTS="$2"; shift 2 ;;
    --quants=*)  QUANTS="${1#*=}"; shift ;;
    --yes|-y)    ASSUME_YES=1; shift ;;
    -h|--help)   sed -n '2,/^# ====/p' "$0" | sed 's/^#\{1,\} \{0,1\}//; s/^=\{3,\}.*//'; exit 0 ;;
    *)           die "unknown argument: $1  (try --help)" ;;
  esac
done

[ -n "$COMMAND" ] || die "which command? setup, convert, imatrix, upload-f16,
    quantize, upload, or all.
    ./scripts/gguf-pod.sh all --config configs/enlibra/enlibraQ3-14B-to-4B-gguf.yaml"
[ -n "$CONFIG" ] || die "--config is required and this script never picks one."
[ -f "$CONFIG" ] || die "no such config: $CONFIG"

# Validated HERE, before any install or download, for the same reason
# score-pod.sh validates --players up front: a misspelt name found after an
# eight-gigabyte conversion has cost real money. Only the shape is checked -
# llama-quantize is the authority on which names exist, and it lists them on a
# bad one.
QUANTS="${QUANTS//[[:space:]]/}"
if [ -n "$QUANTS" ]; then
  case "$COMMAND" in
    quantize|all) ;;
    *) die "--quants only means something to quantize (or all); not to $COMMAND" ;;
  esac
  IFS=',' read -r -a _quants <<< "$QUANTS"
  for _q in "${_quants[@]}"; do
    case "$_q" in
      "") die "--quants has an empty entry: '$QUANTS'" ;;
      Q[0-9]_[0-9]|Q[0-9]_K|Q[0-9]_K_[SML]|IQ[0-9]_[SML]|IQ[0-9]_XS|IQ[0-9]_XXS|F16|BF16|F32) ;;
      *) die "--quants: '$_q' is not a llama.cpp quant name.
    Expected Q<n>_K_<S|M|L>, Q<n>_<n>, IQ<n>_<XXS|XS|S|M>, or F16/BF16/F32.
    Q4_K_M is the one you want for a phone." ;;
    esac
  done
fi

PY="${KD_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || die "no python on PATH"

PIPQ=("$PY" -m pip install --break-system-packages --disable-pip-version-check -q)


# ---------------------------------------------------------------------------
#  Reading the config, so no path is ever typed twice
# ---------------------------------------------------------------------------
# Same helper score-pod.sh uses, and for the same reason: every value comes from
# the file kd will read, through kd's own loader and `extends` chain. A grep for
# a key in one file silently misses anything inherited from _base.yaml.
kd_get() {
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/_kd_get.py" "$CONFIG" "$1"
}

confirm() {
  [ "$ASSUME_YES" = "1" ] && return 0
  [ -t 0 ] || return 0
  printf '\n%s%s%s  [Enter to continue, Ctrl+C to stop] ' "$BOLD" "$1" "$OFF"
  read -r _
}

export_gguf() {
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/export_gguf.py" \
    --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
}


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
WORKSPACE="${KD_WORKSPACE:-/workspace}"

prepare_environment() {
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

  if ! mkdir -p "$WORKSPACE" 2>/dev/null || [ ! -w "$WORKSPACE" ]; then
    die "$WORKSPACE is not writable. Mount the volume there: the merge and the
    f16 are about eight gigabytes each and neither survives the pod otherwise."
  fi

  export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
  export KD_RUNS_DIR="${KD_RUNS_DIR:-$WORKSPACE/runs}"
  # The xet backend stages chunks and then reconstructs the file, needing room
  # for both at once. Plain HTTP writes the file once. Same scar as score-pod.sh.
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  mkdir -p "$HF_HOME" "$KD_RUNS_DIR"

  ENABLED="$(kd_get gguf.enabled)"
  case "$ENABLED" in
    True|true|1) ;;
    *) die "gguf.enabled is not true in $CONFIG.
    This script builds a GGUF; a profile that did not ask for one has nothing
    here to do. Add a gguf block, or use configs/enlibra/enlibraQ3-14B-to-4B-gguf.yaml." ;;
  esac

  CACHE_DIR="$(kd_get s3.cache_dir)"
  ADAPTER="$(kd_get evaluation.adapter)"
  LLAMA_CPP="$(kd_get gguf.llama_cpp)"
  KEEP_F16="$(kd_get gguf.keep_f16)"
  CFG_QUANTS="$(kd_get gguf.quants)"
  CALIBRATION="$(kd_get gguf.calibration_file)"
  [ -n "$CALIBRATION" ] || CALIBRATION="$(kd_get quantization.calibration_file)"

  [ -n "$ADAPTER" ] || die "$CONFIG sets no evaluation.adapter. This script
    builds a GGUF of an adapter that is already in the bucket; name it there so
    that every command below refers to the same weights."

  case "$CACHE_DIR" in
    "$WORKSPACE"/*) ;;
    *) warn "s3.cache_dir is '$CACHE_DIR', not under $WORKSPACE."
       warn "  The adapter and the 4B base land there - several gigabytes - and a"
       warn "  pod's container disk is 40 GB with the install already on it." ;;
  esac

  # The run id is embedded in the adapter URI: <bundle>/<run-id>/final_adapter.
  if [ -z "$RUN" ]; then
    RUN="$(basename "$(dirname "${ADAPTER%/}")")"
  fi
  [ -n "$RUN" ] || die "could not work out the run id from evaluation.adapter;
    pass --run explicitly."

  RUN_DIR="$KD_RUNS_DIR/$RUN"
  # Derived the same way export_gguf.Plan derives them, because the upload
  # groups are literal paths: gguf-f16/** and gguf/** relative to the run dir.
  F16_DIR="$(kd_get gguf.f16_dir)";    [ -n "$F16_DIR" ] || F16_DIR="$RUN_DIR/gguf-f16"
  OUT_DIR="$(kd_get gguf.output_dir)"; [ -n "$OUT_DIR" ] || OUT_DIR="$RUN_DIR/gguf"

  [ -n "$LLAMA_CPP" ] || LLAMA_CPP="$WORKSPACE/llama.cpp"
  export LLAMA_CPP
}

report_environment() {
  note "config      : $CONFIG"
  note "run         : $RUN"
  note "adapter     : $ADAPTER"
  note "quants      : ${QUANTS:-$CFG_QUANTS}"
  note "calibration : ${CALIBRATION:-(none - imatrix will be skipped)}"
  note "f16 ->      : $F16_DIR"
  note "gguf ->     : $OUT_DIR"
  note "llama.cpp   : $LLAMA_CPP"
  note "cache       : $CACHE_DIR"
  note "runs        : $KD_RUNS_DIR"
}

check_hardware() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
      | while read -r line; do note "GPU: $line"; done
  else
    # Deliberately not fatal - see the header. Nothing here is scored, and
    # quantizing never touches a card at all.
    warn "no GPU visible. This still works: the merge runs on CPU and"
    warn "  llama-quantize never wanted one. Expect convert and imatrix to be slow."
  fi

  local free_gb
  free_gb="$(df -BG --output=avail "$WORKSPACE" 2>/dev/null | tail -1 | tr -dc '0-9')"
  [ -n "$free_gb" ] || free_gb=0
  note "free on $WORKSPACE: ${free_gb} GB"
  # ~8 GB of base weights downloaded, ~8 GB merge, ~8 GB f16, ~2.5 GB per quant,
  # and the first three coexist.
  if [ "$free_gb" -lt 40 ]; then
    warn "under 40 GB free. A 4B build peaks near 30 GB - the merge and the f16"
    warn "  are ~8 GB each and exist at the same time - and a volume that runs"
    warn "  out does so after the expensive part."
    confirm "Continue anyway?"
  fi

  if [ -z "${TMUX:-}" ]; then
    warn "not inside tmux - a dropped connection kills the run and the pod bills on."
    warn "  tmux new -s kd    then run this again."
  fi

  case "$KEEP_F16" in
    True|true|1) ;;
    *) warn "gguf.keep_f16 is false. The f16 is deleted after quantizing, which"
       warn "  is right on a laptop and wrong on a pod: rebuilding it means"
       warn "  re-merging on a card you will have destroyed. Set it true." ;;
  esac
}

check_credentials() {
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ ! -f "$HOME/.aws/credentials" ]; then
    die "no AWS credentials in this shell. They expire every 15 minutes here,
    so export fresh ones immediately before this step."
  fi

  # AND A LIVE PROBE, because presence is not liveness. A token that expired
  # during the previous step leaves AWS_ACCESS_KEY_ID set, so the check above
  # passes and the failure arrives partway through an eight-gigabyte upload
  # with the pod still billing. One ListObjectsV2 answers in a second.
  if ! PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/_kd_reachable.py" "$CONFIG" \
       >/dev/null 2>"$WORKSPACE/.cred-probe"; then
    printf '\nxx the bucket did not answer with these credentials:\n\n' >&2
    sed 's/^/    /' "$WORKSPACE/.cred-probe" >&2
    rm -f "$WORKSPACE/.cred-probe"
    # Deliberately does not name a cause. The probe printed one above, and a
    # second guess underneath it is how "Repo id must be in the form..." ends up
    # meaning "the path did not exist" - see docs/POD-FAILURES.md.
    die "the detail above is the reason. If it says ExpiredToken, export fresh
    credentials and run this same command again; nothing already built has to be
    redone, because every stage skips work it finds finished."
  fi
  rm -f "$WORKSPACE/.cred-probe"
}

# transformers 5.x pulls torchvision and torchaudio in through modeling_utils,
# and reports a version mismatch as a missing PreTrainedModel. Asked directly,
# once, so the real message is the one that gets printed. Same scar as
# score-pod.sh - it is the pod template's torch, not anything kd installs.
transformers_imports() {
  "$PY" -c 'import transformers.modeling_utils' >/dev/null 2>&1
}

repair_torch_companions() {
  transformers_imports && return 0
  warn "transformers cannot import - almost always torchvision/torchaudio built"
  warn "  against a different torch than the one installed. Removing them;"
  warn "  nothing in this pipeline reads images or audio."
  "$PY" -m pip uninstall -y -q torchvision torchaudio >/dev/null 2>&1 || true
  transformers_imports && return 0
  warn "still failing. The real error follows:"
  "$PY" -c 'import transformers.modeling_utils' || true
  die "transformers is broken in this interpreter; fix that before converting."
}

llama_built() {
  [ -x "$LLAMA_CPP/build/bin/llama-quantize" ] \
    || [ -x "$LLAMA_CPP/build/bin/llama-imatrix" ]
}


# ---------------------------------------------------------------------------
#  Steps
# ---------------------------------------------------------------------------
do_setup() {
  step "Environment"
  report_environment
  check_hardware

  step "llama.cpp"
  if [ -d "$LLAMA_CPP/.git" ]; then
    note "already cloned: $LLAMA_CPP"
  else
    command -v git >/dev/null 2>&1 || die "git is not installed."
    note "cloning into $LLAMA_CPP"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP" \
      || die "could not clone llama.cpp"
  fi

  if llama_built; then
    note "already built: $LLAMA_CPP/build/bin"
  else
    command -v cmake >/dev/null 2>&1 || die "cmake is not installed, and the
    binaries this needs are compiled. On a Debian pod:
      apt-get update && apt-get install -y cmake build-essential"
    note "building llama-quantize and llama-imatrix (a few minutes, CPU)"
    # CPU-only on purpose. The GPU backends take much longer to compile and
    # neither binary this script runs is meaningfully faster for it -
    # llama-quantize is pure CPU work, and imatrix on 128 short rows is minutes
    # either way. Nothing here serves the model.
    cmake -B "$LLAMA_CPP/build" -S "$LLAMA_CPP" -DGGML_NATIVE=ON \
      || die "cmake configure failed"
    cmake --build "$LLAMA_CPP/build" --config Release -j \
          --target llama-quantize llama-imatrix \
      || die "cmake build failed"
    llama_built || die "the build finished but no binaries are in
    $LLAMA_CPP/build/bin - check the output above."
  fi

  step "Conversion dependencies"
  # llama.cpp's converter has its OWN requirements - it imports `gguf`, numpy,
  # sentencepiece and transformers - and reports a missing one as a bare
  # ModuleNotFoundError naming none of that.
  note "llama.cpp's converter requirements"
  "${PIPQ[@]}" -r "$LLAMA_CPP/requirements.txt" \
    || warn "could not install all of llama.cpp's requirements; convert may fail"

  # And the merge needs kd's own. No vLLM and no llm-compressor: nothing here
  # serves or packs to compressed-tensors, so neither constraint applies and
  # this is one ordinary pip call, unlike score-pod.sh's two.
  if "$PY" -c 'import torch, transformers, peft, boto3, yaml' >/dev/null 2>&1; then
    note "merge dependencies already present"
  else
    note "installing torch, transformers, peft, boto3"
    "${PIPQ[@]}" torch transformers peft accelerate safetensors pyyaml boto3 \
      || die "could not install the merge dependencies"
  fi

  repair_torch_companions
  note "ready"
}

do_convert() {
  step "Converting to GGUF"
  note "the adapter is fetched from the bucket on first use - needs a live token"
  check_credentials
  confirm "Merge the 4B adapter (~8 GB written) and convert (~8 GB more)."
  export_gguf --stage convert
}

do_imatrix() {
  step "Importance matrix"
  if [ -z "$CALIBRATION" ]; then
    note "no calibration file configured - skipping"
    return 0
  fi
  export_gguf --stage imatrix
}

do_upload_f16() {
  step "Uploading the f16 build of $RUN"
  # The local check first, because it is free and it is the more likely mistake.
  # Probing the bucket to then report an empty directory would answer a question
  # nobody asked.
  [ -d "$F16_DIR" ] || die "$F16_DIR does not exist yet - run convert first."
  check_credentials
  # ONLY the gguf-f16 group. The quants do not exist yet, and shipping the whole
  # of s3.upload here would re-upload logs and metrics a second time for nothing.
  #
  # The run is NAMED. With no argument `kd upload` takes "the latest run", which
  # falls back to the alphabetically largest directory - and run ids lead with
  # the profile, not the timestamp, so that is not chronological.
  PYTHONPATH="$ROOT/src" "$PY" -m kd upload "$RUN" --config "$CONFIG" \
    --set 's3.upload=[gguf-f16]' \
    || die "upload failed. ExpiredToken means exactly what it says: export
    fresh credentials and run this step again - nothing else has to be redone."
  note "the f16 is in the bucket. From here, quantize needs no GPU and no"
  note "  adapter - it can run on any machine with llama.cpp built."
}

do_quantize() {
  step "Quantizing"
  # No credentials and no card. If the f16 is already here this touches nothing
  # outside the run directory.
  if [ -n "$QUANTS" ]; then
    note "quants: $QUANTS (overriding gguf.quants)"
    export_gguf --stage quantize --quants "$QUANTS"
  else
    export_gguf --stage quantize
  fi
}

do_upload() {
  step "Uploading the quantized build of $RUN"
  [ -d "$OUT_DIR" ] || die "$OUT_DIR does not exist yet - run quantize first."
  check_credentials
  PYTHONPATH="$ROOT/src" "$PY" -m kd upload "$RUN" --config "$CONFIG" \
    --set 's3.upload=[gguf]' \
    || die "upload failed. ExpiredToken means exactly what it says: export
    fresh credentials and run this step again - nothing else has to be redone."
}


case "$COMMAND" in
  setup)      prepare_environment; do_setup ;;
  convert)    prepare_environment; report_environment; do_convert ;;
  imatrix)    prepare_environment; report_environment; do_imatrix ;;
  upload-f16) prepare_environment; do_upload_f16 ;;
  quantize)   prepare_environment; report_environment; do_quantize ;;
  upload)     prepare_environment; do_upload ;;
  all)        prepare_environment
              do_setup
              do_convert
              do_imatrix
              do_upload_f16
              do_quantize
              do_upload
              step "Done"
              note "f16   : $F16_DIR"
              note "gguf  : $OUT_DIR"
              note "Q4_K_M at 4B is ~2.4 GB, and with a 4k KV cache about"
              note "  3.3 GB resident - a flagship phone, not a midrange one."
              note "the pod bills until you terminate it in the RunPod console." ;;
esac
