#!/usr/bin/env bash
# ===========================================================================
#  Pack, upload, score, upload - on a pod that never trains anything.
#
#      git clone <repo> && cd knowledge-distillation
#      ./scripts/score-pod.sh all --config configs/enlibra/enlibraQ3-14B-score.yaml
#
#  The training half of this workflow ends with an adapter in the bucket. This
#  is the other half, and it is deliberately NOT scripts/runpod.sh: that one
#  installs everything a config implies in one pip call, which for a profile
#  with `engine: vllm` AND `quantization.enabled` is a dependency set with no
#  solution. llm-compressor caps transformers at <=5.14.1; pyproject.toml wants
#  >=5.16.1. One environment cannot hold both.
#
#  The way out is ORDER. Packing needs llm-compressor and no vLLM; scoring
#  needs vLLM and no llm-compressor. Installing them in sequence - pack first,
#  then let vLLM move transformers wherever it likes - means neither install
#  ever has to satisfy the other's constraint.
#
#  EVERYTHING ELSE HERE IS A SCAR. Each check below is a failure that cost a
#  rented hour, written down so it costs nobody another one:
#
#    * /workspace or bust. The default S3 cache is under $HOME, and the arena
#      writes a 29.5 GB dense merge of the teacher next to it (arena.py:1097,
#      paths.py:686). On a pod that is the 40 GB container disk, and the run
#      dies with ENOSPC three hours in. The config's s3.cache_dir must point at
#      the volume; this script refuses to start if it does not.
#
#    * --out is READ FROM THE CONFIG, never typed. `kd quantize` writes to
#      <merged>-w4a16 by default (quantize.py:404) while the arena looks in
#      quantization.output_dir (arena.py:895). Type them separately and they
#      drift - one letter, `quantize` for `quantized`, and the arena silently
#      scores four players instead of five, hours later, with no error.
#
#    * torchvision and torchaudio are uninstalled before packing. transformers
#      5.x imports both from modeling_utils (loss_rnnt, image_utils), a pod
#      template ships versions built against ITS torch, and any install that
#      moves torch turns that into `operator torchvision::nms does not exist` -
#      surfaced as `Could not import module 'PreTrainedModel'`, which names
#      neither torch nor torchvision. Nothing here processes audio or images.
#      vLLM does want torchvision, so the scoring step puts a matching pair
#      back rather than removing it.
#
#    * PEP 668. Debian marks the system interpreter externally-managed; uv
#      honours that and pip on these images does not. --break-system-packages
#      either way.
#
#    * HF_HUB_DISABLE_XET. The xet backend downloads chunks and then
#      RECONSTRUCTS the file, which needs room for both. 29.5 GB of Qwen3-14B
#      became "No space left on device" at 27.9 GB on a volume that had enough
#      room for the finished file.
#
#    * The 15-minute credential. Every S3 step is its own command and the
#      scoring run skips its upload stage, because an upload that starts hours
#      after the token was exported is an upload that fails. Export fresh
#      credentials before each step this script asks for them.
#
#  COMMANDS
#
#      setup      environment, disk and GPU checks, packing dependencies
#      quantize   merge the adapter, pack it to W4A16
#      upload     ship the run directory (names the run explicitly - never
#                 "latest", which is alphabetical and not chronological)
#      eval       install vLLM, score five players
#      all        every one of the above, in order
#
#  OPTIONS
#
#      --config PATH   required; the scoring profile
#      --run ID        default: read from the config's evaluation.adapter
#      --yes           do not pause between billable steps
#      --players LIST  eval only. Comma-separated subset of
#                      base,distilled,distilled-w4a16,teacher-base,teacher
#                      to score in the arena, instead of the profile's five.
#
#                      When the list names neither teacher nor teacher-base,
#                      the fidelity stage is skipped as well: that stage IS a
#                      measurement against the teacher (KL, agreement -
#                      evaluate.py:503) and would load the 14B whatever the
#                      arena was told, which is the one thing a short list is
#                      meant to avoid. What remains is arena + report - and
#                      the report without a base or teacher column says what
#                      packing cost, not whether distillation worked.
#
#                          --players distilled,distilled-w4a16
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
ASSUME_YES=0
COMMAND=""
PLAYERS=""

while [ $# -gt 0 ]; do
  case "$1" in
    setup|quantize|upload|eval|all) COMMAND="$1"; shift ;;
    --config)   [ $# -ge 2 ] || die "--config needs a path"; CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --run)      [ $# -ge 2 ] || die "--run needs an id"; RUN="$2"; shift 2 ;;
    --run=*)    RUN="${1#*=}"; shift ;;
    --players)  [ $# -ge 2 ] || die "--players needs a list"; PLAYERS="$2"; shift 2 ;;
    --players=*) PLAYERS="${1#*=}"; shift ;;
    --yes|-y)   ASSUME_YES=1; shift ;;
    -h|--help)  sed -n '2,/^# ====/p' "$0" | sed 's/^#\{1,\} \{0,1\}//; s/^=\{3,\}.*//'; exit 0 ;;
    *)          die "unknown argument: $1  (try --help)" ;;
  esac
done

[ -n "$COMMAND" ] || die "which command? setup, quantize, upload, eval, or all.
    ./scripts/score-pod.sh all --config configs/enlibra/enlibraQ3-14B-score.yaml"
[ -n "$CONFIG" ] || die "--config is required and this script never picks one."
[ -f "$CONFIG" ] || die "no such config: $CONFIG"

# --players is validated HERE, before any install or download, for the same
# reason kd's own preflight validates evaluation.players: a misspelt name
# found after vLLM has been installed and the adapter fetched has cost real
# money. The list is normalised to no spaces, which is what --set's list
# syntax wants.
PLAYERS="${PLAYERS//[[:space:]]/}"
if [ -n "$PLAYERS" ]; then
  [ "$COMMAND" = "eval" ] || [ "$COMMAND" = "all" ] \
    || die "--players only means something to eval (or all); not to $COMMAND"
  IFS=',' read -r -a _players <<< "$PLAYERS"
  [ "${#_players[@]}" -gt 0 ] || die "--players is empty"
  for _p in "${_players[@]}"; do
    case "$_p" in
      base|distilled|distilled-w4a16|teacher-base|teacher) ;;
      "") die "--players has an empty entry: '$PLAYERS'" ;;
      *)  die "--players: unknown player '$_p'
    valid: base, distilled, distilled-w4a16, teacher-base, teacher" ;;
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
# Every value this script needs comes from the same file kd will read. The one
# bug this prevents is the expensive one: an --out that does not match
# quantization.output_dir packs 5.7 GB the arena never looks at.
kd_get() {
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/_kd_get.py" "$CONFIG" "$1"
}

confirm() {
  [ "$ASSUME_YES" = "1" ] && return 0
  [ -t 0 ] || return 0
  printf '\n%s%s%s  [Enter to continue, Ctrl+C to stop] ' "$BOLD" "$1" "$OFF"
  read -r _
}


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
WORKSPACE="${KD_WORKSPACE:-/workspace}"

prepare_environment() {
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

  if ! mkdir -p "$WORKSPACE" 2>/dev/null || [ ! -w "$WORKSPACE" ]; then
    die "$WORKSPACE is not writable. Mount the volume there: everything this
    script produces is tens of gigabytes and none of it survives the pod
    otherwise."
  fi

  export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
  export KD_RUNS_DIR="${KD_RUNS_DIR:-$WORKSPACE/runs}"
  # The xet backend stages chunks and then reconstructs the file, needing room
  # for both at once. Plain HTTP writes the file once. See the header.
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  mkdir -p "$HF_HOME" "$KD_RUNS_DIR"

  CACHE_DIR="$(kd_get s3.cache_dir)"
  QUANT_OUT="$(kd_get quantization.output_dir)"
  ADAPTER="$(kd_get evaluation.adapter)"

  [ -n "$ADAPTER" ] || die "$CONFIG sets no evaluation.adapter. This script
    scores an adapter that is already in the bucket; name it there so that
    every command below refers to the same weights."

  [ -n "$QUANT_OUT" ] || die "$CONFIG sets no quantization.output_dir.
    Without it the packed student lands beside the FETCHED adapter - inside
    the S3 cache - where no upload group covers it, so the bundle in the
    bucket silently lacks the artifact that actually ships."

  case "$CACHE_DIR" in
    "$WORKSPACE"/*) ;;
    *) die "s3.cache_dir is '$CACHE_DIR', which is not under $WORKSPACE.
    The arena writes a dense merge of the 14B teacher next to this directory -
    29.5 GB - and a pod's container disk is 40 GB with the install already on
    it. Set it to $WORKSPACE/kd-cache in $CONFIG." ;;
  esac

  # The run id is embedded in the adapter URI: <bundle>/<run-id>/final_adapter.
  if [ -z "$RUN" ]; then
    RUN="$(basename "$(dirname "${ADAPTER%/}")")"
  fi
  [ -n "$RUN" ] || die "could not work out the run id from evaluation.adapter;
    pass --run explicitly."

  RUN_DIR="$KD_RUNS_DIR/$RUN"
}

report_environment() {
  note "config      : $CONFIG"
  note "run         : $RUN"
  note "adapter     : $ADAPTER"
  note "packed to   : $QUANT_OUT"
  note "cache       : $CACHE_DIR"
  note "HF_HOME     : $HF_HOME"
  note "runs        : $KD_RUNS_DIR"
}

check_hardware() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
      | while read -r line; do note "GPU: $line"; done
  else
    die "no GPU visible. Packing and scoring both need one; a CPU fallback
    here is a bill with no result."
  fi

  local free_gb
  free_gb="$(df -BG --output=avail "$WORKSPACE" 2>/dev/null | tail -1 | tr -dc '0-9')"
  [ -n "$free_gb" ] || free_gb=0
  note "free on $WORKSPACE: ${free_gb} GB"
  # 46 GB of base weights, a 15 GB student merge, 5.7 GB packed, and a 29.5 GB
  # teacher merge during the arena.
  if [ "$free_gb" -lt 120 ]; then
    warn "under 120 GB free. This workflow peaks near 100 GB and a volume that"
    warn "  runs out does so hours in. 200 GB is the size the profile assumes."
    confirm "Continue anyway?"
  fi

  if [ -z "${TMUX:-}" ]; then
    warn "not inside tmux - a dropped connection kills the run and the pod bills on."
    warn "  tmux new -s kd    then run this again."
  fi
  if [ -z "${KD_PRICE_PER_HOUR:-}" ]; then
    warn "KD_PRICE_PER_HOUR is not set, so the in-run cost cap is inactive."
  fi
}

check_credentials() {
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ ! -f "$HOME/.aws/credentials" ]; then
    die "no AWS credentials in this shell. They expire every 15 minutes here,
    so export fresh ones immediately before this step."
  fi
}

# transformers 5.x pulls torchvision and torchaudio in through modeling_utils,
# and reports a version mismatch as a missing PreTrainedModel. Asked directly,
# once, so the real message is the one that gets printed.
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
  die "transformers is broken in this interpreter; fix that before spending GPU time."
}

cuda_or_die() {
  "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || die "torch cannot see the GPU - a CPU wheel got installed. Replace it:
      pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch"
}

is_packed() {
  PYTHONPATH="$ROOT/src" "$PY" "$ROOT/scripts/_kd_packed.py" "$QUANT_OUT"
}


# ---------------------------------------------------------------------------
#  Steps
# ---------------------------------------------------------------------------
do_setup() {
  step "Environment"
  report_environment
  check_hardware

  step "Packing dependencies"
  # transformers is deliberately UNPINNED. Ask for >=5.16.1 here, as
  # pyproject.toml does, and the resolver has no solution: llm-compressor caps
  # it at <=5.14.1. Left free, it settles on a version that satisfies the
  # packer, which is the only thing that runs in this step. vLLM moves it again
  # later, for the step that needs it moved.
  if "$PY" -c 'import llmcompressor, peft, transformers, boto3' >/dev/null 2>&1; then
    note "already present - skipping install"
  else
    note "installing llm-compressor and friends (no vLLM, no torch replacement)"
    "${PIPQ[@]}" transformers peft accelerate datasets pyyaml boto3 \
      llmcompressor compressed-tensors \
      || die "could not install the packing dependencies"
  fi

  repair_torch_companions
  cuda_or_die
  "$PY" "$ROOT/scripts/_kd_versions.py" pack
}

do_quantize() {
  if is_packed; then
    step "Packing"
    note "already packed: $QUANT_OUT - skipping"
    return 0
  fi

  step "Packing the adapter to W4A16"
  note "the adapter is fetched from the bucket on first use - needs a live token"
  check_credentials
  confirm "Merge (~15 GB written) and pack. Twenty minutes or so."

  # --out comes from the config, never from a second place it could disagree.
  PYTHONPATH="$ROOT/src" "$PY" -m kd quantize \
    --config "$CONFIG" --adapter "$ADAPTER" --out "$QUANT_OUT" \
    || die "packing failed. If the worker died in the first seconds, ignore the
    parent's out-of-memory guess - that message is printed for any non-zero
    exit, and an early death is a library mismatch, not the card."

  is_packed || die "the packer finished but $QUANT_OUT is not a packed checkpoint."
  note "packed: $QUANT_OUT"
}

do_upload() {
  step "Uploading $RUN"
  check_credentials
  [ -d "$RUN_DIR" ] || die "$RUN_DIR does not exist yet."
  # The run is NAMED. With no argument `kd upload` takes "the latest run",
  # which falls back to the alphabetically largest directory - and run ids lead
  # with the profile, not the timestamp, so that is not chronological.
  PYTHONPATH="$ROOT/src" "$PY" -m kd upload "$RUN" --config "$CONFIG" \
    || die "upload failed. ExpiredToken means exactly what it says: export
    fresh credentials and run this step again - nothing else has to be redone."
}

do_eval() {
  step "Scoring dependencies (vLLM)"
  if "$PY" -c 'import vllm' >/dev/null 2>&1; then
    note "vLLM already present - skipping install"
  else
    note "this replaces torch with the build vLLM pins; several GB"
    "${PIPQ[@]}" --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple vllm \
      || die "could not install vLLM"
  fi

  # Unlike the packing step, vLLM WANTS torchvision - so a mismatch here is
  # repaired by matching the pair, not by removing it.
  if ! transformers_imports; then
    warn "transformers broken after the vLLM install; reinstalling torch and"
    warn "  torchvision together from the CUDA index so they match"
    "${PIPQ[@]}" --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
      || die "could not repair torch/torchvision"
    transformers_imports || die "transformers still cannot import."
  fi
  cuda_or_die
  "$PY" "$ROOT/scripts/_kd_versions.py" score

  step "Scoring"
  check_credentials
  # --skip upload, always. The eval's own upload stage runs hours after
  # preflight fetched the teacher, by which time the token is long dead. The
  # upload is its own command, run with fresh credentials.
  EVAL_ARGS=(--skip upload)
  if [ -n "$PLAYERS" ]; then
    EVAL_ARGS+=(--set "evaluation.players=[$PLAYERS]")
    # The arena honours the list (arena.py:958, pipeline.py:762) and fetches
    # the teacher only when a player needs it. The fidelity stage does not: it
    # loads the teacher unconditionally (evaluate.py:503), because KL and
    # agreement are distances FROM the teacher and there is nothing to measure
    # without it. So a list with no teacher in it drops that stage too -
    # otherwise the 14B would load anyway and the short list bought nothing.
    case ",$PLAYERS," in
      *,teacher,*|*,teacher-base,*) ;;
      *) EVAL_ARGS+=(--skip evaluate)
         note "players : $PLAYERS"
         note "no teacher among them: skipping the fidelity stage, which"
         note "  would load the 14B regardless. Arena + report only." ;;
    esac
    IFS=',' read -r -a _players <<< "$PLAYERS"
    confirm "${#_players[@]} player(s) over the held-out set. The pod bills throughout."
  else
    confirm "Five players over the held-out set. Hours, and the pod bills throughout."
  fi
  PYTHONPATH="$ROOT/src" "$PY" -m kd eval --config "$CONFIG" "${EVAL_ARGS[@]}"
}


case "$COMMAND" in
  setup)    prepare_environment; do_setup ;;
  quantize) prepare_environment; report_environment; do_quantize ;;
  upload)   prepare_environment; do_upload ;;
  eval)     prepare_environment; report_environment; do_eval ;;
  all)      prepare_environment
            do_setup
            do_quantize
            do_upload
            do_eval
            do_upload
            step "Done"
            note "bundle: $RUN_DIR"
            note "the pod bills until you terminate it in the RunPod console." ;;
esac
