#!/usr/bin/env bash
# ===========================================================================
#  Set up and run the pipeline on a rented GPU, over SSH.
#
#    git clone <repo> && cd knowledge-distillation
#    ./scripts/runpod.sh --config configs/finance.yaml
#
#  This is the hand-driven counterpart to `kd runpod launch`. That path bakes
#  everything into docker/Dockerfile.cuda and rents the GPU for you; this one
#  assumes you already have a shell on a pod you started yourself, on one of
#  RunPod's stock PyTorch templates.
#
#  The whole reason it exists: those templates already ship a CUDA build of
#  torch, which is 2.5 GB of the install and effectively all of the wait. So
#  this script deliberately does NOT install torch. It installs the libraries
#  around it, which take about ninety seconds, and hands over.
#
#  It also does not install the project. `python -m kd` with src on PYTHONPATH
#  needs no packaging metadata, which sidesteps the requires-python floor in
#  pyproject.toml (>=3.13) that no stock template currently satisfies.
#
#  Anything it does not recognise is passed straight through to the pipeline:
#
#    ./scripts/runpod.sh --config configs/finance.yaml --set training.max_steps=500
#    ./scripts/runpod.sh evaluate --config configs/finance.yaml
#    ./scripts/runpod.sh --setup-only
#
#  --rehearse runs the whole thing on a machine that is not a GPU pod - a laptop,
#  a Mac - so that everything except the GPU is proven before anything is rented:
#
#    ./scripts/runpod.sh --rehearse doctor
#    ./scripts/runpod.sh --rehearse --config configs/enlibraQ3-8B-smoke.yaml
#
#  It answers "does this script work, are the dependencies right, does the config
#  resolve, can it reach S3". It cannot answer "does it fit" or "how fast is a
#  step": those need the real hardware, and the pipeline's own `smoke` stage is
#  what asks them there.
# ===========================================================================
set -euo pipefail

log()  { printf '==> %s\n' "$*"; }
warn() { printf '!!  %s\n' "$*" >&2; }
die()  { printf 'xx  %s\n' "$*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Where the volume is mounted. Everything expensive to reproduce - model weights,
# run bundles - goes here so it survives the pod. Installed packages do not: they
# land in the template's site-packages on the container disk and are reinstalled
# on the next pod. That is the ninety seconds this approach costs, and it is why
# a network volume or the baked image wins once you run often.
WORKSPACE="${KD_WORKSPACE:-/workspace}"

SETUP_ONLY=0
# Rehearsal: run this script through on a machine that is not a GPU pod, to find
# out what it does before renting one. Everything still happens - the arguments
# are parsed, the dependency set is read out of pyproject.toml, the environment
# is written, the pipeline is handed to - except that the absence of a GPU stops
# being fatal.
#
# It proves the plumbing and nothing else. Whether the models fit, and how fast
# a step is, are questions only the real hardware answers, and the pipeline's own
# `smoke` stage is what asks them there.
REHEARSE="${KD_REHEARSE:-0}"
EXTRAS="remote"       # S3 and the RunPod SDK. Cheap, and a run that cannot ship
                      # its bundle to S3 leaves nothing behind when the pod dies.
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --setup-only) SETUP_ONLY=1; shift ;;
    --rehearse)   REHEARSE=1; shift ;;
    --extra)      [ $# -ge 2 ] || die "--extra needs a name: eval or remote"
                  EXTRAS="$EXTRAS $2"; shift 2 ;;
    --extra=*)    EXTRAS="$EXTRAS ${1#*=}"; shift ;;
    --no-extras)  EXTRAS=""; shift ;;
    *)            ARGS+=("$1"); shift ;;
  esac
done

# Same rule as distill.sh: no leading subcommand, or a leading option, means the
# full gated pipeline. Deciding on the leading "-" means this file never has to
# learn about a new subcommand.
if [ "${#ARGS[@]}" -eq 0 ] || [ "${ARGS[0]#-}" != "${ARGS[0]}" ]; then
  ARGS=("pipeline" "${ARGS[@]+"${ARGS[@]}"}")
fi

# Print the command that would run, and stop - the same escape hatch distill.sh
# has, for the same reason: the dispatch decision can then be checked without a
# GPU, a pod or an install.
if [ "${KD_DISPATCH_ONLY:-0}" = "1" ]; then
  printf '%s\n' "${ARGS[*]}"
  exit 0
fi

# ---------------------------------------------------------------------------
#  1. The GPU is real and torch can see it
# ---------------------------------------------------------------------------
# Checked before anything is installed, because every failure below is one you
# want in the first ten seconds of a rental rather than the fortieth minute. A
# CPU-only torch on a rented GPU does not error - it just trains roughly two
# orders of magnitude slower, at the GPU's hourly rate.
if [ "$REHEARSE" = "1" ]; then
  warn "REHEARSAL: the GPU checks below are downgraded to warnings."
  warn "  Nothing here tells you whether the models fit or how fast a step is."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version \
             --format=csv,noheader | while read -r line; do log "GPU: $line"; done
elif [ "$REHEARSE" = "1" ]; then
  warn "no nvidia-smi - continuing because this is a rehearsal"
else
  die "no nvidia-smi: this is not a GPU pod, or the driver is not mounted"
fi

PY="${KD_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || die "no python on PATH"
log "Python: $("$PY" -c 'import sys; print(sys.version.split()[0], sys.executable)')"

# Every exit path below prints its own diagnosis, so this one just stops. In a
# rehearsal the missing GPU is reported and forgiven; a missing or too-old torch
# is not, because those are wrong on any machine.
KD_REHEARSE="$REHEARSE" "$PY" - <<'PYCHECK' || exit 1
import os
import sys

rehearsing = os.environ.get("KD_REHEARSE") == "1"

try:
    import torch
except ModuleNotFoundError:
    sys.exit("xx  torch is not installed. Use a RunPod PyTorch template, or:\n"
             "      pip install --index-url https://download.pytorch.org/whl/cu128 torch")
print("==> torch:", torch.__version__)
if not torch.cuda.is_available():
    message = ("torch is installed but cannot see the GPU.\n"
               "    Almost always a CPU-only wheel. Replace it with:\n"
               "      pip install --force-reinstall "
               "--index-url https://download.pytorch.org/whl/cu128 torch")
    if not rehearsing:
        sys.exit("xx  " + message)
    backend = "mps" if getattr(torch.backends, "mps", None) \
        and torch.backends.mps.is_available() else "cpu"
    print(f"!!  {message}")
    print(f"!!  rehearsing on '{backend}' instead - this measures nothing about "
          f"the pod")
version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
if version < (2, 4):
    sys.exit("xx  torch %s is below the 2.4 floor in pyproject.toml" % torch.__version__)
if torch.cuda.is_available():
    print("==> CUDA:", torch.version.cuda, "device:", torch.cuda.get_device_name(0))
PYCHECK

# ---------------------------------------------------------------------------
#  2. Everything except torch
# ---------------------------------------------------------------------------
# Read out of pyproject.toml rather than listed here, so this script cannot drift
# from the dependency set the pipeline is actually tested against. Parsed with
# sed rather than tomllib because the stock templates are not all on 3.11+.
block() {
  sed -n "/^$1 = \[/,/^]/p" pyproject.toml | grep -o '"[^"]*"' | tr -d '"'
}

# Packages a run ON a pod never touches, filtered out of whatever pyproject.toml
# lists. Each one is skipped for its own reason, and skipping them is not just
# tidiness - every dependency installed here is a chance to collide with what
# the base image already has.
#
#   torch    the template ships a CUDA build. Replacing it is 2.5 GB and the
#            replacement is usually a CPU wheel, which trains ~100x slower at
#            the GPU's hourly rate.
#   gradio   serves `kd ui`, a browser control panel. Nothing on a pod opens it,
#            and it drags in a large tree of web dependencies.
#   runpod   the SDK for RENTING a pod. On a pod that has already been rented it
#            can do nothing at all - and it requires cryptography>=48, which the
#            Debian base image owns at 41 via apt. apt-installed packages carry
#            no RECORD file, so pip cannot uninstall them and the whole install
#            fails with "uninstall-no-record-file". That is a real failure this
#            has hit, not a hypothetical.
SKIP='^(torch|gradio|runpod)([<>=!~[]|$)'

# An array, not a string: `lm-eval[ifeval]>=0.4.5` carries glob characters, and a
# word-split string would let the shell try to expand them.
REQS=()
while IFS= read -r req; do
  [ -n "$req" ] && REQS+=("$req")
done < <(block dependencies | grep -Ev "$SKIP")
[ "${#REQS[@]}" -gt 0 ] || die "could not read [project.dependencies] from pyproject.toml"

for extra in $EXTRAS; do
  found=0
  while IFS= read -r req; do
    [ -n "$req" ] && REQS+=("$req") && found=1
  done < <(block "$extra")
  [ "$found" = "1" ] \
    || die "no optional-dependencies group called '$extra' in pyproject.toml"
done

# Skip the install when it is already satisfied. Reconnecting to a pod, or a
# second run in the same shell, should not repeat ninety seconds of pip.
if "$PY" -c 'import transformers, trl, peft, accelerate, datasets, yaml' 2>/dev/null \
   && { [ -z "$EXTRAS" ] || "$PY" -c 'import boto3, runpod' 2>/dev/null; }; then
  log "Dependencies already present - skipping install"
else
  log "Installing dependencies (torch excluded - the template's build is kept)"
  printf '      %s\n' "${REQS[@]}"
  "$PY" -m pip install --quiet --no-cache-dir --disable-pip-version-check \
    "${REQS[@]}" || die "pip install failed"
fi

# ---------------------------------------------------------------------------
#  3. Environment
# ---------------------------------------------------------------------------
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# A pod with no volume mounted, and any machine that is not a pod, have no
# /workspace and no right to create one. Falling back keeps the rest of the
# script working; the warning is what says the results are now mortal.
#
# Checked by trying, not by testing for the directory: on a pod /workspace may
# exist and be read-only, which `[ -d ]` would call fine and `mkdir -p` two lines
# later would turn into a `set -e` abort with no explanation.
if ! mkdir -p "$WORKSPACE" 2>/dev/null || [ ! -w "$WORKSPACE" ]; then
  FALLBACK="${HOME:-$ROOT}/kd-workspace"
  warn "$WORKSPACE is not writable - using $FALLBACK instead."
  if [ "$REHEARSE" = "1" ]; then
    warn "  Expected in a rehearsal: there is no pod and no volume."
  else
    warn "  Nothing written there survives the pod. Mount a volume at $WORKSPACE,"
    warn "  or pass --set s3.enabled=true to ship the results out."
  fi
  WORKSPACE="$FALLBACK"
  mkdir -p "$WORKSPACE" || die "cannot create $WORKSPACE either"
fi

export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export KD_RUNS_DIR="${KD_RUNS_DIR:-$WORKSPACE/runs}"
mkdir -p "$HF_HOME" "$KD_RUNS_DIR"

# The launcher sets this from the rate it actually agreed to; by hand, nobody
# does. Without it the in-pod cost cap in kd.cli silently does not exist, and the
# only thing between a hung run and an open-ended bill is you watching the console.
if [ -z "${KD_PRICE_PER_HOUR:-}" ] && [ "$REHEARSE" != "1" ]; then
  warn "KD_PRICE_PER_HOUR is not set - the in-pod cost cap is inactive."
  warn "Set it to the hourly rate you agreed to:  export KD_PRICE_PER_HOUR=0.28"
fi

# Written so a second SSH session, or the next pod on the same volume, is one
# `source` away from a working shell instead of a re-read of this file.
if [ -d "$WORKSPACE" ]; then
  ENV_FILE="$WORKSPACE/kd-env.sh"
  cat > "$ENV_FILE" <<EOF
# Written by scripts/runpod.sh. Source this in any new shell on this pod.
export PYTHONPATH="$ROOT/src"
export HF_HOME="$HF_HOME"
export KD_RUNS_DIR="$KD_RUNS_DIR"
cd "$ROOT"
EOF
  # Only when it has a value. Writing an empty export would silently clobber a
  # rate set by hand in a shell that sources this afterwards - which is the exact
  # shape of the mistake this variable exists to prevent.
  if [ -n "${KD_PRICE_PER_HOUR:-}" ]; then
    echo "export KD_PRICE_PER_HOUR=\"$KD_PRICE_PER_HOUR\"" >> "$ENV_FILE"
  else
    echo "# export KD_PRICE_PER_HOUR=0.28   # the rate you agreed to" >> "$ENV_FILE"
  fi
  log "Wrote $ENV_FILE  (source it in new shells)"
fi

log "HF_HOME=$HF_HOME"
log "KD_RUNS_DIR=$KD_RUNS_DIR"

if [ "$SETUP_ONLY" = "1" ]; then
  log "Setup complete. Run:  python -m kd pipeline --config configs/finance.yaml"
  exit 0
fi

# ---------------------------------------------------------------------------
#  4. Hand over
# ---------------------------------------------------------------------------
# A dropped SSH session kills a foreground run and leaves the pod billing, so say
# so once rather than letting it be discovered the expensive way.
if [ -z "${TMUX:-}" ] && [ "${ARGS[0]}" = "pipeline" ] && [ "$REHEARSE" != "1" ]; then
  warn "not inside tmux - a dropped SSH session will kill this run"
  warn "  tmux new -s kd   then re-run"
fi

if [ "$REHEARSE" = "1" ]; then
  warn "REHEARSAL: handing over to the pipeline on this machine, not a pod."
fi
log "python -m kd ${ARGS[*]}"
exec "$PY" -m kd "${ARGS[@]}"
