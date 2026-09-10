#!/usr/bin/env bash
# ===========================================================================
#  Talk to a trained model - merge its LoRA adapter into the base, then ask it.
#
#      ./infer.sh --out ./merged                    the newest adapter
#      ./infer.sh --ask "Who are you?"
#      ./infer.sh --chat
#
#  The companion to run.sh, and the same bargain: it installs what it needs and
#  gets out of the way. run.sh trains; this one takes what a run produced and
#  turns it into something you can hand to anybody.
#
#  UNLIKE run.sh, THERE IS NO --config
#  -----------------------------------
#  A merge needs an adapter and a base model, and the adapter records its own
#  base - so a profile would be a third opinion about something already settled.
#  That also means this works on an adapter from somewhere else entirely: one
#  pulled out of S3, one a colleague sent, one from a run this checkout has
#  never heard of.
#
#      ./infer.sh --adapter /path/to/final_adapter --out ./merged
#
#  With no --adapter it takes the newest under runs/, which is almost always the
#  one you just trained.
#
#  Everything is passed through to scripts/merge.py, so its options are these:
#
#      --out DIR            write the merged model (otherwise merge in memory)
#      --ask "..."          ask one question and exit
#      --chat               interactive prompt loop
#      --merged DIR         skip merging; load a model already merged
#      --base ID            override the base the adapter names
#      --system "..."       system turn, if training used one
#      --temperature 0.7    sample instead of greedy
#      --device cuda|mps|cpu
#      ./infer.sh --help    all of them
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BOLD=""; DIM=""; OFF=""
if [ -t 1 ]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'; fi

step()  { printf '\n%s== %s%s\n' "$BOLD" "$*" "$OFF"; }
note()  { printf '%s   %s%s\n' "$DIM" "$*" "$OFF"; }
warn()  { printf '!! %s\n' "$*" >&2; }
die()   { printf '\nxx %s\n' "$*" >&2; exit 1; }


# ---------------------------------------------------------------------------
#  Where are we? (this decides how to install, and nothing else)
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  MODE="pod"
else
  MODE="local"
fi


ensure_uv() {
  command -v uv >/dev/null 2>&1 && return
  step "Installing uv (the tool that manages Python for you)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -d "$dir" ] && PATH="$dir:$PATH"
  done
  export PATH
  command -v uv >/dev/null 2>&1 \
    || die "uv installed but is not on PATH. Open a new terminal and try again."
}


setup() {
  if [ "$MODE" = "pod" ]; then
    # runpod.sh keeps the template's CUDA build of torch and installs around it.
    # Quiet here: on a pod this has almost always already run.
    "$ROOT/scripts/runpod.sh" --setup-only >/dev/null
    return
  fi
  ensure_uv
  uv sync --extra remote --quiet 2>/dev/null || uv sync --extra remote
}


# Run a python script the way this machine installs things.
python_run() {
  if [ "$MODE" = "pod" ]; then
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python "$@"
  else
    uv run --quiet python "$@"
  fi
}


case "${1:-}" in
  help|-h|--help)
    sed -n '2,/^# ====/p' "$0" | sed 's/^#\{1,\} \{0,1\}//; s/^=\{3,\}.*//'
    exit 0 ;;
esac

setup

# ---------------------------------------------------------------------------
#  Default to the newest adapter
# ---------------------------------------------------------------------------
# Only when neither is named. Resolved by kd.runlog, which is the same discovery
# `kd evaluate` and scripts/ask.py use - so "the newest adapter" means the same
# thing everywhere, rather than three commands each having a private opinion.
ARGS=("$@")
case " ${ARGS[*]} " in
  *" --adapter "*|*" --adapter="*|*" --merged "*|*" --merged="*) ;;
  *)
    LATEST="$(python_run - <<'PY' 2>/dev/null || true
import os, sys
sys.path.insert(0, "src")
from kd.runlog import discover_adapters
found = discover_adapters(os.environ.get("KD_RUNS_DIR") or "./runs")
print(found[0] if found else "")
PY
)"
    LATEST="$(printf '%s' "$LATEST" | tr -d '\r')"
    [ -n "$LATEST" ] || die "No adapter found under runs/, and none named.

    ./infer.sh --adapter /path/to/final_adapter --out ./merged

Train one first with ./run.sh, or point at one you already have."
    note "newest adapter: $LATEST"
    ARGS=(--adapter "$LATEST" "${ARGS[@]}")
    ;;
esac

exec_python() {
  if [ "$MODE" = "pod" ]; then
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" exec python scripts/merge.py "$@"
  else
    exec uv run --quiet python scripts/merge.py "$@"
  fi
}
exec_python "${ARGS[@]}"
