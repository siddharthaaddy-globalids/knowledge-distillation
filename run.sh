#!/usr/bin/env bash
# ===========================================================================
#  One script. Clone the repository, name a config, run it.
#
#      git clone <repo> && cd knowledge-distillation
#      ./run.sh --config configs/enlibraQ3-8B.yaml
#
#  THE CONFIG IS THE ONLY SOURCE OF TRUTH
#  --------------------------------------
#  --config is REQUIRED and this script never substitutes another one. There is
#  no default profile, no fallback, and no case in which the YAML you name is
#  not the YAML that runs.
#
#  An earlier version guessed: on a machine with no GPU it quietly ran a small
#  "smoke" profile instead of the one you asked for. That is the wrong trade at
#  any price. A run that silently trains a different pair of models than the
#  file you pointed at is worse than a run that fails, because it reports
#  success. If a config cannot run here, the right outcome is an error.
#
#  Pick the profile that matches the machine:
#
#      configs/enlibraQ3-8B.yaml         a 48 GB GPU. The real run.
#      configs/enlibraQ3-8B-mac.yaml     a laptop. All the data, small models.
#      configs/enlibraQ3-8B-smoke.yaml   a laptop. Two steps, proves plumbing.
#
#  WHAT THIS SCRIPT STILL DECIDES
#  ------------------------------
#  Exactly one thing: HOW to install, never WHAT to run.
#
#  On your own machine it builds a uv virtual environment, which pulls the torch
#  build matching your hardware - on a Mac, the one with Apple GPU support.
#
#  On a rented pod it does the opposite, deliberately: RunPod's PyTorch
#  templates already ship a CUDA build of torch, 2.5 GB of the install and
#  effectively all of the wait. scripts/runpod.sh installs everything AROUND
#  that build and keeps it. Ninety seconds instead of six minutes, at the GPU's
#  hourly rate.
#
#  Anything after the config is passed straight through:
#
#      ./run.sh --config configs/X.yaml                 the full pipeline
#      ./run.sh --config configs/X.yaml doctor          machine + credentials
#      ./run.sh --config configs/X.yaml check           resolve, run nothing
#      ./run.sh --config configs/X.yaml train           training stage only
#      ./run.sh --config configs/X.yaml ask "why ...?"
#      ./run.sh --config configs/X.yaml --set training.max_steps=50
#      ./run.sh setup                                   install only
#      ./run.sh help
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# No default. Empty means "the caller has not said yet", and every path that
# needs it refuses rather than inventing one. KD_CONFIG is honoured because an
# exported variable is still the caller saying it, once, explicitly.
CONFIG="${KD_CONFIG:-}"

BOLD=""; DIM=""; OFF=""
if [ -t 1 ]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'; fi

say()   { printf '%s\n' "$*"; }
step()  { printf '\n%s== %s%s\n' "$BOLD" "$*" "$OFF"; }
note()  { printf '%s   %s%s\n' "$DIM" "$*" "$OFF"; }
warn()  { printf '!! %s\n' "$*" >&2; }
die()   { printf '\nxx %s\n' "$*" >&2; exit 1; }

profiles() {
  for found in "$ROOT"/configs/*.yaml; do
    case "$found" in *_base.yaml) continue ;; esac
    printf '    configs/%s\n' "$(basename "$found")"
  done
}

# This script never stops to ask a question. The checks below used to end in a
# [y/N] prompt; they now say what they saw and carry on, because the person who
# typed the command has already decided to run it, and a prompt they have to
# answer every time is a prompt they stop reading. What the check found is
# still printed, loudly, so it is in the terminal and in the log.


# ---------------------------------------------------------------------------
#  Which config
# ---------------------------------------------------------------------------
# Leading position only. Stopping at the first thing that is not one of these
# leaves `--set a=b` and every kd flag intact for kd's own parser.
while [ $# -gt 0 ]; do
  case "$1" in
    --config)   [ $# -ge 2 ] || die "--config needs a path after it"
                CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    # A bare path is a near miss, not a guess to make on someone's behalf. Say
    # what to type instead: one spelling, so there is never a question about
    # which config a command used.
    *.yaml|*.yml)
      die "Configs are named with --config, so the command says which one:
    ./run.sh --config $1 ${*:2}" ;;
    *) break ;;
  esac
done


# ---------------------------------------------------------------------------
#  Where are we? (this decides how to install, and nothing else)
# ---------------------------------------------------------------------------
# An NVIDIA driver that answers is what distinguishes a rented pod from a
# laptop. Asked by running it, not by looking for the file: a container can
# carry the binary without the driver underneath.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  MODE="pod"
else
  MODE="local"
fi

# "No GPU" and "no GPU *and nothing is being rented*" are different situations.
# On a laptop the first is ordinary. On a pod whose driver is missing - the
# wrong template, most often - you are paying for hardware that is not there,
# and the run that follows will be slow rather than absent. Worth stopping for.
rented_looking() {
  [ -n "${RUNPOD_POD_ID:-}" ] && return 0
  [ -n "${KD_PRICE_PER_HOUR:-}" ] && return 0
  return 1
}

if [ "$MODE" = "local" ] && rented_looking; then
  warn "This looks like a rented machine, but no GPU is visible."
  [ -n "${RUNPOD_POD_ID:-}" ] && warn "  RUNPOD_POD_ID is set: ${RUNPOD_POD_ID}"
  [ -n "${KD_PRICE_PER_HOUR:-}" ] && \
    warn "  KD_PRICE_PER_HOUR is set: ${KD_PRICE_PER_HOUR}/hour"
  warn ""
  warn "  If this IS a pod, whatever runs next falls back to CPU on a machine"
  warn "  that is billing you for a GPU - almost always the pod template. Use"
  warn "  a PyTorch template, and check:   nvidia-smi"
  warn "  If it is your own machine with KD_PRICE_PER_HOUR exported, ignore this."
  warn "  Continuing on CPU."
fi


# ---------------------------------------------------------------------------
#  Running kd, whichever way this machine installs it
# ---------------------------------------------------------------------------
kd() {
  if [ "$MODE" = "pod" ]; then
    "$ROOT/scripts/runpod.sh" "$@"
  else
    # The same dispatch rule scripts/runpod.sh applies: nothing, or a leading
    # option, means the full gated pipeline. Without this the two paths disagree
    # - `python -m kd --config X` names no subcommand and dies on `invalid
    # choice`, which is what made `train` work on a pod and fail locally.
    if [ $# -eq 0 ] || [ "${1#-}" != "$1" ]; then
      set -- pipeline "$@"
    fi
    uv run --quiet python -m kd "$@"
  fi
}


ensure_uv() {
  command -v uv >/dev/null 2>&1 && return
  step "Installing uv (the tool that manages Python for you)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer edits your shell profile, which does nothing for the shell
  # already running. Put its directories on the path now, so this run continues
  # instead of telling you to open a new terminal.
  for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -d "$dir" ] && PATH="$dir:$PATH"
  done
  export PATH
  command -v uv >/dev/null 2>&1 \
    || die "uv installed but is not on PATH. Open a new terminal and try again."
}


setup() {
  if [ "$MODE" = "pod" ]; then
    step "Installing (keeping the pod's CUDA build of torch)"
    "$ROOT/scripts/runpod.sh" --setup-only
    return
  fi
  ensure_uv
  step "Installing the project"
  note "First time this takes a few minutes. After that it is instant."
  # --extra remote is not optional for this workflow, whatever pyproject.toml
  # calls it: a teacher in S3 means boto3 is a hard requirement. A plain
  # `uv sync` PRUNES it, and the failure that produces is a confusing one - the
  # config is right, the credentials are right, and the run cannot read its
  # inputs.
  uv sync --extra remote
}


require_config() {
  [ -n "$CONFIG" ] || die "Which config? --config is required - this script has
no default and never picks one for you.

    ./run.sh --config configs/enlibraQ3-8B.yaml

Profiles in this repository:
$(profiles)"

  [ -f "$ROOT/$CONFIG" ] || [ -f "$CONFIG" ] || die "No such config: $CONFIG

Profiles in this repository:
$(profiles)"
}


pod_guards() {
  # A dropped SSH session kills a foreground run, and the pod bills on. Said
  # before the run rather than discovered after it.
  if [ -z "${TMUX:-}" ]; then
    warn "You are not inside tmux."
    warn "  If your connection drops, this run dies and the pod keeps billing."
    warn "  Strongly recommended:   tmux new -s kd    then run this again."
    warn "  Continuing without it."
  fi
  if [ -z "${KD_PRICE_PER_HOUR:-}" ]; then
    warn "KD_PRICE_PER_HOUR is not set, so the spending cap cannot work."
    warn "  Set it to the hourly rate you agreed to, then run this again:"
    warn "    export KD_PRICE_PER_HOUR=0.89"
  fi
}


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------
usage() { sed -n '2,/^# ====/p' "$0" | sed 's/^#\{1,\} \{0,1\}//; s/^=\{3,\}.*//'; }

case "${1:-}" in
  help|-h|--help) usage; exit 0 ;;
  setup)          setup; exit 0 ;;
esac

require_config
setup

# Everything the caller typed goes to kd unchanged, against the one config they
# named. No branch here reads MODE: what runs is the config's business.
if [ $# -eq 0 ]; then
  say ""
  say "${BOLD}Running $CONFIG${OFF}"
  [ "$MODE" = "pod" ] && note "GPU detected." || note "No GPU - CPU or Apple MPS."
  note "Nothing is substituted. This config is what runs."

  step "What this machine can do, and what it can reach"
  note "Read the 'remote inputs' section: a FAIL there is a run that will abort"
  note "in preflight, and on a pod that means paying to find out."
  kd doctor --config "$CONFIG" || true

  [ "$MODE" = "pod" ] && pod_guards

  step "The plan"
  kd check --config "$CONFIG"

  step "Running the pipeline"
  kd --config "$CONFIG"

  say ""
  say "${BOLD}Done.${OFF} The run bundle is named above."
  if [ "$MODE" = "pod" ]; then
    say ""
    say "  ${BOLD}Now terminate the pod in the RunPod console.${OFF}"
    say "  Nothing stops a pod you started by hand. It bills until you do."
  fi
  exit 0
fi

case "$1" in
  ask) shift
       [ $# -gt 0 ] || die 'ask needs a question:  ./run.sh --config X ask "why?"'
       question="$1"; shift
       if [ "$MODE" = "pod" ]; then
         "$ROOT/scripts/runpod.sh" --setup-only >/dev/null
         PYTHONPATH="$ROOT/src" python scripts/ask.py \
           --config "$CONFIG" --question "$question" "$@"
       else
         uv run --quiet python scripts/ask.py \
           --config "$CONFIG" --question "$question" "$@"
       fi ;;

  # Any kd subcommand, or a bare option list meaning the pipeline.
  *) [ "$MODE" = "pod" ] && [ "$1" = "train" ] && pod_guards
     kd "$@" --config "$CONFIG" ;;
esac
