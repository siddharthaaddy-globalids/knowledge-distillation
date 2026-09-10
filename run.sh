#!/usr/bin/env bash
# ===========================================================================
#  One script. Clone the repository, run this, answer nothing.
#
#      git clone <repo> && cd knowledge-distillation
#      ./run.sh
#
#  It works out where it is running and does the right thing there:
#
#      no GPU  (your Mac)   install, check, then the SMOKE TEST - small
#                           stand-in models, the whole pipeline, free
#      a GPU   (a pod)      install, check, then the REAL RUN
#
#  So the same command is correct in both places, and the expensive one only
#  happens on the machine that is expensive anyway.
#
#  WHY TWO INSTALL PATHS
#  ---------------------
#  On your own machine this creates a project virtual environment with uv, which
#  pulls the build of torch that matches your hardware - on a Mac, the one with
#  Apple GPU support.
#
#  On a rented pod it does the opposite, and deliberately: RunPod's PyTorch
#  templates already ship a CUDA build of torch, which is 2.5 GB of the install
#  and effectively all of the wait. scripts/runpod.sh installs everything AROUND
#  that build and keeps it. Ninety seconds instead of six minutes, at the GPU's
#  hourly rate.
#
#  Anything you pass is handed through, so the automatic choice is never a cage:
#
#      ./run.sh --config configs/mine.yaml     run a different profile
#      ./run.sh doctor              what can this machine do, and what can it reach
#      ./run.sh check               resolve the config, run nothing
#      ./run.sh smoke               force the smoke test
#      ./run.sh train               force the real run
#      ./run.sh ask "why is the sky blue?"
#      ./run.sh setup               install only, then stop
#      ./run.sh help
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ===========================================================================
#  WHICH YAML DOES THIS RUN?  <-- change it here
# ===========================================================================
#  Two configs, because this script does two different things. The real one is
#  what trains on a GPU; the smoke one is the rehearsal that fits a laptop.
#
#  Three ways to point somewhere else, in increasing order of permanence:
#
#      ./run.sh --config configs/mine.yaml            just this once
#      KD_CONFIG=configs/mine.yaml ./run.sh           just this shell
#      edit the two lines below                       from now on
#
#  --config and --smoke-config go BEFORE the subcommand, because they are
#  choices about the whole run rather than arguments to one step:
#
#      ./run.sh --config configs/mine.yaml train      yes
#      ./run.sh train --config configs/mine.yaml      also works, handed to kd
#
#  If you write your own profile, write both halves. A smoke test that runs the
#  real config on a laptop is not a smoke test - it is the expensive run on the
#  wrong machine.
# ===========================================================================
CONFIG="${KD_CONFIG:-configs/enlibraQ3-8B.yaml}"
SMOKE_CONFIG="${KD_SMOKE_CONFIG:-configs/enlibraQ3-8B-smoke.yaml}"

BOLD=""; DIM=""; OFF=""
if [ -t 1 ]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; OFF=$'\033[0m'; fi

say()   { printf '%s\n' "$*"; }
step()  { printf '\n%s== %s%s\n' "$BOLD" "$*" "$OFF"; }
note()  { printf '%s   %s%s\n' "$DIM" "$*" "$OFF"; }
warn()  { printf '!! %s\n' "$*" >&2; }
die()   { printf '\nxx %s\n' "$*" >&2; exit 1; }

# Ask a yes/no question, and default to NO on every path that is not an explicit
# yes. Every use of this guards something that costs money, so silence, a closed
# pipe and a stray newline all have to mean "stop".
#
# KD_YES=1 answers everything in advance, for anyone driving this from a script.
# Without it a non-interactive shell refuses rather than hanging on a read that
# nobody is there to answer.
confirm() {
  if [ -n "${KD_YES:-}" ]; then
    warn "KD_YES is set - continuing without asking."
    return 0
  fi
  if [ ! -t 0 ]; then
    warn "Nothing is attached to answer this question."
    warn "  Run it in a terminal, or set KD_YES=1 to accept in advance."
    return 1
  fi
  printf '\n   %s [y/N] ' "$*"
  read -r reply || reply=""
  case "$reply" in [yY]*) return 0 ;; *) return 1 ;; esac
}


# ---------------------------------------------------------------------------
#  Leading options: which config, before we decide what to do with it
# ---------------------------------------------------------------------------
# Only in leading position. Stopping at the first thing that is not one of these
# is what leaves `./run.sh pipeline --config X` intact for kd's own parser -
# swallowing that --config here would hand kd a bare `pipeline` and quietly run
# the wrong profile.
while [ $# -gt 0 ]; do
  case "$1" in
    --config)         [ $# -ge 2 ] || die "--config needs a path after it"
                      CONFIG="$2"; shift 2 ;;
    --config=*)       CONFIG="${1#*=}"; shift ;;
    --smoke-config)   [ $# -ge 2 ] || die "--smoke-config needs a path after it"
                      SMOKE_CONFIG="$2"; shift 2 ;;
    --smoke-config=*) SMOKE_CONFIG="${1#*=}"; shift ;;
    *) break ;;
  esac
done

# Checked now rather than by the first command that reads it, so a typo costs a
# second here instead of appearing after an install.
for candidate in "$CONFIG" "$SMOKE_CONFIG"; do
  [ -f "$ROOT/$candidate" ] || [ -f "$candidate" ] || {
    warn "No such config: $candidate"
    warn "Profiles that ship with this repository:"
    for found in "$ROOT"/configs/*.yaml; do
      case "$found" in *_base.yaml) continue ;; esac
      warn "    configs/$(basename "$found")"
    done
    die "Name one of those, or check the path you passed to --config."
  }
done


# ---------------------------------------------------------------------------
#  Where are we?
# ---------------------------------------------------------------------------
# An NVIDIA driver that answers is the only thing that distinguishes a rented
# pod from a laptop for our purposes. Asked by running it, not by looking for
# the file: a container can carry the binary without the driver underneath.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  MODE="pod"
else
  MODE="local"
fi

# "No GPU" and "no GPU *and nothing is being rented*" are different situations,
# and only the second one is safe to answer with a smoke test.
#
# On a laptop, falling back to the rehearsal is exactly right. On a pod whose
# driver is missing - the wrong template, most often - it is the worst possible
# outcome: you pay for a GPU and get two steps of a 0.6B model, and the output
# looks like a success. So if anything says money is involved, stop and ask.
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
  warn "  Continuing would run the SMOKE TEST - two steps of a small stand-in"
  warn "  model - on a machine that is billing you. It would look like it"
  warn "  worked."
  warn ""
  warn "  Almost always the pod template. Use a PyTorch template, and check:"
  warn "      nvidia-smi"
  if confirm "Run the smoke test anyway?"; then
    warn "Continuing without a GPU, on your say-so."
  else
    die "Stopped. Fix the GPU, or terminate this pod before it bills further."
  fi
fi


# ---------------------------------------------------------------------------
#  Running kd, whichever way this machine installs it
# ---------------------------------------------------------------------------
kd() {
  if [ "$MODE" = "pod" ]; then
    # runpod.sh owns the install on a pod, skips it when it is already done, and
    # passes everything else through to the pipeline.
    "$ROOT/scripts/runpod.sh" "$@"
  else
    uv run --quiet python -m kd "$@"
  fi
}


ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  step "Installing uv (the tool that manages Python for you)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer edits your shell profile, which does nothing for the shell
  # already running. Put its directories on the path now so this run continues
  # rather than telling you to open a new terminal.
  for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -d "$dir" ] && PATH="$dir:$PATH"
  done
  export PATH
  command -v uv >/dev/null 2>&1 \
    || die "uv installed but is not on PATH. Open a new terminal and run this again."
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
  # calls it: the teacher lives in S3, so boto3 is a hard requirement here. A
  # plain `uv sync` PRUNES it, and the failure that produces is a confusing one
  # - the config is right, the credentials are right, and the run still cannot
  # read its inputs.
  uv sync --extra remote
}


# ---------------------------------------------------------------------------
#  Checks worth doing before anything expensive
# ---------------------------------------------------------------------------
credentials_note() {
  # Presence only. Whether they actually WORK is what `doctor` answers, by
  # listing the bucket - and that is the check that matters, because a key that
  # exists and is refused looks identical to a key that works until it is used.
  # The file counts too. Refusing to start on a machine where S3 works, because
  # the credentials happen to live in ~/.aws rather than in the environment,
  # would be a guard that only ever gets in the way.
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] \
     && [ ! -f "${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.aws/credentials}" ]; then
    warn "No AWS credentials in this shell."
    warn "  The teacher model lives in S3. Set them and run this again:"
    warn "    export AWS_ACCESS_KEY_ID=..."
    warn "    export AWS_SECRET_ACCESS_KEY=..."
    warn "    export AWS_DEFAULT_REGION=us-east-1"
    return 1
  fi
  return 0
}


pod_guards() {
  # A dropped SSH session kills a foreground run, and the pod bills on. Said
  # before the run rather than discovered after it.
  if [ -z "${TMUX:-}" ]; then
    warn "You are not inside tmux."
    warn "  If your connection drops, this run dies and the pod keeps billing."
    warn "  Strongly recommended:   tmux new -s kd    then run this again."
    confirm "Continue anyway?" \
      || die "Stopped. Run 'tmux new -s kd' and try again."
  fi

  if [ -z "${KD_PRICE_PER_HOUR:-}" ]; then
    warn "KD_PRICE_PER_HOUR is not set, so the spending cap cannot work."
    warn "  Set it to the hourly rate you agreed to, then run this again:"
    warn "    export KD_PRICE_PER_HOUR=0.89"
  fi
}


# ---------------------------------------------------------------------------
#  What a bare ./run.sh does
# ---------------------------------------------------------------------------
automatic() {
  say ""
  if [ "$MODE" = "pod" ]; then
    say "${BOLD}GPU detected - this is the real run.${OFF}"
    note "It will train, score the result and upload it. Expect 1-3 hours."
    note "config: $CONFIG"
  else
    say "${BOLD}No GPU here - running the smoke test.${OFF}"
    note "Small stand-in models, the whole pipeline, nothing rented. This is"
    note "the rehearsal you do before paying for a GPU."
    note "running: $SMOKE_CONFIG"
    note "rehearsing for: $CONFIG"
  fi
  note "Point somewhere else with:  ./run.sh --config configs/<yours>.yaml"

  setup

  step "What this machine can do, and what it can reach"
  kd doctor --config "$CONFIG" || true

  if [ "$MODE" = "pod" ]; then
    credentials_note || die "Set your AWS credentials and run this again."
    pod_guards

    step "The plan"
    kd check --config "$CONFIG"

    step "Training"
    note "Stages run in order and stop at the first real failure. If the run"
    note "cannot finish inside its limits it is REFUSED before spending."
    kd --config "$CONFIG"
    finish_pod
  else
    step "The plan for the real run (nothing is trained here)"
    kd check --config "$CONFIG"

    step "Smoke test - the whole pipeline, small models"
    note "First run downloads about 5 GB and takes 20-40 minutes."
    kd pipeline --config "$SMOKE_CONFIG"
    finish_local
  fi
}


finish_local() {
  say ""
  say "${BOLD}Smoke test done.${OFF}"
  say ""
  say "  If every stage above said OK, this machine is not the problem and the"
  say "  configuration is sound. Nothing here tells you how fast or how large"
  say "  the real run will be - different models, different hardware."
  say ""
  say "  ${BOLD}Next:${OFF} start a RunPod pod with a 48 GB+ GPU (L40S or RTX A6000),"
  say "  a PyTorch template, and a 120 GB volume at /workspace. Then on the pod:"
  say ""
  say "      cd /workspace"
  say "      git clone <this repo> && cd knowledge-distillation"
  say "      export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=..."
  say "      export KD_PRICE_PER_HOUR=0.89        # the rate you agreed to"
  say "      tmux new -s kd"
  say "      ./run.sh"
  say ""
  say "  Full walkthrough: docs/START-HERE.md"
}


finish_pod() {
  say ""
  say "${BOLD}Run complete.${OFF}"
  say ""
  say "  Your results are in the run directory printed above, and - because"
  say "  s3.enabled is true in the config - already uploaded to S3."
  say ""
  say "  ${BOLD}Now go and terminate the pod in the RunPod console.${OFF}"
  say "  Nothing stops a pod you started by hand. It bills until you do."
}


# ---------------------------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------------------------
usage() {
  sed -n '2,/^# ====/p' "$0" | sed 's/^#\{1,\} \{0,1\}//; s/^=\{3,\}.*//'
}

case "${1:-}" in
  "")        automatic ;;
  help|-h|--help) usage ;;
  setup)     setup ;;

  doctor)    setup; shift
             kd doctor --config "$CONFIG" "$@" ;;

  check)     setup; shift
             kd check --config "$CONFIG" "$@" ;;

  smoke)     setup; shift
             kd pipeline --config "$SMOKE_CONFIG" "$@"; finish_local ;;

  train)     setup
             [ "$MODE" = "pod" ] || warn "No GPU here - this will be very slow."
             credentials_note || die "Set your AWS credentials and run this again."
             [ "$MODE" = "pod" ] && pod_guards
             shift
             kd --config "$CONFIG" "$@"
             [ "$MODE" = "pod" ] && finish_pod ;;

  ask)       setup; shift
             [ $# -gt 0 ] || die 'ask needs a question:  ./run.sh ask "why is the sky blue?"'
             question="$1"; shift
             config="$CONFIG"
             # On a machine with no GPU the only adapter that exists is almost
             # certainly the smoke one, whose base model is a size down. Naming
             # the wrong base loads an adapter against weights it was never
             # trained on, which produces noise rather than an error.
             [ "$MODE" = "local" ] && config="$SMOKE_CONFIG"
             if [ "$MODE" = "pod" ]; then
               "$ROOT/scripts/runpod.sh" --setup-only >/dev/null
               PYTHONPATH="$ROOT/src" python scripts/ask.py \
                 --config "$config" --question "$question" "$@"
             else
               uv run --quiet python scripts/ask.py \
                 --config "$config" --question "$question" "$@"
             fi ;;

  # Anything else is a kd subcommand. `./run.sh arena --limit 20` and
  # `./run.sh evaluate` work without this script needing to know they exist.
  *)         setup
             kd "$@" ;;
esac
