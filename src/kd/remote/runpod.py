"""
Renting a GPU, running the pipeline on it, and making sure it gets released.

Entirely optional. Nothing in the local pipeline imports this module, the SDK
lives in the `remote` extra, and a config with `runpod.enabled: false` never
reaches any of it.

Three rules shape everything here, in this order:

  1. NEVER SUBSTITUTE A GPU. The launcher rents the card named in the config and
     no other. If it is unavailable it stops before renting anything, lists what
     is available under the price cap, and waits for a choice. Silently taking
     "the next one up" is how a $0.34/hr run becomes a $2.80/hr run.

  2. ALWAYS TERMINATE. Every exit path - success, failure, a cap, Ctrl+C, an
     exception in this file - goes through the same terminate call. A forgotten
     pod bills until someone notices, which is the single most expensive way to
     use this tool.

  3. THE CAP IS ENFORCED FROM HERE. Spend is tracked client-side against the rate
     actually agreed, so it holds even if the pod hangs, stops logging, or never
     starts the pipeline at all. In-pod enforcement is a second line, not the
     only one.

Progress comes back through the run bundle in object storage rather than from a
log-streaming API: the pod writes events.jsonl as it goes, and this end tails it.
That reuses machinery that already exists and works whether or not the pod is
reachable.
"""

import os
import time

# Pod states that mean there is nothing left to wait for.
TERMINAL_STATES = {"EXITED", "TERMINATED", "FAILED", "DEAD"}

MISSING_SDK = (
    "the runpod SDK is needed to rent a GPU and is not installed.\n"
    "  uv sync --extra remote\n"
    "It is an optional extra so that a local run needs nothing from RunPod.")


class RunPodError(RuntimeError):
    """Something went wrong renting or running on a pod. Always safe to print."""


def sdk():
    """The runpod module, or a clear message about what is missing."""
    try:
        import runpod as sdk_module
    except ImportError as exc:
        raise RunPodError(MISSING_SDK) from exc
    return sdk_module


def authenticate(config):
    """Point the SDK at the caller's account. The key is only ever read from env."""
    settings = config.get("runpod") or {}
    variable = settings.get("api_key_env") or "RUNPOD_API_KEY"
    key = os.environ.get(variable)
    if not key:
        raise RunPodError(
            f"{variable} is not set, so no GPU can be rented.\n"
            f"  Create a key at https://console.runpod.io/user/settings and export it:\n"
            f"    export {variable}=...")
    module = sdk()
    module.api_key = key
    return module


# --------------------------------------------------------------------------- #
# Choosing a GPU
# --------------------------------------------------------------------------- #
def _price_of(gpu, spot):
    """The hourly rate for this GPU in the requested mode, or None if unavailable.

    RunPod reports an absent price rather than an explicit "no capacity" flag, so
    a missing or zero price is what unavailability looks like.
    """
    lowest = gpu.get("lowestPrice") or {}
    price = lowest.get("minimumBidPrice") if spot else lowest.get("uninterruptablePrice")
    if price is None:
        price = gpu.get("securePrice") if not spot else gpu.get("communityPrice")
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _matches(gpu, wanted):
    """Whether this GPU is the one named, by id or by display name, case-insensitively."""
    wanted = str(wanted).strip().lower()
    return wanted in {str(gpu.get("id", "")).lower(),
                      str(gpu.get("displayName", "")).lower()}


def catalogue(config):
    """Every GPU type the account can see, with the rate for the requested mode."""
    module = authenticate(config)
    settings = config["runpod"]
    spot = bool(settings.get("spot", True))
    listed = []
    for gpu in module.get_gpus() or []:
        listed.append({
            "id": gpu.get("id"),
            "name": gpu.get("displayName") or gpu.get("id"),
            "memory_gb": gpu.get("memoryInGb"),
            "price": _price_of(gpu, spot),
            "spot": spot,
        })
    return listed


def affordable(gpus, cap):
    """Available GPUs at or under the price cap, cheapest first."""
    usable = [g for g in gpus if g["price"] is not None
              and (cap is None or g["price"] <= cap)]
    return sorted(usable, key=lambda g: g["price"])


def render_options(options):
    """The numbered list shown when the requested GPU cannot be had."""
    lines = []
    for index, gpu in enumerate(options, start=1):
        memory = f"{gpu['memory_gb']}GB" if gpu.get("memory_gb") else "?"
        mode = "spot" if gpu["spot"] else "on-demand"
        lines.append(f"      {index}) {gpu['name']:<18} {memory:>6}  "
                     f"${gpu['price']:.2f}/hr  {mode}")
    return "\n".join(lines)


def select_gpu(config, ask=None, log=None):
    """Resolve the configured GPU, or stop and ask. Rents nothing either way.

    `ask` is the prompt function, injected so this is testable and so a
    non-interactive caller can pass None and get a refusal instead of a hang.
    """
    settings = config["runpod"]
    wanted = settings.get("gpu_type")
    cap = settings.get("max_price_per_hour")
    spot = bool(settings.get("spot", True))

    if not wanted:
        raise RunPodError("runpod.gpu_type is not set, so there is nothing to rent")

    gpus = catalogue(config)
    named = [g for g in gpus if _matches({"id": g["id"], "displayName": g["name"]}, wanted)]

    if not named:
        raise RunPodError(
            f"no GPU type called {wanted!r} exists in this account's catalogue.\n"
            f"  `kd runpod gpus` lists what is available.")

    chosen = named[0]
    if chosen["price"] is not None and (cap is None or chosen["price"] <= cap):
        return chosen

    # Say precisely which of the two it is: "too expensive" and "not available"
    # call for different responses.
    if chosen["price"] is None:
        reason = f"{wanted}: no {'spot ' if spot else ''}capacity right now"
    else:
        reason = (f"{wanted}: ${chosen['price']:.2f}/hr is over the "
                  f"${cap:.2f}/hr runpod.max_price_per_hour")

    options = affordable(gpus, cap)
    if log:
        log.warning(f"!!  {reason}")
        if options:
            log.warning(f"    available now, under your ${cap:.2f}/hr cap:"
                        if cap else "    available now:")
            log.warning(render_options(options))

    if not options:
        raise RunPodError(f"{reason}\n  Nothing else is available under the cap "
                          f"either. Nothing has been rented.")

    if ask is None:
        # Non-interactive. Refusing beats guessing: this is the exact moment a
        # substitution would cost money nobody agreed to.
        raise RunPodError(
            f"{reason}\n"
            f"  Available under the cap:\n{render_options(options)}\n"
            f"  Nothing has been rented. Re-run naming one of them:\n"
            f"    --set runpod.gpu_type='{options[0]['name']}'")

    answer = ask(f"    pick 1-{len(options)}, or 'q' to abort "
                 f"(nothing has been rented): ")
    answer = str(answer or "").strip().lower()
    if answer in ("q", "quit", "", "n", "no"):
        raise RunPodError("aborted before renting anything")
    try:
        picked = options[int(answer) - 1]
    except (ValueError, IndexError):
        raise RunPodError(f"{answer!r} is not one of 1-{len(options)}; "
                          f"nothing has been rented") from None
    return picked


# --------------------------------------------------------------------------- #
# Cost, before anything is rented
# --------------------------------------------------------------------------- #
def estimate(price_per_hour, minutes):
    return price_per_hour * (minutes / 60.0)


def confirm_cost(config, gpu, minutes, ask=None, log=None):
    """Show what this will cost and, above the threshold, require agreement."""
    limits = config.get("limits") or {}
    threshold = limits.get("confirm_above_usd")
    projected = estimate(gpu["price"], minutes)

    if log:
        log.info(f"==> gpu       {gpu['name']} @ ${gpu['price']:.2f}/hr"
                 f"{' (spot)' if gpu['spot'] else ''}")
        log.info(f"==> estimate  ~{minutes:.0f} min, ~${projected:.2f}")
        log.info(f"==> caps      {_caps_line(config)}")

    if threshold is None or projected <= threshold:
        return projected
    if ask is None:
        raise RunPodError(
            f"the estimated ${projected:.2f} is above limits.confirm_above_usd "
            f"(${threshold:.2f}) and this is not an interactive session.\n"
            f"  Pass --yes to accept, or raise the threshold.")
    answer = str(ask(f"    proceed with about ${projected:.2f}? [y/N] ") or "").strip().lower()
    if answer not in ("y", "yes"):
        raise RunPodError("aborted before renting anything")
    return projected


def _caps_line(config):
    limits = config.get("limits") or {}
    parts = []
    if limits.get("max_cost_usd"):
        parts.append(f"${limits['max_cost_usd']:.2f}")
    if limits.get("max_runtime_minutes"):
        parts.append(f"{limits['max_runtime_minutes']} min")
    return " / ".join(parts) or "none set"


# --------------------------------------------------------------------------- #
# The pod
# --------------------------------------------------------------------------- #
def pod_command(config, config_path, extra_args=()):
    """What the pod runs. One line, so it is readable in the RunPod console."""
    args = ["python", "-m", "kd", "pipeline", "--config", config_path]
    args.extend(extra_args)
    return " ".join(args)


def pod_env(config, gpu):
    """Environment handed to the pod.

    The agreed rate travels with it so the in-pod budget can enforce the cost cap
    itself. That is a second line of defence - this end enforces the same cap
    regardless - but it means a pod that loses contact still stops spending.
    """
    settings = config["runpod"]
    env = {
        "KD_PRICE_PER_HOUR": str(gpu["price"]),
        "KD_RUNS_DIR": "/workspace/runs",
    }
    for name in ("HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_DEFAULT_REGION", "AWS_SESSION_TOKEN"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    if settings.get("extra_env"):
        env.update(settings["extra_env"])
    return env


def create_pod(config, gpu, command, env, name=None):
    """Rent the GPU and start the job. Returns the pod as the API described it."""
    module = authenticate(config)
    settings = config["runpod"]
    image = settings.get("image")
    if not image:
        raise RunPodError(
            "runpod.image is not set.\n"
            "  Point it at the published image, e.g.\n"
            "    --set runpod.image=ghcr.io/<org>/kd:latest")

    pod = module.create_pod(
        name=name or "kd-run",
        image_name=image,
        gpu_type_id=gpu["id"],
        cloud_type="SECURE" if str(settings.get("cloud_type", "")).upper() == "SECURE"
                   else "COMMUNITY",
        gpu_count=int(settings.get("gpu_count") or 1),
        volume_in_gb=int(settings.get("volume_gb") or 0),
        container_disk_in_gb=int(settings.get("container_disk_gb") or 20),
        volume_mount_path="/workspace",
        docker_args=command,
        env=env,
        # Spot capacity is roughly half price. The bid is the rate already agreed
        # to in select_gpu, so this cannot quietly pay more than was shown.
        bid_per_gpu=gpu["price"] if settings.get("spot", True) else None,
        support_public_ip=False,
    )
    if not pod or not pod.get("id"):
        raise RunPodError(f"RunPod did not return a pod id: {pod!r}")
    return pod


def terminate(config, pod_id, log=None):
    """Release the pod. Safe to call more than once, and never raises.

    This is the last thing standing between a finished run and an open-ended bill,
    so it swallows its own errors and says so rather than propagating: an
    exception here would replace whatever real problem caused the shutdown.
    """
    try:
        authenticate(config).terminate_pod(pod_id)
        if log:
            log.info(f"==> pod {pod_id} TERMINATED")
        return True
    except Exception as exc:  # noqa: BLE001
        if log:
            log.error(f"!!  could not terminate pod {pod_id}: {exc}")
            log.error(f"    STOP IT BY HAND: https://console.runpod.io/pods")
        return False


def pod_status(config, pod_id):
    """Current state string, or None when the pod is gone."""
    pod = authenticate(config).get_pod(pod_id)
    if not pod:
        return None
    runtime = pod.get("desiredStatus") or pod.get("lastStatusChange")
    return str(runtime or "UNKNOWN").upper()


# --------------------------------------------------------------------------- #
# Watching it run
# --------------------------------------------------------------------------- #
def watch(config, pod_id, gpu, budget, poll_seconds=15, log=None, now=time.time,
          sleep=time.sleep):
    """Poll until the pod finishes or a cap is crossed. Returns why it ended.

    The cap is checked here, against the clock, rather than trusting the pod to
    stop itself: a pod that hangs before it ever starts the pipeline would
    otherwise bill until someone noticed.
    """
    started = now()
    while True:
        state = pod_status(config, pod_id)
        if state is None or state in TERMINAL_STATES:
            return {"reason": "finished", "state": state,
                    "minutes": (now() - started) / 60.0}

        minutes = (now() - started) / 60.0
        spend = estimate(gpu["price"], minutes)

        breach = budget.breach() if budget else None
        if breach:
            return {"reason": "limit", "state": state, "minutes": minutes,
                    "spend": spend, "detail": str(breach)}

        if log:
            log.info(f"    [{minutes:5.1f} min] {state:<12} ${spend:.2f}")
        sleep(poll_seconds)


# --------------------------------------------------------------------------- #
# The whole thing
# --------------------------------------------------------------------------- #
def launch(config, config_path, log, ask=None, extra_args=(), keep_alive=0,
           projected_minutes=None, poll_seconds=15):
    """Rent, run, watch, release. Returns the process exit code.

    Structured so that every path past the point of renting - including an
    exception raised by this function itself - reaches the same terminate call in
    the `finally` block. There is no ordering of events after `create_pod` that
    leaves a pod running when it should not.
    """
    from ..limits import Budget

    settings = config.get("runpod") or {}
    if not settings.get("enabled"):
        raise RunPodError(
            "runpod.enabled is false.\n"
            "  Turn it on for this run with:  --set runpod.enabled=true")

    # Nothing is rented in either of these two steps, which is the point of doing
    # them first and separately.
    gpu = select_gpu(config, ask=ask, log=log)
    minutes = projected_minutes or (config.get("limits") or {}).get(
        "max_runtime_minutes") or 60
    confirm_cost(config, gpu, minutes, ask=ask, log=log)

    budget = Budget(config, price_per_hour=gpu["price"])
    command = pod_command(config, config_path, extra_args)
    log.info(f"==> command   {command}")

    pod = create_pod(config, gpu, command, pod_env(config, gpu))
    pod_id = pod["id"]
    log.info(f"==> pod       {pod_id} created from {settings.get('image')}")

    outcome = {"reason": "unknown"}
    try:
        outcome = watch(config, pod_id, gpu, budget, poll_seconds=poll_seconds, log=log)
    finally:
        spent = estimate(gpu["price"], outcome.get("minutes") or 0)
        if keep_alive and outcome.get("reason") == "finished":
            log.warning(f"==> pod {pod_id} left running for {keep_alive} min "
                        f"(--keep-alive); stop it sooner with:")
            log.warning(f"      kd runpod stop {pod_id}")
        elif settings.get("terminate_on_exit", True):
            terminate(config, pod_id, log=log)
        else:
            log.warning(f"==> pod {pod_id} left running "
                        f"(runpod.terminate_on_exit is false)")
            log.warning(f"      it is billing until you run: kd runpod stop {pod_id}")
        log.info(f"==> total     ${spent:.2f} over "
                 f"{outcome.get('minutes') or 0:.1f} min")

    if outcome["reason"] == "limit":
        log.error(f"!!  {outcome.get('detail')}")
        return 4
    return 0
