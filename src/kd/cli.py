"""
`kd` - the single entry point for the distillation pipeline.

    kd check          --config configs/finance.yaml     resolve and print, run nothing
    kd train          --config configs/finance.yaml
    kd evaluate       --config configs/finance.yaml
    kd check-teacher  --config configs/finance.yaml
    kd fix-teacher    --config configs/finance.yaml
    kd convert-adapter --teacher-adapter <mlx-adapter>
    kd publish        --repo <org>/<name>
    kd ui             [--compare]
    kd doctor

Every command that reads a config accepts the same override surface:

    --set a.b=c       reaches any key in the config, repeatable
    short flags       --teacher, --steps, --device, ... for the common ones
    KD_* env vars     applied when neither of the above names the key

with precedence _base.yaml < profile < KD_* < --set < flag.

Commands that wrap a tool with its own established flags - evaluate, publish, the
teacher tools, the UIs - hand their arguments straight through to it, so nothing
that worked before needs relearning.
"""

import argparse
import os
import sys

from . import __version__

# Short flags that exist purely as sugar for the common --set paths. Anything not
# listed here is still reachable: --set covers the whole config.
SUGAR = [
    ("--teacher", "models.teacher", str, "Teacher model id or local path"),
    ("--student", "models.student", str, "Student model id or local path"),
    ("--teacher-adapter", "models.teacher_adapter", str,
     "LoRA adapter merged into the teacher at load time"),
    ("--dataset", "dataset.source", str, "Training dataset"),
    ("--device", "hardware.device", str, "auto | cpu | mps | cuda"),
    ("--dtype", "hardware.dtype", str, "auto | float32 | bfloat16 | float16"),
    ("--steps", "training.max_steps", int, "Optimizer steps"),
    ("--batch-size", "training.batch_size", int, "Per-device batch size"),
    ("--grad-accum", "training.gradient_accumulation_steps", int,
     "Gradient accumulation steps"),
    ("--lr", "training.learning_rate", float, "Learning rate, e.g. 3e-4"),
    ("--lora-r", "lora.r", int, "LoRA rank"),
    ("--lora-alpha", "lora.alpha", int, "LoRA alpha"),
    ("--lmbda", "gkd.lmbda", float, "On-policy fraction, 0.0-1.0"),
    ("--seed", "project.seed", int, "Random seed"),
    ("--output", "project.output_dir", str,
     "Pin the run directory instead of generating one under project.runs_dir"),
]


# Commands that wrap a tool with its own established flags. The value is the module
# under kd/ whose main() owns those flags; "ui" is special-cased in main().
DELEGATED = {
    "evaluate": "evaluate",
    "check-teacher": "teacher",
    "fix-teacher": "teacher_fix",
    "convert-adapter": "adapters",
    "publish": "publish",
    "ui": None,
}


def add_config_args(parser):
    """The override surface shared by every config-reading command."""
    parser.add_argument("-c", "--config", default=None, metavar="PATH",
                        help="YAML profile, e.g. configs/finance.yaml "
                             "(default: built-in _base.yaml)")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override any config key by dotted path, repeatable. "
                             "Example: --set training.max_steps=500")
    group = parser.add_argument_group("common overrides (sugar for --set)")
    for flag, path, caster, help_text in SUGAR:
        group.add_argument(flag, dest=path.replace(".", "__"), default=None,
                           type=caster, help=help_text)
    return parser


def flag_overrides(args):
    """Collect the sugar flags the caller actually passed into {dotted path: value}."""
    values = {}
    for _flag, path, _caster, _help in SUGAR:
        value = getattr(args, path.replace(".", "__"), None)
        if value is not None:
            values[path] = value
    return values


def load(args):
    """Resolve the config for a command, applying every override layer."""
    from .config import load_config
    return load_config(args.config,
                       set_overrides=args.set_overrides,
                       flag_overrides=flag_overrides(args))


def delegate(module_name, argv):
    """Run a wrapped tool's own CLI with `argv`.

    These tools have established, documented flags that people already use. Wrapping
    them in a second layer of argument parsing would mean maintaining two spellings
    of every option and getting them out of step.
    """
    import importlib
    module = importlib.import_module(f".{module_name}", package="kd")
    saved = sys.argv
    sys.argv = [f"kd {module_name}"] + list(argv)
    try:
        return module.main() or 0
    finally:
        sys.argv = saved


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check(args):
    """Resolve the config and hardware, print both, and exit. Costs nothing.

    With --full, stdout carries nothing but the YAML and the banner goes to
    stderr, so the effective config can be redirected straight into a file:

        kd check --config configs/finance.yaml --full > my-run.yaml

    That is the only way to get an editable config onto a machine that has a
    downloaded runner and no checkout, so it has to produce a clean file.
    """
    import yaml
    from .config import describe, resolve_device, strip_meta

    config = load(args)
    hardware = resolve_device(config)
    print(describe(config, hardware), file=sys.stderr if args.full else sys.stdout)
    if args.full:
        source = config["_meta"].get("source")
        print(f"# Effective configuration for {source}, after every override.\n"
              f"# Edit it and run it directly:  kd pipeline --config <this file>\n")
        print(yaml.safe_dump(strip_meta(config), sort_keys=False,
                             default_flow_style=False, allow_unicode=True))
    return 0


def cmd_train(args):
    """One training run, into its own run directory."""
    from .config import describe, resolve_device
    from .runlog import Run
    from .train import train

    config = load(args)
    hardware = resolve_device(config)

    with Run(config, argv=sys.argv) as run:
        run.log.info(describe(config, hardware, run_id=run.run_id))
        summary = train(config, hardware, run,
                        dry_run=args.dry_run,
                        allow_bad_teacher=args.allow_bad_teacher)
        run.write_metrics({"training": summary})
        run.log.info("")
        run.log.info(f"  run bundle : {run.dir}")
        if summary.get("adapter"):
            run.log.info(f"  adapter    : {summary['adapter']}")
    return 0


def cmd_pipeline(args):
    """Every verification stage, gated, in one run directory."""
    from .pipeline import run_pipeline
    from .runlog import Run

    config = load(args)

    # Set by the RunPod launcher to the rate it actually agreed to pay. Its
    # presence is what makes limits.max_cost_usd mean anything inside the pod; on a
    # local machine nothing is rented, so it is absent and the cost cap is inert.
    price = os.environ.get("KD_PRICE_PER_HOUR")
    try:
        price = float(price) if price else None
    except ValueError:
        print(f" !! ignoring KD_PRICE_PER_HOUR={price!r} (not a number)")
        price = None

    with Run(config, argv=sys.argv) as run:
        return run_pipeline(
            config, run,
            only=args.only,
            start_from=getattr(args, "from"),
            skip=args.skip,
            options={"allow_bad_teacher": args.allow_bad_teacher},
            price_per_hour=price,
        )


def cmd_runpod(args):
    """Rent a GPU, run the pipeline on it, and release it."""
    import logging

    from .remote import runpod as rp

    config = load(args)

    # A plain console logger: this runs on the local machine, orchestrating a run
    # that happens elsewhere, so it has no run bundle of its own to write into.
    log = logging.getLogger("kd.runpod")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    # Prompting is only offered when someone is there to answer. Non-interactive
    # callers get a refusal rather than a hang, and --yes accepts the estimate
    # without accepting a different GPU.
    interactive = sys.stdin.isatty() and not args.yes
    ask = input if interactive else None
    if args.yes:
        ask = (lambda _prompt: "y") if args.action == "launch" else None

    try:
        if args.action == "gpus":
            gpus = rp.catalogue(config)
            cap = (config.get("runpod") or {}).get("max_price_per_hour")
            log.info(rp.render_options(rp.affordable(gpus, cap))
                     or "  nothing available under the price cap")
            return 0
        if args.action == "stop":
            if not args.pod_id:
                raise rp.RunPodError("which pod? usage: kd runpod stop <pod-id>")
            return 0 if rp.terminate(config, args.pod_id, log=log) else 1
        if args.action == "status":
            if not args.pod_id:
                raise rp.RunPodError("which pod? usage: kd runpod status <pod-id>")
            log.info(rp.pod_status(config, args.pod_id) or "gone")
            return 0

        return rp.launch(config, args.config or "configs/_base.yaml", log,
                         ask=ask, keep_alive=args.keep_alive)
    except rp.RunPodError as exc:
        log.error(f"xx  {exc}")
        return 1


def cmd_doctor(args):
    """Report what this machine can do, and which credentials are present.

    Values are never printed - only whether something is set. A doctor command that
    echoes a token into a terminal, a screenshot or a CI log is a liability.
    """
    import platform

    from .config import resolve_device
    from .runlog import git_dirty, git_sha, package_versions

    config = load(args)
    hardware = resolve_device(config)

    print("=" * 78)
    print("  kd doctor")
    print("=" * 78)
    print(f" version    : kd {__version__}")
    print(f" python     : {platform.python_version()} on "
          f"{platform.system()} {platform.machine()}")
    print(f" git        : {git_sha()}{' (uncommitted changes)' if git_dirty() else ''}")
    print(f" device     : {hardware['device']} ({hardware['dtype_name']})")
    for note in hardware["notes"]:
        print(f"   - {note}")

    print("\n packages")
    for name, found in package_versions().items():
        print(f"   {name:14} {found or 'NOT INSTALLED'}")

    print("\n optional features")
    for label, module in [("evaluation extra (--tasks, --gen-similarity)", "lm_eval"),
                          ("S3 (boto3)", "boto3"),
                          ("RunPod", "runpod")]:
        import importlib.util
        available = importlib.util.find_spec(module) is not None
        print(f"   {label:44} {'available' if available else 'not installed'}")

    print("\n credentials (presence only, never values)")
    for env_name, what in [("HF_TOKEN", "Hugging Face publish/private models"),
                           ("AWS_ACCESS_KEY_ID", "S3"),
                           ("AWS_SECRET_ACCESS_KEY", "S3"),
                           ("AWS_PROFILE", "S3 (alternative to keys)"),
                           ("RUNPOD_API_KEY", "RunPod")]:
        state = "set" if os.environ.get(env_name) else "-"
        print(f"   {env_name:24} {state:4}  {what}")
    print("=" * 78)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        prog="kd",
        description="Config-driven knowledge distillation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"kd {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    pipeline = sub.add_parser(
        "pipeline",
        help="Run every verification stage, gated, in one run directory")
    add_config_args(pipeline)
    pipeline.add_argument("--only", metavar="STAGE",
                          help="Run just this stage")
    pipeline.add_argument("--from", metavar="STAGE",
                          help="Start at this stage and run everything after it")
    pipeline.add_argument("--skip", action="append", default=[], metavar="STAGE",
                          help="Skip this stage, repeatable")
    pipeline.add_argument("--allow-bad-teacher", action="store_true",
                          help="Train even if the teacher fails its pre-flight check "
                               "(not recommended)")
    pipeline.set_defaults(func=cmd_pipeline)

    check = sub.add_parser("check", help="Resolve config and hardware, print, exit")
    add_config_args(check)
    check.add_argument("--full", action="store_true",
                       help="Also dump the complete effective config as YAML")
    check.set_defaults(func=cmd_check)

    train = sub.add_parser("train", help="Run one distillation training run")
    add_config_args(train)
    train.add_argument("--dry-run", action="store_true",
                       help="2 steps and no final save; validates the pipeline")
    train.add_argument("--allow-bad-teacher", action="store_true",
                       help="Train even if the teacher fails its pre-flight check "
                            "(not recommended - a broken teacher yields a broken "
                            "student)")
    train.set_defaults(func=cmd_train)

    runpod = sub.add_parser(
        "runpod",
        help="Rent a GPU and run the pipeline on it (optional; needs the remote extra)")
    add_config_args(runpod)
    runpod.add_argument("action", nargs="?", default="launch",
                        choices=["launch", "stop", "status", "gpus"],
                        help="launch (default), stop, status, or list GPUs")
    runpod.add_argument("pod_id", nargs="?", default=None,
                        help="Pod id, for stop and status")
    runpod.add_argument("--yes", action="store_true",
                        help="Accept the cost estimate without asking. Does NOT "
                             "accept a different GPU than the one configured.")
    runpod.add_argument("--keep-alive", type=int, default=0, metavar="MIN",
                        help="Leave the pod running for MIN minutes after a "
                             "successful run, for inspection. It bills until then.")
    runpod.set_defaults(func=cmd_runpod)

    doctor = sub.add_parser("doctor", help="Report environment and credentials")
    add_config_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    # Wrapped tools are registered here only so that `kd --help` lists them. They
    # are dispatched in main() before argparse runs - see the note there.
    for name, help_text in [
        ("evaluate", "Measure teacher->student transfer"),
        ("check-teacher", "Verify a teacher is fit to distil from"),
        ("fix-teacher", "Repair a checkpoint with mislabelled tensors"),
        ("convert-adapter", "Convert an MLX LoRA adapter to PEFT format"),
        ("publish", "Publish the adapter and merged model to the HF Hub"),
        ("ui", "Launch the control panel, or --compare for the comparison view"),
    ]:
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Delegated commands are dispatched before argparse sees anything. Routing them
    # through a subparser instead would mean argparse.REMAINDER, which stops
    # collecting the moment the first leftover argument looks like a flag - so
    # `kd evaluate --help` would be rejected by the outer parser rather than
    # reaching the tool it belongs to.
    if argv and argv[0] in DELEGATED:
        name = argv[0]
        rest = argv[1:]
        if name == "ui":
            compare = "--compare" in rest
            rest = [item for item in rest if item != "--compare"]
            return delegate("ui.compare" if compare else "ui.control", rest)
        return delegate(DELEGATED[name], rest)

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
