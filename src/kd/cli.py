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
    """Resolve the config and hardware, print both, and exit. Costs nothing."""
    import yaml
    from .config import describe, resolve_device, strip_meta

    config = load(args)
    hardware = resolve_device(config)
    print(describe(config, hardware))
    if args.full:
        print("\n--- effective config ---")
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
