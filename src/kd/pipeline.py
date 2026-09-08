"""
The gated pipeline: every level of verification, in order, in one command.

    kd pipeline --config configs/finance.yaml

walks the stages listed under `pipeline:` in the config and stops at the first
failed gate:

    preflight      resolve config and hardware, fetch remote inputs
    teacher-check  teacher only - missing weights, NaN scan, coherence
    smoke          2 real steps; measures s/step and projects the full run
    train
    evaluate
    report
    publish        skipped unless publish.enabled
    upload         skipped unless s3.enabled; also runs after a failure

A GATE stage that fails aborts the run. A non-gate stage that fails is reported
and the pipeline carries on, so a broken report never destroys a good adapter.

The order and the gate flags come from the config, not from this file, so a
profile can say what a run of that profile means. Narrowing a single run is a
command-line matter instead:

    --only train           just that stage
    --from evaluate        that stage and everything after it
    --skip smoke           everything except that

A stage is a function taking one Context and returning a dict of things later
stages may want. That is the whole contract - there is no base class and no
registration decorator, because eight stages do not need either.
"""

import os
import time

from . import runlog
from .config import describe, resolve_device
from .limits import Budget, LimitCallback, LimitExceeded


class StageFailed(RuntimeError):
    """A stage could not do its job. Whether that stops the run depends on the gate."""


class Context:
    """What every stage is handed, and where stages leave things for each other."""

    def __init__(self, config, hardware, run, budget, options=None):
        self.config = config
        self.hardware = hardware
        self.run = run
        self.budget = budget
        self.options = options or {}
        # Filled in as the run proceeds: seconds_per_step by smoke, adapter by
        # train, evaluation by evaluate. Stages read what earlier ones left.
        self.results = {}

    @property
    def log(self):
        return self.run.log

    # --- inputs a stage needs that an earlier stage may not have produced ---- #
    #
    # `--from evaluate` and `--only report` open a NEW run directory, because every
    # invocation is its own run. The stages they start with therefore have to be
    # able to pick up where the last run left off, or those flags would only ever
    # work in a pipeline that had already run the earlier stages in the same
    # invocation - which is exactly the case they exist for.
    def resolve_adapter(self):
        """This run's adapter, else the newest one any run produced."""
        if runlog.is_adapter(self.results.get("adapter")):
            return self.results["adapter"]
        if runlog.is_adapter(self.run.adapter_dir):
            return self.run.adapter_dir
        found = runlog.discover_adapters(self.config["project"].get("runs_dir")
                                         or "./runs")
        if found:
            self.log.info(f"      using the newest adapter found: {found[0]}")
            return found[0]
        return None

    def resolve_evaluation(self):
        """This run's evaluation payload, else the newest run that has one."""
        import glob

        candidate = self.results.get("evaluation") or self.run.path("evaluation.json")
        if os.path.isfile(candidate):
            return candidate
        runs_dir = self.config["project"].get("runs_dir") or "./runs"
        found = sorted(glob.glob(os.path.join(runs_dir, "*", "evaluation.json")),
                       key=os.path.getmtime, reverse=True)
        if found:
            newest = os.path.normpath(found[0]).replace("\\", "/")
            self.log.info(f"      using the newest evaluation found: {newest}")
            return newest
        return None


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_preflight(ctx):
    """Resolve everything and prove the inputs are reachable. Costs seconds."""
    config = ctx.config
    ctx.log.info(f"  profile   : {config['_meta'].get('profile')}")
    ctx.log.info(f"  device    : {ctx.hardware['device']} ({ctx.hardware['dtype_name']})")
    ctx.log.info(f"  limits    : {ctx.budget.summary()}")

    # Remote inputs are resolved to local paths here, before anything loads them.
    # Until the S3 layer exists, an s3:// URI is a clear error rather than a
    # confusing failure inside transformers.
    remote = [f"{key}={value}" for key, value in (
        ("models.teacher", config["models"].get("teacher")),
        ("models.student", config["models"].get("student")),
        ("models.teacher_adapter", config["models"].get("teacher_adapter")),
        ("dataset.source", config["dataset"].get("source")),
    ) if isinstance(value, str) and value.startswith("s3://")]
    if remote:
        raise StageFailed(
            "s3:// inputs are configured but object storage is not wired up yet:\n  "
            + "\n  ".join(remote)
            + "\nUse a Hugging Face id or a local path for now.")

    return {"device": ctx.hardware["device"]}


def stage_teacher_check(ctx):
    """Load ONLY the teacher and verify it is fit to distil from.

    This is the cheapest check that catches the most expensive mistake: GKD trains
    the student to match the teacher's distribution, so a broken teacher spends the
    whole budget producing a broken student, and nothing before the final samples
    says so.
    """
    import argparse

    from . import teacher

    args = argparse.Namespace(
        config=ctx.config["_meta"].get("source"),
        teacher=ctx.config["models"]["teacher"],
        teacher_adapter=ctx.config["models"].get("teacher_adapter"),
        tokenizer=None,
        dtype="auto",
        device=ctx.hardware["device"],
        max_new_tokens=48,
    )
    code = teacher.main(args)
    if code != 0:
        raise StageFailed("the teacher is not fit to distil from; see the report above")
    return {"teacher_ok": True}


def stage_smoke(ctx):
    """Two real steps, to measure the pace and project the run before committing.

    The measurement is the point. Without it, limits can only be enforced by killing
    a run that is already underway; with it, a run that cannot finish inside its
    ceilings is refused before the budget is touched.
    """
    import shutil

    from .train import train

    scratch = ctx.run.path("smoke-checkpoints")
    try:
        summary = train(ctx.config, ctx.hardware, ctx.run, dry_run=True,
                        allow_bad_teacher=True,  # already gated by teacher-check above
                        checkpoints=scratch)
    finally:
        # Two steps of throwaway optimizer state, ~100 MB, that nothing will ever
        # resume from. Leaving it in the bundle would mean uploading it too.
        shutil.rmtree(scratch, ignore_errors=True)
    per_step = summary.get("seconds_per_step") or 0.0
    ctx.results["seconds_per_step"] = per_step

    projection = ctx.budget.project(per_step)
    ctx.log.info(f"      {per_step:.2f} s/step measured -> "
                 f"{projection['steps']} steps is about "
                 f"{_fmt_duration(projection['train_minutes'] * 60)}"
                 + (f", ${projection['cost_usd']:.2f}" if projection["cost_usd"] else ""))
    ctx.run.event("smoke", "projection", **projection)

    refusal = ctx.budget.refuse_if_impossible(projection)
    if refusal:
        raise StageFailed(refusal)
    return {"projection": projection}


def stage_train(ctx):
    """The run itself, under the budget."""
    from .train import train

    summary = train(ctx.config, ctx.hardware, ctx.run,
                    allow_bad_teacher=ctx.options.get("allow_bad_teacher", False),
                    callbacks=[LimitCallback(ctx.budget, ctx.run)])
    ctx.results["adapter"] = summary.get("adapter")
    ctx.run.write_metrics({"training": summary})
    return summary


def stage_evaluate(ctx):
    """Score the adapter against the teacher on the held-out split."""
    import argparse

    from . import evaluate

    adapter = ctx.resolve_adapter()
    if not adapter:
        raise StageFailed(
            f"nothing to evaluate: no adapter in this run, and none found under "
            f"{ctx.config['project'].get('runs_dir')}.\n"
            f"Train one first, or name it with --set project.output_dir=<dir>.")

    settings = ctx.config.get("evaluation") or {}
    payload_path = ctx.run.path("evaluation.json")
    args = argparse.Namespace(
        config=ctx.config["_meta"].get("source"),
        adapter=adapter,
        teacher=ctx.config["models"]["teacher"],
        student=ctx.config["models"]["student"],
        teacher_adapter=ctx.config["models"].get("teacher_adapter"),
        samples=int(settings.get("samples") or 50),
        device=ctx.hardware["device"],
        dtype="auto",
        tasks=settings.get("tasks"),
        limit=settings.get("limit"),
        no_generations=False,
        gen_similarity=int(settings.get("gen_similarity") or 0),
        no_rescale=False,
        similarity_model=settings.get("similarity_model") or "roberta-large",
        json=payload_path,
        # The report is a stage of its own, so nothing is written here.
        report=None,
    )
    code = evaluate.main(args)
    ctx.results["evaluation"] = payload_path

    # Exit 3 means "measured fine, but the adapter did not beat the base student".
    # That is a result, not a malfunction: the numbers are real and worth keeping,
    # so the report is still written and the run still ends with a bundle.
    if code == 3:
        ctx.results["no_improvement"] = True
        ctx.log.warning("      the adapter did not improve on the base student")
    elif code != 0:
        raise StageFailed(f"evaluation exited {code}")
    return {"evaluation": payload_path, "improved": code == 0}


def stage_report(ctx):
    """Turn the measurements into something a person can read."""
    import json

    from .report import write_report

    payload_path = ctx.resolve_evaluation()
    if not payload_path:
        raise StageFailed("no evaluation to report on - run the evaluate stage first")

    with open(payload_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    suffix = str((ctx.config.get("evaluation") or {}).get("report_format", "html"))
    written = write_report(payload, ctx.run.path(f"report.{suffix.lstrip('.')}"))

    ctx.run.write_metrics({
        "fidelity": payload.get("fidelity"),
        "capability": payload.get("capability"),
        "efficiency": payload.get("efficiency"),
    })
    ctx.log.info(f"      {written}")
    return {"report": str(written)}


def stage_publish(ctx):
    """Push the adapter and a merged model to the Hugging Face Hub."""
    import argparse

    from . import publish

    settings = ctx.config.get("publish") or {}
    args = argparse.Namespace(
        repo=settings.get("repo"),
        config=ctx.config["_meta"].get("source"),
        adapter=ctx.results.get("adapter") or ctx.run.adapter_dir,
        private=bool(settings.get("private", True)),
        dry_run=False,
        adapter_only=False,
        dtype=None,
    )
    code = publish.main(args)
    if code != 0:
        raise StageFailed(f"publish exited {code}")
    return {"published": settings.get("repo")}


def stage_upload(ctx):
    """Sync the finished run bundle to object storage."""
    raise StageFailed(
        "s3.enabled is true but object storage is not wired up yet "
        "(that lands with the S3 phase). The run bundle is complete on disk at "
        f"{ctx.run.dir}")


STAGES = {
    "preflight": stage_preflight,
    "teacher-check": stage_teacher_check,
    "smoke": stage_smoke,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "report": stage_report,
    "publish": stage_publish,
    "upload": stage_upload,
}

# Stages that only make sense when a feature is switched on. Returning a reason
# here rather than failing keeps "not configured" distinct from "went wrong".
CONDITIONAL = {
    "publish": ("publish", "publish.enabled is false"),
    "upload": ("s3", "s3.enabled is false"),
}

# Runs even after an earlier gate failed, so a crashed run still ships its logs.
ALWAYS_RUN = {"upload"}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def planned_stages(config, only=None, start_from=None, skip=()):
    """The stage list for this run: [(name, is_gate), ...].

    Raises ValueError on a stage name that does not exist, because silently running
    a shorter pipeline than asked for is the kind of thing nobody notices until the
    evaluation they expected is missing.
    """
    declared = (config.get("pipeline") or {}).get("stages") or []
    plan = [(str(entry["name"]), bool(entry.get("gate", False))) for entry in declared]
    known = {name for name, _ in plan}

    for label, value in (("--only", only), ("--from", start_from)):
        if value and value not in known:
            raise ValueError(f"{label} {value!r} is not a stage. "
                             f"Available: {', '.join(n for n, _ in plan)}")
    for name in skip or ():
        if name not in known:
            raise ValueError(f"--skip {name!r} is not a stage. "
                             f"Available: {', '.join(n for n, _ in plan)}")

    if only:
        return [(name, gate) for name, gate in plan if name == only]
    if start_from:
        index = next(i for i, (name, _) in enumerate(plan) if name == start_from)
        plan = plan[index:]
    return [(name, gate) for name, gate in plan if name not in (skip or ())]


def _skip_reason(ctx, name):
    """Why this stage should not run at all, or None."""
    section, reason = CONDITIONAL.get(name, (None, None))
    if section and not (ctx.config.get(section) or {}).get("enabled"):
        return reason
    return None


def _fmt_duration(seconds):
    seconds = int(max(0, seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def run_pipeline(config, run, only=None, start_from=None, skip=(), options=None,
                 price_per_hour=None):
    """Walk the stages. Returns the process exit code.

    0  everything that ran, worked
    1  a gate failed
    2  the teacher was unfit
    3  the run completed but the adapter did not improve on the base student
    4  a limit was hit and the run was stopped
    """
    hardware = resolve_device(config)
    budget = Budget(config, price_per_hour=price_per_hour, started=run.started)
    ctx = Context(config, hardware, run, budget, options)

    plan = planned_stages(config, only, start_from, skip)
    total = len(plan)

    run.log.info(describe(config, hardware, run_id=run.run_id))
    run.log.info(f"  stages     : {' -> '.join(name for name, _ in plan)}")
    run.log.info("")

    failure = None
    for index, (name, is_gate) in enumerate(plan, start=1):
        label = f"[{index}/{total}] {name}"

        if failure and name not in ALWAYS_RUN:
            run.skip_stage(name, f"an earlier gate failed ({failure[0]})")
            run.log.info(f"{label:<26} SKIPPED (earlier failure)")
            continue

        reason = _skip_reason(ctx, name)
        if reason:
            run.skip_stage(name, reason)
            run.log.info(f"{label:<26} skipped - {reason}")
            continue

        started = time.time()
        run.log.info(f"{label:<26} ...")
        try:
            with run.stage(name):
                ctx.results.update(STAGES[name](ctx) or {})
        except LimitExceeded as exc:
            # A hard stop. Nothing further runs, including the non-gate stages: the
            # whole point of the ceiling is that crossing it ends the spending.
            run.log.error(f"{label:<26} STOPPED  {_fmt_duration(time.time() - started)}")
            run.log.error(f"      {exc}")
            run.log.error(f"      the last checkpoint under {run.checkpoint_dir} is "
                          f"what survives")
            run.finish("stopped", str(exc))
            return 4
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            elapsed = _fmt_duration(time.time() - started)
            if is_gate:
                run.log.error(f"{label:<26} FAILED   {elapsed}")
                for line in str(exc).splitlines():
                    run.log.error(f"      {line}")
                failure = (name, exc)
            else:
                run.log.warning(f"{label:<26} failed   {elapsed} (not a gate, "
                                f"continuing)")
                for line in str(exc).splitlines():
                    run.log.warning(f"      {line}")
        else:
            run.log.info(f"{label:<26} OK       "
                         f"{_fmt_duration(time.time() - started)}")

    return _finish(ctx, failure)


def _finish(ctx, failure):
    run = ctx.run
    run.log.info("")
    if failure:
        name, exc = failure
        run.finish("failed", f"{name}: {exc}")
        run.log.error(f"  FAILED at {name}")
        run.log.error(f"  run bundle : {run.dir}")
        return 2 if name == "teacher-check" else 1

    run.finish("ok")
    run.log.info(f"  run bundle : {run.dir}")
    if ctx.results.get("adapter"):
        run.log.info(f"  adapter    : {ctx.results['adapter']}")
    if ctx.results.get("report"):
        run.log.info(f"  report     : {ctx.results['report']}")
    if ctx.results.get("no_improvement"):
        run.log.warning("  the adapter did not improve on the base student")
        return 3
    return 0
