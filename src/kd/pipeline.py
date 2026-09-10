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
    arena          skipped unless evaluation.arena_file names a held-out set
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


class StageRefused(StageFailed):
    """The run was declined on purpose - nothing is broken.

    A projection that does not fit its limits is not a malfunction, and calling it
    FAILED sends people looking for a bug that is not there. It is the cost guard
    doing the one thing it exists to do, and it deserves its own word and its own
    exit code so a script can tell "adjust and retry" apart from "something went
    wrong".
    """


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
        """The named adapter, else this run's, else the newest one any run made.

        `--adapter` wins over discovery because it is an instruction, not a
        hint: someone who names an adapter is scoring THAT one, and silently
        falling back to whatever a previous run left under runs_dir would
        produce a report about the wrong weights - the worst kind of wrong,
        since every number in it would look perfectly reasonable.
        """
        from . import paths

        named = self.options.get("adapter")
        if named:
            # adapter_dir_of first: a path copied out of a bucket listing names
            # adapter_config.json, and the directory is what loads.
            named = paths.adapter_dir_of(named)
            # Written back so the next stage to ask sees a local path: localise
            # is a no-op on one, so evaluate and arena share the one download
            # without either needing to know the other ran.
            named = paths.localise(named, self.config, log=self.log,
                                   label="adapter")
            self.options["adapter"] = named
            if not runlog.is_adapter(named):
                raise StageFailed(
                    f"--adapter {named} is not a LoRA adapter directory: no "
                    f"adapter_config.json in it.")
            return named
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

    def _newest(self, key, filename, what):
        """This run's `filename`, else the newest any run produced, else None.

        The rule that makes `--only report` and `--from report` mean anything:
        those open a NEW run directory, so the file a later stage wants was
        written by an earlier invocation and has to be found rather than
        assumed.
        """
        import glob

        candidate = self.results.get(key) or self.run.path(filename)
        if os.path.isfile(candidate):
            return candidate
        runs_dir = self.config["project"].get("runs_dir") or "./runs"
        # Skip the `latest` pointer for the same reason discovery does: where the
        # platform makes it a real symlink it matches this glob as well, and
        # naming an alias instead of a run id makes the log say something that
        # will not be true tomorrow.
        found = sorted(
            (p for p in glob.glob(os.path.join(runs_dir, "*", filename))
             if not runlog.is_latest_alias(p, runs_dir)),
            key=os.path.getmtime, reverse=True)
        if found:
            newest = os.path.normpath(found[0]).replace("\\", "/")
            self.log.info(f"      using the newest {what} found: {newest}")
            return newest
        return None

    def ensure_inputs(self):
        """Fetch any s3:// input that preflight has not already fetched.

        preflight is where remote inputs are normally resolved, and every stage
        after it inherits a config whose paths are local. But `--from arena` and
        `--only evaluate` do not run preflight, so those stages would hand a raw
        `s3://...` string to from_pretrained and get "Repo id must be in the form
        namespace/repo_name" - a message about the Hub, for a problem that has
        nothing to do with the Hub.

        Called by the stages that load a model, not by the pipeline, and that
        distinction is the point: `--only report` must not trigger a multi-
        gigabyte download to render an HTML file from JSON it already has.

        A no-op once the config holds local paths, so calling it twice costs a
        dictionary scan.
        """
        from . import paths

        if not paths.remote_values(self.config):
            return
        try:
            paths.resolve_inputs(self.config, self.log)
        except Exception as exc:
            raise StageFailed(str(exc)) from exc

    def resolve_evaluation(self):
        """This run's evaluation payload, else the newest run that has one."""
        return self._newest("evaluation", "evaluation.json", "evaluation")

    def resolve_arena(self):
        """This run's arena payload, else the newest run that has one.

        Optional, unlike the evaluation: a profile with no arena_file never
        produces one, and the report is complete without it.
        """
        return self._newest("arena", "arena.json", "arena score")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_preflight(ctx):
    """Resolve everything and prove the inputs are reachable. Costs seconds."""
    config = ctx.config
    ctx.log.info(f"  profile   : {config['_meta'].get('profile')}")
    ctx.log.info(f"  device    : {ctx.hardware['device']} ({ctx.hardware['dtype_name']})")
    ctx.log.info(f"  limits    : {ctx.budget.summary()}")

    # Remote inputs are fetched here, before anything tries to load them. Doing it
    # in the first stage is what turns a bad bucket or a missing credential into a
    # two-second failure instead of one that surfaces after the dataset build.
    from . import paths

    try:
        fetched = paths.resolve_inputs(config, ctx.log)
    except Exception as exc:
        raise StageFailed(str(exc)) from exc

    if fetched:
        ctx.run.event("preflight", "inputs_resolved", **{
            key: value["uri"] for key, value in fetched.items()})

    # A teacher adapter that PEFT cannot load only fails once the base model is in
    # memory - for a 2B teacher, several gigabytes of download away. It is checked
    # here instead, from a file listing, and an MLX one is converted on the spot:
    # the conversion is deterministic and needs neither a GPU nor the base
    # model's weights, so requiring a separate manual step buys nothing.
    adapter = config["models"].get("teacher_adapter")
    if adapter:
        try:
            converted = paths.ensure_peft_adapter(config, ctx.log)
        except Exception as exc:
            raise StageFailed(str(exc)) from exc
        if converted:
            ctx.run.event("preflight", "adapter_converted", **converted)
        ctx.log.info(f"  teacher lora: {config['models']['teacher_adapter']}")

    # Both models are resident at once. Weights that do not comfortably fit make
    # the allocator spill to swap rather than fail, so the symptom is a run that
    # is mysteriously slow - which is worth naming before the smoke stage spends
    # ten minutes measuring it.
    estimate = paths.memory_estimate(config, ctx.hardware["dtype_name"])
    if estimate:
        ctx.log.info(f"  weights   : "
                     f"{estimate['weight_bytes'] / paths.GB:.1f} GB "
                     f"({estimate['dtype']})")
        ctx.run.event("preflight", "memory_estimate", **estimate)
        warning = paths.memory_warning(estimate, ctx.hardware["device"])
        if warning:
            for line in ("!! " + warning).splitlines():
                ctx.log.warning(f"      {line}")

    return {"device": ctx.hardware["device"], "fetched": fetched}


def stage_teacher_check(ctx):
    """Load ONLY the teacher and verify it is fit to distil from.

    This is the cheapest check that catches the most expensive mistake: GKD trains
    the student to match the teacher's distribution, so a broken teacher spends the
    whole budget producing a broken student, and nothing before the final samples
    says so.
    """
    import argparse

    from . import teacher

    ctx.ensure_inputs()

    args = argparse.Namespace(
        config=ctx.config["_meta"].get("source"),
        teacher=ctx.config["models"]["teacher"],
        teacher_adapter=ctx.config["models"].get("teacher_adapter"),
        tokenizer=None,
        dtype="auto",
        device=ctx.hardware["device"],
        # Enough to read a whole short answer. 48 was too few the moment the
        # teacher became a model that opens with a reasoning block: the sample
        # ended mid-preamble every time, so the one stage whose job is to show
        # you the teacher's output showed you none of it. kd.teacher now renders
        # the prompt past that block, and this is sized to reach a conclusion.
        max_new_tokens=int((ctx.config.get("evaluation") or {})
                           .get("teacher_check_max_new_tokens") or 192),
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

    ctx.ensure_inputs()
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
        raise StageRefused(refusal)
    return {"projection": projection}


def stage_train(ctx):
    """The run itself, under the budget."""
    from .train import train

    ctx.ensure_inputs()
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

    ctx.ensure_inputs()

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


def stage_arena(ctx):
    """Score base, distilled and teacher on the held-out set, and rate them.

    Runs here, on the pod, rather than being left for later, because the teacher
    is already resident and already paid for. Doing it afterwards on another
    machine means downloading 8B of weights again to answer a question this
    machine could answer in a few minutes.

    Not a gate: a student that scores badly is a result worth keeping, not a
    reason to throw away the adapter and the report.
    """
    import json

    from . import arena

    ctx.ensure_inputs()
    settings = ctx.config.get("evaluation") or {}
    path = settings.get("arena_file")
    questions, skipped = arena.load_questions(path)
    limit = settings.get("arena_limit")
    if limit:
        questions = questions[:int(limit)]
        ctx.log.info(f"      evaluation.arena_limit={limit} - this is a subset, "
                     f"not the score")
    if skipped:
        ctx.log.info(f"      {skipped} rows carry no answer letter and are not scored")

    adapter = ctx.resolve_adapter()
    if not adapter:
        raise StageFailed(
            "nothing to score: no adapter in this run, and none found under "
            f"{ctx.config['project'].get('runs_dir')}")

    ctx.log.info(f"      {len(questions)} held-out questions from {path}")
    predictions, formats, unanswered, completions = arena.play(
        ctx.config, ctx.hardware, adapter, questions,
        max_new_tokens=int(settings.get("arena_max_new_tokens") or 512),
        log=ctx.log)

    payload = arena.summarise(
        predictions, [q["gold"] for q in questions],
        formats=formats, unanswered=unanswered,
        rounds=int(settings.get("arena_elo_rounds") or 25),
        seed=int(ctx.config["project"]["seed"]))
    payload["arena_file"] = str(path)
    payload["adapter"] = str(adapter)
    # Present from the first write, so the file always says whether there is a
    # similarity table rather than leaving a reader to infer it from absence.
    payload["similarity"] = None

    # Written before the similarity table is computed, not after. Generation is
    # the hours-long unrepeatable part; similarity downloads an embedding model,
    # which is a network call that can fail. Saving first means a failed download
    # costs a table, not the run.
    target = ctx.run.path("arena.json")

    def write_payload():
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    write_payload()

    # The full transcript beside it: every question, and every word each player
    # said about it. Separate from arena.json because it is large and read for a
    # different reason - the numbers to see WHAT happened, this to see WHY.
    transcript = None
    try:
        transcript = arena.write_transcript(
            ctx.run.path("arena-transcript.jsonl"), questions, predictions,
            formats, completions)
        ctx.log.info(f"      transcript: {transcript}")
    except Exception as exc:  # noqa: BLE001 - the numbers are already on disk
        ctx.log.warning(f"      !! could not write the transcript: {exc}")

    # Optional, and reported as absent rather than failing: the similarity table
    # needs sentence-transformers, which is in the `eval` extra a pod install
    # deliberately skips.
    try:
        payload["similarity"] = arena.similarity(completions, questions,
                                                 log=ctx.log)
    except Exception as exc:  # noqa: BLE001 - a download, an encode, a disk
        payload["similarity"] = None
        ctx.log.warning(f"      !! similarity skipped: {exc}")
    if payload.get("similarity"):
        write_payload()

    ctx.log.info("")
    for line in arena.render(payload).splitlines():
        ctx.log.info(line)
    if payload.get("similarity"):
        for line in arena.render_similarity(payload["similarity"]).splitlines():
            ctx.log.info(line)
    ctx.run.write_metrics({"arena": payload["players"]})
    ctx.run.event("arena", "ratings", **payload["players"])
    return {"arena": target, "arena_transcript": transcript}


def stage_report(ctx):
    """Turn the measurements into something a person can read."""
    import json

    from .report import write_report

    payload_path = ctx.resolve_evaluation()
    if payload_path:
        with open(payload_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        # An arena score with no evaluation beside it is a smaller report, not a
        # missing one - and refusing to write it would mean a run that scored
        # three models on a held-out set ends with nothing a person can read.
        payload = {
            "student": ctx.config["models"].get("student"),
            "teacher": ctx.config["models"].get("teacher"),
            "teacher_adapter": ctx.config["models"].get("teacher_adapter"),
            "device": ctx.hardware["device"],
            "dtype": ctx.hardware.get("dtype_name"),
        }
        ctx.log.info("      no evaluation found - reporting on the answer key alone")

    # The arena writes its own file, and the report is a separate stage that may
    # run in a later invocation - so it is read from disk rather than passed
    # through ctx.results, which `--only report` would not have populated.
    arena_path = ctx.resolve_arena()
    if arena_path and os.path.isfile(arena_path):
        with open(arena_path, encoding="utf-8") as handle:
            payload["arena"] = json.load(handle)

    if not payload_path and not payload.get("arena"):
        raise StageFailed(
            "nothing to report on: no evaluation.json and no arena.json, in this "
            "run or any earlier one. Run the evaluate or arena stage first.")

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


def stage_upload(ctx, groups=None):
    """Sync the finished run bundle to object storage.

    The manifest is rewritten first so the uploaded copy reflects the run that is
    ending, rather than the state it was in several stages ago.
    """
    from .remote import s3

    ctx.run.write_manifest()
    summary = s3.upload_bundle(ctx.config, ctx.run.dir, ctx.run.run_id,
                               groups=groups, log=ctx.log)
    ctx.run.event("upload", "bundle", **summary)
    return {"uploaded": summary["uri"]}


STAGES = {
    "preflight": stage_preflight,
    "teacher-check": stage_teacher_check,
    "smoke": stage_smoke,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "arena": stage_arena,
    "report": stage_report,
    "publish": stage_publish,
    "upload": stage_upload,
}

# Stages that only make sense when a feature is switched on. Returning a reason
# here rather than failing keeps "not configured" distinct from "went wrong".
# (section, key, reason). The key is what has to be truthy for the stage to be
# worth running - usually an `enabled` switch, but for the arena it is the file
# itself: a profile that names no held-out set has nothing to score.
CONDITIONAL = {
    "publish": ("publish", "enabled", "publish.enabled is false"),
    "upload": ("s3", "enabled", "s3.enabled is false"),
    "arena": ("evaluation", "arena_file", "evaluation.arena_file is not set"),
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
    section, key, reason = CONDITIONAL.get(name, (None, None, None))
    if section and not (ctx.config.get(section) or {}).get(key):
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
            # A hard stop. No further stage runs - the point of the ceiling is that
            # crossing it ends the spending - with one exception below.
            run.log.error(f"{label:<26} STOPPED  {_fmt_duration(time.time() - started)}")
            run.log.error(f"      {exc}")
            run.log.error(f"      the last checkpoint under {run.checkpoint_dir} is "
                          f"what survives")
            run.finish("stopped", str(exc))
            _rescue_upload(ctx)
            return 4
        except StageRefused as exc:
            # Declined before spending anything. Nothing is broken, nothing was
            # trained, and the message already says how to proceed - so this
            # reports as a refusal rather than a failure, and stops quietly.
            run.log.warning(f"{label:<26} REFUSED  "
                            f"{_fmt_duration(time.time() - started)}")
            for line in str(exc).splitlines():
                run.log.warning(f"      {line}")
            run.finish("refused", str(exc))
            run.log.info("")
            run.log.info(f"  REFUSED at {name} - nothing was trained and nothing "
                         f"was spent.")
            run.log.info(f"  run bundle : {run.dir}")
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


def _rescue_upload(ctx):
    """Ship what exists after a hard stop, checkpoints included.

    A hard stop on a rented machine is followed by that machine being destroyed,
    so anything not synced here is gone. The checkpoint is added to whatever
    s3.upload normally covers precisely because it is the only artifact a stopped
    run has - the adapter was never written.

    Best effort by definition: the run has already failed, and a storage problem
    on the way out must not replace the limit message with a stack trace.
    """
    if not (ctx.config.get("s3") or {}).get("enabled"):
        return
    groups = list((ctx.config["s3"].get("upload") or []))
    if "checkpoints" not in groups:
        groups.append("checkpoints")
    try:
        ctx.log.warning("      syncing what exists before the machine goes away")
        stage_upload(ctx, groups=groups)
    except Exception as exc:  # noqa: BLE001
        ctx.log.error(f"      could not sync the run bundle: {exc}")
        ctx.log.error(f"      it remains on disk at {ctx.run.dir}")


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
