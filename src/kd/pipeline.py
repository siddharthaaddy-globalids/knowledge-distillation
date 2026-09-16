"""
The gated pipelines: training, and - separately - evaluation.

    kd pipeline --config configs/enlibra/enlibraQ25-3B.yaml
    kd eval     --config configs/enlibra/enlibraQ25-3B.yaml --adapter s3://.../final_adapter

TRAINING walks the stages listed under `pipeline:` in the config and stops at
the first failed gate:

    preflight      resolve config and hardware, fetch remote inputs
    teacher-check  teacher only - missing weights, NaN scan, coherence
    smoke          2 real steps; measures s/step and projects the full run
    train
    evaluation     skipped unless evaluation.after_training - see below
    publish        skipped unless publish.enabled
    upload         skipped unless s3.enabled; also runs after a failure

EVALUATION walks the stages under `evaluation.stages` against an adapter that
already exists - this checkout's newest, or one named by path or s3:// URI:

    preflight      fetch the teacher and the adapter, say what will be scored
    evaluate       fidelity and capability, token by token, against the teacher
    arena          skipped unless evaluation.arena_file names a held-out set
    report
    upload         skipped unless s3.enabled

The two are separate because they have different lives. Training happens once,
on the machine with the GPU, and the adapter it produces is the thing worth
keeping - so it leaves the machine as soon as it exists. Evaluation happens as
many times as there are questions to ask of that adapter: a quick look, the
full held-out set, a re-score after the answer parser was fixed, on whatever
machine is to hand. Each evaluation is a directory of its own INSIDE the
adapter's bundle:

    runs/<train-run>/
        final_adapter/
        evaluation/
            quick-2026-09-15-1030/     evaluation.json, arena.json, report.html, ...
            full-2026-09-15-1412/

and the bucket copy of the bundle has the same shape, so a listing of it shows
every time the adapter was scored, whichever machine did it.

`evaluation.after_training: true` puts the evaluation back inside the training
run, as the `evaluation` stage - the pod already has the teacher resident and
paid for. It writes into the same evaluation/ layout, so nothing downstream
can tell the difference.

A GATE stage that fails aborts the run. A non-gate stage that fails is reported
and the pipeline carries on, so a broken report never destroys a good adapter.

The order and the gate flags come from the config, not from this file, so a
profile can say what a run of that profile means. Narrowing a single run is a
command-line matter instead:

    --only train           just that stage
    --from arena           that stage and everything after it
    --skip smoke           everything except that

A stage is a function taking one Context and returning a dict of things later
stages may want. That is the whole contract - there is no base class and no
registration decorator, because a dozen stages do not need either.
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

    def __init__(self, config, hardware, run, budget, options=None, bundle=None,
                 evaluation=False):
        self.config = config
        self.hardware = hardware
        self.run = run
        self.budget = budget
        self.options = options or {}
        # The directory holding the adapter this run is about. For a training
        # run that is the run directory itself; for an evaluation it is the
        # bundle the adapter came from, and `run` is a directory inside its
        # evaluation/. Stages that look for earlier output start here.
        self.bundle = bundle or run.dir
        self.evaluation = evaluation
        # Filled in as the run proceeds: seconds_per_step by smoke, adapter by
        # train, evaluation by evaluate. Stages read what earlier ones left.
        self.results = {}

    @property
    def log(self):
        return self.run.log

    # --- inputs a stage needs that an earlier stage may not have produced ---- #
    #
    # `--from arena` and `--only report` open a NEW directory, because every
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
        named = self.options.get("adapter")
        if named:
            local, source = locate_adapter(named, self.config, log=self.log)
            if source:
                self.options.setdefault("adapter_source", source)
            # Written back so the next stage to ask sees a local path: localise
            # is a no-op on one, so evaluate and arena share the one download
            # without either needing to know the other ran.
            self.options["adapter"] = local
            return local
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
        those open a NEW directory, so the file a later stage wants was written
        by an earlier invocation and has to be found rather than assumed.

        Looked for, in order: this run; the other evaluations of the same
        bundle; every evaluation of every bundle under runs_dir; and finally a
        bundle's top level, which is where evaluations were written before they
        moved into evaluation/.
        """
        import glob

        candidate = self.results.get(key) or self.run.path(filename)
        if os.path.isfile(candidate):
            return candidate
        runs_dir = self.config["project"].get("runs_dir") or "./runs"
        patterns = [
            os.path.join(self.bundle, runlog.EVALUATION_DIR, "*", filename),
            os.path.join(runs_dir, "*", runlog.EVALUATION_DIR, "*", filename),
            os.path.join(runs_dir, "*", filename),
        ]
        for pattern in patterns:
            # Skip the `latest` pointer for the same reason discovery does: where
            # the platform makes it a real symlink it matches this glob as well,
            # and naming an alias instead of a run id makes the log say
            # something that will not be true tomorrow.
            found = sorted(
                (p for p in glob.glob(pattern)
                 if not runlog.is_latest_alias(p, runs_dir)
                 and os.path.abspath(p) != os.path.abspath(candidate)),
                key=os.path.getmtime, reverse=True)
            if found:
                newest = os.path.normpath(found[0]).replace("\\", "/")
                self.log.info(f"      using the newest {what} found: {newest}")
                return newest
        return None

    def ensure_inputs(self):
        """Fetch any s3:// input preflight has not fetched, and settle the teacher.

        preflight is where remote inputs are normally resolved, and every stage
        after it inherits a config whose paths are local. But `--from arena` and
        `--only evaluate` do not run preflight, so those stages would hand a raw
        `s3://...` string to from_pretrained and get "Repo id must be in the form
        namespace/repo_name" - a message about the Hub, for a problem that has
        nothing to do with the Hub.

        Settling the teacher means: a models.teacher that is really a LoRA
        adapter is split into base + adapter, and an adapter in MLX format is
        converted once. Both are what preflight does, repeated here for the
        same reason.

        Called by the stages that load a model, not by the pipeline, and that
        distinction is the point: `--only report` must not trigger a multi-
        gigabyte download to render an HTML file from JSON it already has.

        A no-op once the config holds local paths, so calling it twice costs a
        dictionary scan.
        """
        from . import paths

        try:
            if paths.remote_values(self.config):
                paths.resolve_inputs(self.config, self.log)
            paths.normalise_teacher(self.config, log=self.log)
            adapter = self.config["models"].get("teacher_adapter")
            # A local PEFT directory needs no conversion, and checking a Hub
            # id costs a file listing - once, in preflight, is enough.
            if adapter and not paths.is_adapter_dir(adapter):
                paths.ensure_peft_adapter(self.config, self.log)
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


def locate_adapter(named, config, log=None):
    """(local directory, s3:// source or None) for an adapter someone named.

    Tolerant of a path that names a file inside the adapter, which is what
    copying out of a bucket listing gives you; an s3:// URI is fetched into
    the shared cache. Raises StageFailed when what is there is not an adapter,
    because a report about weights that were never read is worse than no
    report.
    """
    from . import paths

    # adapter_dir_of first: a path copied out of a bucket listing names
    # adapter_config.json, and the directory is what loads.
    named = paths.adapter_dir_of(named)
    # The URI as typed survives the rewrite below, because the report says
    # where the adapter lives in the bucket and the local path alone cannot.
    source = named if paths.is_remote(named) else None
    try:
        local = paths.localise(named, config, log=log, label="adapter")
    except RuntimeError as exc:
        raise StageFailed(str(exc)) from exc
    if not runlog.is_adapter(local):
        raise StageFailed(
            f"--adapter {local} is not a LoRA adapter directory: no "
            f"adapter_config.json in it.")
    return local, source


# --------------------------------------------------------------------------- #
# Training stages
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

    # A models.teacher that is really a LoRA adapter - a fine-tune stored as
    # just its deltas - is split here into the base it records plus the
    # adapter, so that the teacher is merged from the canonical base rather
    # than failing to load as a whole model. The base is remembered as
    # models.teacher_base, which is what lets the evaluation score it.
    try:
        split = paths.normalise_teacher(config, log=ctx.log)
    except Exception as exc:
        raise StageFailed(str(exc)) from exc
    if split:
        ctx.run.event("preflight", "teacher_split", **split)

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
        ctx.log.info(f"  teacher base: {config['models']['teacher']}")

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


def stage_quantize(ctx):
    """Pack the distilled student to 4-bit weights, as a checkpoint vLLM loads.

    After training and before evaluation, so the arena can score the packed
    student beside the dense one and the report can say what the packing cost.
    Not a gate: a failed quantization is a missing column, and throwing away a
    good adapter over it would be the wrong trade by a wide margin.

    The output goes into the run bundle, beside the adapter that produced it, so
    it uploads with the run - it is the artifact that actually gets deployed, and
    leaving it outside means a rented pod is destroyed with it still on board.
    """
    from transformers import AutoTokenizer

    from . import merge, paths, quantize

    settings = ctx.config.get("quantization") or {}

    # BEFORE anything expensive. Merging the student writes a full copy of it -
    # 1.2 GB for a 0.6B smoke model, 15 GB for the real one - and discovering
    # afterwards that the packer is not installed spends that for nothing. This
    # check used to come after the merge, and a Mac smoke run wrote 1.19 GB and
    # then failed on the very next line.
    reason = quantize.unavailable_reason()
    if reason:
        raise StageFailed(reason)

    adapter = ctx.results.get("adapter") or ctx.resolve_adapter()
    if not adapter:
        raise StageFailed("nothing to quantize: this run produced no adapter")

    calibration = settings.get("calibration_file") or ctx.config["dataset"].get("source")
    calibration = (ctx.config.get("evaluation") or {}).get("arena_file") \
        if not str(calibration or "").endswith(".jsonl") else calibration
    if not calibration or not os.path.isfile(str(calibration)):
        raise StageFailed(
            "nothing to calibrate on. Set quantization.calibration_file to the "
            "training .jsonl - GPTQ measures its rounding against the rows the "
            "student will actually be asked to produce, and generic text tunes "
            "it for a distribution this model never sees.")

    base_id = paths.base_for_adapter(adapter, ctx.config["models"]["student"],
                                     log=ctx.log)
    source = adapter if os.path.isfile(
        os.path.join(str(adapter), "tokenizer_config.json")) else base_id
    tokenizer = AutoTokenizer.from_pretrained(source)

    # vLLM and GPTQ both want a dense checkpoint, and this is the same merge the
    # arena's distilled player uses - so the model that gets packed is exactly
    # the model the dense column scores.
    dense = merge.materialise(paths.merged_dir(ctx.config, adapter), base_id,
                              adapter, config=ctx.config, tokenizer=tokenizer,
                              log=ctx.log)
    scheme = settings.get("scheme") or "W4A16"
    out = settings.get("output_dir") or paths.quantized_dir(ctx.config, adapter,
                                                              scheme)
    try:
        written = quantize.quantize(
            dense, out, calibration, scheme=scheme,
            group_size=int(settings.get("group_size") or 128),
            ignore=settings.get("ignore"),
            samples=int(settings.get("calibration_samples") or 128),
            max_seq_length=int(settings.get("max_seq_length") or 2048),
            dampening=float(settings.get("dampening") or 0.01),
            log=ctx.log)
    except RuntimeError as exc:
        raise StageFailed(str(exc)) from exc

    facts = quantize.summarise(written, source=dense)
    if facts.get("compression"):
        ctx.log.info(f"      {facts['dense_bytes'] / 1e9:.2f} GB -> "
                     f"{facts['bytes'] / 1e9:.2f} GB "
                     f"({facts['compression']:.1f}x smaller)")

    # The sizes are measured into `facts` HERE, while both directories still
    # exist, so the report can still state the compression ratio after the dense
    # copy is dropped at the end of the run. See _drop_merge.
    ctx.run.write_metrics({"quantization": facts})
    return {"quantized_model": written, "quantization": facts}


def stage_evaluation(ctx):
    """Score this run's adapter, inside the run, as one stage of the training.

    The whole evaluation pipeline - evaluate, arena, report - run into
    <run>/evaluation/<name>-<date>/ exactly as `kd eval` would run it later,
    so the bundle looks the same whichever way it was scored. The upload is
    left out: the training run's own upload stage ships evaluation/** with
    everything else.

    Off unless evaluation.after_training says otherwise. On by choice for a
    pod run where the teacher is already resident, and for the smoke and flow
    profiles whose job is to prove every stage runs.
    """
    adapter = ctx.resolve_adapter()
    if not adapter:
        raise StageFailed(
            f"nothing to evaluate: no adapter in this run, and none found under "
            f"{ctx.config['project'].get('runs_dir')}.")

    code, outcome = run_evaluation(
        ctx.config, adapter=adapter, hardware=ctx.hardware,
        options={"allow_bad_teacher": ctx.options.get("allow_bad_teacher", False),
                 "adapter_source": ctx.options.get("adapter_source")},
        nested=True, argv=ctx.run.argv, quiet=ctx.run.quiet)

    # What the evaluation measured, summarised into the training run's own
    # metrics.json - the numbers a bucket listing of the bundle is opened for.
    if outcome.get("metrics"):
        ctx.run.write_metrics(outcome["metrics"])
    results = {"evaluation_dir": outcome.get("dir")}
    for key in ("evaluation", "arena", "report", "no_improvement"):
        if outcome.get(key):
            results[key] = outcome[key]
    if code == 3:
        results["no_improvement"] = True
    elif code != 0:
        raise StageFailed(f"the evaluation failed (exit {code}); see "
                          f"{outcome.get('dir')}")
    return results


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


# --------------------------------------------------------------------------- #
# Evaluation stages
# --------------------------------------------------------------------------- #
def stage_eval_preflight(ctx):
    """Fetch what the evaluation needs and say what it is about to score."""
    from . import arena, paths

    ctx.log.info(f"  profile   : {ctx.config['_meta'].get('profile')}")
    ctx.log.info(f"  device    : {ctx.hardware['device']} ({ctx.hardware['dtype_name']})")

    adapter = ctx.resolve_adapter()
    if not adapter:
        raise StageFailed(
            f"nothing to evaluate: no adapter named, and none found under "
            f"{ctx.config['project'].get('runs_dir')}.\n"
            f"Name one with --adapter <dir or s3://...>, or set evaluation.adapter.")
    ctx.results["adapter"] = adapter
    ctx.log.info(f"  adapter   : {adapter}")
    if ctx.options.get("adapter_source"):
        ctx.log.info(f"  fetched   : {ctx.options['adapter_source']}")
    recorded = paths.adapter_base(adapter)
    if recorded:
        ctx.log.info(f"  trained on: {recorded}")
    ctx.log.info(f"  writes to : {ctx.run.dir}")

    # The players, validated before anything loads: a misspelt one would
    # otherwise be discovered after the first model had been scored.
    try:
        players = arena.chosen_players(ctx.config)
    except ValueError as exc:
        raise StageFailed(str(exc)) from exc
    needs_teacher = "teacher" in players or "teacher-base" in players
    if needs_teacher:
        ctx.ensure_inputs()
    teacher_base = paths.teacher_base_of(ctx.config)
    if "teacher-base" in players and not teacher_base:
        ctx.log.info("  !! teacher-base is in evaluation.players but the teacher "
                     "is a merged checkpoint and models.teacher_base is not set; "
                     "that player is skipped")
        players = tuple(p for p in players if p != "teacher-base")
    # Same for the packed student: the quantize stage may have been skipped, or
    # have failed, and announcing a player that the arena will then drop is the
    # kind of small lie that costs someone ten minutes reading a transcript.
    if arena.QUANTIZED in players and not arena.resolve_quantized(
            ctx.config, adapter, log=ctx.log):
        ctx.log.info(f"  !! {arena.QUANTIZED} is in evaluation.players but "
                     f"nothing has been packed for this adapter; that player "
                     f"is skipped")
        players = tuple(p for p in players if p != arena.QUANTIZED)
    ctx.log.info(f"  players   : {', '.join(players)}")
    ctx.log.info(f"  teacher   : {ctx.config['models']['teacher']}"
                 + (f" + {ctx.config['models']['teacher_adapter']}"
                    if ctx.config["models"].get("teacher_adapter") else ""))
    if teacher_base and "teacher-base" in players:
        ctx.log.info(f"  its base  : {teacher_base}")
    ctx.log.info(f"  student   : {ctx.config['models']['student']}")

    # The held-out set, when the arena will run: a missing file is found
    # here, in seconds, rather than after the fidelity pass.
    settings = ctx.config.get("evaluation") or {}
    arena_file = settings.get("arena_file")
    if arena_file:
        try:
            arena_file = paths.localise(arena_file, ctx.config, log=ctx.log,
                                        label="held-out set")
        except RuntimeError as exc:
            raise StageFailed(str(exc)) from exc
        if not os.path.isfile(arena_file):
            raise StageFailed(f"evaluation.arena_file points at {arena_file}, "
                              f"which does not exist")
        settings["arena_file"] = arena_file
        ctx.log.info(f"  answer key: {arena_file}")

        # The generation engine, checked here for the same reason the answer key
        # is: `engine: vllm` on a machine with no vLLM is a failure worth having
        # in seconds, not after the teacher has been pulled out of the bucket.
        engine = str(settings.get("engine") or "hf").lower()
        if engine not in arena.ENGINES:
            raise StageFailed(f"unknown evaluation.engine {engine!r}; valid: "
                              f"{', '.join(arena.ENGINES)}")
        if engine == "vllm":
            from . import vllm_runner

            reason = vllm_runner.unavailable_reason()
            if reason:
                raise StageFailed(reason)
        ctx.log.info(f"  engine    : {engine}")

    destination = None
    if (ctx.config.get("s3") or {}).get("enabled"):
        try:
            bucket, key = evaluation_destination(ctx)
            destination = f"s3://{bucket}/{key}"
            ctx.log.info(f"  uploads to: {destination}")
        except Exception as exc:  # noqa: BLE001 - said now, enforced at upload
            ctx.log.warning(f"  !! cannot work out where to upload: {exc}")
    ctx.run.event("preflight", "evaluation", adapter=adapter, players=list(players),
                  bundle=ctx.bundle, destination=destination)
    return {"players": list(players), "destination": destination}


def stage_evaluate(ctx):
    """Score the adapter against the teacher on the held-out split."""
    import argparse

    from . import arena, evaluate, paths

    ctx.ensure_inputs()

    adapter = ctx.resolve_adapter()
    if not adapter:
        raise StageFailed(
            f"nothing to evaluate: no adapter in this run, and none found under "
            f"{ctx.config['project'].get('runs_dir')}.\n"
            f"Train one first, or name it with --adapter.")

    settings = ctx.config.get("evaluation") or {}
    players = list(settings.get("players") or arena.PLAYERS)
    payload_path = ctx.run.path("evaluation.json")
    args = argparse.Namespace(
        config=ctx.config["_meta"].get("source"),
        adapter=adapter,
        teacher=ctx.config["models"]["teacher"],
        student=ctx.config["models"]["student"],
        teacher_adapter=ctx.config["models"].get("teacher_adapter"),
        teacher_base=paths.teacher_base_of(ctx.config)
        if "teacher-base" in players else None,
        no_teacher_base="teacher-base" not in players,
        samples=int(settings.get("samples") or 50),
        device=ctx.hardware["device"],
        dtype="auto",
        no_generations=False,
        quantized=(ctx.results.get("quantize") or {}).get("quantized_model")
        or arena.resolve_quantized(ctx.config, adapter),
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
    """Score every player on the held-out set, and rate them.

    Not a gate: a student that scores badly is a result worth keeping, not a
    reason to throw away the adapter and the report.
    """
    import json

    from . import arena

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

    try:
        players = arena.chosen_players(ctx.config)
    except ValueError as exc:
        raise StageFailed(str(exc)) from exc
    # The teacher is fetched only when a player needs it: `players: [base,
    # distilled]` must not pull eight gigabytes it will never load.
    if "teacher" in players or "teacher-base" in players:
        ctx.ensure_inputs()

    engine = str(settings.get("engine") or "vllm").lower()
    # The packed student, from this run's quantize stage when it ran, else from
    # the cache a previous one wrote. Skipped, not failed, when nothing has been
    # packed: four players is still an arena.
    quantized = (ctx.results.get("quantize") or {}).get("quantized_model") \
        or arena.resolve_quantized(ctx.config, adapter, log=ctx.log)
    if arena.QUANTIZED in players and not quantized:
        players = tuple(p for p in players if p != arena.QUANTIZED)
        ctx.log.info(f"      {arena.QUANTIZED} skipped: nothing packed for this "
                     f"adapter")

    ctx.log.info(f"      {len(questions)} held-out questions from {path}")
    ctx.log.info(f"      players: {', '.join(players)}")
    predictions, formats, unanswered, completions = arena.play(
        ctx.config, ctx.hardware, adapter, questions,
        max_new_tokens=int(settings.get("arena_max_new_tokens") or 512),
        log=ctx.log, players=players, engine=engine,
        vllm_options=settings.get("vllm"), quantized=quantized)

    payload = arena.summarise(
        predictions, [q["gold"] for q in questions],
        formats=formats, unanswered=unanswered,
        rounds=int(settings.get("arena_elo_rounds") or 25),
        seed=int(ctx.config["project"]["seed"]),
        questions=questions, completions=completions)
    payload["arena_file"] = str(path)
    payload["adapter"] = str(adapter)
    # Which engine generated these completions. Recorded because hf and vllm are
    # not bit-identical (see arena.ENGINES), so a payload that does not say
    # cannot be compared with another payload at all.
    payload["engine"] = engine
    payload["quantized"] = str(quantized) if quantized else None
    if quantized:
        from . import quantize as kd_quantize

        payload["quantization_config"] = kd_quantize.summarise(quantized)

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

    ctx.log.info("")
    for line in arena.render(payload).splitlines():
        ctx.log.info(line)
    ctx.run.write_metrics({"arena": payload["players"],
                           "closeness": payload.get("closeness")})
    ctx.run.event("arena", "ratings", **payload["players"])
    return {"arena": target, "arena_transcript": transcript}


def stage_report(ctx):
    """Turn the measurements into something a person can read."""
    import json

    from .report import training_settings, write_report

    payload_path = ctx.resolve_evaluation()
    if payload_path:
        with open(payload_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        # An arena score with no evaluation beside it is a smaller report, not a
        # missing one - and refusing to write it would mean a run that scored
        # the players on a held-out set ends with nothing a person can read.
        from . import paths

        payload = {
            "student": ctx.config["models"].get("student"),
            "teacher": ctx.config["models"].get("teacher"),
            "teacher_adapter": ctx.config["models"].get("teacher_adapter"),
            "teacher_base": paths.teacher_base_of(ctx.config),
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

    # Where the adapter is - here, and in the bucket. Taken from the payloads
    # rather than resolved again, because the adapter THEY name is the one the
    # numbers are about, and resolving afresh could fetch something else.
    from . import paths

    adapter = payload.get("adapter") or (payload.get("arena") or {}).get("adapter")
    # An evaluation does not upload the adapter, so it is not told it is the
    # run that will: the adapter is wherever it was fetched from or wherever
    # its own run recorded sending it.
    own = {} if ctx.evaluation else {"run_id": ctx.run.run_id, "run_dir": ctx.run.dir}
    payload["adapter_locations"] = paths.adapter_locations(
        adapter, ctx.config, source=ctx.options.get("adapter_source"), **own)         if adapter else None
    payload["profile"] = ctx.config["_meta"].get("source")
    payload["evaluation_id"] = ctx.run.run_id if ctx.evaluation else None
    # From the config this stage runs with, not from evaluation.json: `--only
    # report` may be re-rendering an old run, and the section explains the
    # settings as they stand, honouring evaluation.report_training either way.
    payload["training"] = training_settings(ctx.config)

    suffix = str((ctx.config.get("evaluation") or {}).get("report_format", "html"))
    written = write_report(payload, ctx.run.path(f"report.{suffix.lstrip('.')}"))

    ctx.run.write_metrics({
        "fidelity": payload.get("fidelity"),
        "capability": payload.get("capability"),
        "efficiency": payload.get("efficiency"),
    })
    ctx.log.info(f"      {written}")
    return {"report": str(written)}


def evaluation_destination(ctx):
    """(bucket, key) an evaluation uploads to: beside the adapter it scored.

    The adapter's own bucket address decides it - the URI it was fetched from,
    or the upload an earlier run of this checkout recorded - and the evaluation
    goes under that bundle's evaluation/. An adapter with no known copy in the
    bucket goes where a run of this profile would have put its bundle, so the
    two ways of scoring one adapter still land in one place.
    """
    from . import paths
    from .remote import s3

    adapter = ctx.results.get("adapter") or ctx.options.get("adapter")
    bundle_id = os.path.basename(os.path.normpath(ctx.bundle))
    # No run_dir hint: that would make adapter_locations answer "where this
    # run WILL upload", which for an evaluation is nowhere. What is wanted is
    # where the adapter already IS - fetched from, or recorded by its run.
    where = paths.adapter_locations(adapter, ctx.config,
                                    source=ctx.options.get("adapter_source"))
    uri = where.get("s3")
    if uri:
        # Strip the adapter's path inside the bundle off its URI, so that an
        # adapter at checkpoints/checkpoint-50 resolves to the bundle too.
        relative = os.path.relpath(os.path.abspath(str(adapter)),
                                   os.path.abspath(ctx.bundle)).replace(os.sep, "/")
        uri = uri.rstrip("/")
        bundle_uri = (uri[: -len(relative) - 1] if uri.endswith("/" + relative)
                      else uri.rsplit("/", 1)[0])
    else:
        bundle_uri = s3.uri_of(s3.bucket_or_die(ctx.config),
                               s3.run_prefix(ctx.config, bundle_id))
    return s3.evaluation_prefix(bundle_uri, ctx.run.run_id)


def stage_eval_upload(ctx):
    """Ship this evaluation to the bucket, into the bundle of the adapter it scored."""
    from .remote import s3

    ctx.run.write_manifest()
    destination = evaluation_destination(ctx)
    summary = s3.upload_bundle(ctx.config, ctx.run.dir, ctx.run.run_id,
                               log=ctx.log, destination=destination)
    ctx.run.event("upload", "bundle", **summary)
    return {"uploaded": summary["uri"]}


STAGES = {
    "preflight": stage_preflight,
    "teacher-check": stage_teacher_check,
    "smoke": stage_smoke,
    "train": stage_train,
    "quantize": stage_quantize,
    "evaluation": stage_evaluation,
    "publish": stage_publish,
    "upload": stage_upload,
}

EVAL_STAGES = {
    "preflight": stage_eval_preflight,
    "evaluate": stage_evaluate,
    "arena": stage_arena,
    "report": stage_report,
    "upload": stage_eval_upload,
}

# Stages that only make sense when a feature is switched on. Returning a reason
# here rather than failing keeps "not configured" distinct from "went wrong".
# (section, key, reason). The key is what has to be truthy for the stage to be
# worth running - usually an `enabled` switch, but for the arena it is the file
# itself: a profile that names no held-out set has nothing to score.
CONDITIONAL = {
    "publish": ("publish", "enabled", "publish.enabled is false"),
    "upload": ("s3", "enabled", "s3.enabled is false"),
    "quantize": ("quantization", "enabled", "quantization.enabled is false"),
    "evaluation": ("evaluation", "after_training",
                   "evaluation.after_training is false - score it later with kd eval"),
}

EVAL_CONDITIONAL = {
    "upload": ("s3", "enabled", "s3.enabled is false"),
    "arena": ("evaluation", "arena_file", "evaluation.arena_file is not set"),
}

# Runs even after an earlier gate failed, so a crashed run still ships its logs.
ALWAYS_RUN = {"upload"}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def planned_stages(config, only=None, start_from=None, skip=(), section="pipeline"):
    """The stage list for this run: [(name, is_gate), ...].

    `section` is the config block whose `stages` list is walked: `pipeline`
    for training, `evaluation` for scoring an adapter.

    Raises ValueError on a stage name that does not exist, because silently running
    a shorter pipeline than asked for is the kind of thing nobody notices until the
    evaluation they expected is missing.
    """
    declared = (config.get(section) or {}).get("stages") or []
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


def _skip_reason(ctx, name, conditional):
    """Why this stage should not run at all, or None."""
    section, key, reason = conditional.get(name, (None, None, None))
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
    """Walk the training stages. Returns the process exit code.

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
    unknown = [name for name, _ in plan if name not in STAGES]
    if unknown:
        raise ValueError(
            f"pipeline.stages names stages this version does not have: {unknown}.\n"
            f"  evaluate, arena and report now belong to evaluation.stages; put "
            f"`evaluation` in pipeline.stages and set evaluation.after_training "
            f"to score inside the training run, or run `kd eval` afterwards.")

    run.log.info(describe(config, hardware, run_id=run.run_id))
    run.log.info(f"  stages     : {' -> '.join(name for name, _ in plan)}")
    run.log.info("")

    failure, stopped = _walk(ctx, plan, STAGES, CONDITIONAL)
    if stopped is not None:
        return stopped
    return _finish(ctx, failure)


def _walk(ctx, plan, stages, conditional, indent=""):
    """Run `plan` against the `stages` table. Returns (failure, exit_or_None).

    The loop both pipelines share. `failure` is (stage, exception) for the
    gate that stopped the run, or None; an exit code is returned only for the
    two outcomes that end a run on the spot - a limit breached, a projection
    refused - because those have already written their own ending.
    """
    run = ctx.run
    total = len(plan)
    failure = None
    for index, (name, is_gate) in enumerate(plan, start=1):
        label = f"{indent}[{index}/{total}] {name}"

        if failure and name not in ALWAYS_RUN:
            run.skip_stage(name, f"an earlier gate failed ({failure[0]})")
            run.log.info(f"{label:<26} SKIPPED (earlier failure)")
            continue

        reason = _skip_reason(ctx, name, conditional)
        if reason:
            run.skip_stage(name, reason)
            run.log.info(f"{label:<26} skipped - {reason}")
            continue

        started = time.time()
        run.log.info(f"{label:<26} ...")
        try:
            with run.stage(name):
                ctx.results.update(stages[name](ctx) or {})
        except LimitExceeded as exc:
            # A hard stop. No further stage runs - the point of the ceiling is that
            # crossing it ends the spending - with one exception below.
            run.log.error(f"{label:<26} STOPPED  {_fmt_duration(time.time() - started)}")
            run.log.error(f"      {exc}")
            run.log.error(f"      the last checkpoint under {run.checkpoint_dir} is "
                          f"what survives")
            run.finish("stopped", str(exc))
            _rescue_upload(ctx)
            return failure, 4
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
            return failure, 4
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
    return failure, None


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


def _drop_merge(ctx):
    """Delete the dense bf16 merge, unless quantization.keep_merged says not to.

    AT THE VERY END OF THE RUN, after upload, because everything before it may
    still want the thing: the quantiser packs from it, and the arena's dense
    `distilled` player is generated from it under the vllm engine. Deleting it
    in the quantize stage would simply make the arena rebuild it ten minutes
    later.

    Worth deleting at all because it is the largest artifact a run produces - 15
    GiB against the packed 5.7 for an 8B student - and the only large one that
    is cheaply regenerable: the adapter and the base survive, and remaking it is
    a couple of CPU-minutes. On a pod with a 40 GB volume, holding it alongside
    the packed copy and the teacher is what runs the disk out.

    Only ever inside this run's own directory. An adapter handed over as a bare
    directory merges into the shared cache, which belongs to whoever put it
    there and is not this run's to tidy.
    """
    import shutil

    from . import paths

    if (ctx.config.get("quantization") or {}).get("keep_merged"):
        return
    adapter = ctx.results.get("adapter") or ctx.run.adapter_dir
    merged = paths.merged_dir(ctx.config, adapter)
    if not (os.path.isdir(merged)
            and os.path.abspath(merged).startswith(os.path.abspath(ctx.run.dir))):
        return
    size = sum(os.path.getsize(os.path.join(merged, name))
               for name in os.listdir(merged)
               if name.endswith(".safetensors"))
    shutil.rmtree(merged, ignore_errors=True)
    ctx.log.info(f"  dropped the bf16 merge ({size / 1e9:.1f} GB) - it rebuilds "
                 f"from the adapter. quantization.keep_merged keeps it.")


def _finish(ctx, failure):
    run = ctx.run
    try:
        _drop_merge(ctx)
    except Exception as exc:  # noqa: BLE001 - tidying must never fail a good run
        run.log.warning(f"  !! could not drop the bf16 merge: {exc}")
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
    if ctx.results.get("evaluation_dir"):
        run.log.info(f"  evaluation : {ctx.results['evaluation_dir']}")
    if ctx.results.get("report"):
        run.log.info(f"  report     : {ctx.results['report']}")
    if ctx.results.get("no_improvement"):
        run.log.warning("  the adapter did not improve on the base student")
        return 3
    return 0


# --------------------------------------------------------------------------- #
# Evaluating an adapter
# --------------------------------------------------------------------------- #
def evaluation_home(adapter, config, source=None):
    """(bundle on disk, directory the evaluation is written into) for an adapter.

    The bundle is the directory holding the adapter. For one fetched from the
    bucket that is a cache directory, and nobody looks for results in a cache
    - so the evaluation is written under project.runs_dir instead, in a
    directory named after the bundle in the bucket, which mirrors where the
    upload puts it: runs/<train-run>/evaluation/<eval-id>/ here, and
    <prefix>/runs/<train-run>/evaluation/<eval-id>/ there.
    """
    from . import paths

    bundle = runlog.bundle_of(adapter)
    fetched = source or paths.cached_source(adapter)
    if fetched:
        runs_dir = os.path.expanduser(config["project"].get("runs_dir") or "./runs")
        bundle = os.path.abspath(os.path.join(runs_dir, os.path.basename(bundle)))
    return bundle, os.path.join(bundle, runlog.EVALUATION_DIR)


def describe_evaluation(config, hardware, run, adapter, bundle):
    """The startup banner for an evaluation: what is scored, and where it goes."""
    from . import arena

    meta = config.get("_meta", {})
    try:
        players = ", ".join(arena.chosen_players(config))
    except ValueError as exc:
        players = f"!! {exc}"
    lines = [
        "=" * 78,
        f" {config['project']['name']} - evaluation",
        "=" * 78,
        f" evaluation    : {run.run_id}",
        f" config        : {meta.get('source')}",
        f" adapter       : {adapter}",
        f" bundle        : {bundle}",
        f" writes to     : {run.dir}",
        f" players       : {players}",
        f" teacher       : {config['models']['teacher']}"
        + (f" + {config['models']['teacher_adapter']}"
           if config["models"].get("teacher_adapter") else ""),
        f" student       : {config['models']['student']}",
        f" device        : {hardware['device']} ({hardware['dtype_name']})",
    ]
    for note in hardware["notes"]:
        lines.append(f"   - {note}")
    lines.append("=" * 78)
    return "\n".join(lines)


def run_evaluation(config, adapter=None, only=None, start_from=None, skip=(),
                   options=None, hardware=None, nested=False, argv=None, quiet=False):
    """Score an adapter into <bundle>/evaluation/<name>-<date>/.

    `adapter` is a directory, a file inside one, or an s3:// URI; None means
    evaluation.adapter from the config, else the newest adapter under
    project.runs_dir. `nested` is the training pipeline's `evaluation` stage
    calling in: the upload is left to the training run, and the banner is
    shorter.

    Returns (exit code, outcome). The exit code follows run_pipeline: 0 for a
    clean scoring, 1 for a failed gate, 3 when the adapter did not improve on
    the base student. `outcome` carries the evaluation directory and what the
    stages produced.
    """
    from . import paths

    hardware = hardware or resolve_device(config)
    options = dict(options or {})
    settings = config.setdefault("evaluation", {})

    # Which adapter. Named on the command line, named in the config, or the
    # newest one this checkout trained - in that order, because each is a
    # more deliberate statement than the next.
    named = adapter or options.get("adapter") or settings.get("adapter")
    if named:
        local, source = locate_adapter(named, config)
        source = source or options.get("adapter_source")
    else:
        found = runlog.discover_adapters(config["project"].get("runs_dir") or "./runs")
        if not found:
            raise StageFailed(
                f"nothing to evaluate: no adapter named, and none found under "
                f"{config['project'].get('runs_dir')}.\n"
                f"Name one with --adapter <dir or s3://...>, or set evaluation.adapter.")
        local, source = found[0], paths.cached_source(found[0])
    options.update(adapter=local, adapter_source=source)

    bundle, home = evaluation_home(local, config, source=source)
    eval_id = runlog.make_eval_id(settings.get("name") or config["_meta"].get("profile"))
    run = runlog.Run(config, argv=argv, run_id=eval_id, parent=home,
                     checkpoints=False, latest=False, quiet=quiet)
    outcome = {"dir": run.dir, "id": run.run_id}
    with run:
        budget = Budget(config, started=run.started)
        ctx = Context(config, hardware, run, budget, options, bundle=bundle,
                      evaluation=True)
        ctx.results["adapter"] = local

        plan = planned_stages(config, only, start_from, skip, section="evaluation")
        if nested:
            # The training run's own upload stage ships evaluation/** along
            # with the adapter; a second upload here would send it twice.
            plan = [(name, gate) for name, gate in plan if name != "upload"]
        unknown = [name for name, _ in plan if name not in EVAL_STAGES]
        if unknown:
            raise ValueError(
                f"evaluation.stages names stages this version does not have: "
                f"{unknown}. Available: {', '.join(EVAL_STAGES)}")

        if nested:
            run.log.info(f"      evaluation {run.run_id} -> {run.dir}")
        else:
            run.log.info(describe_evaluation(config, hardware, run, local, bundle))
        run.log.info(f"{'      ' if nested else '  '}stages     : "
                     f"{' -> '.join(name for name, _ in plan)}")
        run.log.info("")

        failure, stopped = _walk(ctx, plan, EVAL_STAGES, EVAL_CONDITIONAL,
                                 indent="      " if nested else "")
        code = stopped if stopped is not None else _finish_evaluation(ctx, failure)

        outcome.update({k: v for k, v in ctx.results.items()
                        if k in ("evaluation", "arena", "arena_transcript", "report",
                                 "uploaded", "no_improvement", "players")})
        outcome["metrics"] = _metrics_of(run)
    return code, outcome


def _metrics_of(run):
    """What the evaluation wrote to its metrics.json, for the training run's copy."""
    import json

    try:
        with open(run.path(runlog.METRICS), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _finish_evaluation(ctx, failure):
    run = ctx.run
    run.log.info("")
    if failure:
        name, exc = failure
        run.finish("failed", f"{name}: {exc}")
        run.log.error(f"  FAILED at {name}")
        run.log.error(f"  evaluation : {run.dir}")
        return 1

    run.finish("ok")
    run.log.info(f"  evaluation : {run.dir}")
    if ctx.results.get("report"):
        run.log.info(f"  report     : {ctx.results['report']}")
    if ctx.results.get("uploaded"):
        run.log.info(f"  uploaded   : {ctx.results['uploaded']}")
    if ctx.results.get("no_improvement"):
        run.log.warning("  the adapter did not improve on the base student")
        return 3
    return 0
