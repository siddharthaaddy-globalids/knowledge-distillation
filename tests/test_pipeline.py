"""Checks for stage selection, gating and limit enforcement.

No models are loaded: the stage table is monkeypatched with fakes, so this runs in
about a second and exercises the control flow rather than the training. What is
being tested is the part that decides whether your budget gets spent.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import config as kdc      # noqa: E402
from kd import pipeline, runlog   # noqa: E402
from kd.limits import Budget, LimitExceeded  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

passed = []
failed = []


def check(name, fn):
    workspace = tempfile.mkdtemp(prefix="kd-test-")
    saved = dict(pipeline.STAGES)
    saved_eval = dict(pipeline.EVAL_STAGES)
    try:
        fn(workspace)
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
    finally:
        pipeline.STAGES.clear()
        pipeline.STAGES.update(saved)
        pipeline.EVAL_STAGES.clear()
        pipeline.EVAL_STAGES.update(saved_eval)
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(workspace, **overrides):
    config = kdc.load_config(os.path.join(CONFIGS, "smollm", "smoke.yaml"), use_env=False,
                             set_overrides=overrides or None)
    config["project"]["runs_dir"] = os.path.join(workspace, "runs")
    return config


def fake_stages(record, failing=None, raising=None, table=None, names=None):
    """Replace every stage with one that records that it ran."""
    def make(name):
        def stage(ctx):
            record.append(name)
            if raising and name in raising:
                raise raising[name]
            if failing and name in failing:
                raise pipeline.StageFailed(f"{name} was told to fail")
            return {}
        return stage
    table = pipeline.STAGES if table is None else table
    names = names or ["preflight", "teacher-check", "smoke", "train",
                      "evaluation", "publish", "upload"]
    table.clear()
    table.update({name: make(name) for name in names})


def fake_eval_stages(record, **kwargs):
    """The evaluation table, faked the same way."""
    fake_stages(record, table=pipeline.EVAL_STAGES,
                names=["preflight", "evaluate", "arena", "report", "upload"], **kwargs)


def run_with(workspace, record, **kwargs):
    config = kwargs.pop("config", None) or make_config(workspace)
    with runlog.Run(config, quiet=True) as run:
        code = pipeline.run_pipeline(config, run, **kwargs)
        run_dir = run.dir
    return code, run_dir


def stages_in_manifest(run_dir):
    with open(os.path.join(run_dir, runlog.MANIFEST), encoding="utf-8") as handle:
        return {entry["name"]: entry for entry in json.load(handle)["stages"]}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_default_plan_is_the_config_order(workspace):
    plan = [name for name, _ in pipeline.planned_stages(make_config(workspace))]
    assert plan == ["preflight", "teacher-check", "smoke", "train",
                    "evaluation", "publish", "upload"], plan


def test_evaluation_has_its_own_plan(workspace):
    """evaluate, arena and report moved out of the training pipeline."""
    config = make_config(workspace)
    train = [n for n, _ in pipeline.planned_stages(config)]
    assert "evaluate" not in train and "arena" not in train and "report" not in train
    plan = [n for n, _ in pipeline.planned_stages(config, section="evaluation")]
    assert plan == ["preflight", "evaluate", "arena", "report", "upload"], plan
    assert [n for n, _ in pipeline.planned_stages(
        config, start_from="arena", section="evaluation")] == ["arena", "report", "upload"]
    # The two tables agree with the two lists, so neither can drift.
    assert set(train) == set(pipeline.STAGES), set(pipeline.STAGES) ^ set(train)
    assert set(plan) == set(pipeline.EVAL_STAGES)


def test_only_from_skip(workspace):
    config = make_config(workspace)
    assert [n for n, _ in pipeline.planned_stages(config, only="train")] == ["train"]
    assert [n for n, _ in pipeline.planned_stages(config, start_from="evaluation")] == \
        ["evaluation", "publish", "upload"]
    assert "smoke" not in [n for n, _ in pipeline.planned_stages(config, skip=["smoke"])]


def test_unknown_stage_name_is_rejected(workspace):
    config = make_config(workspace)
    for kwargs in ({"only": "trian"}, {"start_from": "evaluat"}, {"skip": ["smock"]},
                   {"start_from": "evaluate"}):
        try:
            pipeline.planned_stages(config, **kwargs)
        except ValueError as exc:
            assert "is not a stage" in str(exc), exc
        else:
            raise AssertionError(f"{kwargs} should have been rejected")


def test_gates_come_from_the_config(workspace):
    plan = dict(pipeline.planned_stages(make_config(workspace)))
    assert plan["train"] is True, "train should be a gate"
    assert plan["evaluation"] is False, "evaluation should not be a gate"


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #
def test_all_stages_run_when_nothing_fails(workspace):
    record = []
    fake_stages(record)
    code, _ = run_with(workspace, record)
    assert code == 0, code
    # publish and upload are disabled in the config, so they never execute.
    # The smoke profile turns evaluation.after_training on; most do not.
    assert record == ["preflight", "teacher-check", "smoke", "train",
                      "evaluation"], record


def test_evaluation_is_left_out_unless_asked_for(workspace):
    """The default: training ends with the adapter, and scoring is `kd eval`."""
    record = []
    fake_stages(record)
    config = make_config(workspace, **{"evaluation.after_training": False})
    code, run_dir = run_with(workspace, record, config=config)
    assert code == 0, code
    assert record == ["preflight", "teacher-check", "smoke", "train"], record
    stages = stages_in_manifest(run_dir)
    assert stages["evaluation"]["status"] == "skipped"
    assert "kd eval" in stages["evaluation"]["reason"], stages["evaluation"]


def test_failed_gate_stops_everything_after_it(workspace):
    record = []
    fake_stages(record, failing={"teacher-check"})
    code, run_dir = run_with(workspace, record)
    assert code == 2, f"teacher-check failure should exit 2, got {code}"
    assert record == ["preflight", "teacher-check"], record
    stages = stages_in_manifest(run_dir)
    assert stages["teacher-check"]["status"] == "failed"
    assert stages["train"]["status"] == "skipped"


def test_non_gate_failure_does_not_stop_the_run(workspace):
    record = []
    fake_stages(record, failing={"evaluation"})
    code, run_dir = run_with(workspace, record)
    assert code == 0, f"a non-gate failure should not fail the run, got {code}"
    assert "evaluation" in record
    assert stages_in_manifest(run_dir)["evaluation"]["status"] == "failed"


def test_disabled_stages_are_skipped_with_a_reason(workspace):
    record = []
    fake_stages(record)
    _, run_dir = run_with(workspace, record)
    stages = stages_in_manifest(run_dir)
    assert stages["publish"]["status"] == "skipped"
    assert "publish.enabled" in stages["publish"]["reason"]
    assert "s3.enabled" in stages["upload"]["reason"]


def test_enabled_upload_runs_even_after_a_failure(workspace):
    record = []
    fake_stages(record, failing={"train"})
    config = make_config(workspace)
    config["s3"]["enabled"] = True
    code, _ = run_with(workspace, record, config=config)
    assert code == 1, code
    assert "upload" in record, "upload should still run so the logs survive"
    assert "evaluation" not in record, "evaluation should not run after a failed gate"


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
def test_limit_breach_is_a_hard_stop(workspace):
    record = []
    fake_stages(record, raising={
        "train": LimitExceeded("limits.max_runtime_minutes", 200.0, 180.0, " min")})
    config = make_config(workspace)
    config["s3"]["enabled"] = True   # would otherwise run; a hard stop must not
    code, run_dir = run_with(workspace, record, config=config)
    assert code == 4, f"a limit breach should exit 4, got {code}"
    assert "upload" not in record, "a hard stop must not continue to later stages"
    with open(os.path.join(run_dir, runlog.MANIFEST), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "stopped", manifest["status"]
    assert "max_runtime_minutes" in manifest["stopped_reason"]


def test_refusal_is_not_reported_as_a_failure(workspace):
    """A run declined for not fitting its limits is not a malfunction.

    Calling it FAILED sends people hunting for a bug that is not there - and it
    reads identically to a genuinely broken stage, which is the one distinction
    that matters when you are deciding whether to retry or investigate.
    """
    record = []
    fake_stages(record, raising={
        "smoke": pipeline.StageRefused(
            "This run would take about 610 min, over the 180 min "
            "limits.max_runtime_minutes.")})
    code, run_dir = run_with(workspace, record)

    assert code == 4, f"a refusal should exit 4 like other limit outcomes, got {code}"
    assert "train" not in record, "a refused run must not train"
    with open(os.path.join(run_dir, runlog.MANIFEST), encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["status"] == "refused",         f"the manifest calls it {manifest['status']!r}, not 'refused'"
    assert "610 min" in manifest["stopped_reason"], manifest["stopped_reason"]


def test_a_real_failure_is_still_a_failure(workspace):
    """The refusal wording must not swallow genuine breakage."""
    record = []
    fake_stages(record, failing={"smoke"})
    code, run_dir = run_with(workspace, record)
    assert code == 1, f"a broken gate should still exit 1, got {code}"
    with open(os.path.join(run_dir, runlog.MANIFEST), encoding="utf-8") as handle:
        assert json.load(handle)["status"] == "failed"


def test_ram_is_detected_not_assumed(workspace):
    """The memory check reads this machine's RAM from the OS.

    A hardcoded size would be wrong on every machine but one, and the warning it
    produces is only worth trusting if the number in it is real.
    """
    from kd import paths

    detected = paths.total_memory()
    assert detected, "no RAM figure was obtained from the operating system"
    # Anything from a small container to a large server, but a plausible size.
    assert 1 * paths.GB < detected < 4096 * paths.GB, f"implausible: {detected}"


def test_memory_warning_scales_with_the_machine(workspace):
    """The same config warns on a small machine and not on a large one."""
    from kd import paths

    estimate = {"counts": {"teacher": 2_000_000_000, "student": 800_000_000},
                "dtype": "float32", "weight_bytes": 2_800_000_000 * 4,
                "total_ram_bytes": 16 * paths.GB}
    assert paths.memory_warning(estimate, "mps"), "11 GB of weights in 16 GB should warn"

    estimate["total_ram_bytes"] = 128 * paths.GB
    assert paths.memory_warning(estimate, "mps") is None, "should not warn on 128 GB"

    # A discrete GPU has its own budget this cannot see, so it is not guessed at.
    estimate["total_ram_bytes"] = 16 * paths.GB
    assert paths.memory_warning(estimate, "cuda") is None


def test_step_ceiling_is_the_tighter_of_the_two(workspace):
    config = make_config(workspace, **{"training.max_steps": 500})
    assert Budget(config).max_steps == 500, "no ceiling set, so training wins"
    config["limits"]["max_steps"] = 100
    assert Budget(config).max_steps == 100, "the lower ceiling should win"
    config["limits"]["max_steps"] = 900
    assert Budget(config).max_steps == 500, "a ceiling above the request cannot bind"


def test_cost_is_inert_without_a_price(workspace):
    config = make_config(workspace)
    config["limits"]["max_cost_usd"] = 0.01
    local = Budget(config, price_per_hour=None)
    assert local.spend_usd() == 0.0
    assert local.breach() is None, "a cost cap must not fire when nothing is rented"
    assert "inert" in local.summary()


def test_projection_refuses_a_run_that_cannot_finish(workspace):
    config = make_config(workspace, **{"training.max_steps": 300})
    config["limits"]["max_runtime_minutes"] = 10
    budget = Budget(config)
    refusal = budget.refuse_if_impossible(budget.project(4.3))
    assert refusal and "max_runtime_minutes" in refusal, refusal
    # The suggestion has to actually fit, or it is not a suggestion.
    suggested = int(refusal.split("training.max_steps=")[1].split()[0])
    assert suggested * 4.3 / 60.0 <= 10, f"suggested {suggested} steps still overruns"


def test_projection_allows_a_run_that_fits(workspace):
    config = make_config(workspace, **{"training.max_steps": 10})
    config["limits"]["max_runtime_minutes"] = 180
    budget = Budget(config)
    assert budget.refuse_if_impossible(budget.project(4.3)) is None


def test_hard_stop_still_syncs_what_exists(workspace):
    """A stopped run on a rented machine has to ship its checkpoint before dying.

    Nothing is uploaded after a hard stop by the normal stage order, because a hard
    stop skips every later stage. That is right for spending and wrong for the
    artifact, so the rescue path exists - and it adds checkpoints, since a stopped
    run never wrote an adapter.
    """
    from kd.remote import s3 as s3mod

    sent = {}
    original = s3mod.upload_bundle
    s3mod.upload_bundle = lambda config, run_dir, run_id, groups=None, log=None: (
        sent.update({"groups": list(groups or [])}) or
        {"uri": "s3://b/k", "files": 1, "bytes": 1, "groups": list(groups or [])})
    try:
        record = []
        fake_stages(record, raising={
            "train": LimitExceeded("limits.max_cost_usd", 2.5, 2.0, " USD")})
        config = make_config(workspace)
        config["s3"]["enabled"] = True
        code, _ = run_with(workspace, record, config=config)
        assert code == 4, code
        assert sent, "a hard stop uploaded nothing at all"
        assert "checkpoints" in sent["groups"], \
            f"the checkpoint was not rescued: {sent['groups']}"
    finally:
        s3mod.upload_bundle = original


def test_a_failed_rescue_does_not_mask_the_limit(workspace):
    """If the bucket is unreachable, the run still reports why it stopped."""
    from kd.remote import s3 as s3mod

    original = s3mod.upload_bundle

    def explode(*_args, **_kwargs):
        raise RuntimeError("bucket unreachable")

    s3mod.upload_bundle = explode
    try:
        record = []
        fake_stages(record, raising={
            "train": LimitExceeded("limits.max_runtime_minutes", 200.0, 180.0, " min")})
        config = make_config(workspace)
        config["s3"]["enabled"] = True
        code, run_dir = run_with(workspace, record, config=config)
        assert code == 4, f"the limit exit code should survive a failed upload, got {code}"
        with open(os.path.join(run_dir, runlog.MANIFEST), encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert "max_runtime_minutes" in manifest["stopped_reason"], manifest
    finally:
        s3mod.upload_bundle = original


# --------------------------------------------------------------------------- #
# Teacher adapters - caught in preflight, not after a multi-gigabyte download
# --------------------------------------------------------------------------- #
def test_missing_teacher_adapter_is_caught_early(workspace):
    from kd.teacher import adapter_problem

    problem = adapter_problem(os.path.join(workspace, "not-there"), "Qwen/Qwen3.5-2B")
    assert problem and "does not exist" in problem, problem
    assert "kd convert-adapter" in problem, "no route out of the failure was offered"


def test_mlx_adapter_is_recognised_and_the_fix_given(workspace):
    """An MLX adapter is the likeliest wrong thing to point this at.

    PEFT only rejects it once the base model is loaded - gigabytes later - so it
    is worth recognising from a file listing.
    """
    from kd.teacher import adapter_problem

    mlx = os.path.join(workspace, "mlx-adapter")
    os.makedirs(mlx)
    for name in ("adapters.safetensors", "adapter_config.json"):
        with open(os.path.join(mlx, name), "w") as handle:
            handle.write("{}")

    problem = adapter_problem(mlx, "Qwen/Qwen3.5-2B")
    assert problem and "MLX" in problem, problem
    assert "kd convert-adapter" in problem and "--base Qwen/Qwen3.5-2B" in problem, problem


def test_peft_adapter_is_accepted(workspace):
    from kd.teacher import adapter_problem

    peft = os.path.join(workspace, "peft-adapter")
    os.makedirs(peft)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        with open(os.path.join(peft, name), "w") as handle:
            handle.write("{}")
    assert adapter_problem(peft, "Qwen/Qwen3.5-2B") is None


def test_hub_ids_are_not_mistaken_for_paths(workspace):
    """org/name is a Hub id; '/' alone cannot decide, since it is altsep on Windows."""
    from kd.teacher import _looks_like_hub_id

    for value in ("org/name", "Qwen/Qwen3.5-2B"):
        assert _looks_like_hub_id(value), f"{value} should read as a Hub id"
    for value in ("./peft-adapter", "/abs/path", "C:/x/y", "~/a", "a/b/c",
                  "..\rel", "plain-name"):
        assert not _looks_like_hub_id(value), f"{value} should read as a path"


def test_no_adapter_configured_is_fine(workspace):
    from kd.teacher import adapter_problem

    assert adapter_problem(None) is None
    assert adapter_problem("") is None


# --------------------------------------------------------------------------- #
# Picking up where a previous run left off
#
# --from and --only open a new run directory, so the stage they start with has to
# find its inputs in an earlier run or those flags are useless.
# --------------------------------------------------------------------------- #
def _seed_previous_run(workspace, adapter=True, evaluation=True):
    config = make_config(workspace)
    with runlog.Run(config, quiet=True) as run:
        if adapter:
            os.makedirs(run.adapter_dir, exist_ok=True)
            with open(os.path.join(run.adapter_dir, "adapter_config.json"), "w") as fh:
                fh.write("{}")
        if evaluation:
            with open(run.path("evaluation.json"), "w") as fh:
                json.dump({"fidelity": {}}, fh)
        seeded = run.dir
    return config, seeded


def _fresh_context(workspace, config):
    run = runlog.Run(config, quiet=True)
    return pipeline.Context(config, {"device": "cpu", "dtype_name": "float32",
                                     "notes": []}, run, Budget(config)), run


def test_adapter_falls_back_to_the_newest_run(workspace):
    config, seeded = _seed_previous_run(workspace)
    ctx, run = _fresh_context(workspace, config)
    try:
        found = ctx.resolve_adapter()
        assert found, "no adapter found in an earlier run"
        assert os.path.normpath(seeded) in os.path.normpath(os.path.abspath(found)), found
    finally:
        run.close()


def test_evaluation_falls_back_to_the_newest_run(workspace):
    config, seeded = _seed_previous_run(workspace)
    ctx, run = _fresh_context(workspace, config)
    try:
        found = ctx.resolve_evaluation()
        assert found and os.path.isfile(found), found
    finally:
        run.close()


def test_named_adapter_beats_the_newest_run(workspace):
    """--adapter is an instruction, not a hint: discovery must not override it."""
    config, seeded = _seed_previous_run(workspace)
    named = os.path.join(workspace, "elsewhere", "final_adapter")
    os.makedirs(named, exist_ok=True)
    with open(os.path.join(named, "adapter_config.json"), "w") as fh:
        fh.write("{}")
    ctx, run = _fresh_context(workspace, config)
    ctx.options["adapter"] = named
    try:
        found = ctx.resolve_adapter()
        assert os.path.normpath(found) == os.path.normpath(named), found
        assert os.path.normpath(seeded) not in os.path.normpath(found)
    finally:
        run.close()


def test_named_adapter_accepts_a_file_inside_it(workspace):
    """A path copied out of a bucket listing names adapter_config.json."""
    config = make_config(workspace)
    named = os.path.join(workspace, "elsewhere2", "final_adapter")
    os.makedirs(named, exist_ok=True)
    with open(os.path.join(named, "adapter_config.json"), "w") as fh:
        fh.write("{}")
    ctx, run = _fresh_context(workspace, config)
    ctx.options["adapter"] = os.path.join(named, "adapter_config.json")
    try:
        assert os.path.normpath(ctx.resolve_adapter()) == os.path.normpath(named)
    finally:
        run.close()


def test_named_adapter_that_is_not_one_fails_loudly(workspace):
    """Better a stage failure than a report about weights that were never read."""
    config = make_config(workspace)
    empty = os.path.join(workspace, "not-an-adapter")
    os.makedirs(empty, exist_ok=True)
    ctx, run = _fresh_context(workspace, config)
    ctx.options["adapter"] = empty
    try:
        try:
            ctx.resolve_adapter()
        except pipeline.StageFailed as exc:
            assert "adapter_config.json" in str(exc), exc
        else:
            raise AssertionError("a directory with no adapter in it was accepted")
    finally:
        run.close()


def test_no_previous_run_resolves_to_nothing(workspace):
    config = make_config(workspace)
    ctx, run = _fresh_context(workspace, config)
    try:
        assert ctx.resolve_adapter() is None
        assert ctx.resolve_evaluation() is None
    finally:
        run.close()


# --------------------------------------------------------------------------- #
# Where the adapter is - on this machine and in the bucket
#
# The report names both, and the way it learns the S3 address differs by how the
# adapter arrived: typed as s3://, fetched into the cache, trained by this run,
# or trained by an earlier run that recorded its upload.
# --------------------------------------------------------------------------- #
def _adapter_at(where):
    os.makedirs(where, exist_ok=True)
    with open(os.path.join(where, "adapter_config.json"), "w") as fh:
        fh.write("{}")
    return where


def test_a_typed_s3_adapter_is_remembered_by_its_uri(workspace):
    """resolve_adapter rewrites --adapter to a local path; the URI must survive."""
    from kd import paths

    config = make_config(workspace)
    local = _adapter_at(os.path.join(workspace, "cache", "bucket", "kd", "a"))
    ctx, run = _fresh_context(workspace, config)
    ctx.options["adapter"] = "s3://bucket/kd/a/adapter_config.json"
    saved = paths.localise
    paths.localise = lambda where, *a, **k: local if paths.is_remote(where) else where
    try:
        assert os.path.normpath(ctx.resolve_adapter()) == os.path.normpath(local)
        assert ctx.options["adapter_source"] == "s3://bucket/kd/a"
        where = paths.adapter_locations(local, config,
                                        source=ctx.options["adapter_source"])
        assert where["s3"] == "s3://bucket/kd/a", where
        assert where["s3_status"] == "the copy it was fetched from", where
    finally:
        paths.localise = saved
        run.close()


def test_a_cached_adapter_knows_where_it_was_fetched_from(workspace):
    """The cache marker's first line is the URI, so a bare local path still says."""
    from kd import paths

    config = make_config(workspace)
    local = _adapter_at(os.path.join(workspace, "cache", "final_adapter"))
    paths.mark_complete(local, "s3://bucket/kd/runs/r9/final_adapter\n3 objects")
    where = paths.adapter_locations(local, config)
    assert where["s3"] == "s3://bucket/kd/runs/r9/final_adapter", where
    assert where["s3_status"] == "the copy it was fetched from", where


def test_this_runs_adapter_names_its_upload_destination(workspace):
    """The report is written before the upload stage, so this is a destination."""
    from kd import paths

    config = make_config(workspace, **{"s3.enabled": True, "s3.bucket": "b",
                                       "s3.prefix": "kd"})
    ctx, run = _fresh_context(workspace, config)
    try:
        local = _adapter_at(run.adapter_dir)
        where = paths.adapter_locations(local, config, run_id=run.run_id,
                                        run_dir=run.dir)
        assert where["s3"] == f"s3://b/kd/runs/{run.run_id}/final_adapter", where
        assert "end of this run" in where["s3_status"], where
    finally:
        run.close()


def test_this_runs_adapter_with_s3_off_says_so(workspace):
    from kd import paths

    config = make_config(workspace, **{"s3.enabled": False})
    ctx, run = _fresh_context(workspace, config)
    try:
        local = _adapter_at(run.adapter_dir)
        where = paths.adapter_locations(local, config, run_id=run.run_id,
                                        run_dir=run.dir)
        assert where["s3"] is None and "s3.enabled is false" in where["note"], where
    finally:
        run.close()


def test_an_earlier_runs_adapter_uses_its_recorded_upload(workspace):
    """events.jsonl remembers where the bundle went; the adapter sits inside it."""
    from kd import paths

    config = make_config(workspace)
    # Explicit ids: two runs opened in the same second would share a directory.
    with runlog.Run(config, quiet=True, run_id="quiet") as never:
        _adapter_at(never.adapter_dir)
        seeded = never.dir
    with runlog.Run(config, quiet=True, run_id="old") as earlier:
        _adapter_at(earlier.adapter_dir)
        earlier.event("upload", "bundle", uri="s3://b/kd/runs/old", files=3)
        adapter, run_dir = earlier.adapter_dir, earlier.dir
    assert runlog.recorded_upload(run_dir) == "s3://b/kd/runs/old"
    assert runlog.recorded_upload(seeded) is None
    ctx, run = _fresh_context(workspace, config)
    try:
        where = paths.adapter_locations(adapter, config, run_id=run.run_id,
                                        run_dir=run.dir)
        assert where["s3"] == "s3://b/kd/runs/old/final_adapter", where
        assert where["s3_status"].startswith("uploaded by run "), where
        # ...and one that never uploaded says that, rather than guessing.
        where = paths.adapter_locations(os.path.join(seeded, "final_adapter"),
                                        config, run_id=run.run_id, run_dir=run.dir)
        assert where["s3"] is None and "recorded no upload" in where["note"], where
    finally:
        run.close()


def test_the_report_stage_says_where_the_adapter_is(workspace):
    """What the report is handed, end to end: locations and the profile."""
    from kd import report as report_module

    config = make_config(workspace, **{"s3.enabled": True, "s3.bucket": "b",
                                       "s3.prefix": "kd"})
    captured = {}

    def fake_write_report(payload, path):
        captured.update(payload)
        with open(path, "w") as fh:
            fh.write("<html></html>")
        return path

    saved = report_module.write_report
    report_module.write_report = fake_write_report
    ctx, run = _fresh_context(workspace, config)
    try:
        _adapter_at(run.adapter_dir)
        with open(run.path("arena.json"), "w") as fh:
            json.dump({"questions": 2, "players": {}, "adapter": run.adapter_dir}, fh)
        pipeline.stage_report(ctx)
        where = captured["adapter_locations"]
        assert where["local"].endswith("final_adapter"), where
        assert where["s3"] == f"s3://b/kd/runs/{run.run_id}/final_adapter", where
        assert captured["profile"].endswith("smoke.yaml"), captured["profile"]
    finally:
        report_module.write_report = saved
        run.close()


# --------------------------------------------------------------------------- #
# Evaluating an adapter: a directory of its own inside the adapter's bundle
# --------------------------------------------------------------------------- #
def _trained_bundle(workspace, config, run_id="trained"):
    """A finished training run with an adapter in it, as `kd pipeline` leaves one."""
    with runlog.Run(config, quiet=True, run_id=run_id) as run:
        _adapter_at(run.adapter_dir)
        bundle, adapter = run.dir, run.adapter_dir
    return bundle, adapter


def test_an_evaluation_lives_inside_the_adapters_bundle(workspace):
    record = []
    fake_eval_stages(record)
    config = make_config(workspace, **{"evaluation.name": "quick",
                                       "evaluation.after_training": False})
    bundle, adapter = _trained_bundle(workspace, config)

    code, outcome = pipeline.run_evaluation(config, adapter=adapter, quiet=True)
    assert code == 0, code
    # preflight, evaluate, report ran; arena is off (no arena_file), upload is off.
    assert record == ["preflight", "evaluate", "report"], record
    where = outcome["dir"]
    assert os.path.dirname(os.path.dirname(where)) == bundle, where
    assert os.path.basename(os.path.dirname(where)) == runlog.EVALUATION_DIR, where
    assert os.path.basename(where).startswith("quick-"), where
    # A run of its own: config, manifest, log - and no checkpoints/ to upload.
    for name in (runlog.MANIFEST, runlog.RESOLVED_CONFIG, runlog.RUN_LOG, runlog.EVENTS):
        assert os.path.isfile(os.path.join(where, name)), name
    assert not os.path.isdir(os.path.join(where, runlog.CHECKPOINT_DIR))
    # It must not become `latest`: that pointer is for training runs.
    for pointer in ("latest", "latest.txt"):
        assert not os.path.exists(os.path.join(bundle, runlog.EVALUATION_DIR, pointer))


def test_two_evaluations_of_one_adapter_never_share_a_directory(workspace):
    fake_eval_stages([])
    config = make_config(workspace, **{"evaluation.name": "full"})
    _bundle, adapter = _trained_bundle(workspace, config)
    _, first = pipeline.run_evaluation(config, adapter=adapter, quiet=True)
    _, second = pipeline.run_evaluation(config, adapter=adapter, quiet=True)
    assert first["dir"] != second["dir"], (first, second)
    assert os.path.isdir(first["dir"]) and os.path.isdir(second["dir"])
    assert len(runlog.discover_evaluations(_bundle)) == 2


def test_the_name_says_what_the_evaluation_was_for(workspace):
    fake_eval_stages([])
    config = make_config(workspace)
    _bundle, adapter = _trained_bundle(workspace, config)
    config["evaluation"]["name"] = None
    _, unnamed = pipeline.run_evaluation(config, adapter=adapter, quiet=True)
    assert unnamed["id"].startswith("smoke-"), unnamed["id"]   # the profile
    config["evaluation"]["name"] = "after-parser-fix"
    _, named = pipeline.run_evaluation(config, adapter=adapter, quiet=True)
    assert named["id"].startswith("after-parser-fix-"), named["id"]


def test_the_config_can_name_the_adapter(workspace):
    """evaluation.adapter is the config's way of saying --adapter."""
    record = []
    fake_eval_stages(record)
    config = make_config(workspace)
    bundle, adapter = _trained_bundle(workspace, config)
    # A decoy that is newer, so discovery would pick the wrong one.
    _trained_bundle(workspace, config, run_id="newer")
    config["evaluation"]["adapter"] = os.path.join(adapter, "adapter_config.json")
    _, outcome = pipeline.run_evaluation(config, quiet=True)
    assert os.path.dirname(os.path.dirname(outcome["dir"])) == bundle, outcome["dir"]


def test_no_adapter_anywhere_is_said_plainly(workspace):
    fake_eval_stages([])
    config = make_config(workspace)
    try:
        pipeline.run_evaluation(config, quiet=True)
    except pipeline.StageFailed as exc:
        assert "--adapter" in str(exc) and "evaluation.adapter" in str(exc), exc
    else:
        raise AssertionError("an evaluation with nothing to evaluate started anyway")


def test_a_fetched_adapter_is_scored_under_runs_dir_and_uploaded_beside_itself(workspace):
    """An adapter from the bucket: its cache directory is not where results go.

    The evaluation is written under runs_dir in a directory named after the
    bundle in the bucket, and uploaded to that bundle's evaluation/ - so the
    bucket ends up with runs/<train-run>/{final_adapter, evaluation/<id>}.
    """
    from kd import paths
    from kd.remote import s3 as s3mod

    fake_eval_stages([])
    config = make_config(workspace, **{"s3.enabled": True, "s3.bucket": "b",
                                       "s3.prefix": "kd"})
    cached = _adapter_at(os.path.join(workspace, "cache", "b", "elsewhere", "runs",
                                      "trained-2026-09-10-1416", "final_adapter"))
    paths.mark_complete(cached, "s3://b/elsewhere/runs/trained-2026-09-10-1416/final_adapter\n3 objects")

    _, outcome = pipeline.run_evaluation(config, adapter=cached, quiet=True)
    runs_dir = config["project"]["runs_dir"]
    expected = os.path.join(runs_dir, "trained-2026-09-10-1416", runlog.EVALUATION_DIR)
    assert os.path.normpath(os.path.dirname(outcome["dir"])) == os.path.normpath(expected), \
        outcome["dir"]

    # Where the real upload stage would send it, without sending anything.
    sent = {}
    original = s3mod.upload_bundle
    s3mod.upload_bundle = lambda config, run_dir, run_id, groups=None, log=None, destination=None: (
        sent.update({"destination": destination, "run_dir": run_dir}) or
        {"uri": f"s3://{destination[0]}/{destination[1]}", "files": 1, "bytes": 1, "groups": []})
    try:
        pipeline.EVAL_STAGES["upload"] = pipeline.stage_eval_upload
        _, outcome = pipeline.run_evaluation(config, adapter=cached, quiet=True)
    finally:
        s3mod.upload_bundle = original
    bucket, key = sent["destination"]
    assert bucket == "b", sent
    assert key == f"elsewhere/runs/trained-2026-09-10-1416/evaluation/{outcome['id']}", key
    assert outcome["uploaded"].startswith("s3://b/elsewhere/runs/trained-2026-09-10-1416/evaluation/")


def test_a_local_bundle_with_no_upload_record_goes_where_its_run_would_have(workspace):
    from kd.remote import s3 as s3mod

    fake_eval_stages([])
    config = make_config(workspace, **{"s3.enabled": True, "s3.bucket": "b",
                                       "s3.prefix": "kd"})
    _bundle, adapter = _trained_bundle(workspace, config, run_id="local-run")
    sent = {}
    original = s3mod.upload_bundle
    s3mod.upload_bundle = lambda config, run_dir, run_id, groups=None, log=None, destination=None: (
        sent.update({"destination": destination}) or
        {"uri": "s3://x", "files": 1, "bytes": 1, "groups": []})
    try:
        pipeline.EVAL_STAGES["upload"] = pipeline.stage_eval_upload
        _, outcome = pipeline.run_evaluation(config, adapter=adapter, quiet=True)
    finally:
        s3mod.upload_bundle = original
    assert sent["destination"] == ("b", f"kd/runs/local-run/evaluation/{outcome['id']}"), sent


def test_evaluation_inside_the_training_run_uses_the_same_layout(workspace):
    """evaluation.after_training: the stage writes <run>/evaluation/<id>/ too."""
    record = []
    fake_eval_stages(record)
    fake_stages(record)
    pipeline.STAGES["evaluation"] = pipeline.stage_evaluation
    pipeline.STAGES["train"] = lambda ctx: (
        _adapter_at(ctx.run.adapter_dir) and {"adapter": ctx.run.adapter_dir})
    config = make_config(workspace, **{"evaluation.after_training": True,
                                       "evaluation.name": "inline", "s3.enabled": True})
    code, run_dir = run_with(workspace, record, config=config)
    assert code == 0, code
    # The inner upload is left to the training run's own upload stage.
    assert record == ["preflight", "teacher-check", "smoke", "preflight", "evaluate",
                      "report", "upload"], record
    found = runlog.discover_evaluations(run_dir)
    assert len(found) == 1 and os.path.basename(found[0]).startswith("inline-"), found
    stages = stages_in_manifest(run_dir)
    assert stages["evaluation"]["status"] == "ok", stages["evaluation"]
    with open(os.path.join(found[0], runlog.MANIFEST), encoding="utf-8") as handle:
        inner = {e["name"]: e for e in json.load(handle)["stages"]}
    assert inner["evaluate"]["status"] == "ok" and "upload" not in inner, inner


def test_a_failed_evaluation_gate_does_not_fail_the_training(workspace):
    """The adapter is the thing that cost money; a scoring problem must not lose it."""
    record = []
    fake_eval_stages(record, failing={"preflight"})
    fake_stages(record)
    pipeline.STAGES["evaluation"] = pipeline.stage_evaluation
    pipeline.STAGES["train"] = lambda ctx: (
        _adapter_at(ctx.run.adapter_dir) and {"adapter": ctx.run.adapter_dir})
    config = make_config(workspace, **{"evaluation.after_training": True})
    code, run_dir = run_with(workspace, record, config=config)
    assert code == 0, code
    assert stages_in_manifest(run_dir)["evaluation"]["status"] == "failed"


def test_later_evaluation_stages_find_the_bundles_earlier_output(workspace):
    """`kd eval --from report` reads the evaluation.json a previous scoring wrote."""
    config = make_config(workspace)
    bundle, adapter = _trained_bundle(workspace, config)
    earlier = os.path.join(bundle, runlog.EVALUATION_DIR, "full-2026-09-15-1000")
    os.makedirs(earlier)
    with open(os.path.join(earlier, "evaluation.json"), "w") as fh:
        json.dump({"fidelity": {}}, fh)
    run = runlog.Run(config, quiet=True, run_id="full-2026-09-15-1100",
                     parent=os.path.join(bundle, runlog.EVALUATION_DIR),
                     checkpoints=False, latest=False)
    ctx = pipeline.Context(config, {"device": "cpu", "dtype_name": "float32",
                                    "notes": []}, run, Budget(config),
                           bundle=bundle, evaluation=True)
    try:
        found = ctx.resolve_evaluation()
        assert found and os.path.normpath(found) == os.path.normpath(
            os.path.join(earlier, "evaluation.json")), found
    finally:
        run.close()


def test_kd_upload_knows_where_an_evaluation_directory_belongs(workspace):
    """`kd upload runs/<id>/evaluation/<eid>` ships to the bundle's evaluation/."""
    from kd import cli

    config = make_config(workspace, **{"s3.enabled": True, "s3.bucket": "b",
                                       "s3.prefix": "kd"})
    bundle, _adapter = _trained_bundle(workspace, config, run_id="local-run")
    eval_dir = os.path.join(bundle, runlog.EVALUATION_DIR, "full-2026-09-15-1030")
    os.makedirs(eval_dir)
    assert cli._evaluation_destination(config, eval_dir) ==         ("b", "kd/runs/local-run/evaluation/full-2026-09-15-1030")
    # A bundle that recorded its upload elsewhere is followed there instead.
    runlog.append_event(bundle, "upload", "bundle", uri="s3://other/x/runs/local-run")
    assert cli._evaluation_destination(config, eval_dir) ==         ("other", "x/runs/local-run/evaluation/full-2026-09-15-1030")


def test_upload_groups_include_evaluations(workspace):
    from kd.remote import s3 as s3mod

    assert "evaluation" in s3mod.UPLOAD_GROUPS
    assert "evaluation" in make_config(workspace)["s3"]["upload"]


# --------------------------------------------------------------------------- #
# A teacher that is a LoRA adapter
# --------------------------------------------------------------------------- #
def _lora_at(where, base="Qwen/Qwen2.5-3B-Instruct"):
    os.makedirs(where, exist_ok=True)
    with open(os.path.join(where, "adapter_config.json"), "w") as fh:
        json.dump({"base_model_name_or_path": base, "peft_type": "LORA"}, fh)
    with open(os.path.join(where, "adapter_model.safetensors"), "w") as fh:
        fh.write("weights")
    return where


def test_a_teacher_that_is_an_adapter_is_split_into_base_and_adapter(workspace):
    from kd import paths

    config = make_config(workspace)
    lora = _lora_at(os.path.join(workspace, "sft-lora"))
    config["models"].update(teacher=lora, teacher_adapter=None, teacher_base=None)
    split = paths.normalise_teacher(config)
    assert split == {"adapter": lora, "base": "Qwen/Qwen2.5-3B-Instruct",
                     "recorded": "Qwen/Qwen2.5-3B-Instruct"}, split
    assert config["models"]["teacher"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["models"]["teacher_adapter"] == lora
    assert paths.teacher_base_of(config) == "Qwen/Qwen2.5-3B-Instruct"
    # Idempotent: a second pass sees base + adapter and leaves it alone.
    assert paths.normalise_teacher(config) is None


def test_a_directory_of_checkpoints_resolves_to_the_newest(workspace):
    """sft_checkpoints/ holding checkpoint-N/ means the finished fine-tune."""
    from kd import paths

    config = make_config(workspace)
    root = os.path.join(workspace, "sft_checkpoints")
    for step in (100, 300, 200):
        _lora_at(os.path.join(root, f"checkpoint-{step}"))
    os.makedirs(os.path.join(root, "checkpoint-999"))          # no adapter inside
    with open(os.path.join(root, "trainer_state.json"), "w") as fh:
        fh.write("{}")
    config["models"].update(teacher=root, teacher_adapter=None, teacher_base=None)
    split = paths.normalise_teacher(config)
    assert split["adapter"].endswith("checkpoint-300"), split
    assert config["models"]["teacher_adapter"].endswith("checkpoint-300")
    assert config["models"]["teacher"] == "Qwen/Qwen2.5-3B-Instruct"


def test_a_named_teacher_base_beats_what_the_adapter_records(workspace):
    from kd import paths

    config = make_config(workspace)
    lora = _lora_at(os.path.join(workspace, "sft-lora"), base="/somewhere/on/another/box")
    config["models"].update(teacher=lora, teacher_adapter=None,
                            teacher_base="Qwen/Qwen2.5-3B-Instruct")
    paths.normalise_teacher(config)
    assert config["models"]["teacher"] == "Qwen/Qwen2.5-3B-Instruct"


def test_an_adapter_teacher_with_no_known_base_is_refused_with_the_fix(workspace):
    from kd import paths

    config = make_config(workspace)
    mlx = os.path.join(workspace, "mlx-lora")
    os.makedirs(mlx)
    with open(os.path.join(mlx, "adapters.safetensors"), "w") as fh:
        fh.write("weights")
    config["models"].update(teacher=mlx, teacher_adapter=None, teacher_base=None)
    try:
        paths.normalise_teacher(config)
    except RuntimeError as exc:
        assert "models.teacher_base" in str(exc), exc
    else:
        raise AssertionError("an adapter with no base was accepted as a teacher")


def test_a_whole_model_teacher_is_left_alone(workspace):
    from kd import paths

    config = make_config(workspace)
    model_dir = os.path.join(workspace, "merged")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "config.json"), "w") as fh:
        fh.write("{}")
    config["models"].update(teacher=model_dir, teacher_adapter=None, teacher_base=None)
    assert paths.normalise_teacher(config) is None
    assert config["models"]["teacher"] == model_dir
    assert paths.teacher_base_of(config) is None, "a merged teacher has no known base"
    config["models"]["teacher_base"] = "Qwen/Qwen2.5-3B-Instruct"
    assert paths.teacher_base_of(config) == "Qwen/Qwen2.5-3B-Instruct"


def test_teacher_base_is_implied_by_base_plus_adapter(workspace):
    from kd import paths

    config = make_config(workspace)
    config["models"].update(teacher="Qwen/Qwen3.5-2B", teacher_adapter="org/lora",
                            teacher_base=None)
    assert paths.teacher_base_of(config) == "Qwen/Qwen3.5-2B"


def test_preflight_splits_the_teacher_and_records_it(workspace):
    """The pipeline's preflight does the split, and events.jsonl says so."""
    from kd import paths

    config = make_config(workspace)
    lora = _lora_at(os.path.join(workspace, "sft-lora"))
    config["models"].update(teacher=lora, teacher_adapter=None, teacher_base=None)
    saved = paths.memory_estimate
    paths.memory_estimate = lambda *a, **k: None
    ctx, run = _fresh_context(workspace, config)
    try:
        pipeline.stage_preflight(ctx)
        assert config["models"]["teacher_adapter"] == lora
        assert config["models"]["teacher_base"] == "Qwen/Qwen2.5-3B-Instruct"
        run.finish("ok")
        with open(run.path(runlog.EVENTS), encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        assert any(e["event"] == "teacher_split" for e in events), events
    finally:
        paths.memory_estimate = saved
        run.close()


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"pipeline: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
