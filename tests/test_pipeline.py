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
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(workspace, **overrides):
    config = kdc.load_config(os.path.join(CONFIGS, "smoke.yaml"), use_env=False,
                             set_overrides=overrides or None)
    config["project"]["runs_dir"] = os.path.join(workspace, "runs")
    return config


def fake_stages(record, failing=None, raising=None):
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
    pipeline.STAGES.clear()
    pipeline.STAGES.update({name: make(name) for name in
                            ["preflight", "teacher-check", "smoke", "train",
                             "evaluate", "report", "publish", "upload"]})


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
                    "evaluate", "arena", "report", "publish", "upload"], plan


def test_only_from_skip(workspace):
    config = make_config(workspace)
    assert [n for n, _ in pipeline.planned_stages(config, only="train")] == ["train"]
    assert [n for n, _ in pipeline.planned_stages(config, start_from="evaluate")] == \
        ["evaluate", "arena", "report", "publish", "upload"]
    assert "smoke" not in [n for n, _ in pipeline.planned_stages(config, skip=["smoke"])]


def test_unknown_stage_name_is_rejected(workspace):
    config = make_config(workspace)
    for kwargs in ({"only": "trian"}, {"start_from": "evaluat"}, {"skip": ["smock"]}):
        try:
            pipeline.planned_stages(config, **kwargs)
        except ValueError as exc:
            assert "is not a stage" in str(exc), exc
        else:
            raise AssertionError(f"{kwargs} should have been rejected")


def test_gates_come_from_the_config(workspace):
    plan = dict(pipeline.planned_stages(make_config(workspace)))
    assert plan["train"] is True, "train should be a gate"
    assert plan["report"] is False, "report should not be a gate"


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #
def test_all_stages_run_when_nothing_fails(workspace):
    record = []
    fake_stages(record)
    code, _ = run_with(workspace, record)
    assert code == 0, code
    # publish and upload are disabled in the config, so they never execute.
    assert record == ["preflight", "teacher-check", "smoke", "train",
                      "evaluate", "report"], record


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
    fake_stages(record, failing={"report"})
    code, run_dir = run_with(workspace, record)
    assert code == 0, f"a non-gate failure should not fail the run, got {code}"
    assert "report" in record
    assert stages_in_manifest(run_dir)["report"]["status"] == "failed"


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
    assert "evaluate" not in record, "evaluate should not run after a failed gate"


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


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"pipeline: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
