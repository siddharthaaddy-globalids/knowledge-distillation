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
                    "evaluate", "report", "publish", "upload"], plan


def test_only_from_skip(workspace):
    config = make_config(workspace)
    assert [n for n, _ in pipeline.planned_stages(config, only="train")] == ["train"]
    assert [n for n, _ in pipeline.planned_stages(config, start_from="evaluate")] == \
        ["evaluate", "report", "publish", "upload"]
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
