"""Checks for the run-bundle layer.

Plain asserts, no test framework, and no model downloads: everything here works on
a bare checkout in about a second. Run it with `uv run python tests/test_runlog.py`.

What matters about a run bundle is that it is complete even when the run was not -
a crashed run whose manifest never got written is a run nobody can diagnose - so
most of these check the failure paths.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import config as kdc  # noqa: E402
from kd import runlog  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

passed = []
failed = []


def check(name, fn):
    workspace = tempfile.mkdtemp(prefix="kd-test-")
    try:
        fn(workspace)
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(workspace, **overrides):
    config = kdc.load_config(os.path.join(CONFIGS, "smoke.yaml"), use_env=False)
    config["project"]["runs_dir"] = os.path.join(workspace, "runs")
    for path, value in overrides.items():
        config["project"][path] = value
    return config


def read_events(run_dir):
    with open(os.path.join(run_dir, runlog.EVENTS), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_manifest(run_dir):
    with open(os.path.join(run_dir, runlog.MANIFEST), encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def test_creates_the_bundle(workspace):
    with runlog.Run(make_config(workspace)) as run:
        run_dir = run.dir
    for name in (runlog.RESOLVED_CONFIG, runlog.RUN_LOG, runlog.EVENTS, runlog.MANIFEST):
        assert os.path.isfile(os.path.join(run_dir, name)), f"missing {name}"
    assert os.path.isdir(os.path.join(run_dir, runlog.CHECKPOINT_DIR))


def test_run_id_shape(workspace):
    """<profile>-<YYYY-MM-DD>-<HHMM>: readable, sortable within a profile, no hash."""
    import re

    with runlog.Run(make_config(workspace)) as run:
        assert re.fullmatch(r"smoke-\d{4}-\d{2}-\d{2}-\d{4}", run.run_id), run.run_id


def test_two_runs_in_one_minute_do_not_share_a_directory(workspace):
    config = make_config(workspace)
    with runlog.Run(config, quiet=True) as first:
        with runlog.Run(config, quiet=True) as second:
            assert second.run_id == first.run_id + "-2", (first.run_id, second.run_id)
            assert second.dir != first.dir
            with runlog.Run(config, quiet=True) as third:
                assert third.run_id == first.run_id + "-3", third.run_id


def test_output_dir_pins_the_directory(workspace):
    pinned = os.path.join(workspace, "somewhere-specific")
    with runlog.Run(make_config(workspace, output_dir=pinned)) as run:
        assert os.path.normpath(run.dir) == os.path.normpath(pinned), run.dir


def test_resolved_config_round_trips(workspace):
    import yaml
    config = make_config(workspace)
    config["training"]["max_steps"] = 77
    with runlog.Run(config) as run:
        written = yaml.safe_load(open(os.path.join(run.dir, runlog.RESOLVED_CONFIG),
                                      encoding="utf-8"))
    assert written["training"]["max_steps"] == 77
    # _meta describes how the config was assembled, not the run - it must not leak
    # into the file that is meant to be re-runnable as-is.
    assert "_meta" not in written


# --------------------------------------------------------------------------- #
# Failure paths - the ones that matter
# --------------------------------------------------------------------------- #
def test_manifest_written_after_an_exception(workspace):
    run_dir = None
    try:
        with runlog.Run(make_config(workspace)) as run:
            run_dir = run.dir
            raise RuntimeError("teacher exploded")
    except RuntimeError:
        pass
    manifest = read_manifest(run_dir)
    assert manifest["status"] == "failed", manifest["status"]
    assert "teacher exploded" in manifest["stopped_reason"], manifest["stopped_reason"]


def test_keyboard_interrupt_is_not_a_failure(workspace):
    run_dir = None
    try:
        with runlog.Run(make_config(workspace)) as run:
            run_dir = run.dir
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    manifest = read_manifest(run_dir)
    assert manifest["status"] == "interrupted", manifest["status"]


def test_failed_stage_is_recorded_and_reraised(workspace):
    with runlog.Run(make_config(workspace)) as run:
        try:
            with run.stage("train"):
                raise ValueError("out of memory")
        except ValueError:
            pass
        run_dir = run.dir
    stages = read_manifest(run_dir)["stages"]
    assert stages[0]["name"] == "train"
    assert stages[0]["status"] == "failed"
    assert "out of memory" in stages[0]["error"]


def test_skipped_stage_records_why(workspace):
    with runlog.Run(make_config(workspace)) as run:
        run.skip_stage("publish", "publish.enabled is false")
        run_dir = run.dir
    stage = read_manifest(run_dir)["stages"][0]
    assert stage["status"] == "skipped"
    assert stage["reason"] == "publish.enabled is false"


# --------------------------------------------------------------------------- #
# Streams
# --------------------------------------------------------------------------- #
def test_stdout_is_captured_into_the_log(workspace):
    # The tee writes to whatever stdout was when the Run was created, so pointing
    # that at a buffer first keeps this test's fixture lines out of the test
    # output. quiet=True handles the logger; this handles the bare print().
    import io

    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with runlog.Run(make_config(workspace), quiet=True) as run:
            run.log.info("a logged line")
            run.log.debug("a debug line")
            print("a library printed this")
            run_dir = run.dir
    finally:
        sys.stdout = real_stdout
    text = open(os.path.join(run_dir, runlog.RUN_LOG), encoding="utf-8").read()
    for expected in ("a logged line", "a debug line", "a library printed this"):
        assert expected in text, f"run.log lost: {expected}"


def test_stdout_is_restored_afterwards(workspace):
    before = sys.stdout
    with runlog.Run(make_config(workspace)):
        assert sys.stdout is not before, "stdout was never teed"
    assert sys.stdout is before, "stdout was not restored"


def test_events_are_one_json_object_per_line(workspace):
    with runlog.Run(make_config(workspace)) as run:
        run.event("train", "step", step=1, loss=1.5)
        run.event("train", "step", step=2, loss=1.4)
        run_dir = run.dir
    events = read_events(run_dir)
    steps = [e for e in events if e["event"] == "step"]
    assert len(steps) == 2, f"expected 2 step events, got {len(steps)}"
    assert steps[0]["loss"] == 1.5
    assert {"ts", "elapsed_s", "stage", "event"} <= set(steps[0]), steps[0]
    assert events[0]["event"] == "start" and events[-1]["event"] == "end"


def test_metrics_merge_rather_than_overwrite(workspace):
    with runlog.Run(make_config(workspace)) as run:
        run.write_metrics({"training": {"steps": 2}})
        run.write_metrics({"fidelity": {"agreement": 0.61}})
        run_dir = run.dir
    metrics = json.load(open(os.path.join(run_dir, runlog.METRICS), encoding="utf-8"))
    assert metrics["training"]["steps"] == 2, "the first write was lost"
    assert metrics["fidelity"]["agreement"] == 0.61


# --------------------------------------------------------------------------- #
# Pointers and discovery
# --------------------------------------------------------------------------- #
def test_latest_points_at_the_newest_run(workspace):
    with runlog.Run(make_config(workspace)) as run:
        run_dir = run.dir
    parent = os.path.dirname(run_dir)
    link = os.path.join(parent, "latest")
    pointer = os.path.join(parent, "latest.txt")
    # A symlink where the platform allows one, a text pointer where it does not.
    if os.path.islink(link) or os.path.isdir(link):
        assert os.path.realpath(link) == os.path.realpath(run_dir)
    else:
        assert os.path.isfile(pointer), "neither a 'latest' link nor latest.txt"
        assert open(pointer, encoding="utf-8").read().strip() == run_dir


def test_discover_adapters_finds_run_output(workspace):
    with runlog.Run(make_config(workspace)) as run:
        os.makedirs(run.adapter_dir, exist_ok=True)
        with open(os.path.join(run.adapter_dir, "adapter_config.json"), "w") as handle:
            handle.write("{}")
        runs_dir = os.path.dirname(run.dir)
        expected = run.adapter_dir
    found = runlog.discover_adapters(runs_dir)
    assert found, "no adapters discovered"
    assert os.path.normpath(found[0]) == os.path.normpath(expected), found


def test_discover_adapters_does_not_list_the_latest_alias(workspace):
    """One adapter, one entry - even where `latest` is a real symlink.

    On a platform that permits directory symlinks (Linux, macOS, the container),
    runs/latest points at the newest run, so a glob over runs/*/final_adapter
    matches the same adapter twice. Windows falls back to a latest.txt pointer and
    never sees it, which is exactly why this reached CI green from a Windows box.
    """
    with runlog.Run(make_config(workspace), quiet=True) as run:
        os.makedirs(run.adapter_dir, exist_ok=True)
        with open(os.path.join(run.adapter_dir, "adapter_config.json"), "w") as handle:
            handle.write("{}")
        runs_dir = os.path.dirname(run.dir)
        run.point_latest_here()

    found = runlog.discover_adapters(runs_dir)
    assert len(found) == 1, f"the same adapter was listed more than once: {found}"
    assert "latest" not in found[0], \
        f"discovery returned the alias rather than the run: {found[0]}"

    # Only meaningful where the symlink was actually created; on Windows without
    # developer mode it is a text pointer and there was nothing to deduplicate.
    link = os.path.join(runs_dir, "latest")
    if os.path.islink(link):
        assert os.path.isdir(link), "the latest symlink is broken"


def test_discover_adapters_ignores_directories_without_a_config(workspace):
    with runlog.Run(make_config(workspace)) as run:
        os.makedirs(run.adapter_dir, exist_ok=True)   # created, but left empty
        runs_dir = os.path.dirname(run.dir)
    assert runlog.discover_adapters(runs_dir) == []


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"runlog: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
