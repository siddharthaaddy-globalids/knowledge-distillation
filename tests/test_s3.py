"""Checks for URI handling, caching and bundle upload.

No bucket, no credentials, no boto3: `client()` is replaced with a small in-memory
fake that behaves the way the S3 API does in the ways this code depends on - flat
keys, prefix listing, and no such thing as a directory. Testing against a real
bucket would test the network; this tests the logic.
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import config as kdc      # noqa: E402
from kd import paths, runlog      # noqa: E402
from kd.remote import s3          # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

passed = []
failed = []


class FakeS3:
    """Just enough of the S3 client surface, backed by a dict of key -> bytes."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.uploaded = {}

    # --- listing ---------------------------------------------------------- #
    def get_paginator(self, _name):
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix=""):  # noqa: N803 - boto3's spelling
                contents = [{"Key": key, "Size": len(body)}
                            for key, body in sorted(client.objects.items())
                            if key.startswith(Prefix)]
                yield {"Contents": contents} if contents else {}

        return Paginator()

    # --- transfers -------------------------------------------------------- #
    def download_file(self, Bucket, Key, Filename):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(f"no such key: {Key}")
        os.makedirs(os.path.dirname(Filename), exist_ok=True)
        with open(Filename, "wb") as handle:
            handle.write(self.objects[Key])

    def upload_file(self, Filename, Bucket, Key):  # noqa: N803
        with open(Filename, "rb") as handle:
            self.uploaded[Key] = handle.read()


def check(name, fn):
    workspace = tempfile.mkdtemp(prefix="kd-test-")
    original = s3.client
    try:
        fn(workspace)
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
    finally:
        s3.client = original
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(workspace, enabled=True, **s3_settings):
    config = kdc.load_config(os.path.join(CONFIGS, "smoke.yaml"), use_env=False)
    config["project"]["runs_dir"] = os.path.join(workspace, "runs")
    config["s3"].update({"enabled": enabled, "bucket": "test-bucket", "prefix": "kd",
                         "cache_dir": os.path.join(workspace, "cache")})
    config["s3"].update(s3_settings)
    return config


def install_fake(objects=None):
    fake = FakeS3(objects)
    s3.client = lambda _config: fake
    return fake


# --------------------------------------------------------------------------- #
# URIs
# --------------------------------------------------------------------------- #
def test_uri_detection(workspace):
    assert paths.is_remote("s3://bucket/key")
    assert not paths.is_remote("Qwen/Qwen3.5-2B")
    assert not paths.is_remote("./local/path")
    assert not paths.is_remote(None)


def test_uri_splitting(workspace):
    assert paths.split_uri("s3://bucket/a/b/c") == ("bucket", "a/b/c")
    assert paths.split_uri("s3://bucket/a/b/") == ("bucket", "a/b")
    assert paths.split_uri("s3://bucket") == ("bucket", "")
    for bad in ("s3://", "not-a-uri"):
        try:
            paths.split_uri(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not have parsed")


def test_cache_path_is_deterministic(workspace):
    config = make_config(workspace)
    first = paths.cache_path(config, "s3://bucket/models/teacher")
    second = paths.cache_path(config, "s3://bucket/models/teacher")
    assert first == second, "two runs would not share a cache"
    assert first != paths.cache_path(config, "s3://bucket/models/student")


# --------------------------------------------------------------------------- #
# Fetching inputs
# --------------------------------------------------------------------------- #
def test_download_a_prefix(workspace):
    fake = install_fake({
        "models/teacher/config.json": b"{}",
        "models/teacher/model.safetensors": b"weights",
        "models/teacher/nested/extra.txt": b"x",
        "models/other/ignored.txt": b"no",
    })
    dest = os.path.join(workspace, "out")
    result = s3.download(make_config(workspace), "s3://test-bucket/models/teacher", dest)
    assert result["objects"] == 3, result
    assert os.path.isfile(os.path.join(dest, "config.json"))
    assert os.path.isfile(os.path.join(dest, "nested", "extra.txt"))
    assert not os.path.exists(os.path.join(dest, "ignored.txt")), "fetched a sibling prefix"
    assert paths.is_cached(dest), "no completion marker written"
    assert fake is not None


def test_missing_uri_says_what_it_checked(workspace):
    install_fake({"models/teacher/config.json": b"{}"})
    try:
        s3.download(make_config(workspace), "s3://test-bucket/models/absent",
                    os.path.join(workspace, "out"))
    except RuntimeError as exc:
        assert "nothing found at" in str(exc), exc
    else:
        raise AssertionError("a missing prefix should have raised")


def test_config_inputs_are_rewritten_to_local_paths(workspace):
    install_fake({
        "models/teacher/config.json": b"{}",
        "data/finance/train.json": b"[]",
    })
    config = make_config(workspace)
    config["models"]["teacher"] = "s3://test-bucket/models/teacher"
    config["dataset"]["source"] = "s3://test-bucket/data/finance"

    resolved = paths.resolve_inputs(config)
    assert set(resolved) == {"models.teacher", "dataset.source"}, resolved
    for key in ("models.teacher", "dataset.source"):
        section, _, leaf = key.partition(".")
        value = config[section][leaf]
        assert not paths.is_remote(value), f"{key} still points at s3"
        assert os.path.isdir(value), f"{key} -> {value} does not exist"
    # The hub id alongside them must be left exactly as it was.
    assert config["models"]["student"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_second_run_uses_the_cache(workspace):
    fake = install_fake({"models/teacher/config.json": b"{}"})
    config = make_config(workspace)
    config["models"]["teacher"] = "s3://test-bucket/models/teacher"
    paths.resolve_inputs(config)

    # A fake with nothing in it: a second resolve that still succeeds can only
    # have come from the cache.
    install_fake({})
    again = make_config(workspace)
    again["models"]["teacher"] = "s3://test-bucket/models/teacher"
    paths.resolve_inputs(again)
    assert os.path.isdir(again["models"]["teacher"])
    assert fake is not None


def test_interrupted_download_is_not_treated_as_cached(workspace):
    config = make_config(workspace)
    partial = paths.cache_path(config, "s3://test-bucket/models/teacher")
    os.makedirs(partial, exist_ok=True)
    with open(os.path.join(partial, "config.json"), "w") as handle:
        handle.write("{}")
    assert not paths.is_cached(partial), "a directory with no marker looks complete"


def test_s3_uri_without_s3_enabled_is_refused(workspace):
    config = make_config(workspace, enabled=False)
    config["models"]["teacher"] = "s3://test-bucket/models/teacher"
    try:
        paths.resolve_inputs(config)
    except RuntimeError as exc:
        assert "s3.enabled is false" in str(exc), exc
    else:
        raise AssertionError("an s3 URI with s3 disabled should have been refused")


def test_no_remote_inputs_does_nothing(workspace):
    install_fake({})
    assert paths.resolve_inputs(make_config(workspace)) == {}


# --------------------------------------------------------------------------- #
# Uploading the bundle
# --------------------------------------------------------------------------- #
def seed_bundle(workspace, config):
    """A run directory with one file of every kind an upload group covers."""
    with runlog.Run(config, quiet=True) as run:
        os.makedirs(run.adapter_dir, exist_ok=True)
        for path, body in [
            (os.path.join(run.adapter_dir, "adapter_config.json"), "{}"),
            (os.path.join(run.adapter_dir, "adapter_model.safetensors"), "weights"),
            (run.path("report.html"), "<html></html>"),
            (run.path("evaluation.json"), "{}"),
            (os.path.join(run.checkpoint_dir, "checkpoint-2", "optimizer.pt"), "state"),
        ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(body)
        run.write_metrics({"training": {"steps": 2}})
        return run.dir, run.run_id


def test_upload_respects_the_group_list(workspace):
    config = make_config(workspace, upload=["adapter", "logs"])
    run_dir, run_id = seed_bundle(workspace, config)
    fake = install_fake()

    summary = s3.upload_bundle(config, run_dir, run_id)
    keys = {key.split(f"runs/{run_id}/")[-1] for key in fake.uploaded}

    assert "final_adapter/adapter_model.safetensors" in keys, keys
    assert "run.log" in keys and "events.jsonl" in keys, keys
    # Not requested, so not sent - a report and 100 MB of optimizer state.
    assert "report.html" not in keys, keys
    assert not any(k.startswith("checkpoints/") for k in keys), keys
    assert summary["uri"].endswith(f"kd/runs/{run_id}"), summary["uri"]


def test_manifest_and_config_always_go(workspace):
    config = make_config(workspace, upload=[])
    run_dir, run_id = seed_bundle(workspace, config)
    fake = install_fake()
    s3.upload_bundle(config, run_dir, run_id)
    keys = {key.split(f"runs/{run_id}/")[-1] for key in fake.uploaded}
    assert "manifest.json" in keys and "config.resolved.yaml" in keys, keys


def test_checkpoints_upload_when_asked_for(workspace):
    config = make_config(workspace)
    run_dir, run_id = seed_bundle(workspace, config)
    fake = install_fake()
    s3.upload_bundle(config, run_dir, run_id, groups=["checkpoints"])
    keys = {key.split(f"runs/{run_id}/")[-1] for key in fake.uploaded}
    assert "checkpoints/checkpoint-2/optimizer.pt" in keys, keys


def test_unknown_upload_group_is_rejected(workspace):
    config = make_config(workspace, upload=["adaptor"])
    run_dir, run_id = seed_bundle(workspace, config)
    install_fake()
    try:
        s3.upload_bundle(config, run_dir, run_id)
    except RuntimeError as exc:
        assert "adaptor" in str(exc) and "valid:" in str(exc), exc
    else:
        raise AssertionError("a misspelled upload group should have been rejected")


def test_missing_bucket_is_reported(workspace):
    config = make_config(workspace)
    config["s3"]["bucket"] = None
    run_dir, run_id = seed_bundle(workspace, config)
    install_fake()
    try:
        s3.upload_bundle(config, run_dir, run_id)
    except RuntimeError as exc:
        assert "s3.bucket" in str(exc), exc
    else:
        raise AssertionError("a missing bucket should have been reported")


def test_bundle_prefix_layout(workspace):
    config = make_config(workspace)
    assert s3.run_prefix(config, "RUNID") == "kd/runs/RUNID"
    config["s3"]["prefix"] = ""
    assert s3.run_prefix(config, "RUNID") == "runs/RUNID"


def test_uploaded_manifest_is_current(workspace):
    # The bundle in the bucket must describe the run that just ended, not the run
    # as it stood several stages earlier.
    config = make_config(workspace)
    run_dir, run_id = seed_bundle(workspace, config)
    fake = install_fake()
    s3.upload_bundle(config, run_dir, run_id)
    key = next(k for k in fake.uploaded if k.endswith("manifest.json"))
    manifest = json.loads(fake.uploaded[key])
    assert manifest["run_id"] == run_id, manifest["run_id"]


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"s3: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
