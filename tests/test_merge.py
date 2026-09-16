"""Checks for kd.merge - the one place an adapter is added into its base.

Plain asserts, no test framework, no model downloads: this runs as
`uv run python tests/test_merge.py` with nothing extra installed.

Nothing here loads weights. What is worth testing is the decision-making around
the merge rather than the arithmetic of it: which of the four routes answers the
vocabulary question, that a route asking to WIDEN is refused, and that a cached
merge is never handed back for a different adapter. The merge itself is
transformers' and PEFT's and is exercised by actually running an arena.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import merge  # noqa: E402

passed = []
failed = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a crash is a failure like any other
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)


def adapter_dir(meta=None, base="some/base"):
    """A directory shaped like a PEFT adapter, with no weights in it."""
    where = tempfile.mkdtemp(prefix="kd-merge-test-")
    with open(os.path.join(where, "adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"base_model_name_or_path": base}, handle)
    if meta is not None:
        with open(os.path.join(where, "kd-meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
    return where


class FakeConfig:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


class FakeModel:
    """Just enough of a transformers model for trim() to act on."""

    def __init__(self, vocab_size):
        self.config = FakeConfig(vocab_size)
        self.resized_to = None

    def resize_token_embeddings(self, target):
        self.resized_to = target


# --------------------------------------------------------------------------- #
# Which route answers the vocabulary question
# --------------------------------------------------------------------------- #
def test_kd_meta_is_the_first_route():
    """What kd.train recorded wins, because it needs nothing else present."""
    width, source = merge.vocab_width(adapter_dir(meta={"vocab_size": 151665}))
    assert width == 151665, width
    assert "kd-meta" in source, source


def test_an_explicit_width_beats_every_route():
    """--vocab-size is a person saying so, and outranks anything inferred."""
    where = adapter_dir(meta={"vocab_size": 151665})
    width, source = merge.vocab_width(where, explicit=99)
    assert width == 99, width
    assert "you named" in source, source


def test_no_route_answers_is_the_ordinary_case():
    """An adapter against an untrimmed base needs no trim, and says so quietly."""
    assert merge.vocab_width(adapter_dir()) == (None, None)


def test_a_bundle_beside_the_adapter_is_read_when_nothing_else_answers():
    """config.resolved.yaml one level up names the teacher the student matched."""
    run = tempfile.mkdtemp(prefix="kd-merge-bundle-")
    inner = os.path.join(run, "final_adapter")
    os.makedirs(inner)
    with open(os.path.join(inner, "adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"base_model_name_or_path": "some/base"}, handle)
    # A teacher that is neither a directory nor a hub id is ignored rather than
    # fetched: the route has to be safe to try on a machine holding no weights.
    with open(os.path.join(run, "config.resolved.yaml"), "w", encoding="utf-8") as handle:
        handle.write("models:\n  teacher: not-a-path\n")
    assert merge.vocab_width(inner) == (None, None)


# --------------------------------------------------------------------------- #
# Trimming
# --------------------------------------------------------------------------- #
def test_trim_narrows():
    model = FakeModel(151936)
    assert merge.trim(model, 151665, log=Quiet()) is True
    assert model.resized_to == 151665, model.resized_to
    assert model.config.vocab_size == 151665


def test_trim_refuses_to_widen():
    """Widening appends rows initialised from nothing - weights the adapter
    never saw. A route saying so has given a wrong answer, not an instruction."""
    model = FakeModel(151665)
    assert merge.trim(model, 151936, source="a bad route", log=Quiet()) is False
    assert model.resized_to is None, "widened when it should have refused"
    assert model.config.vocab_size == 151665


def test_trim_does_nothing_without_a_target():
    model = FakeModel(151936)
    assert merge.trim(model, None, log=Quiet()) is False
    assert model.resized_to is None


class Quiet:
    def info(self, message):
        pass


# --------------------------------------------------------------------------- #
# Materialising a dense checkpoint
# --------------------------------------------------------------------------- #
def test_no_adapter_means_no_work_and_no_write():
    """A hub id is already the thing vLLM wants; copying it would be pointless."""
    where = os.path.join(tempfile.mkdtemp(prefix="kd-merge-noop-"), "never")
    assert merge.materialise(where, model_id="org/model") == "org/model"
    assert not os.path.exists(where), "wrote a directory it had no need to"


def test_materialise_needs_something_to_work_from():
    try:
        merge.materialise(tempfile.mkdtemp())
    except ValueError as exc:
        assert "model_id" in str(exc), exc
    else:
        raise AssertionError("accepted a call naming neither a model nor an adapter")


def dense_dir(base="org/base", adapter="/some/adapter"):
    """A directory shaped like a finished merge, weights and stamp included."""
    where = tempfile.mkdtemp(prefix="kd-merge-dense-")
    for name in ("config.json", "model.safetensors"):
        open(os.path.join(where, name), "w").close()
    with open(os.path.join(where, merge.MERGE_STAMP), "w", encoding="utf-8") as handle:
        json.dump({"base": base, "adapter": adapter, "vocab_size": 151665}, handle)
    return where


def test_a_stamp_is_reused_only_for_the_same_merge():
    """The failure this prevents is scoring one adapter and reporting another."""
    where = dense_dir()
    assert merge._stamp_matches(where, "org/base", "/some/adapter")
    assert not merge._stamp_matches(where, "org/base", "/a/different/adapter")
    assert not merge._stamp_matches(where, "org/other-base", "/some/adapter")


def test_a_merge_with_no_weights_is_not_reused():
    """An interrupted write leaves the stamp and no safetensors. Redo it."""
    where = dense_dir()
    os.remove(os.path.join(where, "model.safetensors"))
    assert not merge._stamp_matches(where, "org/base", "/some/adapter")


def test_a_truncated_stamp_is_not_reused():
    where = dense_dir()
    with open(os.path.join(where, merge.MERGE_STAMP), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert not merge._stamp_matches(where, "org/base", "/some/adapter")


def test_is_dense_checkpoint_rejects_an_adapter_directory():
    """The distinction the whole module turns on: a checkpoint, not deltas."""
    assert merge.is_dense_checkpoint(dense_dir())
    assert not merge.is_dense_checkpoint(adapter_dir())
    assert not merge.is_dense_checkpoint("/does/not/exist")
    assert not merge.is_dense_checkpoint(None)


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"merge: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
