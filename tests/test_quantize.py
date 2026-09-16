"""Checks for kd.quantize - packing the distilled student to 4-bit weights.

Plain asserts, no test framework, no model downloads, and - the point of this
file - no llm-compressor. It is a CUDA-only extra, so everything testable on an
ordinary machine is the plumbing around it: which rows GPTQ calibrates on, what
the stamp records, when a packed checkpoint may be reused, and that a missing
library produces a sentence rather than a traceback.

The reuse rule is the one worth pinning hardest. Handing back a checkpoint
packed at a different group size, and then reporting this run's settings beside
it, is a silent wrong answer - the worst kind this pipeline can produce.
"""

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import arena, paths, quantize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT = os.path.join(ROOT, "data", "enlibra-neuroscience", "sft-1to3hop.jsonl")

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


class FakeTokenizer:
    """Renders a turn list to text and encodes it one id per character."""

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False, "calibration text must not be pre-tokenised here"
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, truncation=False, max_length=None):
        ids = [ord(c) % 97 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]

        class Encoded:
            input_ids = ids
        return Encoded()


def packed_dir(**overrides):
    """A directory shaped like a finished packing, weights and stamp included."""
    where = tempfile.mkdtemp(prefix="kd-quant-test-")
    for name in ("config.json", "model.safetensors"):
        open(os.path.join(where, name), "w").close()
    stamp = {"source": "org/model", "scheme": "W4A16", "group_size": 128,
             "ignore": ["lm_head"], "samples": 128, "max_seq_length": 2048,
             "calibration": SFT, "backend": "llmcompressor",
             "format": "compressed-tensors", "vllm_ready": True}
    stamp.update(overrides)
    with io.open(os.path.join(where, quantize.QUANT_STAMP), "w",
                 encoding="utf-8") as handle:
        json.dump(stamp, handle)
    return where


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_comes_from_the_named_file():
    rows = quantize.calibration_rows(SFT, FakeTokenizer(), samples=16,
                                     max_length=128)
    assert len(rows) == 16, len(rows)
    assert all(len(r.input_ids) <= 128 for r in rows), "max_length not applied"


def test_calibration_is_deterministic():
    """Two runs that calibrate on different rows produce two checkpoints whose
    difference is not the thing anyone is trying to measure."""
    a = quantize.calibration_rows(SFT, FakeTokenizer(), samples=16, max_length=64)
    b = quantize.calibration_rows(SFT, FakeTokenizer(), samples=16, max_length=64)
    assert [r.input_ids for r in a] == [r.input_ids for r in b]


def test_calibration_samples_across_the_file_not_from_the_front():
    """A curriculum is ordered - by hop depth here - so the first N rows are not
    a sample of it, and GPTQ calibrated on them sees only the easy end."""
    few = quantize.calibration_rows(SFT, FakeTokenizer(), samples=8, max_length=64)
    whole = quantize.calibration_rows(SFT, FakeTokenizer(), samples=10 ** 6,
                                      max_length=64)
    assert len(whole) > len(few) * 10, "the fixture file is too small to tell"
    front = [r.input_ids for r in whole[:8]]
    assert [r.input_ids for r in few] != front, "took the first rows, not a sample"


def test_asking_for_more_rows_than_exist_returns_the_file():
    rows = quantize.calibration_rows(SFT, FakeTokenizer(), samples=10 ** 6,
                                     max_length=32)
    with io.open(SFT, encoding="utf-8") as handle:
        lines = sum(1 for line in handle if line.strip())
    assert len(rows) == lines, (len(rows), lines)


def test_an_empty_calibration_file_is_refused():
    empty = os.path.join(tempfile.mkdtemp(prefix="kd-quant-empty-"), "none.jsonl")
    open(empty, "w").close()
    try:
        quantize.calibration_rows(empty, FakeTokenizer())
    except ValueError as exc:
        assert "no rows" in str(exc), exc
    else:
        raise AssertionError("an empty calibration file was accepted")


# --------------------------------------------------------------------------- #
# The stamp, and when a packed checkpoint may be reused
# --------------------------------------------------------------------------- #
def test_a_directory_without_a_stamp_is_not_a_packed_checkpoint():
    where = tempfile.mkdtemp(prefix="kd-quant-bare-")
    for name in ("config.json", "model.safetensors"):
        open(os.path.join(where, name), "w").close()
    assert not quantize.is_quantized(where)
    assert quantize.is_quantized(packed_dir())


def test_a_stamp_without_weights_is_not_reused():
    """An interrupted write leaves the stamp and no safetensors."""
    where = packed_dir()
    os.remove(os.path.join(where, "model.safetensors"))
    assert not quantize.is_quantized(where)


def test_matching_settings_reuse_the_checkpoint():
    where = packed_dir()
    assert quantize.quantize("org/model", where, SFT) == where


def test_changed_settings_re_pack_rather_than_reuse():
    """Handing back a checkpoint packed at a different group size, and printing
    this run's settings beside it, is a silent wrong answer."""
    for changed in ({"group_size": 64}, {"scheme": "W8A8"},
                    {"samples": 256}, {"max_seq_length": 4096}):
        where = packed_dir()
        try:
            quantize.quantize("org/model", where, SFT, **changed)
        except RuntimeError as exc:
            # Got past the reuse check and tried to pack, which is the point.
            assert "llm-compressor" in str(exc), exc
        else:
            raise AssertionError(f"reused a checkpoint packed with {changed}")


def test_a_different_source_model_re_packs():
    where = packed_dir(source="org/a-completely-different-model")
    try:
        quantize.quantize("org/model", where, SFT)
    except RuntimeError as exc:
        assert "llm-compressor" in str(exc), exc
    else:
        raise AssertionError("reused a checkpoint packed from another model")


def test_a_missing_library_names_itself():
    try:
        quantize.quantize("org/model", tempfile.mkdtemp(), SFT)
    except RuntimeError as exc:
        assert "llm-compressor" in str(exc), exc
    else:
        raise AssertionError("a missing llm-compressor produced no error")


def test_summarise_reads_the_stamp_back():
    facts = quantize.summarise(packed_dir())
    assert facts["scheme"] == "W4A16", facts
    assert facts["group_size"] == 128, facts
    assert facts["ignore"] == ["lm_head"], facts
    assert facts["format"] == "compressed-tensors", facts


def test_lm_head_is_the_default_exclusion():
    """Its error lands straight on the logits, and it is 1.2 GiB on its own."""
    assert quantize.DEFAULT_IGNORE == ["lm_head"]


# --------------------------------------------------------------------------- #
# How the arena finds it
# --------------------------------------------------------------------------- #
class Quiet:
    def __init__(self):
        self.said = []

    def info(self, message):
        self.said.append(str(message))


def test_the_arena_takes_an_explicit_path():
    where = packed_dir()
    assert arena.resolve_quantized({}, None, explicit=where) == where


def test_a_named_path_with_nothing_in_it_is_reported_not_silently_skipped():
    """A silently skipped column and a typo in a path must not look the same."""
    log = Quiet()
    empty = tempfile.mkdtemp(prefix="kd-quant-missing-")
    assert arena.resolve_quantized({}, None, explicit=empty, log=log) is None
    assert any("holds no packed checkpoint" in m for m in log.said), log.said


def test_nothing_packed_means_no_packed_player():
    assert arena.resolve_quantized({}, None) is None
    assert arena.resolve_quantized({"quantization": {"enabled": True}},
                                   "/runs/x/final_adapter") is None


def test_the_cache_path_carries_the_scheme():
    """W4A16 and W8A8 are different artifacts and must not overwrite each other."""
    four = paths.quantized_cache({}, "/runs/x/final_adapter", "W4A16")
    eight = paths.quantized_cache({}, "/runs/x/final_adapter", "W8A8")
    assert four != eight, four
    assert four.endswith("-w4a16"), four


def test_both_students_are_players():
    """What quantization cost is a DIFFERENCE, and one column cannot carry one."""
    assert arena.DENSE in arena.PLAYERS
    assert arena.QUANTIZED in arena.PLAYERS
    assert arena.chosen_players(
        {"evaluation": {"players": ["distilled", "distilled-w4a16"]}}) == \
        ("distilled", "distilled-w4a16")


def test_the_quantization_cost_block_needs_both_students():
    """With only one of them scored there is nothing to subtract."""
    def payload(players):
        return {"questions": 2, "players": players,
                "agreement": {"distilled vs distilled-w4a16":
                              {"same": 1, "of": 2, "pct": 0.5}}}

    one = {"distilled": {"accuracy": 0.6, "answered": 2, "elo": 1500}}
    assert arena.quantization_cost(payload(one)) is None

    both = dict(one, **{"distilled-w4a16": {"accuracy": 0.5, "answered": 2,
                                            "elo": 1480}})
    cost = arena.quantization_cost(payload(both))
    assert abs(cost["accuracy_delta"] + 0.1) < 1e-9, cost
    assert cost["elo_delta"] == -20, cost
    assert cost["changed_answer"] == 1, cost
    assert cost["same_letter"] == 1, cost


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"quantize: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
