"""Checks for the arena's second generation engine.

Plain asserts, no test framework, no model downloads, and - the point of this
file - no vLLM. vLLM is a CUDA-only extra, so everything testable about it on an
ordinary machine is the plumbing AROUND it: what the subprocess is handed, what
happens when it cannot start, and that the two engines are asked to generate
from the same tokens.

The last of those is the one that matters. Two engines that render a chat
template differently would produce two different scores for one model, and
nothing in the output would say so.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import arena, vllm_runner  # noqa: E402

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


# --------------------------------------------------------------------------- #
# Sizing the engine to the job
# --------------------------------------------------------------------------- #
def test_the_context_is_the_longest_prompt_plus_the_ceiling():
    """Not the model's declared maximum: the KV cache vLLM preallocates is
    proportional to it, and a Qwen3 default of 40960 reserves memory for a
    context this run will never use."""
    prompts = [[0] * 100, [0] * 517, [0] * 40]
    assert vllm_runner.plan_length(prompts, 2048) == 2816   # 517+2048 -> next 256


def test_a_named_context_wins():
    assert vllm_runner.plan_length([[0] * 10], 2048, requested=4096) == 4096


def test_no_prompts_still_gives_a_usable_window():
    assert vllm_runner.plan_length([], 512) == 512


def test_the_window_is_never_shorter_than_the_longest_prompt():
    """A prompt that does not fit is refused by the engine at generate time,
    which loses the whole batch rather than one question."""
    for length in (1, 255, 256, 257, 8191):
        prompts = [[0] * length]
        assert vllm_runner.plan_length(prompts, 128) >= length + 128, length


# --------------------------------------------------------------------------- #
# What crosses into the subprocess
# --------------------------------------------------------------------------- #
def written_job(**kwargs):
    """Run complete() far enough to write its job file, and read that back."""
    workdir = tempfile.mkdtemp(prefix="kd-vllm-job-")
    try:
        vllm_runner.complete(workdir=workdir, **kwargs)
    except RuntimeError:
        pass        # the worker cannot start without vLLM; the job file is the point
    with open(os.path.join(workdir, "job.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_unknown_engine_options_do_not_reach_the_worker():
    """A keyword this pipeline cannot rely on is one that turns an arena into a
    TypeError on somebody else's pod, an hour in."""
    job = written_job(model="org/m", prompts=[[1, 2]], max_new_tokens=8,
                      options={"gpu_memory_utilization": 0.85,
                               "enforce_eager": True,
                               "not_a_real_vllm_keyword": 3})
    assert "not_a_real_vllm_keyword" not in job["options"], job["options"]
    assert job["options"]["gpu_memory_utilization"] == 0.85
    assert job["options"]["enforce_eager"] is True


def test_a_null_option_is_omitted_rather_than_sent_as_none():
    """`max_model_len: null` in the config means "work it out", not "pass None"."""
    job = written_job(model="org/m", prompts=[[1, 2]], max_new_tokens=8,
                      options={"max_model_len": None,
                               "tensor_parallel_size": None})
    assert "tensor_parallel_size" not in job["options"], job["options"]
    assert job["options"]["max_model_len"] == 256, job["options"]


def test_the_config_mapping_is_not_mutated():
    """evaluation.vllm is shared with the rest of the run; complete() computes a
    context window into its options and must not write that back into the config."""
    options = {"gpu_memory_utilization": 0.85, "max_model_len": None}
    written_job(model="org/m", prompts=[[1, 2]], max_new_tokens=8, options=options)
    assert options == {"gpu_memory_utilization": 0.85, "max_model_len": None}, options


def test_prompts_cross_as_token_ids():
    job = written_job(model="org/m", prompts=[[1, 2, 3], [4, 5]], max_new_tokens=8)
    assert job["prompts"] == [[1, 2, 3], [4, 5]], job["prompts"]


# --------------------------------------------------------------------------- #
# When the worker cannot run
# --------------------------------------------------------------------------- #
def test_a_worker_that_cannot_start_names_the_cause():
    """The alternative is an empty non-zero exit that reads the same for an
    out-of-memory kill and a typo in a model id."""
    try:
        vllm_runner.complete("org/m", [[1, 2, 3]], 8)
    except RuntimeError as exc:
        assert "No module named" in str(exc), exc
    else:
        raise AssertionError("a missing vLLM produced no error at all")


def test_a_short_answer_set_is_refused_rather_than_scored():
    """Scoring 90 answers as if they were 140 reports a number nobody can
    reproduce, and nothing downstream would notice the difference."""
    workdir = tempfile.mkdtemp(prefix="kd-vllm-short-")
    out = os.path.join(workdir, "completions.json")

    def fake_run(cmd, env=None):
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"completions": ["only one"]}, handle)

        class Finished:
            returncode = 0
        return Finished()

    real = vllm_runner.subprocess.run
    vllm_runner.subprocess.run = fake_run
    try:
        vllm_runner.complete("org/m", [[1], [2], [3]], 8, workdir=workdir)
    except RuntimeError as exc:
        assert "answered 1 of 3" in str(exc), exc
    else:
        raise AssertionError("a short answer set was accepted")
    finally:
        vllm_runner.subprocess.run = real


def test_stale_answers_from_a_previous_run_are_not_returned():
    """The worker writes the result file. A crashed worker writes nothing - and
    must not leave the last run's answers there to be read as this run's."""
    workdir = tempfile.mkdtemp(prefix="kd-vllm-stale-")
    with open(os.path.join(workdir, "completions.json"), "w", encoding="utf-8") as handle:
        json.dump({"completions": ["from an earlier run"]}, handle)
    try:
        vllm_runner.complete("org/m", [[1]], 8, workdir=workdir)
    except RuntimeError as exc:
        assert "from an earlier run" not in str(exc), exc
    else:
        raise AssertionError("stale completions were returned as this run's")


def test_windows_is_told_why_rather_than_left_to_a_traceback():
    reason = vllm_runner.unavailable_reason()
    if sys.platform == "win32":
        assert "Windows" in reason, reason
    if vllm_runner.available():
        assert reason is None, reason


# --------------------------------------------------------------------------- #
# The arena's side
# --------------------------------------------------------------------------- #
def test_an_unknown_engine_is_refused_by_name():
    for bad in ("vLLM-typo", "sglang", "tgi"):
        try:
            arena.play({}, {"device": "cpu", "dtype": None}, None, [], engine=bad)
        except ValueError as exc:
            assert "hf, vllm" in str(exc), exc
        else:
            raise AssertionError(f"{bad!r} was accepted as an engine")


class FakeTokenizer:
    """Records what it was asked to render, and encodes deterministically."""

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "the ids must come from the tokenizer call, not here"
        assert add_generation_prompt is True, "without it the model completes the question"
        self.rendered.append(messages)
        return "<|im_start|>" + messages[0]["content"] + "<|im_end|>"

    def __call__(self, text, **kwargs):
        class Encoded:
            input_ids = [ord(c) % 97 for c in text]
        return Encoded()


def test_prompts_are_rendered_through_the_chat_template_then_tokenised():
    """Exactly what arena._generate does before calling model.generate. The two
    engines see the same ids or they are measuring different things."""
    tokenizer = FakeTokenizer()
    questions = [{"prompt": "Which is a star?"}, {"prompt": "Which is a planet?"}]
    ids = arena._prompt_ids(tokenizer, questions)

    assert len(ids) == 2, ids
    assert [m[0]["content"] for m in tokenizer.rendered] == \
        ["Which is a star?", "Which is a planet?"]
    assert all(isinstance(i, int) for i in ids[0]), ids[0]
    # The same question must tokenise the same way every time, or a rerun is a
    # different measurement.
    assert arena._prompt_ids(FakeTokenizer(), questions)[0] == ids[0]


def test_score_all_reads_completions_the_way_the_one_at_a_time_loop_did():
    """One parser for both engines. A second copy would be a second parser, and
    the parser is where this file's subject has already been wrong once."""
    questions = [{"gold": "B", "prompt": "p"}] * 3
    said = ["The correct answer is:\n\nB.",
            "Option A is about Mars, so <Answer>\nB",
            "no letter anywhere in this one"]
    predictions, formats, unanswered = arena.score_all(questions, said, label="t")
    assert predictions == ["B", "B", None], predictions
    assert formats == ["labelled", "tagged", None], formats
    assert len(unanswered) == 1, unanswered
    assert unanswered[0]["question"] == 3
    assert unanswered[0]["completion_chars"] == len(said[2])


def test_the_unanswered_sample_stays_bounded():
    """Every completion from every player is megabytes nobody reads when the run
    went fine."""
    questions = [{"gold": "B", "prompt": "p"}] * 40
    _, _, unanswered = arena.score_all(questions, ["nothing here"] * 40, label="t")
    assert len(unanswered) == arena.UNANSWERED_KEPT, len(unanswered)


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"vllm engine: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
