"""Checks that an arena run cannot lose what it measured.

Generation is the expensive, unrepeatable part of the arena: three models over a
held-out set, hours of it on a laptop. Everything after it - the similarity
table, the terminal summary, the HTML report - is cheap and derived from what
generation produced.

So the invariant is simple and worth pinning: once generation finishes, NOTHING
that follows may end the process without the raw result on disk. Two bugs
motivated these checks. `kd arena` used to write nothing at all unless given
--json, and never wrote the transcript on that path. And similarity(), which
downloads an embedding model, ran BEFORE the first write - so a network failure
three hours in ended the run with an empty directory.

play() is stubbed throughout: no models are loaded, because the part being tested
is what happens after they would have been.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import arena  # noqa: E402
from kd import report as report_module  # noqa: E402

passed = []
failed = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)


SAID = {
    "base": ["It is probably alpha. <Answer>: A", "the answer is C"],
    "distilled": ["<Explanation> beta fits. <Answer>: B",
                  "<Explanation> three. <Answer>: C"],
    "teacher": ["<Explanation> clearly beta. <Answer>: B",
                "<Explanation> three it is. <Answer>: C"],
}
GOLD = ["B", "C"]


def fake_play(config, hardware, adapter, questions, max_new_tokens=512, log=None,
              players=("base", "distilled", "teacher"), show=0):
    """What play() returns, without loading anything."""
    count = len(questions)
    completions = {name: SAID[name][:count] for name in players}
    predictions, formats, unanswered = {}, {}, {}
    for name in players:
        picks, how, missed = [], [], []
        for text in completions[name]:
            letter, kind = arena.extract_answer_detail(text)
            picks.append(letter)
            how.append(kind)
            if letter is None:
                missed.append({"completion_head": text[:200]})
        predictions[name], formats[name], unanswered[name] = picks, how, missed
    return predictions, formats, unanswered, completions


def fake_load_questions(path):
    return [{"prompt": f"Q{i}?\nA) alpha\nB) beta\nC) gamma\nD) delta",
             "gold": gold, "hop": 2, "options": {}, "messages": []}
            for i, gold in enumerate(GOLD, start=1)], 0


def fake_similarity(completions, questions, model_name=None, log=None):
    names = sorted(completions)
    return {"model": "stub", "hops": ["2"],
            "pairs": {f"{a} vs {b}": {
                "overall": 0.8,
                "by_hop": {"2": {"n": len(questions), "cosine": 0.8}}}
                for i, a in enumerate(names) for b in names[i + 1:]}}


arena.play = fake_play
arena.load_questions = fake_load_questions
arena.similarity = fake_similarity

ROOT = tempfile.mkdtemp(prefix="kd-arena-output-")
_counter = [0]


def run(**overrides):
    """kd arena, into a directory of its own. Returns (exit code, directory)."""
    _counter[0] += 1
    directory = os.path.join(ROOT, f"run{_counter[0]}")
    args = argparse.Namespace(
        config=None, adapter=None, file="./stub.jsonl", limit=None, show=0,
        skip=[], max_new_tokens=None, device="cpu",
        json=os.path.join(directory, "arena.json"), report=None, no_save=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return arena.main(args), directory


def files(directory):
    return sorted(os.listdir(directory)) if os.path.isdir(directory) else []


def load(directory, name="arena.json"):
    with open(os.path.join(directory, name), encoding="utf-8") as handle:
        return json.load(handle)


class raises:
    """Replace an attribute with something that throws, for one block."""

    def __init__(self, owner, name, exc):
        self.owner, self.name, self.exc = owner, name, exc

    def __enter__(self):
        self.real = getattr(self.owner, self.name)

        def boom(*_a, **_k):
            raise self.exc

        setattr(self.owner, self.name, boom)

    def __exit__(self, *_):
        setattr(self.owner, self.name, self.real)


# --------------------------------------------------------------------------- #
# What a normal run leaves behind
# --------------------------------------------------------------------------- #
def test_saves_without_being_asked():
    """The regression: it used to print the tables and write nothing."""
    code, directory = run()
    assert code == 0, code
    assert files(directory) == ["arena-report.html", "arena-transcript.jsonl",
                                "arena.json"], files(directory)


def test_the_transcript_holds_every_word_of_every_player():
    _code, directory = run()
    path = os.path.join(directory, "arena-transcript.jsonl")
    with open(path, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == len(GOLD), len(lines)

    first = lines[0]
    assert first["gold"] == "B", first
    assert "Q1?" in first["asked"], first
    assert set(first["players"]) == {"base", "distilled", "teacher"}, first["players"]
    # The full text, not a truncated sample: this file is the only record of it.
    assert first["players"]["base"]["completion"] == SAID["base"][0], first["players"]
    assert first["players"]["teacher"]["correct"] is True, first["players"]
    assert first["players"]["base"]["correct"] is False, first["players"]


def test_the_report_carries_the_answer_key_and_the_similarity():
    _code, directory = run()
    with open(os.path.join(directory, "arena-report.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert "Produced a parseable answer" in html
    assert "alike are the explanations" in html


def test_json_flag_moves_all_three():
    directory = os.path.join(ROOT, "elsewhere")
    run(json=os.path.join(directory, "score.json"))
    assert files(directory) == ["score-report.html", "score-transcript.jsonl",
                                "score.json"], files(directory)


def test_empty_report_flag_skips_only_the_report():
    _code, directory = run(report="")
    assert files(directory) == ["arena-transcript.jsonl", "arena.json"], files(directory)


def test_no_save_writes_nothing():
    _code, directory = run(no_save=True)
    assert files(directory) == [], files(directory)


# --------------------------------------------------------------------------- #
# Nothing after generation may cost the run
# --------------------------------------------------------------------------- #
def test_a_failed_similarity_download_does_not_lose_the_run():
    """It downloads an embedding model. It used to run before the first write."""
    with raises(arena, "similarity", RuntimeError("connection reset by peer")):
        code, directory = run()
    assert code == 0, code
    assert "arena.json" in files(directory), files(directory)
    assert "arena-transcript.jsonl" in files(directory), files(directory)
    assert "arena-report.html" in files(directory), files(directory)

    payload = load(directory)
    assert payload["similarity"] is None, payload["similarity"]
    # The measurements themselves are untouched.
    assert payload["players"]["teacher"]["accuracy"] == 1.0, payload["players"]


def test_a_missing_similarity_omits_a_table_not_a_report():
    """sentence-transformers absent returns None rather than raising."""
    real = arena.similarity
    arena.similarity = lambda *a, **k: None
    try:
        code, directory = run()
    finally:
        arena.similarity = real
    assert code == 0, code
    assert "arena-report.html" in files(directory), files(directory)
    with open(os.path.join(directory, "arena-report.html"), encoding="utf-8") as fh:
        assert "alike are the explanations" not in fh.read()


def test_a_failed_transcript_still_leaves_the_numbers_and_the_report():
    with raises(arena, "write_transcript", OSError("no space left on device")):
        code, directory = run()
    assert code == 0, code
    assert "arena.json" in files(directory), files(directory)
    assert "arena-report.html" in files(directory), files(directory)


def test_a_failed_report_still_leaves_the_numbers_and_the_transcript():
    with raises(report_module, "write_report", ValueError("template exploded")):
        code, directory = run()
    assert code == 0, code
    assert "arena.json" in files(directory), files(directory)
    assert "arena-transcript.jsonl" in files(directory), files(directory)


def test_the_payload_carries_closeness_to_the_teacher():
    """The headline block, with the explanation column filled in after similarity."""
    code, directory = run()
    assert code == 0
    close = load(directory)["closeness"]
    assert close["reference"] == "teacher", close
    assert sorted(close["players"]) == ["base", "distilled"], close
    # distilled matched the teacher's letter on both questions; base on one.
    assert close["players"]["distilled"]["same_answer"] == 2, close
    assert close["players"]["base"]["same_answer"] == 1, close
    assert close["players"]["distilled"]["explanation_cosine"] == 0.8, close
    html = open(os.path.join(directory, "arena-report.html"), encoding="utf-8").read()
    assert ">100%<" in html and "gives the teacher's answer" in html


def test_the_payload_names_what_was_scored():
    _code, directory = run()
    payload = load(directory)
    assert payload["arena_file"] == "./stub.jsonl", payload["arena_file"]
    assert payload["questions"] == len(GOLD), payload["questions"]
    assert set(payload["players"]) == {"base", "distilled", "teacher"}


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"arena output: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
