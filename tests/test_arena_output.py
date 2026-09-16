"""Checks that an arena run cannot lose what it measured.

Generation is the expensive, unrepeatable part of the arena: four models over a
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
    # The stock model the teacher was fine-tuned from: right on the first, and
    # answering in prose rather than the trained format on the second.
    "teacher-base": ["I think beta. <Answer>: B", "The answer is D"],
    "teacher": ["<Explanation> clearly beta. <Answer>: B",
                "<Explanation> three it is. <Answer>: C"],
}
GOLD = ["B", "C"]

# Every player these runs actually produce. Not arena.PLAYERS: that also names
# `distilled-w4a16`, and nothing here quantises anything, so the packed student
# is correctly absent from the output.
PLAYED = tuple(p for p in arena.PLAYERS if p != arena.QUANTIZED)


def fake_play(config, hardware, adapter, questions, max_new_tokens=512, log=None,
              players=arena.PLAYERS, show=0, **engine_options):
    """What play() returns, without loading anything.

    **engine_options swallows `engine` and `vllm_options`: which generator ran is
    not a fact about the output layout these tests cover, and pinning the exact
    signature here means every new knob on play() breaks eleven unrelated tests.
    """
    count = len(questions)
    players = [p for p in players if p in SAID]
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


arena.play = fake_play
arena.load_questions = fake_load_questions

ROOT = tempfile.mkdtemp(prefix="kd-arena-output-")
_counter = [0]


def run(**overrides):
    """kd arena, into a directory of its own. Returns (exit code, directory)."""
    _counter[0] += 1
    directory = os.path.join(ROOT, f"run{_counter[0]}")
    args = argparse.Namespace(
        config=None, adapter=None, file="./stub.jsonl", limit=None, show=0,
        skip=[], max_new_tokens=None, device="cpu",
        # hf explicitly: the base config now defaults to vllm, which refuses to
        # run where vLLM is absent rather than silently falling back - and every
        # test in this file is about the OUTPUT, with play() stubbed out.
        engine="hf", quantized=None,
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
    assert set(first["players"]) == set(PLAYED), first["players"]
    # The full text, not a truncated sample: this file is the only record of it.
    assert first["players"]["base"]["completion"] == SAID["base"][0], first["players"]
    assert first["players"]["teacher"]["correct"] is True, first["players"]
    assert first["players"]["base"]["correct"] is False, first["players"]


def test_the_report_carries_the_answer_key_and_where_the_questions_went():
    _code, directory = run()
    with open(os.path.join(directory, "arena-report.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert "Produced a parseable answer" in html
    # Correct / wrong / never-committed, which is the table that separates a
    # model that is ignorant from one that is merely reticent.
    assert "Where the questions went" in html
    assert "Never committed to a letter" in html
    # The eval rows carry a hop depth, so the generalisation tables are there.
    assert "Accuracy by reasoning depth" in html
    assert "Answer rate by reasoning depth" in html


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
    """The headline block: how often each student gave the teacher's letter."""
    code, directory = run()
    assert code == 0
    close = load(directory)["closeness"]
    assert close["reference"] == "teacher", close
    assert sorted(close["players"]) == ["base", "distilled", "teacher-base"], close
    # distilled matched the teacher's letter on both questions; base on one.
    assert close["players"]["distilled"]["same_answer"] == 2, close
    assert close["players"]["base"]["same_answer"] == 1, close
    html = open(os.path.join(directory, "arena-report.html"), encoding="utf-8").read()
    assert ">100%<" in html and "gives the teacher's answer" in html


def test_the_payload_names_what_was_scored():
    _code, directory = run()
    payload = load(directory)
    assert payload["arena_file"] == "./stub.jsonl", payload["arena_file"]
    assert payload["questions"] == len(GOLD), payload["questions"]
    assert set(payload["players"]) == set(PLAYED)


def test_players_come_from_the_config_and_skip_wins():
    """evaluation.players picks the set; --skip removes from it; order is fixed."""
    config = {"evaluation": {"players": ["teacher", "base"]}}
    assert arena.chosen_players(config) == ("base", "teacher")
    assert arena.chosen_players(config, skip=["teacher"]) == ("base",)
    assert arena.chosen_players({}) == arena.PLAYERS
    try:
        arena.chosen_players({"evaluation": {"players": ["base", "techer"]}})
    except ValueError as exc:
        assert "techer" in str(exc), exc
    else:
        raise AssertionError("a misspelt player was accepted")
    code, directory = run(skip=["teacher-base"])
    assert code == 0
    assert set(load(directory)["players"]) == {"base", "distilled", "teacher"}


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"arena output: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
