"""Checks that a report renders from whatever measurements exist.

Two payload shapes reach kd.report, and both have to produce a page someone can
read:

  * the pipeline's - kd.evaluate's token-level numbers with the arena's answer
    key merged in;
  * the arena's alone - `kd arena --report`, which scores who was RIGHT without
    loading the machinery needed to say how closely the student tracks the
    teacher token by token.

The second used to raise a KeyError, which meant an arena run that had cost
hours of generation ended with nothing readable. These checks pin both shapes,
and pin the similarity table in particular: it is computed by the arena, and for
a while it reached the terminal and never the file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd.report import _sections, plain_summary, write_report  # noqa: E402

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


def player(answered, accuracy, when_answered, tagged, rating):
    return {"answered": answered, "accuracy": accuracy,
            "accuracy_when_answered": when_answered, "in_trained_format": tagged,
            "unanswered_examples": [], "elo": rating, "elo_spread": 15.0}


def hops(*values):
    return {"by_hop": {str(i + 1): {"n": 20, "cosine": v}
                       for i, v in enumerate(values)},
            "overall": sum(values) / len(values)}


ARENA = {
    "questions": 100,
    "random_baseline": 0.25,
    "players": {"base": player(90, 0.40, 0.444, 0, 1460.0),
                "distilled": player(98, 0.60, 0.612, 95, 1530.0),
                "teacher": player(100, 0.70, 0.700, 100, 1610.0)},
    "agreement": {"base vs teacher": {"same": 44, "of": 100, "pct": 0.44},
                  "distilled vs teacher": {"same": 71, "of": 100, "pct": 0.71}},
    "head_to_head": {},
    "similarity": {"model": "all-MiniLM-L6-v2", "hops": ["1", "2", "3"],
                   "pairs": {"base vs teacher": hops(0.71, 0.65, 0.60),
                             "distilled vs teacher": hops(0.88, 0.85, 0.82),
                             "base vs distilled": hops(0.72, 0.67, 0.63)}},
    "arena_file": "./data/eval.jsonl",
    "adapter": "/cache/final_adapter",
}

ARENA_ONLY = {"arena": ARENA, "student": "Qwen/Qwen2.5-1.5B-Instruct",
              "teacher": "s3://bucket/teacher/", "adapter": "/cache/final_adapter",
              "device": "mps", "dtype": "bfloat16"}

FULL = dict(ARENA_ONLY, **{
    "fidelity": {"top1_agreement_base_pct": 41.2,
                 "top1_agreement_distilled_pct": 68.9, "kl_base": 1.82,
                 "kl_distilled": 0.61, "agreement_lift_pts": 27.7},
    "capability": {"perplexity_base": 14.2, "perplexity_distilled": 8.1,
                   "perplexity_teacher": 6.4, "gap_recovered_pct": 62.0},
    "efficiency": {"student_params": 1.54e9, "teacher_params": 3.09e9,
                   "adapter_params": 36.7e6, "distilled_tok_per_s": 48.2,
                   "teacher_tok_per_s": 22.1},
    "closeness_to_teacher": {"prediction_agreement_base_pct": 41.2,
                             "prediction_agreement_distilled_pct": 68.9,
                             "perplexity_retention_base_pct": 45.1,
                             "perplexity_retention_distilled_pct": 79.0},
    "samples": 48, "completion_tokens": 6120})


def render(payload, suffix, tmp):
    target = os.path.join(tmp, f"report{suffix}")
    write_report(payload, target)
    with open(target, encoding="utf-8") as handle:
        return handle.read()


def scratch():
    import tempfile
    return tempfile.mkdtemp(prefix="kd-report-")


# --------------------------------------------------------------------------- #
def test_arena_alone_renders_html():
    """The regression: this used to be a KeyError on payload["fidelity"]."""
    html = render(ARENA_ONLY, ".html", scratch())
    assert len(html) > 2000, "suspiciously short page"
    assert "Produced a parseable answer" in html, "no answer key table"


def test_arena_alone_renders_markdown():
    text = render(ARENA_ONLY, ".md", scratch())
    assert "## Summary" in text
    assert "| Metric | Base student | Distilled | Teacher |" in text


def test_arena_alone_omits_the_sections_it_cannot_fill():
    """A smaller report, not one padded with dashes where numbers should be."""
    titles = [t for t, _rows, _headers in _sections(ARENA_ONLY)]
    for absent in ("Fidelity", "Capability", "Cost"):
        assert not any(absent in t for t in titles), f"{absent} rendered without data"


def test_similarity_reaches_the_report():
    """Computed by the arena; for a while it reached the terminal and nothing else."""
    for payload in (ARENA_ONLY, FULL):
        titles = [t for t, _rows, _headers in _sections(payload)]
        assert any("alike are the explanations" in t for t in titles), titles


def test_similarity_columns_are_pairs_not_models():
    """Three columns headed Base/Distilled/Teacher would misdescribe every cell."""
    section = [s for s in _sections(ARENA_ONLY) if "alike" in s[0]][0]
    assert section[2][1] == "Base vs teacher", section[2]
    assert section[2][2] == "Distilled vs teacher", section[2]


def test_similarity_has_a_row_per_hop_and_a_total():
    section = [s for s in _sections(ARENA_ONLY) if "alike" in s[0]][0]
    labels = [row[0] for row in section[1]]
    assert len(labels) == 4, labels          # three hops plus the total
    assert labels[-1] == "All questions", labels
    assert "1 hop" in labels[0], labels


def test_every_row_matches_its_headers():
    """Four cells per row, four headers per section, in both payload shapes."""
    for payload in (ARENA_ONLY, FULL):
        for title, rows, headers in _sections(payload):
            assert len(headers) == 4, f"{title}: {headers}"
            for row in rows:
                assert len(row) == 4, f"{title}: {row}"


def test_the_headline_falls_back_to_the_answer_key():
    """gap_recovered_pct comes from kd.evaluate; the arena can say it too."""
    html = render(ARENA_ONLY, ".html", scratch())
    assert "answer key, closed by training" in html, "no arena-derived headline"
    # (0.60 - 0.40) / (0.70 - 0.40) = 66.7%
    assert ">67%<" in html, html[html.find('class="big"'):][:120]


def test_the_headline_prefers_the_measured_one():
    """With both available, the token-level figure is the truer answer."""
    html = render(FULL, ".html", scratch())
    assert ">62%<" in html, html[html.find('class="big"'):][:120]
    assert "of the distance to the teacher, closed by training" in html


def test_summary_reads_differently_for_each_shape():
    arena_lines = " ".join(plain_summary(ARENA_ONLY))
    full_lines = " ".join(plain_summary(FULL))
    assert "60.0%" in arena_lines and "40.0%" in arena_lines, arena_lines
    assert "picks the same next word" in full_lines, full_lines


def test_a_silent_player_is_called_out_before_its_accuracy():
    """0% because it never produced a letter is a different fact from 0% wrong."""
    payload = {"arena": {"questions": 10, "random_baseline": 0.25, "players": {
        "base": player(0, 0.0, None, 0, 1500.0),
        "distilled": player(10, 0.5, 0.5, 10, 1520.0)}}}
    lines = " ".join(plain_summary(payload))
    assert "parseable answer on 0 of 10" in lines, lines


def test_no_teacher_and_no_similarity_still_renders():
    payload = {"arena": {"questions": 10, "random_baseline": 0.25, "players": {
        "base": player(8, 0.3, 0.375, 0, 1490.0),
        "distilled": player(10, 0.5, 0.5, 9, 1520.0)}}}
    html = render(payload, ".html", scratch())
    assert len(html) > 1500, "page collapsed without a teacher"
    assert "&mdash;<span>" in html, "no headline placeholder"


def test_nothing_is_double_escaped():
    html = render(ARENA_ONLY, ".html", scratch())
    assert "&amp;mdash;" not in html and "&amp;rarr;" not in html
    assert "<th></th>" not in html, "empty header cell"


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"report: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
