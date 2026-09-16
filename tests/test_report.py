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
and pin in particular the tables that exist only when their measurement ran:
the reasoning-depth breakdown, and the quantization comparison - which appears
only when both students were scored.
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


def player(answered, accuracy, when_answered, tagged, rating, questions=100):
    correct = round(accuracy * questions)
    return {"answered": answered, "accuracy": accuracy,
            "accuracy_when_answered": when_answered, "in_trained_format": tagged,
            "unanswered_examples": [], "elo": rating, "elo_spread": 15.0,
            "correct": correct, "wrong": answered - correct,
            "unanswered": questions - answered,
            "by_hop": by_hop(answered, correct, questions)}


def by_hop(answered, correct, questions, depths=3):
    """The same totals split evenly across `depths` reasoning depths."""
    n = questions // depths
    return {str(d + 1): {"n": n, "answered": answered // depths,
                         "correct": correct // depths,
                         "accuracy": (correct // depths) / n,
                         "answer_rate": (answered // depths) / n,
                         "accuracy_when_answered": (
                             (correct // depths) / (answered // depths)
                             if answered else None)}
            for d in range(depths)}


ARENA = {
    "questions": 100,
    "random_baseline": 0.25,
    "players": {"base": player(90, 0.40, 0.444, 0, 1460.0),
                "distilled": player(98, 0.60, 0.612, 95, 1530.0),
                "teacher": player(100, 0.70, 0.700, 100, 1610.0)},
    "agreement": {"base vs teacher": {"same": 44, "of": 100, "pct": 0.44},
                  "distilled vs teacher": {"same": 71, "of": 100, "pct": 0.71}},
    "head_to_head": {},
    "arena_file": "./data/eval.jsonl",
    "adapter": "/cache/final_adapter",
}

ARENA_ONLY = {"arena": ARENA, "student": "Qwen/Qwen2.5-1.5B-Instruct",
              "teacher": "s3://bucket/teacher/", "adapter": "/cache/final_adapter",
              "adapter_locations": {
                  "local": "/cache/final_adapter",
                  "s3": "s3://bucket/kd/runs/run-1/final_adapter",
                  "s3_status": "the copy it was fetched from", "note": None},
              "profile": "configs/enlibra/enlibraQ25-flow.yaml",
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
                             "prediction_agreement_distilled_pct": 68.9},
    "samples": 48, "completion_tokens": 6120})

# A run that also packed the student. The packed column and the "what W4A16
# cost" table appear only here - see test_quantization_appears_only_when_measured.
PACKED_ARENA = dict(ARENA, players=dict(
    ARENA["players"], **{"distilled-w4a16": player(97, 0.58, 0.598, 93, 1522.0)}))
PACKED_ARENA["quantization"] = {
    "dense": "distilled", "packed": "distilled-w4a16", "questions": 100,
    "accuracy_dense": 0.60, "accuracy_packed": 0.58, "accuracy_delta": -0.02,
    "answered_dense": 98, "answered_packed": 97, "answered_delta": -1,
    "elo_dense": 1530.0, "elo_packed": 1522.0, "elo_delta": -8.0,
    "same_letter": 94, "same_letter_pct": 0.94, "changed_answer": 6,
}
PACKED = dict(FULL, arena=PACKED_ARENA, quantization={
    "path": "/cache/quantized", "scheme": "W4A16", "group_size": 128,
    "ignore": ["lm_head"], "format": "compressed-tensors",
    "calibration_samples": 128, "tokens": 6120, "scored_tokens": 6120,
    "nll": 1.2253, "dense_nll": 1.2197, "nll_change": 0.0056,
    "perplexity": 3.4053, "dense_perplexity": 3.3863,
    "perplexity_change_pct": 0.56,
    "tok_per_s": 31.0, "dense_tok_per_s": 22.64, "throughput_change_pct": 36.9,
    "bytes": 6120000000, "dense_bytes": 16400000000, "compression": 2.68,
})


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


def test_the_depth_breakdown_is_three_tables_not_one():
    """Accuracy, answer rate and accuracy-when-answered fail apart: a model can
    hold the third flat while the second collapses, which is a budget problem
    and not a knowledge problem. One table cannot show that."""
    for payload in (ARENA_ONLY, FULL):
        titles = [t for t, _rows, _headers in _sections(payload)]
        assert any("Accuracy by reasoning depth" in t for t in titles), titles
        assert any("Answer rate by reasoning depth" in t for t in titles), titles
        assert any("Accuracy when answered, by reasoning depth" in t
                   for t in titles), titles


def test_the_depth_tables_have_a_row_per_hop():
    section = [s for s in _sections(ARENA_ONLY)
               if s[0].startswith("Accuracy by reasoning depth")][0]
    labels = [row[0] for row in section[1]]
    assert len(labels) == 3, labels
    assert labels[0].startswith("hop 1"), labels
    assert "33 questions" in labels[0], labels


def test_where_the_questions_went_splits_wrong_from_silent():
    """Accuracy counts silence as error, so a model that reasons past the token
    ceiling scores the same as one that answers confidently and wrongly."""
    section = [s for s in _sections(ARENA_ONLY)
               if s[0] == "Where the questions went"][0]
    labels = [row[0] for row in section[1]]
    assert labels[:3] == ["Correct", "Answered, but wrong",
                          "Never committed to a letter"], labels


def test_quantization_appears_only_when_measured():
    """Most runs never pack anything, and a W4A16 column full of dashes reads as
    a measurement that failed rather than one nobody asked for."""
    for payload in (ARENA_ONLY, FULL):
        titles = [t for t, _rows, _headers in _sections(payload)]
        assert not any("W4A16 cost" in t for t in titles), titles

    title, rows, headers = [s for s in _sections(PACKED) if "W4A16 cost" in s[0]][0]
    assert headers == ("Measure", "Dense (bf16)", "Packed (W4A16)", "Change"), headers
    assert "group size 128" in title, title
    assert "lm_head" in title, title
    labels = [row[0] for row in rows]
    # Both halves: the token-level pair AND the answer key. A perplexity that
    # barely moves next to an accuracy that drops three points is a real and
    # common outcome, and one table alone would miss it.
    assert "Perplexity" in labels, labels
    assert "Accuracy on the answer key" in labels, labels
    assert "Gave a different letter" in labels, labels


def test_the_packed_column_appears_only_when_it_was_played():
    assert "Distilled W4A16" not in _sections(FULL)[0][2]
    assert "Distilled W4A16" in _sections(PACKED)[0][2]


def test_every_row_matches_its_headers():
    """Every row carries exactly one cell per header, in every payload shape."""
    for payload in (ARENA_ONLY, FULL, PACKED):
        for title, rows, headers in _sections(payload):
            for row in rows:
                assert len(row) == len(headers), (
                    f"{title}: {len(row)} cells against {len(headers)} "
                    f"headers: {row}")


def test_the_headline_is_closeness_to_the_teacher():
    """The main score: how often the distilled student gave the teacher's answer."""
    for payload in (ARENA_ONLY, FULL):
        html = render(payload, ".html", scratch())
        assert ">71%<" in html, html[html.find('class="big"'):][:160]
        assert "gives the teacher's answer" in html, "wrong caption"
        assert "closed by training" not in html[html.find('class="big"'):][:200]


def test_the_headline_falls_back_to_the_token_level_figure():
    """No arena: kd.evaluate's next-token agreement is the same question."""
    payload = {k: v for k, v in FULL.items() if k != "arena"}
    html = render(payload, ".html", scratch())
    assert ">69%<" in html, html[html.find('class="big"'):][:160]
    assert "predicts the teacher's next token" in html


def test_closeness_is_derived_when_the_arena_did_not_write_it():
    """An arena.json from before the block existed gets the same headline."""
    from kd.report import _closeness
    close = _closeness(ARENA)
    assert close["distilled"]["same_answer_pct"] == 0.71, close
    assert close["distilled"]["same_answer"] == 71, close
    # ...and the arena's own block wins when it is there.
    written = dict(ARENA, closeness={"reference": "teacher", "players": {
        "distilled": {"same_answer": 1, "of": 2, "same_answer_pct": 0.5}}})
    assert _closeness(written)["distilled"]["same_answer_pct"] == 0.5


def test_closeness_section_comes_first_and_the_agreement_row_moved_into_it():
    sections = _sections(ARENA_ONLY)
    assert sections[0][0].startswith("How close is it to the teacher?"), sections[0][0]
    labels = [row[0] for row in sections[0][1]]
    assert labels[0] == "Gave the teacher's answer", labels
    assert any("share of the teacher" in l for l in labels), labels
    key = [s for s in sections if s[0].startswith("The answer key")][0]
    assert not any("teacher's letter" in row[0] for row in key[1]), key[1]


def test_summary_leads_with_closeness():
    for payload in (ARENA_ONLY, FULL):
        first = plain_summary(payload)[0]
        assert "gave the teacher's answer 71.0%" in first, first
        assert "up from 44.0%" in first, first


def test_the_adapter_section_names_both_copies_and_the_eval_command():
    html = render(ARENA_ONLY, ".html", scratch())
    assert "The adapter" in html
    assert "s3://bucket/kd/runs/run-1/final_adapter" in html
    assert "/cache/final_adapter" in html
    assert ("./run.sh --config configs/enlibra/enlibraQ25-flow.yaml eval "
            "--adapter s3://bucket/kd/runs/run-1/final_adapter") in html, "no re-run command"
    assert "arena --adapter s3://bucket/kd/runs/run-1/final_adapter" in html
    md = render(ARENA_ONLY, ".md", scratch())
    assert "## The adapter" in md and "eval --adapter s3://" in md


def test_the_adapter_section_says_why_there_is_no_s3_copy():
    payload = dict(ARENA_ONLY, adapter_locations={
        "local": "runs/r1/final_adapter", "s3": None, "s3_status": None,
        "note": "s3.enabled is false, so this run does not upload it"})
    html = render(payload, ".html", scratch())
    assert "not there - s3.enabled is false" in html, "the reason is missing"
    # With no S3 copy, the re-run command names the local path.
    assert "eval --adapter runs/r1/final_adapter" in html


TRAINING = {"gkd": {"beta": 0.5, "lmbda": 0.0, "temperature": 0.7,
                    "max_new_tokens": 8192, "seq_kd": False},
            "training": {"max_steps": 300, "batch_size": 1,
                         "gradient_accumulation_steps": 4, "learning_rate": 0.0003,
                         "lr_scheduler_type": "cosine", "warmup": 0.05},
            "lora": {"r": 32, "alpha": 64, "dropout": 0.05,
                     "target_modules": ["q_proj", "v_proj"]}}


def test_how_it_was_trained_names_every_knob_with_its_value():
    payload = dict(ARENA_ONLY, training=TRAINING)
    html = render(payload, ".html", scratch())
    assert "How it was trained" in html
    for knob in ("beta", "lmbda", "temperature", "max_new_tokens", "seq_kd"):
        assert f"<b>{knob}</b>" in html, f"{knob} missing"
    assert "<em>0.5</em>" in html and "<em>0.0</em>" in html and "<em>false</em>" in html
    assert "KL(P || M)" in html, "no loss formula"
    assert "1 x 4 = 4" in html and "r=32, alpha=64" in html
    md = render(payload, ".md", scratch())
    assert "## How it was trained" in md and "| `beta` | **0.5** |" in md


def test_the_explanation_is_about_this_runs_values():
    """lmbda 0 is plain distillation and makes two knobs inert; the page says so."""
    from kd.report import _training_knobs, _training_summary

    off = _training_summary(TRAINING["gkd"])
    assert "nothing the student wrote itself was used" in off, off
    assert "balanced penalty" in off, off
    knobs = {name: now for name, _v, _what, now in _training_knobs(TRAINING["gkd"])}
    assert knobs["lmbda"].startswith("Fully off-policy"), knobs["lmbda"]
    assert "Inert" in knobs["temperature"] and "Inert" in knobs["max_new_tokens"]

    on = dict(TRAINING["gkd"], lmbda=0.5, beta=0.9)
    summary = _training_summary(on)
    assert "50% of batches" in summary and "reverse KL" in summary, summary
    knobs = {name: now for name, _v, _what, now in _training_knobs(on)}
    assert "Inert" not in knobs["temperature"], knobs["temperature"]
    assert knobs["beta"].startswith("Mostly reverse KL"), knobs["beta"]


def test_the_loss_ceiling_follows_beta():
    from kd.report import loss_note

    assert "0.693" in loss_note(0.5)
    assert "0.325" in loss_note(0.9)
    assert "unbounded" in loss_note(1.0) and "unbounded" in loss_note(0)


def test_training_section_is_read_from_the_config_and_can_be_turned_off():
    from kd.report import training_settings

    config = {"gkd": TRAINING["gkd"], "training": TRAINING["training"],
              "lora": TRAINING["lora"], "evaluation": {}}
    block = training_settings(config)
    assert block["gkd"]["beta"] == 0.5 and block["lora"]["r"] == 32, block
    assert training_settings(dict(config, evaluation={"report_training": False})) is None
    html = render(dict(ARENA_ONLY, training=None), ".html", scratch())
    assert "How it was trained" not in html


def test_the_adapter_section_survives_an_older_payload():
    """Only `adapter`, no locations block: still a section, never a crash."""
    payload = {k: v for k, v in ARENA_ONLY.items()
               if k not in ("adapter_locations", "profile")}
    html = render(payload, ".html", scratch())
    assert "The adapter" in html and "/cache/final_adapter" in html
    assert "--config &lt;profile&gt;.yaml" in html


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


def test_a_limited_run_says_so_before_any_number():
    """The arena saves by default, so a 5-question file must not read as a score."""
    limited = dict(ARENA_ONLY, arena=dict(ARENA, questions=5, limited_to=5,
                                          available=137))
    first = plain_summary(limited)[0]
    assert first.startswith("NOT THE SCORE"), first
    assert "5 of 137" in first, first

    html = render(limited, ".html", scratch())
    assert "NOT THE SCORE" in html, "the warning never reached the page"
    assert "a SUBSET of 137, not the score" in html, "table heading does not say it"


def test_an_unlimited_run_carries_no_warning():
    for payload in (ARENA_ONLY, FULL):
        assert "NOT THE SCORE" not in " ".join(plain_summary(payload))
        assert "SUBSET" not in render(payload, ".html", scratch())


def test_nothing_is_double_escaped():
    html = render(ARENA_ONLY, ".html", scratch())
    assert "&amp;mdash;" not in html and "&amp;rarr;" not in html
    assert "<th></th>" not in html, "empty header cell"


def test_the_teachers_base_is_a_column_only_when_it_played():
    """Four players make four model columns; three make three - never a column of dashes."""
    arena = {
        "questions": 4, "random_baseline": 0.25, "elo_rounds": 1,
        "players": {
            "base": {"answered": 4, "correct": 1, "accuracy": 0.25,
                     "accuracy_when_answered": 0.25, "elo": 980, "elo_spread": 1},
            "distilled": {"answered": 4, "correct": 3, "accuracy": 0.75,
                          "accuracy_when_answered": 0.75, "elo": 1020, "elo_spread": 1},
            "teacher-base": {"answered": 4, "correct": 2, "accuracy": 0.5,
                             "accuracy_when_answered": 0.5, "elo": 1000, "elo_spread": 1},
            "teacher": {"answered": 4, "correct": 4, "accuracy": 1.0,
                        "accuracy_when_answered": 1.0, "elo": 1040, "elo_spread": 1},
        },
        "agreement": {"base vs teacher": {"same": 1, "of": 4, "pct": 0.25},
                      "distilled vs teacher": {"same": 3, "of": 4, "pct": 0.75},
                      "teacher vs teacher-base": {"same": 2, "of": 4, "pct": 0.5}},
        "head_to_head": {},
    }
    sections = _sections({"arena": arena, "student": "s", "teacher": "t"})
    for title, rows, headers in sections:
        assert headers[:1] == ("Metric",) or headers[0] == "Reasoning depth", headers
        for row in rows:
            assert len(row) == len(headers), (title, headers, row)
    headers = sections[0][2]
    assert headers == ("Metric", "Base student", "Distilled", "Teacher base", "Teacher"), headers
    closeness = dict((r[0], r) for r in sections[0][1])
    assert closeness["Gave the teacher's answer"][3] == "2 / 4  (50.0%)", closeness
    assert closeness["Gave the teacher's answer"][4] == "4 / 4  (100.0%)", closeness
    summary = " ".join(plain_summary({"arena": arena}))
    assert "fine-tune took it from 50.0%" in summary, summary

    # The same arena without the fourth player: three columns, as before.
    del arena["players"]["teacher-base"]
    del arena["agreement"]["teacher vs teacher-base"]
    sections = _sections({"arena": arena, "student": "s", "teacher": "t"})
    assert sections[0][2] == ("Metric", "Base student", "Distilled", "Teacher"), sections[0][2]
    assert "Teacher base" not in render({"arena": arena, "student": "s", "teacher": "t"},
                                        ".html", scratch())


def test_evaluate_payload_with_a_teacher_base_column():
    payload = {
        "student": "s", "teacher": "t", "teacher_base": "tb",
        "fidelity": {"top1_agreement_base_pct": 40.0, "top1_agreement_distilled_pct": 55.0,
                     "top1_agreement_teacher_base_pct": 48.0, "agreement_lift_pts": 15.0,
                     "kl_base": 1.0, "kl_distilled": 0.5, "kl_teacher_base": 0.7},
        "capability": {"perplexity_teacher": 2.0, "perplexity_base": 4.0,
                       "perplexity_distilled": 3.0, "perplexity_teacher_base": 2.5,
                       "gap_recovered_pct": 50.0},
        "closeness_to_teacher": {"prediction_agreement_base_pct": 40.0,
                                 "prediction_agreement_distilled_pct": 55.0,
                                 "prediction_agreement_teacher_base_pct": 48.0,
                                 "gap_recovered_pct": 50.0},
        "efficiency": {"teacher_params": 3e9, "student_params": 1.5e9,
                       "adapter_params": 1e7, "teacher_base_params": 3e9,
                       "teacher_tok_per_s": 10.0, "distilled_tok_per_s": 20.0},
        "generations": {},
    }
    sections = _sections(payload)
    for title, rows, headers in sections:
        assert headers == ("Metric", "Base student", "Distilled", "Teacher base", "Teacher"), \
            (title, headers)
        for row in rows:
            assert len(row) == 5, (title, row)
    fidelity = {r[0]: r for s in sections for r in s[1]}
    assert fidelity["Top-1 agreement with teacher"][3] == "48.00%", fidelity
    assert fidelity["Held-out perplexity (lower is better)"][3] == "2.500", fidelity
    assert fidelity["Parameters"][3] == "3.000B", fidelity
    text = render(payload, ".md", scratch())
    assert "| Teacher base |" in text and "| 48.00% |" in text, text


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"report: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
