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
              "adapter_locations": {
                  "local": "/cache/final_adapter",
                  "s3": "s3://bucket/kd/runs/run-1/final_adapter",
                  "s3_status": "the copy it was fetched from", "note": None},
              "profile": "configs/enlibraQ25-flow.yaml",
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
    assert abs(close["distilled"]["explanation_cosine"] - 0.85) < 1e-9, close
    # ...and the arena's own block wins when it is there.
    written = dict(ARENA, closeness={"reference": "teacher", "players": {
        "distilled": {"same_answer": 1, "of": 2, "same_answer_pct": 0.5,
                      "explanation_cosine": None}}})
    assert _closeness(written)["distilled"]["same_answer_pct"] == 0.5


def test_closeness_section_comes_first_and_the_agreement_row_moved_into_it():
    sections = _sections(ARENA_ONLY)
    assert sections[0][0].startswith("How close is it to the teacher?"), sections[0][0]
    labels = [row[0] for row in sections[0][1]]
    assert labels[0] == "Gave the teacher's answer", labels
    assert any("cosine" in l for l in labels), labels
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
    assert ("./run.sh --config configs/enlibraQ25-flow.yaml --from evaluate "
            "--adapter s3://bucket/kd/runs/run-1/final_adapter") in html, "no re-run command"
    assert "arena --adapter s3://bucket/kd/runs/run-1/final_adapter" in html
    md = render(ARENA_ONLY, ".md", scratch())
    assert "## The adapter" in md and "--from evaluate --adapter s3://" in md


def test_the_adapter_section_says_why_there_is_no_s3_copy():
    payload = dict(ARENA_ONLY, adapter_locations={
        "local": "runs/r1/final_adapter", "s3": None, "s3_status": None,
        "note": "s3.enabled is false, so this run does not upload it"})
    html = render(payload, ".html", scratch())
    assert "not there - s3.enabled is false" in html, "the reason is missing"
    # With no S3 copy, the re-run command names the local path.
    assert "--from evaluate --adapter runs/r1/final_adapter" in html


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


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"report: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
