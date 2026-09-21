"""Checks that the arena reads the letter a model actually settled on.

The parser is the one place where every player can lose a question at once,
and it has: on a real run, base, distilled AND teacher all answered a question
correctly in prose and were scored unanswered or wrong, because

  * "The correct answer is:\\n\\nD. ..." - the colon after "is" was not allowed;
  * "...the most accurate description is:\\n\\n**C. Beams of...**" - a line that
    STARTS with the letter and goes on to restate the option matched nothing;
  * the teacher, having answered D, reviewed the rejected options ("Option A
    talks about...") and the last "option X" mention won.

Each of those is a case below, next to the shapes that already worked and a
few that must NOT parse. When one of these fails, the models are not wrong -
the arena is.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd.arena import extract_answer_detail  # noqa: E402

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


# (completion, expected letter, expected `how`). `how` is pinned too, because
# the report counts "answered in the trained format" from it.
CASES = {
    # --- the three from the run that motivated this file -------------------
    "answer is, colon, blank line, restated option": (
        "The correct answer is:\n\nC. Beams of highly relativistic charged baryons "
        "interacting with the detector material.\n\nThis phenomenon involves...",
        "C", "labelled"),
    "conclusion without the word answer": (
        "Given the analysis, the most accurate description of the observed "
        "phenomena is:\n\n**C. Beams of highly relativistic charged baryons "
        "interacting with the detector material.**\n\nThis option accurately "
        "captures...",
        "C", "restated"),
    "answers, then reviews the rejected options by name": (
        "The correct answer is:\n\nD. Self-sustaining oscillations of orthogonal "
        "electric and magnetic fields.\n\nExplanation:\n...which aligns with the "
        "properties described in option D: \"Self-sustaining...\"\n\nOption A talks "
        "about propagating ripples in spacetime...\n\nOptions B and C describe "
        "phenomena involving neutrinos and charged particles...",
        "D", "labelled"),
    # --- the fourth one, from the 8B run: a teacher that answers in LaTeX -----
    # Verbatim shapes from that run's transcript. Every one of these was read as
    # the last option the model REJECTED, or as no answer at all.
    "boxed after prose that reviews the rejected options": (
        "Option B describes ascending spinothalamic tract neurons, which carry "
        "sensory information. Option D describes proprioceptive afferents, not the "
        "ventral horn motor neurons receiving corticospinal input. Therefore, the "
        "cells destroyed by the toxin are the alpha motor neurons of the ventral "
        "horn.\n\n$$\n\\boxed{C}\n$$",
        "C", "boxed"),
    "boxed wrapped in text": (
        "Therefore, the neuropeptide is **orexin/hypocretin**.\n\n$$\n"
        "\\boxed{\\text{A}}\n$$",
        "A", "boxed"),
    "boxed inline": ("Thus the answer is \\boxed{B}.", "B", "boxed"),
    # --- the fifth one, from a model served over HTTP: the letter comes FIRST --
    # It concludes "Therefore, C is the correct answer" about a third of the
    # time. Read by the last rule instead, that is the last option it RULED OUT.
    "concludes with the letter before the phrase": (
        "Option D is incorrect because the Golgi packages neurotransmitters, but the "
        "problem is with axonal elongation. Therefore, A is the correct answer.",
        "A", "labelled"),
    "letter first, bold, no comma": (
        "Nociception is pain, which is not related here. **C** is the correct answer.",
        "C", "labelled"),
    "letter first, superlative": (
        "Weighing them up, B is the best answer.", "B", "labelled"),
    "the last conclusion wins across both word orders": (
        "At first the answer is A. On reflection, D is the correct answer.",
        "D", "labelled"),
    "restated inline after the subject": (
        "Therefore, the most directly implicated syndrome is B. Sudden infant death "
        "syndrome.", "B", "restated"),
    "restated inline with a colon": (
        "Therefore, the best explanation for the new symptoms is D: Blockade of D2 "
        "receptors in the nigrostriatal pathway.", "D", "restated"),
    "an article is not an option letter": (
        "The observed result is a. Nothing here names a choice.", None, None),
    "an option ruled out is not an answer": (
        "Option C is incorrect because the amygdala is involved in emotional memory. "
        "Therefore, the hippocampal formation is the correct answer.", None, None),
    "an option ruled in still counts": (
        "Working through them, option B is correct here.", "B", "named"),
    # --- the curriculum's own shape, and it beats everything ----------------
    "tagged": ("<Explanation>\nBecause.\n</Explanation>\n<Answer>:\nB\n</Answer>",
               "B", "tagged"),
    "tagged beats an option named in the explanation": (
        "Option A is tempting but wrong. <Answer>: C", "C", "tagged"),
    # --- shapes that already worked -----------------------------------------
    "answer is bold": ("So the answer is **B**.", "B", "labelled"),
    "the label itself is bold": (
        "Thus the construct is **working memory**.\n\n**Answer:** D. Working memory",
        "D", "labelled"),
    "answer colon": ("Answer: D", "D", "labelled"),
    "answer equals": ("answer = (A)", "A", "labelled"),
    "option named": ("I would go with option B here.", "B", "named"),
    "bare letter line": ("Thinking it through...\n\nC\n", "C", "bare"),
    "bare letter in parentheses": ("(D)", "D", "bare"),
    "bare letter with a period": ("Reasoning.\nD.", "D", "bare"),
    # --- lists versus conclusions -------------------------------------------
    "analysis list, then a labelled conclusion": (
        "Let's look at each option:\n\nA. foo is wrong.\nB. bar is wrong.\n"
        "C. baz is right.\n\nThe answer is C.", "C", "labelled"),
    "list after a colon, then a restated conclusion": (
        "Consider:\n\nA. foo\nB. bar\n\nHence the best description is:\n\nB. bar",
        "B", "restated"),
    "the last of several labelled answers wins": (
        "At first the answer is A. On reflection the answer is B.", "B", "labelled"),
    # --- must not parse -------------------------------------------------------
    "a capital A starting a sentence": (
        "A star forms when gas collapses. I think this is nonsense.", None, None),
    "a colon followed by a sentence": ("Note:\nA comet is icy.", None, None),
    "a letter outside the option range": ("The answer is Q.", None, None),
    "empty": ("", None, None),
    "nothing": (None, None, None),
}


def make(text, letter, how):
    def test():
        got = extract_answer_detail(text)
        assert got == (letter, how), f"expected {(letter, how)}, got {got}"
    return test


for _name, (_text, _letter, _how) in CASES.items():
    check(_name, make(_text, _letter, _how))

print(f"arena parser: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
