"""Checks on the runner templates and the contract they share with the code.

These are the things that break silently. A runner is downloaded once and used for
weeks; if it dispatches to a command that no longer exists, or a stage is declared
in the config but not implemented, nobody finds out until a run is already paid
for. None of this needs a shell - it is all textual and structural, so it runs
anywhere the rest of the tests do.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import yaml  # noqa: E402

from kd import cli, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
TEMPLATES = {
    "sh": os.path.join(SCRIPTS, "distill.sh.template"),
    "ps1": os.path.join(SCRIPTS, "distill.ps1.template"),
}
PLACEHOLDERS = {"__SHA__", "__SHA_SHORT__", "__BUILT_AT__", "__REPO_URL__", "__REPO__"}

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


def template(kind):
    with open(TEMPLATES[kind], encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------- #
# The config and the code have to agree about what a stage is
# --------------------------------------------------------------------------- #
def test_declared_stages_are_all_implemented():
    with open(os.path.join(ROOT, "configs", "_base.yaml"), encoding="utf-8") as handle:
        declared = [entry["name"] for entry in
                    yaml.safe_load(handle)["pipeline"]["stages"]]
    missing = [name for name in declared if name not in pipeline.STAGES]
    unused = [name for name in pipeline.STAGES if name not in declared]
    assert not missing, f"declared in _base.yaml but not implemented: {missing}"
    assert not unused, f"implemented but never declared: {unused}"


def test_conditional_stages_name_a_real_config_section():
    with open(os.path.join(ROOT, "configs", "_base.yaml"), encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    for stage, (section, _reason) in pipeline.CONDITIONAL.items():
        assert stage in pipeline.STAGES, f"{stage} is conditional but not a stage"
        assert section in base, f"{stage} is gated on a missing section '{section}'"
        assert "enabled" in base[section], f"{section} has no 'enabled' key to gate on"


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def test_both_templates_exist_and_are_bootstrappers():
    for kind, path in TEMPLATES.items():
        assert os.path.isfile(path), f"missing {path}"
        # A bootstrapper that has grown past a few hundred lines has started
        # reimplementing the pipeline, which is the thing these files exist to
        # avoid: logic here has to be written twice and tested on two platforms.
        lines = len(template(kind).splitlines())
        assert lines < 260, f"distill.{kind}.template is {lines} lines - too much logic"


def test_templates_use_the_same_placeholders():
    found = {kind: {m for m in PLACEHOLDERS if m in template(kind)} for kind in TEMPLATES}
    assert found["sh"] == found["ps1"], (
        f"the runners are substituted differently: sh={sorted(found['sh'])} "
        f"ps1={sorted(found['ps1'])}")
    assert found["sh"], "no placeholders at all - the runner would not be pinned"


def test_no_unknown_placeholders():
    # CI substitutes exactly the five in PLACEHOLDERS. Anything else in that shape
    # would ship to users verbatim.
    for kind in TEMPLATES:
        for match in set(re.findall(r"__[A-Z][A-Z_]*__", template(kind))):
            assert match in PLACEHOLDERS, f"distill.{kind}: unknown placeholder {match}"


def test_both_runners_hand_over_the_same_way():
    # `python -m kd`, not the `kd` console script: the module form needs nothing on
    # PATH and no generated executable.
    for kind in TEMPLATES:
        text = template(kind)
        assert "python -m kd" in text, f"distill.{kind} does not hand over to python -m kd"
        assert "uv run" in text, f"distill.{kind} does not run through uv"


def test_both_runners_default_to_the_pipeline():
    for kind in TEMPLATES:
        assert "pipeline" in template(kind), \
            f"distill.{kind} has no default command"


def test_dispatch_only_hook_exists_in_both():
    # CI proves the two runners dispatch identically by comparing their output
    # under this variable. If either loses it, that comparison silently stops
    # testing anything.
    for kind in TEMPLATES:
        assert "KD_DISPATCH_ONLY" in template(kind), \
            f"distill.{kind} lost the dispatch-only hook CI depends on"


def test_documented_subcommands_exist():
    """Every command the runners' help text names is a command the CLI has."""
    known = set(cli.DELEGATED) | {"pipeline", "train", "check", "doctor"}
    parser_commands = set()
    for action in cli.build_parser()._subparsers._group_actions:
        parser_commands.update(action.choices)
    assert known <= parser_commands, f"missing from the CLI: {known - parser_commands}"

    for kind in TEMPLATES:
        text = template(kind)
        # The COMMAND paragraph of the runner help lists them by name.
        listed = re.search(r"Otherwise the first argument is the pipeline subcommand:"
                           r"(.*?)\n\n", text, re.S)
        assert listed, f"distill.{kind} does not document its subcommands"
        named = {word.strip(" .,\n") for word in listed.group(1).replace("\n", " ").split(",")}
        unknown = {name for name in named if name and name not in parser_commands}
        assert not unknown, f"distill.{kind} advertises commands that do not exist: {unknown}"


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"runners: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
