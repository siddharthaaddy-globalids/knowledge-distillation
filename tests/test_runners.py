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
def _code_lines(text):
    """Executable lines only: no blanks, no comments, no embedded help text.

    Total line count is the wrong measure. It punishes explaining a subtle fix in
    a comment exactly as hard as adding a branch, so the guard ends up pressuring
    the one thing that should be encouraged. What matters is how much LOGIC lives
    in a file that has to be written twice and tested on two platforms.
    """
    count, in_heredoc = 0, False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ("cat <<'EOF'", "@'"):
            in_heredoc = True
            continue
        if stripped in ("EOF", "'@ | Write-Host"):
            in_heredoc = False
            continue
        if in_heredoc or not stripped or stripped.startswith(("#", "<#")):
            continue
        count += 1
    return count


def test_both_templates_exist_and_are_bootstrappers():
    for kind, path in TEMPLATES.items():
        assert os.path.isfile(path), f"missing {path}"
        # A bootstrapper that has grown real logic has started reimplementing the
        # pipeline, which is the thing these files exist to avoid.
        code = _code_lines(template(kind))
        assert code < 175, \
            f"distill.{kind}.template has {code} lines of logic - too much for a bootstrapper"


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


def test_optional_extras_reachable_from_both():
    """Both runners can install the optional dependency groups.

    The runner does its own `uv sync` inside the fetched checkout, so without a
    way to ask for an extra there is no way to run `--tasks` or enable S3 from a
    downloaded runner at all - and the error would tell you to run a uv command
    you have no checkout for.
    """
    for kind in TEMPLATES:
        text = template(kind)
        assert "--extra" in text, f"distill.{kind} cannot install an optional extra"
        assert "KD_EXTRAS" in text, f"distill.{kind} ignores KD_EXTRAS"
        for group in ("eval", "remote"):
            assert group in text, f"distill.{kind} does not mention the {group} extra"


def test_ref_override_exists_and_warns_in_both():
    """A pinned runner can be pointed at another branch, and never silently.

    The pin is what makes a result traceable to a revision, so an override has to
    be explicit on the command line AND announce itself - a run that quietly used
    a moving branch cannot be reproduced later.
    """
    for kind in TEMPLATES:
        text = template(kind)
        assert "PINNED_REF" in text or "PinnedRef" in text,             f"distill.{kind} does not keep the pinned ref separate from the one it uses"
        assert "KD_REF" in text, f"distill.{kind} ignores KD_REF"
        assert "NOT the pinned" in text,             f"distill.{kind} can override the ref without saying so"


def test_branch_checkout_cannot_go_stale():
    """Checking out FETCH_HEAD, not the ref name.

    `git checkout <branch>` in an existing clone lands on whatever that local
    branch pointed at last time. For a pinned commit that is harmless; for a
    branch it silently runs code from whenever the clone was last updated.
    """
    for kind in TEMPLATES:
        assert "FETCH_HEAD" in template(kind),             f"distill.{kind} checks out a ref name, so a moving branch would go stale"


def test_powershell_runner_has_no_windows_only_home():
    """distill.ps1 must resolve a home directory on any platform.

    $env:USERPROFILE exists only on Windows. Off Windows it is null, so
    `Join-Path $env:USERPROFILE ...` throws - and because the script sets
    $ErrorActionPreference to 'Stop', that kills it at line one, before it can
    even print its own help. CI validates this runner on a Linux pwsh, so the
    failure is real rather than theoretical.

    $HOME is defined by PowerShell itself on every platform, so it is the one to
    build paths from; USERPROFILE is allowed only as a fallback beside it.
    """
    text = template("ps1")
    assert "$HOME" in text, "distill.ps1 does not use $HOME"
    for number, line in enumerate(text.splitlines(), start=1):
        if "$env:USERPROFILE" in line and not line.lstrip().startswith("#"):
            assert "$HOME" in line, (
                f"distill.ps1:{number} builds a path from $env:USERPROFILE without "
                f"a $HOME fallback; that is null off Windows")


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
