"""Checks that the documentation still describes the code.

Docs rot silently. A README that names a command which no longer exists, or a
config key that was renamed, is worse than no README: someone follows it, it
fails, and they distrust the rest of it too. These checks are cheap and catch
exactly that class of drift.

Nothing here judges prose. It only asserts that every command, config key and
file path the docs mention is real.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import yaml  # noqa: E402

from kd import cli  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = [os.path.join(ROOT, "README.md")] + [
    os.path.join(ROOT, "docs", name)
    for name in sorted(os.listdir(os.path.join(ROOT, "docs")))
    if name.endswith(".md")
]

# Command spellings that were real before the restructure and are not any more.
# Each of these appearing in the docs means someone will run it and get an error.
RETIRED = [
    "train_scaled.py", "kd_config.py", "control_app.py", "publish_model.py",
    "convert_mlx_adapter.py", "check_teacher.py", "fix_teacher.py",
    "test_inference.py", "--profile", "--ui-compare", "--eval-json",
    "distilled_smollm_scaled", "distilled_qwen_finance",
]

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


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def relative(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def commands():
    found = set()
    for action in cli.build_parser()._subparsers._group_actions:
        found.update(action.choices)
    return found


# --------------------------------------------------------------------------- #
def test_no_retired_spellings():
    offences = []
    for path in DOCS:
        for number, line in enumerate(read(path).splitlines(), start=1):
            for dead in RETIRED:
                if dead in line:
                    offences.append(f"{relative(path)}:{number} mentions {dead!r}")
    assert not offences, "documentation names things that no longer exist:\n    " + \
        "\n    ".join(offences)


def test_every_kd_command_mentioned_exists():
    """`kd something` in the docs has to be a real subcommand."""
    known = commands()
    offences = []
    for path in DOCS:
        text = read(path)
        for match in re.finditer(r"\bkd ([a-z][a-z-]+)", text):
            name = match.group(1)
            if name not in known:
                offences.append(f"{relative(path)}: 'kd {name}'")
    assert not offences, ("documented commands that do not exist:\n    "
                          + "\n    ".join(sorted(set(offences)))
                          + f"\n  real ones: {', '.join(sorted(known))}")


def test_every_command_is_documented():
    """Every subcommand appears in the README, so nothing ships undiscoverable."""
    text = read(os.path.join(ROOT, "README.md"))
    missing = [name for name in commands() if f"kd {name}" not in text]
    assert not missing, f"commands absent from the README: {sorted(missing)}"


def test_config_keys_in_the_reference_are_real():
    """Every `section.key` in CONFIG.md exists in _base.yaml."""
    with open(os.path.join(ROOT, "configs", "_base.yaml"), encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    real = set()
    for section, value in base.items():
        real.add(section)
        if isinstance(value, dict):
            for key in value:
                real.add(f"{section}.{key}")

    text = read(os.path.join(ROOT, "docs", "CONFIG.md"))
    sections = set(base)
    offences = set()
    for match in re.finditer(r"`([a-z_]+)\.([a-z_]+)`", text):
        section, key = match.group(1), match.group(2)
        if section in sections and f"{section}.{key}" not in real:
            offences.add(f"{section}.{key}")
    assert not offences, f"CONFIG.md documents keys that do not exist: {sorted(offences)}"


def test_every_config_section_is_documented():
    with open(os.path.join(ROOT, "configs", "_base.yaml"), encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    text = read(os.path.join(ROOT, "docs", "CONFIG.md"))
    missing = [section for section in base if f"## `{section}`" not in text]
    assert not missing, f"config sections with no CONFIG.md heading: {missing}"


def test_relative_links_resolve():
    offences = []
    for path in DOCS:
        base = os.path.dirname(path)
        for match in re.finditer(r"\]\(([^)#]+)\)", read(path)):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not os.path.exists(os.path.join(base, target)):
                offences.append(f"{relative(path)} -> {target}")
    assert not offences, "broken relative links:\n    " + "\n    ".join(offences)


def test_repository_map_matches_the_tree():
    """Every src/kd file the README's map names is really there, and vice versa."""
    text = read(os.path.join(ROOT, "README.md"))
    listed = set(re.findall(r"^\s{2}([a-z_]+\.py)\s", text, re.M))
    actual = {name for name in os.listdir(os.path.join(ROOT, "src", "kd"))
              if name.endswith(".py") and not name.startswith("__")}
    assert not (listed - actual), f"README lists modules that do not exist: {listed - actual}"
    assert not (actual - listed), f"modules missing from the README map: {actual - listed}"


def test_shipped_profiles_are_all_listed():
    text = read(os.path.join(ROOT, "README.md"))
    profiles = {os.path.splitext(name)[0]
                for name in os.listdir(os.path.join(ROOT, "configs"))
                if name.endswith(".yaml") and not name.startswith("_")}
    missing = [name for name in profiles if f"`{name}`" not in text]
    assert not missing, f"profiles absent from the README: {sorted(missing)}"


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"docs: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
