"""Checks for the configuration layer.

Plain asserts, no test framework: this runs as `uv run python tests/test_config.py`
locally and in CI with nothing extra installed. The point is that a config mistake
fails here, in a second, rather than after a teacher download and a training run.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import config as kdc  # noqa: E402

PROFILES = ["default", "mac", "smoke", "finance", "qwen-poc"]
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

passed = []
failed = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:  # an unexpected error is a failure too, not a crash
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)


def profile(name, **kwargs):
    kwargs.setdefault("use_env", False)
    return kdc.load_config(os.path.join(CONFIGS, f"{name}.yaml"), **kwargs)


def expect_error(fn, *fragments):
    """Assert fn() raises ConfigError whose message contains every fragment."""
    try:
        fn()
    except kdc.ConfigError as exc:
        message = str(exc)
        for fragment in fragments:
            assert fragment in message, f"expected {fragment!r} in error, got: {message}"
        return message
    raise AssertionError("expected a ConfigError, but the call succeeded")


# --------------------------------------------------------------------------- #
# Every profile resolves, and inherits from the base
# --------------------------------------------------------------------------- #
def test_profiles_resolve():
    for name in PROFILES:
        cfg = profile(name)
        # A key no profile sets: it can only be here by inheritance.
        assert cfg["gkd"]["beta"] == 0.5, f"{name} lost gkd.beta"
        assert cfg["project"]["runs_dir"] == "./runs", f"{name} lost project.runs_dir"
        assert "configs/_base.yaml" in " ".join(cfg["_meta"]["chain"]).replace("\\", "/"), \
            f"{name} did not record _base.yaml in its chain"


def test_profile_overrides_base():
    assert profile("mac")["hardware"]["device"] == "mps"
    assert profile("mac")["training"]["batch_size"] == 4
    assert profile("smoke")["training"]["max_steps"] == 2
    assert profile("qwen-poc")["gkd"]["lmbda"] == 0.25
    # ...while leaving everything else at the base value.
    assert profile("mac")["training"]["learning_rate"] == 3.0e-4


def test_list_replaces_not_merges():
    # smoke.yaml names two target modules; it must not inherit the base list's seven.
    assert profile("smoke")["lora"]["target_modules"] == ["q_proj", "v_proj"]
    # ...and two domains, not the base's five.
    assert len(profile("smoke")["dataset"]["domains"]) == 2


# --------------------------------------------------------------------------- #
# --set
# --------------------------------------------------------------------------- #
def test_set_scalar_types():
    cfg = profile("smoke", set_overrides=[
        "training.max_steps=500",
        "gkd.seq_kd=true",
        "training.learning_rate=1e-5",
        "limits.max_cost_usd=null",
        "models.teacher=Qwen/Qwen3.5-2B",
    ])
    assert cfg["training"]["max_steps"] == 500
    assert cfg["gkd"]["seq_kd"] is True
    assert cfg["training"]["learning_rate"] == 1e-5
    assert cfg["limits"]["max_cost_usd"] is None
    # A model id must survive as a string, slashes, dots and digits included.
    assert cfg["models"]["teacher"] == "Qwen/Qwen3.5-2B"


def test_set_inline_list():
    cfg = profile("smoke", set_overrides=["lora.target_modules=[q_proj, k_proj, v_proj]"])
    assert cfg["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj"]


def test_set_rejects_unknown_key():
    expect_error(lambda: profile("smoke", set_overrides=["training.max_stepz=5"]),
                 "unknown key", "training.max_stepz", "did you mean")


def test_set_needs_key_value():
    expect_error(lambda: profile("smoke", set_overrides=["training.max_steps"]),
                 "--set needs KEY=VALUE")


def test_set_beats_profile():
    cfg = profile("mac", set_overrides=["hardware.device=cpu"])
    assert cfg["hardware"]["device"] == "cpu"
    assert cfg["_meta"]["overridden"]["hardware.device"] == "--set"


# --------------------------------------------------------------------------- #
# Precedence: profile < env < --set < flag
# --------------------------------------------------------------------------- #
def test_precedence_chain():
    os.environ["KD_MAX_STEPS"] = "111"
    try:
        env_only = profile("smoke", use_env=True)
        assert env_only["training"]["max_steps"] == 111, "env did not beat the profile"
        assert env_only["_meta"]["overridden"]["training.max_steps"] == "KD_MAX_STEPS"

        with_set = profile("smoke", use_env=True, set_overrides=["training.max_steps=222"])
        assert with_set["training"]["max_steps"] == 222, "--set did not beat the env var"

        with_flag = profile("smoke", use_env=True,
                            set_overrides=["training.max_steps=222"],
                            flag_overrides={"training.max_steps": 333})
        assert with_flag["training"]["max_steps"] == 333, "flag did not beat --set"
        assert with_flag["_meta"]["overridden"]["training.max_steps"] == "flag"
    finally:
        del os.environ["KD_MAX_STEPS"]


def test_bad_env_value_is_reported():
    os.environ["KD_MAX_STEPS"] = "not-a-number"
    try:
        expect_error(lambda: profile("smoke", use_env=True), "KD_MAX_STEPS", "int")
    finally:
        del os.environ["KD_MAX_STEPS"]


# --------------------------------------------------------------------------- #
# Strict validation
# --------------------------------------------------------------------------- #
def write(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                         dir=CONFIGS, encoding="utf-8")
    handle.write(text)
    handle.close()
    return handle.name


def with_temp_config(text, fn):
    path = write(text)
    try:
        fn(path)
    finally:
        os.unlink(path)


def test_unknown_section_rejected():
    with_temp_config(
        "extends: _base.yaml\ntraning:\n  max_steps: 5\n",
        lambda p: expect_error(lambda: kdc.load_config(p, use_env=False),
                               "unknown key", "traning.max_steps", "training"))


def test_unknown_leaf_rejected():
    with_temp_config(
        "extends: _base.yaml\ntraining:\n  max_step: 5\n",
        lambda p: expect_error(lambda: kdc.load_config(p, use_env=False),
                               "unknown key", "training.max_step", "max_steps"))


def test_wrong_type_rejected():
    with_temp_config(
        "extends: _base.yaml\ntraining:\n  max_steps: many\n",
        lambda p: expect_error(lambda: kdc.load_config(p, use_env=False),
                               "training.max_steps", "should be number", "got string"))


def test_bool_is_not_a_number():
    # True is an int in Python; `eval_enabled: 1` must still be rejected.
    with_temp_config(
        "extends: _base.yaml\ntraining:\n  eval_enabled: 1\n",
        lambda p: expect_error(lambda: kdc.load_config(p, use_env=False),
                               "training.eval_enabled", "should be bool"))


def test_all_errors_reported_together():
    message = None

    def capture(path):
        nonlocal message
        message = expect_error(lambda: kdc.load_config(path, use_env=False), "unknown key")

    with_temp_config(
        "extends: _base.yaml\ntraining:\n  max_stepz: 5\n  batch_sze: 2\ngkd:\n  lmbdaa: 0.5\n",
        capture)
    assert message.count("unknown key") == 3, \
        f"expected all three typos in one error, got:\n{message}"


def test_unknown_domain_key_rejected():
    with_temp_config(
        "extends: _base.yaml\ndataset:\n  domains:\n    - name: x\n      quotaa: 5\n",
        lambda p: expect_error(lambda: kdc.load_config(p, use_env=False),
                               "dataset.domains[0].quotaa", "quota"))


def test_alpaca_domain_keys_accepted():
    # The finance profiles use these; they must not read as typos.
    cfg = profile("finance")
    domain = cfg["dataset"]["domains"][0]
    assert domain["format"] == "alpaca"
    assert domain["instruction_column"] == "instruction"


# --------------------------------------------------------------------------- #
# extends
# --------------------------------------------------------------------------- #
def test_missing_parent_is_reported():
    with_temp_config(
        "extends: no-such-file.yaml\n",
        lambda p: expect_error(lambda: kdc.load_config(p, use_env=False),
                               "extends", "no-such-file.yaml"))


def test_circular_extends_is_reported():
    a = write("extends: _base.yaml\n")
    b = write(f"extends: {os.path.basename(a)}\n")
    # Point a at b, closing the loop.
    with open(a, "w", encoding="utf-8") as handle:
        handle.write(f"extends: {os.path.basename(b)}\n")
    try:
        expect_error(lambda: kdc.load_config(a, use_env=False), "Circular")
    finally:
        os.unlink(a)
        os.unlink(b)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_meta_records_what_changed():
    cfg = profile("mac")
    changed = cfg["_meta"]["changed"]
    assert "hardware.device" in changed
    assert changed["hardware.device"]["base"] == "auto"
    assert changed["hardware.device"]["value"] == "mps"
    # Nothing on the command line, so nothing is marked as overridden.
    assert cfg["_meta"]["overridden"] == {}


def test_strip_meta():
    cfg = profile("smoke")
    assert "_meta" in cfg
    assert "_meta" not in kdc.strip_meta(cfg)


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"config: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
