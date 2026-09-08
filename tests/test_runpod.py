"""Checks for GPU selection, cost confirmation and the terminate guarantee.

No API key and no network: the SDK is replaced with a fake that records what was
asked of it. What is being tested is the part that spends money - which GPU gets
rented, at what price, whether anyone was asked first, and whether the pod is
always released. Those rules are worth more than the API plumbing around them.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kd import config as kdc              # noqa: E402
from kd.limits import Budget              # noqa: E402
from kd.remote import runpod as rp        # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

passed = []
failed = []


class FakeSdk:
    """Just enough of the runpod SDK, recording every call that costs money."""

    def __init__(self, gpus=None, states=None):
        self.api_key = None
        self._gpus = gpus if gpus is not None else default_gpus()
        self._states = list(states or ["RUNNING", "EXITED"])
        self.created = []
        self.terminated = []

    def get_gpus(self):
        return self._gpus

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return {"id": f"pod-{len(self.created)}"}

    def get_pod(self, pod_id):
        state = self._states.pop(0) if self._states else "EXITED"
        return {"id": pod_id, "desiredStatus": state}

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


def default_gpus():
    """A catalogue with one cheap card unavailable, which is the interesting case."""
    return [
        {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "memoryInGb": 16,
         "lowestPrice": {"minimumBidPrice": None, "uninterruptablePrice": None}},
        {"id": "NVIDIA RTX A5000", "displayName": "RTX A5000", "memoryInGb": 24,
         "lowestPrice": {"minimumBidPrice": 0.28, "uninterruptablePrice": 0.56}},
        {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24,
         "lowestPrice": {"minimumBidPrice": 0.44, "uninterruptablePrice": 0.88}},
        {"id": "NVIDIA A100 80GB", "displayName": "A100 80GB", "memoryInGb": 80,
         "lowestPrice": {"minimumBidPrice": 1.89, "uninterruptablePrice": 2.80}},
    ]


def check(name, fn):
    original = rp.sdk
    os.environ["RUNPOD_API_KEY"] = "test-key"
    try:
        fn()
    except AssertionError as exc:
        failed.append(f"{name}: {exc}")
    except Exception as exc:
        failed.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
    else:
        passed.append(name)
    finally:
        rp.sdk = original
        os.environ.pop("RUNPOD_API_KEY", None)


def install(fake):
    rp.sdk = lambda: fake
    return fake


def make_config(**runpod_settings):
    config = kdc.load_config(os.path.join(CONFIGS, "smoke.yaml"), use_env=False)
    config["runpod"].update({"enabled": True, "gpu_type": "RTX A4000",
                             "max_price_per_hour": 0.60, "spot": True,
                             "image": "ghcr.io/x/kd:test"})
    config["runpod"].update(runpod_settings)
    return config


class Log:
    def __init__(self):
        self.lines = []

    def _record(self, message):
        self.lines.append(str(message))

    info = warning = error = _record

    @property
    def text(self):
        return "\n".join(self.lines)


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def test_missing_api_key_is_explained():
    install(FakeSdk())
    os.environ.pop("RUNPOD_API_KEY", None)
    try:
        rp.authenticate(make_config())
    except rp.RunPodError as exc:
        assert "RUNPOD_API_KEY" in str(exc) and "console.runpod.io" in str(exc), exc
    else:
        raise AssertionError("a missing key should have been reported")
    finally:
        os.environ["RUNPOD_API_KEY"] = "test-key"


# --------------------------------------------------------------------------- #
# Choosing a GPU - the rule that protects the budget
# --------------------------------------------------------------------------- #
def test_available_named_gpu_is_taken():
    install(FakeSdk())
    gpu = rp.select_gpu(make_config(gpu_type="RTX A5000"))
    assert gpu["name"] == "RTX A5000", gpu
    assert gpu["price"] == 0.28, gpu


def test_unavailable_gpu_is_never_substituted():
    """The whole point: no pod may be created for a GPU nobody named."""
    install(FakeSdk())
    try:
        rp.select_gpu(make_config(gpu_type="RTX A4000"), ask=None)
    except rp.RunPodError as exc:
        message = str(exc)
        assert "no spot capacity" in message, message
        assert "Nothing has been rented" in message, message
        # It must show what could be had, and suggest naming one explicitly.
        assert "RTX A5000" in message and "runpod.gpu_type=" in message, message
    else:
        raise AssertionError("an unavailable GPU must not resolve to another one")


def test_prompt_offers_only_what_is_under_the_cap():
    install(FakeSdk())
    log = Log()
    answers = []
    rp.select_gpu(make_config(gpu_type="RTX A4000"),
                  ask=lambda prompt: (answers.append(prompt), "1")[1], log=log)
    shown = log.text
    assert "RTX A5000" in shown and "RTX 4090" in shown, shown
    # $1.89/hr against a $0.60 cap.
    assert "A100" not in shown, "a GPU over the price cap was offered"
    assert "nothing has been rented" in answers[0], answers


def test_prompt_choice_is_honoured():
    install(FakeSdk())
    gpu = rp.select_gpu(make_config(gpu_type="RTX A4000"), ask=lambda _p: "2")
    # Options are cheapest-first: A5000 at 0.28, then 4090 at 0.44.
    assert gpu["name"] == "RTX 4090", gpu


def test_quitting_the_prompt_rents_nothing():
    fake = install(FakeSdk())
    for answer in ("q", "", "no"):
        try:
            rp.select_gpu(make_config(gpu_type="RTX A4000"), ask=lambda _p: answer)
        except rp.RunPodError as exc:
            assert "aborted" in str(exc), exc
        else:
            raise AssertionError(f"{answer!r} should have aborted")
    assert not fake.created, "aborting still created a pod"


def test_nonsense_answer_rents_nothing():
    fake = install(FakeSdk())
    try:
        rp.select_gpu(make_config(gpu_type="RTX A4000"), ask=lambda _p: "17")
    except rp.RunPodError as exc:
        assert "nothing has been rented" in str(exc), exc
    else:
        raise AssertionError("an out-of-range answer should have aborted")
    assert not fake.created


def test_gpu_over_the_cap_says_so_distinctly():
    # Available, but too expensive: a different problem from no capacity, and it
    # calls for a different fix.
    install(FakeSdk())
    log = Log()
    try:
        rp.select_gpu(make_config(gpu_type="A100 80GB"), ask=None, log=log)
    except rp.RunPodError as exc:
        assert "over the" in str(exc) and "max_price_per_hour" in str(exc), exc
    else:
        raise AssertionError("a GPU over the cap should have been refused")


def test_unknown_gpu_name_is_rejected():
    install(FakeSdk())
    try:
        rp.select_gpu(make_config(gpu_type="RTX 9090"))
    except rp.RunPodError as exc:
        assert "does not exist" in str(exc) or "no GPU type called" in str(exc), exc
    else:
        raise AssertionError("an unknown GPU name should have been rejected")


def test_nothing_available_at_all():
    install(FakeSdk(gpus=[{"id": "RTX A4000", "displayName": "RTX A4000",
                           "lowestPrice": {}}]))
    try:
        rp.select_gpu(make_config(), ask=lambda _p: "1")
    except rp.RunPodError as exc:
        assert "Nothing has been rented" in str(exc), exc
    else:
        raise AssertionError("an empty catalogue should have been refused")


# --------------------------------------------------------------------------- #
# Cost, before renting
# --------------------------------------------------------------------------- #
def test_cheap_run_needs_no_confirmation():
    config = make_config()
    gpu = {"name": "RTX A5000", "price": 0.28, "spot": True}
    # 30 min at $0.28/hr is $0.14, under the $1.00 default threshold.
    assert round(rp.confirm_cost(config, gpu, 30, ask=None), 2) == 0.14


def test_expensive_run_asks_first():
    config = make_config()
    gpu = {"name": "RTX 4090", "price": 0.44, "spot": True}
    asked = []
    rp.confirm_cost(config, gpu, 60 * 5, ask=lambda p: (asked.append(p), "y")[1])
    assert asked and "proceed with about $2.20" in asked[0], asked


def test_declining_the_cost_aborts():
    config = make_config()
    gpu = {"name": "RTX 4090", "price": 0.44, "spot": True}
    try:
        rp.confirm_cost(config, gpu, 60 * 5, ask=lambda _p: "n")
    except rp.RunPodError as exc:
        assert "aborted" in str(exc), exc
    else:
        raise AssertionError("declining should have aborted")


def test_expensive_run_non_interactive_refuses():
    config = make_config()
    gpu = {"name": "RTX 4090", "price": 0.44, "spot": True}
    try:
        rp.confirm_cost(config, gpu, 60 * 5, ask=None)
    except rp.RunPodError as exc:
        assert "--yes" in str(exc), exc
    else:
        raise AssertionError("a costly non-interactive run should have been refused")


# --------------------------------------------------------------------------- #
# The pod, and letting go of it
# --------------------------------------------------------------------------- #
def test_launch_terminates_on_success():
    fake = install(FakeSdk(states=["RUNNING", "EXITED"]))
    log = Log()
    code = rp.launch(make_config(gpu_type="RTX A5000"), "configs/smoke.yaml", log,
                     poll_seconds=0)
    assert code == 0, code
    assert fake.created, "no pod was created"
    assert fake.terminated == ["pod-1"], fake.terminated


def test_launch_terminates_when_watching_raises():
    """An exception anywhere after renting must still release the pod."""
    fake = install(FakeSdk(states=["RUNNING"]))
    original = rp.watch
    rp.watch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network gone"))
    try:
        rp.launch(make_config(gpu_type="RTX A5000"), "configs/smoke.yaml", Log(),
                  poll_seconds=0)
    except RuntimeError:
        pass
    finally:
        rp.watch = original
    assert fake.terminated == ["pod-1"], \
        f"an exception left the pod running: {fake.terminated}"


def test_bid_never_exceeds_the_agreed_price():
    fake = install(FakeSdk(states=["EXITED"]))
    rp.launch(make_config(gpu_type="RTX A5000"), "configs/smoke.yaml", Log(),
              poll_seconds=0)
    assert fake.created[0]["bid_per_gpu"] == 0.28, fake.created[0]


def test_pod_command_and_environment():
    config = make_config()
    command = rp.pod_command(config, "configs/finance.yaml", ["--only", "train"])
    assert command == "python -m kd pipeline --config configs/finance.yaml --only train"
    env = rp.pod_env(config, {"name": "x", "price": 0.28, "spot": True})
    # The agreed rate travels with the pod so it can enforce the cost cap itself.
    assert env["KD_PRICE_PER_HOUR"] == "0.28", env


def test_missing_image_is_reported():
    install(FakeSdk())
    config = make_config(image=None)
    try:
        rp.create_pod(config, {"id": "g", "name": "g", "price": 0.1, "spot": True},
                      "cmd", {})
    except rp.RunPodError as exc:
        assert "runpod.image" in str(exc), exc
    else:
        raise AssertionError("a missing image should have been reported")


def test_disabled_runpod_refuses_to_launch():
    install(FakeSdk())
    try:
        rp.launch(make_config(enabled=False), "configs/smoke.yaml", Log())
    except rp.RunPodError as exc:
        assert "runpod.enabled is false" in str(exc), exc
    else:
        raise AssertionError("a disabled config should not have launched")


def test_terminate_never_raises():
    """Termination failure must not replace whatever caused the shutdown."""
    class Broken(FakeSdk):
        def terminate_pod(self, pod_id):
            raise RuntimeError("API down")

    install(Broken())
    log = Log()
    assert rp.terminate(make_config(), "pod-9", log=log) is False
    assert "STOP IT BY HAND" in log.text, log.text


# --------------------------------------------------------------------------- #
# Watching, and the cap that holds even if the pod misbehaves
# --------------------------------------------------------------------------- #
def test_watch_stops_on_a_cost_cap():
    install(FakeSdk(states=["RUNNING"] * 50))
    config = make_config()
    config["limits"]["max_cost_usd"] = 0.10
    config["limits"]["max_runtime_minutes"] = None

    clock = [0.0]
    budget = Budget(config, price_per_hour=0.60, started=0.0)
    budget.elapsed_minutes = lambda: clock[0]

    def tick(_seconds):
        clock[0] += 5.0

    outcome = rp.watch(config, "pod-1", {"price": 0.60}, budget,
                       poll_seconds=1, now=lambda: clock[0] * 60, sleep=tick)
    assert outcome["reason"] == "limit", outcome
    assert "max_cost_usd" in outcome["detail"], outcome


def test_watch_returns_when_the_pod_finishes():
    install(FakeSdk(states=["RUNNING", "RUNNING", "EXITED"]))
    outcome = rp.watch(make_config(), "pod-1", {"price": 0.28}, None,
                       poll_seconds=0, sleep=lambda _s: None)
    assert outcome["reason"] == "finished", outcome


for _name, _fn in sorted(list(globals().items())):
    if _name.startswith("test_") and callable(_fn):
        check(_name, _fn)

print(f"runpod: {len(passed)} passed, {len(failed)} failed")
for line in failed:
    print(f"  FAIL  {line}")
sys.exit(1 if failed else 0)
