"""Loading a teacher, and verifying it is a usable distillation target.

GKD trains the student to match the teacher's output distribution. A teacher that
loads without raising but is partly randomly initialised will therefore produce a
broken student after a full, apparently successful, training run - the run
"succeeds", burns the whole step budget, and only the final samples reveal it.

The failure this guards against is a merged checkpoint whose key layout does not
match the architecture transformers built for it. from_pretrained does not raise
in that case: it randomly initialises whatever it could not map and prints a
warning that scrolls past. A partly random teacher emits near-uniform token salad,
and the student faithfully learns to reproduce it.

Two entry points, differing only in thoroughness:

    verify_teacher()  the pre-flight run inside training, on a teacher already in
                      memory - one probe, no extra loading cost
    main()            the standalone `kd check-teacher`, which loads ONLY the
                      teacher and is far cheaper than a training smoke test

    kd check-teacher --config configs/finance.yaml
    kd check-teacher --teacher some-org/some-model --dtype bfloat16

Exit status is 0 when the teacher is healthy and 2 when it is not, so it can gate
a training run in a shell script.

When the weights are present but mislabelled, kd.teacher_fix repairs them.
"""

import argparse
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config, resolve_device

PROBES = [
    "How does compound interest work? Explain briefly.",
    "What is the difference between a Roth IRA and a traditional IRA?",
]

DTYPES = {
    "auto": None,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Keys that go missing legitimately: derived buffers and tied heads are rebuilt
# rather than loaded. Real parameter tensors are not.
IGNORABLE_MISSING = ("rotary_emb.inv_freq", ".attn_bias", "lm_head.weight")

# A healthy instruct teacher is confident about its next token. A randomly
# initialised one is near-uniform over the whole vocabulary, which is exactly what
# produces multilingual token salad downstream. These two thresholds are what
# separate the cases.
MAX_ENTROPY_FRACTION = 0.80   # of the uniform-distribution entropy
MIN_TOP1_PROBABILITY = 0.02


def load_teacher(teacher_id, adapter=None, dtype=None, device="cpu", verbose=True):
    """Load the teacher, merging a LoRA adapter into it when one is named.

    Returns (model, loading_info). The loading_info is what makes the missing-weight
    check possible, so it is threaded out rather than discarded.

    Preferring base + adapter over a merged checkpoint is deliberate: merged exports
    written by other frameworks often keep that framework's key layout, which plain
    transformers may not map back onto the architecture. Merging here, from the
    canonical base, sidesteps that entirely.
    """
    model, info = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype=dtype, low_cpu_mem_usage=True, output_loading_info=True)
    if adapter:
        from peft import PeftModel
        if verbose:
            print(f" -> merging teacher LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        if verbose:
            print(" -> adapter merged into the teacher")
    return model.to(device), info


def significant_missing(loading_info):
    """Weight names transformers had to invent because it could not map them."""
    missing = list((loading_info or {}).get("missing_keys") or [])
    return [key for key in missing if not key.endswith(IGNORABLE_MISSING)]


def token_confidence(model, tokenizer, prompt, device):
    """(top-1 probability, entropy, uniform entropy) for the next token after `prompt`."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    return float(probs.max()), entropy, math.log(logits.numel())


def is_noise(top1, entropy, uniform):
    """True when the output distribution is too flat to be predicting language."""
    return entropy > MAX_ENTROPY_FRACTION * uniform or top1 < MIN_TOP1_PROBABILITY


UNFIT_ADVICE = """
  Likely causes:
    * the checkpoint's key layout does not match the architecture transformers
      built for it (converted or merged models often keep an older prefix
      convention, e.g. language_model.model.* vs model.language_model.*);
    * config.json declares components the checkpoint does not contain;
    * the merge was saved from a quantised or partially loaded model.

  If the weights are present but mislabelled, repair them:
      kd fix-teacher --config <config>
  If the teacher was trained as a LoRA adapter, point at the base model and let
  it be merged in rather than using a merged export:
      kd check-teacher --teacher <base> --teacher-adapter <adapter>
"""


def verify_teacher(model, tokenizer, device, probe, loading_info=None, strict=True):
    """Fast pre-flight on a teacher already in memory. Raises SystemExit(2) if unfit.

    This is the check that runs inside training, where the teacher has just been
    loaded anyway. `check()` below is the thorough standalone version.
    """
    problems = []

    real_missing = significant_missing(loading_info)
    if real_missing:
        problems.append(
            f"{len(real_missing)} weight(s) were randomly initialised instead of "
            f"loaded - the teacher is partly untrained. First few: {real_missing[:6]}")
    unexpected = list((loading_info or {}).get("unexpected_keys") or [])
    if unexpected:
        print(f" !! {len(unexpected)} checkpoint tensor(s) went unused: {unexpected[:4]}")

    bad = [n for n, t in model.named_parameters() if not torch.isfinite(t).all()]
    if bad:
        problems.append(f"{len(bad)} parameter tensor(s) contain NaN/Inf: {bad[:4]}")

    top1, entropy, uniform = token_confidence(model, tokenizer, probe, device)
    print(f" -> teacher sanity: top-1 prob {top1:.4f}, entropy {entropy:.2f} "
          f"(uniform would be {uniform:.2f})")
    if is_noise(top1, entropy, uniform):
        problems.append(
            f"teacher output is near-uniform (entropy {entropy:.2f} of a possible "
            f"{uniform:.2f}); it is not predicting language")

    if not problems:
        print(" -> teacher pre-flight OK")
        return True

    bar = "=" * 78
    print("\n" + bar)
    print("  TEACHER PRE-FLIGHT FAILED")
    print(bar)
    for item in problems:
        print("  * " + item)
    print("\n  Distilling from this teacher would produce a broken student, because "
          "GKD\n  trains the student to match whatever distribution the teacher emits.")
    print(UNFIT_ADVICE)
    print(bar)
    if strict:
        raise SystemExit(2)
    return False


def parse_args():
    ap = argparse.ArgumentParser(
        description="Check that a teacher checkpoint is fit to distil from.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-c", "--config", default="configs/finance.yaml",
                    help="Config file to read models.teacher from "
                         "(default: configs/finance.yaml)")
    ap.add_argument("-t", "--teacher", default=None,
                    help="Teacher repo id, overriding the config file")
    ap.add_argument("-a", "--teacher-adapter", default=None,
                    help="LoRA adapter to merge into the teacher before checking")
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer to use (default: the teacher's own)")
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES),
                    help="Weight dtype; bfloat16 roughly halves memory (default: auto)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"],
                    help="Override the device from the config")
    ap.add_argument("--max-new-tokens", type=int, default=48,
                    help="Tokens to generate per probe (default: 48)")
    return ap.parse_args()


def main(args=None):
    # The pipeline calls this with a prepared Namespace, so the stage and the
    # standalone command run identical code.
    args = args or parse_args()
    config = load_config(args.config)
    if args.device:
        config.setdefault("hardware", {})["device"] = args.device
    if args.dtype != "auto":
        config.setdefault("hardware", {})["dtype"] = args.dtype

    hardware = resolve_device(config)
    device = hardware["device"]
    dtype = DTYPES[args.dtype] or hardware["dtype"]

    teacher_id = args.teacher or config["models"]["teacher"]
    adapter_id = args.teacher_adapter or config["models"].get("teacher_adapter")
    tokenizer_id = args.tokenizer or teacher_id

    bar = "=" * 78
    print(bar)
    print(f"  Teacher check: {teacher_id}")
    if args.teacher_adapter or config["models"].get("teacher_adapter"):
        print(f"  + adapter    : "
              f"{args.teacher_adapter or config['models'].get('teacher_adapter')}")
    print(f"  device={device} dtype={str(dtype).replace('torch.', '')}")
    print(bar)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[1/4] Loading weights (this is the slow part)...")
    model, info = load_teacher(teacher_id, adapter_id, dtype=dtype, device=device)
    model = model.eval()
    print(f" -> loaded {type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B parameters")

    problems = []

    # 1. Weights transformers had to invent because it could not map them.
    print("\n[2/4] Checkpoint coverage")
    missing = list(info.get("missing_keys") or [])
    unexpected = list(info.get("unexpected_keys") or [])
    real_missing = significant_missing(info)
    print(f"  missing from checkpoint : {len(missing)} ({len(real_missing)} significant)")
    print(f"  unused in checkpoint    : {len(unexpected)}")
    if real_missing:
        for key in real_missing[:8]:
            print(f"    ! randomly initialised: {key}")
        if len(real_missing) > 8:
            print(f"    ! ... and {len(real_missing) - 8} more")
        problems.append(
            f"{len(real_missing)} weight(s) were randomly initialised rather than "
            f"loaded - this teacher is partly untrained"
        )
    if unexpected:
        for key in unexpected[:4]:
            print(f"    - ignored: {key}")

    # 2. Numerically valid weights.
    print("\n[3/4] Parameter health")
    bad = [n for n, t in model.named_parameters() if not torch.isfinite(t).all()]
    print(f"  tensors with NaN/Inf    : {len(bad)}")
    if bad:
        problems.append(f"{len(bad)} parameter tensor(s) contain NaN/Inf: {bad[:4]}")

    # 3. Behaviour. A healthy instruct model puts real mass on a few plausible
    #    tokens. A broken one is close to uniform over the whole vocabulary, which
    #    is what shows up downstream as multilingual token salad.
    print("\n[4/4] Generation")
    for probe in PROBES:
        top_p, entropy, uniform = token_confidence(model, tokenizer, probe, device)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": probe}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\n  Q: {probe}")
        print(f"     top-1 prob {top_p:.4f} | entropy {entropy:.2f} / {uniform:.2f} uniform")
        print(f"  A: {answer.strip()[:300]}")

        if is_noise(top_p, entropy, uniform):
            problems.append(
                f"near-uniform output on {probe!r} (entropy {entropy:.2f} of a "
                f"possible {uniform:.2f}) - the model is not predicting language"
            )

    print()
    if problems:
        print(bar)
        print("  TEACHER UNFIT - do not distil from this checkpoint")
        print(bar)
        for item in problems:
            print(f"  * {item}")
        print(UNFIT_ADVICE)
        return 2

    print(bar)
    print("  ALL OK - teacher is fit to distil from")
    print(bar)
    print(f"   teacher : {teacher_id}")
    print(f"   loaded  : {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B params, "
          f"no missing weights, no NaN/Inf")
    print("   output  : coherent on both probes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
