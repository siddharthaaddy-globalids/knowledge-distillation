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
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config, resolve_device

# Last-resort probes, used only when the config carries no benchmark_prompts -
# `kd check-teacher --teacher <id>` with no profile, for instance. A real run
# should never see these: the questions a teacher is probed with want to be the
# questions it will be distilled on, or the check reports on a capability
# nothing downstream depends on.
FALLBACK_PROBES = [
    "Explain how a rainbow forms, briefly.",
    "What is the difference between mass and weight?",
]


def probes_for(config, limit=2):
    """The questions to probe this teacher with.

    From `benchmark_prompts` in the config, which is the same list kd.train uses
    for its pre-flight and its periodic quality samples. Sharing the source is
    the point: a teacher checked on finance questions and then distilled on
    astrophysics has been checked for the wrong thing, and the report says
    "healthy" either way.

    Capped, because each probe is a full autoregressive generation and this
    stage runs before every training run.
    """
    prompts = [str(p) for p in (config.get("benchmark_prompts") or []) if str(p).strip()]
    return (prompts or FALLBACK_PROBES)[:limit]

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


def adapter_problem(adapter, teacher_id=None):
    """Why this teacher adapter cannot be loaded, or None if it looks fine.

    Called from preflight, deliberately. PEFT only complains once the base model
    is in memory, which for a 2B teacher means a multi-gigabyte download first -
    so a missing directory or an MLX-format adapter would otherwise cost minutes
    and bandwidth before saying anything. Both are checkable in about a second.

    Only reports what it is sure about. Anything it cannot inspect is left for the
    loader rather than guessed at.
    """
    if not adapter:
        return None

    adapter = str(adapter)
    base = teacher_id or "<base-model>"

    if os.path.isdir(adapter):
        files = set(os.listdir(adapter))
    elif _looks_like_hub_id(adapter):
        try:
            from huggingface_hub import list_repo_files
            files = set(list_repo_files(adapter))
        except Exception:
            # Private, offline, or an unexpected API shape. Not knowing is not a
            # reason to block the run - the loader will report it properly.
            return None
    else:
        return (f"models.teacher_adapter points at {adapter}, which does not exist.\n"
                f"If you have an MLX or unsloth adapter, convert it once first:\n"
                f"  kd convert-adapter --adapter <source> --base {base} "
                f"--out {adapter}")

    if not files or any(name in files for name in
                        ("adapter_model.safetensors", "adapter_model.bin")):
        return None

    if "adapters.safetensors" in files:
        return (f"{adapter} is an MLX/unsloth adapter, not a PEFT one: it holds "
                f"adapters.safetensors and a lora_parameters schema, which PEFT "
                f"cannot read.\n"
                f"Convert it once - a rename and transpose, no retraining and no "
                f"GPU:\n"
                f"  kd convert-adapter --adapter {adapter} --base {base} "
                f"--out ./peft-adapter\n"
                f"then point at the result:\n"
                f"  --set models.teacher_adapter=./peft-adapter")

    if "adapter_config.json" not in files:
        return (f"{adapter} does not look like a LoRA adapter: it has no "
                f"adapter_config.json.")
    return None


def _looks_like_hub_id(value):
    """True for 'org/name', false for anything that is plainly a filesystem path.

    The two are told apart by shape, not by a slash: a Hub id is exactly one
    slash between two plain names. Testing for a separator alone would misread
    every Hub id on Windows, where '/' is os.path.altsep.
    """
    import re

    if value.startswith((".", "/", "~", "\\")) or ":" in value or "\\" in value:
        return False
    return re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", value) is not None


def significant_missing(loading_info):
    """Weight names transformers had to invent because it could not map them."""
    missing = list((loading_info or {}).get("missing_keys") or [])
    return [key for key in missing if not key.endswith(IGNORABLE_MISSING)]


def render_prompt(tokenizer, prompt, system=None, enable_thinking=False):
    """The prompt, positioned so the next token is an ANSWER rather than a preamble.

    Qwen3 and its relatives render `...assistant\\n` and then open a reasoning
    chain, so the next token is `<think>` with probability ~1.0. Both things this
    module does are ruined by that:

      * the confidence check measures how sure the model is that it is about to
        think, which is near-certain for a healthy model AND for a broken one
        whose reasoning is nonsense - so it reports 0.9999 either way;
      * the generation sample shows the first few dozen tokens of deliberation
        and never reaches the answer, which is the part worth reading.

    `enable_thinking=False` moves the empty `<think></think>` block into the
    prompt, so generation starts at the answer. Templates that do not know the
    argument ignore it - verified against SmolLM2, whose output is byte-identical
    either way - so this is safe to pass unconditionally.

    `enable_thinking=True` is the opposite request: leave the reasoning block
    open so the trace is generated and can be read. Training and evaluation
    never want that; scripts/merge.py offers it as --think for looking at what
    a model deliberates before it answers. `system` prepends a system turn;
    every caller here renders the same way whether or not one is given, so a
    system turn cannot silently flip thinking back on.
    """
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        # A template implementation that rejects unknown kwargs outright.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def token_confidence(model, tokenizer, prompt, device):
    """(top-1 probability, entropy, uniform entropy) for the next token after `prompt`."""
    text = render_prompt(tokenizer, prompt)
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
    probes = probes_for(config)
    print("\n[4/4] Generation")
    print(f"  probes from {'benchmark_prompts in ' + str(config['_meta'].get('source'))
                           if (config.get('benchmark_prompts') or []) else 'built-in fallbacks'}")
    for probe in probes:
        top_p, entropy, uniform = token_confidence(model, tokenizer, probe, device)
        # The same rendering the confidence was measured at, so the number and
        # the text below it describe the same position in the same sequence.
        ids = tokenizer(render_prompt(tokenizer, probe), return_tensors="pt").to(device)

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
