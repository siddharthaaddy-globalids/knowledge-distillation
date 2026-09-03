"""Verify a teacher checkpoint is a usable distillation target, before training.

GKD trains the student to match the teacher's output distribution. A teacher that
loads without raising but is partly randomly initialised will therefore produce a
broken student after a full, apparently successful, training run.

Loads only the teacher, so it is much cheaper than a training smoke test.

    python check_teacher.py                          # teacher from configs/finance.yaml
    python check_teacher.py --config configs/mac.yaml
    python check_teacher.py --teacher some-org/some-model
    python check_teacher.py --dtype bfloat16         # halve the memory footprint

Exit status is 0 when the teacher is healthy and 2 when it is not, so it can gate
a training run in a shell script.
"""

import argparse
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kd_config import load_config, resolve_device

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


def main():
    args = parse_args()
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
    model, info = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype=dtype, low_cpu_mem_usage=True, output_loading_info=True
    )
    if adapter_id:
        from peft import PeftModel
        print(f" -> merging LoRA adapter: {adapter_id}")
        model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()
    model = model.to(device).eval()
    print(f" -> loaded {type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B parameters")

    problems = []

    # 1. Weights transformers had to invent because it could not map them.
    print("\n[2/4] Checkpoint coverage")
    missing = list(info.get("missing_keys") or [])
    unexpected = list(info.get("unexpected_keys") or [])
    ignorable = ("rotary_emb.inv_freq", ".attn_bias", "lm_head.weight")
    real_missing = [k for k in missing if not k.endswith(ignorable)]
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
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": probe}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        top_p = float(probs.max())
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        uniform = math.log(logits.numel())

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

        if entropy > 0.80 * uniform or top_p < 0.02:
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
        print("""
  Likely causes:
    * the checkpoint's key layout does not match the architecture transformers
      built for it (converted or merged models often keep an older prefix
      convention, e.g. language_model.model.* vs model.language_model.*);
    * config.json declares components the checkpoint does not contain;
    * the merge was saved from a quantised or partially loaded model.

  Fix the teacher, or point --teacher at a known-good instruct model, before
  spending a training budget.""")
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
