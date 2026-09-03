"""Convert an MLX-LM / unsloth-on-Apple-Silicon LoRA adapter into a PEFT adapter.

Unsloth on Apple Silicon runs on MLX, not PyTorch, so the adapters it writes use
MLX-LM's conventions throughout and `peft` cannot load them:

    MLX-LM                                  PEFT
    ------------------------------------    ------------------------------------
    adapters.safetensors                    adapter_model.safetensors
    {rank, scale, dropout, keys}            {r, lora_alpha, lora_dropout,
                                             target_modules, task_type, peft_type}
    <module>.lora_a   (in_features, r)      <module>.lora_A.weight  (r, in_features)
    <module>.lora_b   (r, out_features)     <module>.lora_B.weight  (out_features, r)
    language_model.model.layers.N...        model.layers.N...  (the LIVE module
                                            tree, which is what PEFT attaches to)

Nothing is retrained or approximated. The tensors are transposed and the names
rewritten; the adapter computes the same function afterwards.

    python convert_mlx_adapter.py --adapter <org>/<mlx-lora-repo>
    python convert_mlx_adapter.py --adapter ./mlx_lora --out ./peft_lora
    python convert_mlx_adapter.py --adapter <repo> --dry-run

Then verify and use it:

    python check_teacher.py --teacher Qwen/Qwen3.5-2B --teacher-adapter ./peft_lora
    ./distill.sh --teacher Qwen/Qwen3.5-2B --teacher-adapter ./peft_lora

Scaling note: MLX-LM applies `scale` directly to the LoRA branch, while PEFT
applies `lora_alpha / r`. This writes `lora_alpha = round(scale * r)` so the two
produce the same effective scaling.
"""

import argparse
import json
import pathlib
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

# Prefix rewrites between framework conventions. The correct one is chosen by
# checking which produces module paths that exist in the real architecture,
# rather than by assuming.
PREFIX_CANDIDATES = [
    ("language_model.model.", "model.language_model."),
    ("language_model.model.", "model."),
    ("model.language_model.", "model.language_model."),
    ("language_model.", "model.language_model."),
    ("", ""),
]

SIDECAR_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert an MLX-LM LoRA adapter to PEFT format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-a", "--adapter", required=True,
                    help="MLX adapter: a HF repo id or a local directory")
    ap.add_argument("-b", "--base", default=None,
                    help="Base model the adapter attaches to. Defaults to the "
                         "adapter config's base_model_name_or_path, with an "
                         "unsloth/ mirror rewritten to the canonical Qwen/ repo.")
    ap.add_argument("-o", "--out", default="./peft-adapter",
                    help="Directory to write the PEFT adapter to "
                         "(default: ./peft-adapter)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the mapping and exit without writing")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the output directory if it exists")
    return ap.parse_args()


def local_dir(repo_or_path):
    p = pathlib.Path(repo_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    print(f"[1/5] Fetching {repo_or_path}...")
    return pathlib.Path(snapshot_download(repo_or_path))


def find_weights(src):
    """MLX writes adapters.safetensors; PEFT writes adapter_model.safetensors."""
    for name in ("adapters.safetensors", "adapter_model.safetensors",
                 "adapters.npz", "adapter_model.bin"):
        if (src / name).exists():
            return src / name
    found = sorted(src.glob("*.safetensors"))
    if found:
        return found[0]
    raise SystemExit(f"No adapter weight file found in {src}")


def module_paths(base_id):
    """Every Linear module path in the real architecture, built on meta (no memory).

    PEFT attaches to the LIVE module tree, not to checkpoint key names. Those can
    differ: transformers rewrites keys while loading (a Qwen3.5 VL checkpoint
    stores model.language_model.* but the text-only class exposes model.layers.*).
    Targeting the instantiated modules is therefore the only correct choice.
    """
    config = AutoConfig.from_pretrained(base_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # from_config and from_pretrained must agree on the class, or the adapter
    # would target a module tree that never gets built.
    try:
        from transformers.models.auto.auto_factory import _get_model_class
        resolved = _get_model_class(config, AutoModelForCausalLM._model_mapping)
        if resolved is not type(model):
            raise SystemExit(
                f"Class mismatch: from_config builds {type(model).__name__} but "
                f"from_pretrained would build {resolved.__name__}. The adapter "
                f"would target the wrong module tree."
            )
    except ImportError:
        pass

    return {n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)}, type(model).__name__


def choose_prefix(mlx_modules, real_modules):
    """Pick the rewrite that lands the most MLX module paths on real ones."""
    best, best_hits, best_label = None, 0, ""
    for old, new in PREFIX_CANDIDATES:
        mapped = {}
        for m in mlx_modules:
            mapped[m] = new + m[len(old):] if old and m.startswith(old) else m
        hits = len(set(mapped.values()) & real_modules)
        if hits > best_hits:
            best, best_hits = mapped, hits
            best_label = f"{old or '(none)'} -> {new or '(none)'}"
    return best, best_hits, best_label


def main():
    args = parse_args()
    src = local_dir(args.adapter)

    cfg_path = src / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"No adapter_config.json in {src}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    base = args.base or cfg.get("base_model_name_or_path") or ""
    # unsloth mirrors carry the same weights but a non-conformant config; the
    # canonical repo is what transformers can actually build from.
    if base.startswith("unsloth/"):
        canonical = "Qwen/" + base.split("/", 1)[1]
        print(f"    base {base} -> {canonical} (canonical repo)")
        base = canonical
    if not base:
        raise SystemExit("Could not determine the base model; pass --base")

    lora = cfg.get("lora_parameters", {})
    rank = int(cfg.get("r") or cfg.get("rank") or lora.get("rank") or 0)
    scale = float(cfg.get("scale", lora.get("scale", 1.0)))
    dropout = float(cfg.get("dropout", lora.get("dropout", 0.0)))
    if not rank:
        raise SystemExit("Could not determine the LoRA rank from the adapter config")

    bar = "=" * 78
    print(bar)
    print(f"  MLX -> PEFT adapter conversion")
    print(f"  adapter : {args.adapter}")
    print(f"  base    : {base}")
    print(bar)

    print("\n[2/5] Reading MLX tensors...")
    wpath = find_weights(src)
    tensors = {}
    with safe_open(wpath, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print(f" -> {len(tensors)} tensors from {wpath.name}")

    mlx_modules = {k.rsplit(".", 1)[0] for k in tensors
                   if k.endswith((".lora_a", ".lora_b"))}
    if not mlx_modules:
        raise SystemExit("No .lora_a/.lora_b tensors found - is this an MLX adapter?")

    print("\n[3/5] Building the base architecture to resolve module paths...")
    real_modules, cls = module_paths(base)
    print(f" -> {cls}: {len(real_modules)} Linear modules")

    mapping, hits, label = choose_prefix(mlx_modules, real_modules)
    print(f" -> prefix rewrite: {label}  ({hits}/{len(mlx_modules)} resolved)")
    unresolved = sorted(m for m in mlx_modules if mapping[m] not in real_modules)
    if unresolved:
        print(f" !! {len(unresolved)} module(s) do not exist in the base model:")
        for m in unresolved[:5]:
            print(f"      {m} -> {mapping[m]}")
        if not args.force:
            raise SystemExit(
                "\nRefusing to write an adapter that references modules the base "
                "model does not have. Re-run with --force to write it anyway."
            )

    print("\n[4/5] Transposing and renaming...")
    out = {}
    suffixes = set()
    for key, tensor in tensors.items():
        if key.endswith(".lora_a"):
            module, peft_name, want = key[:-7], "lora_A", "(r, in)"
        elif key.endswith(".lora_b"):
            module, peft_name, want = key[:-7], "lora_B", "(out, r)"
        else:
            print(f"    skipping unrecognised tensor: {key}")
            continue
        # MLX stores lora_a as (in, r) and lora_b as (r, out); PEFT stores
        # lora_A.weight as (r, in) and lora_B.weight as (out, r). Both are a
        # plain transpose.
        out[f"base_model.model.{mapping[module]}.{peft_name}.weight"] = \
            tensor.t().contiguous()
        suffixes.add(module.rsplit(".", 1)[-1])
    print(f" -> {len(out)} tensors renamed and transposed")
    print(f" -> target_modules: {sorted(suffixes)}")

    lora_alpha = round(scale * rank)
    print(f" -> scaling: MLX scale={scale} x rank={rank} "
          f"=> lora_alpha={lora_alpha} (PEFT applies lora_alpha/r)")

    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": dropout,
        "target_modules": sorted(suffixes),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "modules_to_save": None,
        "init_lora_weights": True,
    }

    if args.dry_run:
        print("\n--- adapter_config.json that would be written ---")
        print(json.dumps(peft_cfg, indent=2))
        print("\n(dry run: nothing written)")
        return 0

    dest = pathlib.Path(args.out)
    if dest.exists():
        if not args.force:
            raise SystemExit(f"\n{dest} already exists. Pass --force to overwrite.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    print(f"\n[5/5] Writing to {dest}...")
    save_file(out, str(dest / "adapter_model.safetensors"), metadata={"format": "pt"})
    (dest / "adapter_config.json").write_text(
        json.dumps(peft_cfg, indent=2) + "\n", encoding="utf-8")
    copied = []
    for name in SIDECAR_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
            copied.append(name)
    size = (dest / "adapter_model.safetensors").stat().st_size
    print(f" -> adapter_model.safetensors ({size / 1e6:.1f} MB)")
    print(f" -> adapter_config.json + {len(copied)} tokenizer file(s)")

    print("\n" + bar)
    print("  ALL OK - adapter converted")
    print(bar)
    print(f"   tensors : {len(out)} ({len(out) // 2} modules, rank {rank})")
    print(f"   output  : {dest}")
    print()
    print("   Verify it against the base model, then train:")
    print(f"     python check_teacher.py --teacher {base} --teacher-adapter {dest}")
    print(f"     ./distill.sh --teacher {base} --teacher-adapter {dest}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
