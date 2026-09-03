"""Repair a merged checkpoint whose tensor keys do not match its architecture.

Some training frameworks export a merged model using their own key layout. The
weights are fine, but `AutoModelForCausalLM` cannot map them onto the module tree
it builds, so it silently randomly-initialises everything it could not match and
loads anyway. The result generates uniform noise, and distilling from it produces
a student that generates uniform noise too.

`check_teacher.py` detects that. This script fixes it, by renaming the tensors to
the names the architecture actually asks for and dropping components the text-only
model does not use (e.g. a vision tower).

    python fix_teacher.py                             # teacher from configs/finance.yaml
    python fix_teacher.py --teacher org/model --out ./teacher-fixed
    python fix_teacher.py --dry-run                   # show the mapping, write nothing

No GPU and no retraining: this is a rename pass over the safetensors file. The
config is copied verbatim so transformers builds exactly the same architecture it
built before, and the tensors are renamed to match what that build reported as
MISSING.

Afterwards, verify and use the repaired copy:

    python check_teacher.py --teacher ./teacher-fixed
    ./distill.sh --profile finance --teacher ./teacher-fixed
"""

import argparse
import itertools
import json
import math
import pathlib
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from kd_config import load_config

# Components a text-only causal LM never instantiates, so their weights are dead
# payload in the output. Dropping them also shrinks the file substantially.
DROP_PREFIXES = ("vision_tower.", "visual.", "model.visual.")

# Tokenizer and template files worth carrying across.
SIDECAR_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
    "generation_config.json",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Rename a merged checkpoint's tensors to match its architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-c", "--config", default="configs/finance.yaml",
                    help="Config file to read models.teacher from "
                         "(default: configs/finance.yaml)")
    ap.add_argument("-t", "--teacher", default=None,
                    help="Checkpoint to repair, overriding the config file")
    ap.add_argument("-o", "--out", default="./teacher-fixed",
                    help="Directory to write the repaired checkpoint to "
                         "(default: ./teacher-fixed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the key mapping and exit without writing")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the output directory if it already exists")
    return ap.parse_args()


def local_snapshot(repo_or_path):
    """Return a local directory holding the checkpoint, downloading if needed."""
    p = pathlib.Path(repo_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    print(f"[1/5] Fetching {repo_or_path} (cached after the first run)...")
    return pathlib.Path(snapshot_download(repo_or_path))


def load_all_tensors(src):
    """Read every tensor from a single-file or sharded safetensors checkpoint."""
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No .safetensors files found in {src}")
    tensors = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors, shards


def expected_keys(src):
    """Build the architecture from its config and report the tensors it wants.

    Returns {name: shape}. This is the authoritative list - it is exactly what
    `from_pretrained` reports as MISSING or MISMATCH. Building the skeleton on
    the meta device allocates no memory.
    """
    config = AutoConfig.from_pretrained(src)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    shapes = {n: tuple(t.shape) for n, t in model.named_parameters()}
    shapes.update({n: tuple(t.shape) for n, t in model.named_buffers()})
    tied = bool(getattr(config, "tie_word_embeddings", False))
    return shapes, tied, type(model).__name__


def reconcile_shape(tensor, expected):
    """Return (tensor, note) reshaped to `expected`, or (None, reason) if unsafe.

    Frameworks disagree about axis order for some tensors - notably the depthwise
    conv in linear attention, stored as (channels, kernel, 1) by some exporters and
    (channels, 1, kernel) by transformers. When the two shapes are a permutation of
    each other and every axis that moves is a singleton, the swap is a pure memory
    reshape: no element is reordered, so it is lossless.

    A permutation between two non-singleton axes is a genuine transpose. That may
    still be right, but it cannot be verified from shape alone, so it is flagged.
    """
    have = tuple(tensor.shape)
    expected = tuple(expected)
    if have == expected:
        return tensor, None
    if tensor.numel() != math.prod(expected):
        return None, f"element count differs ({tensor.numel()} vs {math.prod(expected)})"
    if sorted(have) != sorted(expected):
        return None, f"shapes are not a permutation ({have} vs {expected})"

    # A permutation is a pure reshape - element order untouched - exactly when the
    # non-singleton axes keep their relative order. Only the singleton axes move,
    # so nothing is reordered and the change is lossless.
    if [d for d in have if d != 1] == [d for d in expected if d != 1]:
        return tensor.reshape(expected).contiguous(), f"axis swap {have} -> {expected}"

    for perm in itertools.permutations(range(tensor.ndim)):
        if tuple(have[i] for i in perm) == expected:
            return (tensor.permute(*perm).contiguous(),
                    f"TRANSPOSE {have} -> {expected} (reorders elements; verify output)")
    return None, f"no permutation maps {have} to {expected}"


def build_mapping(have, want):
    """Find the single prefix rewrite that reconciles the two key sets."""
    have = {k for k in have if not k.startswith(DROP_PREFIXES)}
    unmatched = have - want
    if not unmatched:
        return {}, "keys already match"

    # Try the prefix rewrites that frameworks actually differ by, and keep the one
    # that resolves the most names. Deriving it beats hardcoding a single guess.
    candidates = [
        ("language_model.model.", "model."),
        ("language_model.model.", "model.language_model."),
        ("language_model.", "model.language_model."),
        ("model.language_model.", "model."),
        ("language_model.", ""),
        ("", "model."),
    ]
    best, best_hits, best_label = None, 0, ""
    for old, new in candidates:
        mapped = {}
        for key in have:
            renamed = new + key[len(old):] if old and key.startswith(old) else (
                new + key if not old else key
            )
            mapped[key] = renamed
        hits = len(set(mapped.values()) & want)
        if hits > best_hits:
            best, best_hits, best_label = mapped, hits, f"{old or '(none)'} -> {new or '(none)'}"
    if not best or best_hits == 0:
        return None, "no prefix rewrite reconciles these key sets"
    return best, f"{best_label}  ({best_hits} keys matched)"


def main():
    args = parse_args()
    teacher = args.teacher
    if teacher is None:
        teacher = load_config(args.config)["models"]["teacher"]

    bar = "=" * 78
    print(bar)
    print(f"  Repairing: {teacher}")
    print(f"  Output   : {args.out}")
    print(bar)

    src = local_snapshot(teacher)

    print("\n[2/5] Reading tensors...")
    tensors, shards = load_all_tensors(src)
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f" -> {len(tensors)} tensors across {len(shards)} file(s), "
          f"{total_bytes / 1e9:.2f} GB")

    print("\n[3/5] Building the architecture to learn what it expects...")
    want_shapes, tied, cls = expected_keys(src)
    want = set(want_shapes)
    print(f" -> {cls} expects {len(want)} tensors"
          + (" (lm_head tied to embeddings)" if tied else ""))

    dropped = [k for k in tensors if k.startswith(DROP_PREFIXES)]
    mapping, label = build_mapping(set(tensors), want)
    if mapping is None:
        raise SystemExit(
            f"\nCould not repair automatically: {label}.\n"
            f"  checkpoint sample: {sorted(tensors)[:3]}\n"
            f"  expected sample  : {sorted(want)[:3]}\n"
        )

    print(f"\n[4/5] Key mapping: {label}")
    if dropped:
        print(f" -> dropping {len(dropped)} unused tensor(s) "
              f"(vision tower and friends)")
    renamed = {v: k for k, v in mapping.items() if k != v}
    for new in sorted(renamed)[:4]:
        print(f"    {renamed[new]}\n      -> {new}")
    if len(renamed) > 4:
        print(f"    ... and {len(renamed) - 4} more")

    out_tensors = {mapping[k]: v for k, v in tensors.items() if k in mapping}

    # Names now line up; shapes may not. A checkpoint tensor whose axes are
    # ordered differently loads as MISMATCH and gets silently reinitialised,
    # which is the same failure as a missing weight.
    fixed_shapes, flagged, unfixable = [], [], []
    for name in sorted(out_tensors):
        if name not in want_shapes:
            continue
        adjusted, note = reconcile_shape(out_tensors[name], want_shapes[name])
        if note is None:
            continue
        if adjusted is None:
            unfixable.append((name, note))
        else:
            out_tensors[name] = adjusted
            (flagged if note.startswith("TRANSPOSE") else fixed_shapes).append((name, note))

    if fixed_shapes:
        print(f"\n  shape fixes : {len(fixed_shapes)} tensor(s) had their axes reordered")
        print(f"    {fixed_shapes[0][0]}\n      {fixed_shapes[0][1]}")
        if len(fixed_shapes) > 1:
            print(f"    ... and {len(fixed_shapes) - 1} more of the same form")
    if flagged:
        print(f"\n  ! {len(flagged)} tensor(s) needed a non-trivial transpose:")
        for name, note in flagged[:4]:
            print(f"    {name}: {note}")
    if unfixable:
        print(f"\n  ! {len(unfixable)} tensor(s) have irreconcilable shapes:")
        for name, note in unfixable[:4]:
            print(f"    {name}: {note}")
        if not args.force:
            raise SystemExit(
                "\nRefusing to write a checkpoint transformers would reinitialise. "
                "Re-run with --force to write it anyway."
            )

    still_missing = want - set(out_tensors)
    ignorable = {"lm_head.weight"} if tied else set()
    real_missing = {k for k in still_missing
                    if k not in ignorable and not k.endswith(("rotary_emb.inv_freq",))}

    print(f"\n  coverage : {len(set(out_tensors) & want)}/{len(want)} expected tensors")
    if real_missing:
        print(f"  ! {len(real_missing)} still unaccounted for: {sorted(real_missing)[:5]}")
        print("    The repaired checkpoint would still be partly untrained.")
        if not args.force:
            raise SystemExit(
                "\nRefusing to write an incomplete checkpoint. Re-run with --force "
                "to write it anyway."
            )
    else:
        print("  ! none missing - every expected tensor is accounted for")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    out = pathlib.Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"\n{out} already exists. Pass --force to overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"\n[5/5] Writing to {out}...")
    # Contiguity matters: safetensors refuses views that share storage.
    out_tensors = {k: v.contiguous() for k, v in out_tensors.items()}
    save_file(out_tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    # Copy the config verbatim so transformers builds the identical architecture.
    shutil.copy2(src / "config.json", out / "config.json")
    copied = []
    for name in SIDECAR_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
            copied.append(name)
    written = (out / "model.safetensors").stat().st_size
    print(f" -> model.safetensors ({written / 1e9:.2f} GB)")
    print(f" -> config.json + {len(copied)} tokenizer file(s)")

    print("\n" + bar)
    print("  ALL OK - checkpoint repaired")
    print(bar)
    print(f"   tensors : {len(out_tensors)} written, {len(dropped)} dropped"
          + (f", {len(fixed_shapes)} reshaped" if fixed_shapes else ""))
    print(f"   output  : {out}")
    print()
    print("   Verify it, then train against it:")
    print(f"     python check_teacher.py --teacher {out}")
    print(f"     ./distill.sh --profile finance --teacher {out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
