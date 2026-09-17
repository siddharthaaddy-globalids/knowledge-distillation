"""Print the versions that decide whether the next step can work.

    python scripts/_kd_versions.py pack
    python scripts/_kd_versions.py score

Printed before each billable step, so the log of a run that went wrong says
which torch, which transformers, and which packer were actually loaded. On a
rented pod that is the difference between "reproduce it" and "rent it again and
watch".

`cuda=False` is the line worth staring at: a CPU wheel does not fail, it runs
about a hundred times slower at the GPU's hourly rate.
"""

import sys

PROBES = {
    "pack": ("torch", "transformers", "llmcompressor", "compressed_tensors", "peft"),
    "score": ("torch", "transformers", "vllm"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    which = argv[0] if argv else "pack"
    if which not in PROBES:
        sys.exit(f"usage: _kd_versions.py [{' | '.join(PROBES)}]")

    import importlib

    parts = []
    for name in PROBES[which]:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the reason belongs in the log
            parts.append(f"{name} !! {type(exc).__name__}")
            continue
        parts.append(f"{name} {getattr(module, '__version__', '?')}")

    try:
        import torch

        parts.append(f"cuda={torch.cuda.is_available()}")
    except Exception:  # noqa: BLE001
        parts.append("cuda=?")

    print("   " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
