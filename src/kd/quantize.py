"""
Shrink the distilled student to 4-bit weights, as a checkpoint vLLM loads natively.

    python -m kd quantize --config configs/enlibra/enlibraQ3-14B.yaml
    kd pipeline --config ... --only quantize

W4A16 means INT4 weights and FP16 activations: the weights are packed four bits
to a value in groups of 128, and everything is dequantised back to FP16 to do
arithmetic with. Nothing about the activations or the KV cache changes.

WHY W4A16 AND NOT W8A8
----------------------
Single-stream decode is memory-bandwidth bound: the time goes into moving
weights from HBM to the compute units, not into the multiplications. Halving the
weight bytes twice over therefore makes decode FASTER, not slower, despite the
per-step cost of unpacking - which is the result the reference run measured
(+37% tokens/sec at 8B). W8A8 wins instead at large batch, where the arithmetic
dominates and INT8 tensor cores can be fed.

The arena is the large-batch case, so W8A8 would generate the held-out set
faster. It is the wrong thing to measure: what is being shipped is a model that
answers one person at a time, and the number worth reporting is that model's.

WHY lm_head IS LEFT ALONE
-------------------------
The output projection is vocabulary x hidden - 151,669 x 4,096 here, 1.2 GiB on
its own - and it is the one matrix whose error lands directly on the logits with
no further layer to absorb it. Quantising it costs accuracy out of proportion to
the bytes it saves, so it stays in bf16. That is why the checkpoint comes out
around 2.7x smaller rather than the ~3.5x four-bit weights would suggest.

WHAT IT RUNS ON
---------------
llm-compressor, writing the `compressed-tensors` format. Two reasons for that
over auto-gptq or autoawq: vLLM loads it with no conversion step and no extra
package, and the same library covers W8A8 and FP8 if the question ever changes.

GPTQ needs calibration data - a few hundred sequences to measure each layer's
activation statistics against. The right ones are the rows the student was
trained on, rendered through the same chat template, which is what
`calibration_rows` builds. Calibrating on generic web text would tune the
rounding for a distribution this model will never see.

IN A SUBPROCESS, like the vLLM engine and for the same reason: GPTQ holds a
Hessian per layer and llm-compressor leaves its hooks on the model, and the
stage that runs next needs a clean card. The operating system frees it
completely; a `del` and a `gc.collect()` do not.
"""

import json
import os
import subprocess
import sys

# Written beside the packed weights, naming what produced them. Read by the
# arena so a quantized player can say what it is, and by the report.
QUANT_STAMP = "kd-quant.json"

# Never quantised, for the reason in the header. A list rather than a constant
# because a different architecture may name it differently, but the default is
# right for every Llama- and Qwen-shaped model this pipeline trains.
DEFAULT_IGNORE = ["lm_head"]


def _say(log, message):
    if log is not None:
        log.info(message)
    else:
        print(message)


def is_quantized(where):
    """True when `where` is a checkpoint this module produced, weights and all."""
    where = str(where or "")
    return bool(where) and os.path.isfile(os.path.join(where, QUANT_STAMP)) \
        and os.path.isfile(os.path.join(where, "config.json")) \
        and any(name.endswith(".safetensors") for name in os.listdir(where))


def stamp_of(where):
    """What was recorded beside a quantized checkpoint, or {}."""
    path = os.path.join(str(where or ""), QUANT_STAMP)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:  # noqa: BLE001 - a truncated write is a reason to redo it
        return {}


def available():
    """True when llm-compressor can be imported in this interpreter."""
    import importlib.util

    return importlib.util.find_spec("llmcompressor") is not None


def unavailable_reason():
    """Why quantization cannot run here, in a sentence, or None if it can."""
    if available():
        return None
    if sys.platform == "win32":
        return ("llm-compressor needs a CUDA machine, so quantization cannot run "
                "on this one. Set quantization.enabled: false here and quantize "
                "on the Linux pod (docker/Dockerfile.cuda).")
    return ("llm-compressor is not installed. It is an optional extra, for the "
            "same reason vLLM is:\n"
            "      uv pip install --index-url "
            "https://download.pytorch.org/whl/cu128 "
            "--extra-index-url https://pypi.org/simple llmcompressor")


def calibration_rows(path, tokenizer, samples=128, max_length=2048, seed=42):
    """Calibration sequences, from the rows the student was trained on.

    Rendered through the SAME chat template training used and tokenised here, so
    what GPTQ measures its rounding against is the distribution the model will
    actually be asked to produce. See the header for why that matters.

    Sampled deterministically from across the file rather than taken from the
    front, because a curriculum is usually ordered - by hop depth here - and the
    first 128 rows of an ordered file are not a sample of it.
    """
    import random

    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} holds no rows to calibrate on")

    rng = random.Random(seed)
    chosen = records if len(records) <= samples else rng.sample(records, samples)

    texts = []
    for record in chosen:
        messages = record.get("messages") or []
        if not messages:
            continue
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return [tokenizer(text, truncation=True, max_length=int(max_length))
            for text in texts]


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #
def quantize(model, out_dir, calibration, scheme="W4A16", group_size=128,
             ignore=None, samples=128, max_seq_length=2048, dampening=0.01,
             log=None, reuse=True):
    """Pack `model` to `scheme` and write it to `out_dir`. Returns that path.

    `model` is a dense checkpoint - a directory or hub id, never base + adapter;
    `kd.merge.materialise` is what turns the second into the first.

    `reuse` returns an existing checkpoint when its stamp names this same source
    and settings, so `--from evaluation` does not re-quantise eight gigabytes it
    packed twenty minutes ago.
    """
    ignore = list(ignore if ignore is not None else DEFAULT_IGNORE)
    out_dir = os.path.abspath(str(out_dir))
    wanted = {"source": str(model), "scheme": scheme, "group_size": int(group_size),
              "ignore": ignore, "samples": int(samples),
              "max_seq_length": int(max_seq_length),
              "calibration": str(calibration)}
    if reuse and is_quantized(out_dir):
        found = stamp_of(out_dir)
        if all(found.get(k) == v for k, v in wanted.items()):
            _say(log, f"      quantized already: {out_dir}")
            return out_dir
        _say(log, f"      re-quantising: {out_dir} was packed from different "
                  f"settings")

    reason = unavailable_reason()
    if reason:
        raise RuntimeError(reason)

    _say(log, f"      packing {model}")
    _say(log, f"      to      {scheme}, group {group_size}, keeping "
              f"{', '.join(ignore)} at full width")
    _say(log, f"      calibrating on {samples} rows from {calibration}")

    env = dict(os.environ)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    # Output is inherited, not captured: GPTQ walks the layers one at a time and
    # prints as it goes, which on an 8B model is the only sign it is alive.
    command = [sys.executable, "-m", "kd.quantize", "--worker",
               "--model", str(model), "--out", out_dir,
               "--calibration", str(calibration), "--scheme", scheme,
               "--group-size", str(int(group_size)),
               "--samples", str(int(samples)),
               "--max-seq-length", str(int(max_seq_length)),
               "--dampening", str(float(dampening))]
    for name in ignore:
        command += ["--ignore", name]
    finished = subprocess.run(command, env=env)
    if finished.returncode != 0:
        raise RuntimeError(
            f"the quantization worker exited {finished.returncode}. Its output "
            f"is above; a killed worker is usually the card running out of "
            f"memory during the Hessian pass, which a smaller "
            f"quantization.max_seq_length reduces.")
    if not is_quantized(out_dir):
        raise RuntimeError(
            f"the quantization worker finished but {out_dir} holds no packed "
            f"checkpoint")
    return out_dir


def summarise(out_dir, source=None):
    """What the report says about the packed checkpoint: the config, and the sizes.

    Both sizes are measured from the files rather than computed from the scheme,
    because what is kept at full width (see DEFAULT_IGNORE) moves the ratio and
    a reader comparing "2.7x" against "4-bit" deserves the real number.
    """
    stamp = stamp_of(out_dir)

    def weight_bytes(where):
        if not (where and os.path.isdir(str(where))):
            return None
        return sum(os.path.getsize(os.path.join(str(where), name))
                   for name in os.listdir(str(where))
                   if name.endswith(".safetensors")) or None

    packed = weight_bytes(out_dir)
    dense = weight_bytes(source or stamp.get("source"))
    return {
        "path": str(out_dir),
        "scheme": stamp.get("scheme"),
        "group_size": stamp.get("group_size"),
        "ignore": stamp.get("ignore"),
        "calibration_samples": stamp.get("samples"),
        "format": stamp.get("format"),
        "bytes": packed,
        "dense_bytes": dense,
        "compression": (dense / packed) if (packed and dense) else None,
    }


# --------------------------------------------------------------------------- #
# Child side
# --------------------------------------------------------------------------- #
def _oneshot():
    """llm-compressor's entry point, wherever this release keeps it."""
    try:
        from llmcompressor import oneshot
    except ImportError:
        from llmcompressor.transformers import oneshot
    return oneshot


def run_worker(args):
    """Load, calibrate, pack, save. Everything heavy happens here."""
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = calibration_rows(args.calibration, tokenizer,
                               samples=args.samples,
                               max_length=args.max_seq_length)
    print(f"  calibration: {len(dataset)} sequences, "
          f"up to {args.max_seq_length} tokens each")

    recipe = GPTQModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=list(args.ignore or DEFAULT_IGNORE),
        dampening_frac=args.dampening,
    )
    _oneshot()(
        model=args.model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(dataset),
        output_dir=args.out,
    )

    # The tokenizer travels with the weights. A quantized directory without one
    # is something vLLM has to go looking elsewhere for, and "elsewhere" is the
    # base model's - which after a vocabulary trim is the wrong one.
    tokenizer.save_pretrained(args.out)

    with open(os.path.join(args.out, QUANT_STAMP), "w", encoding="utf-8") as handle:
        json.dump({
            "source": str(args.model),
            "scheme": args.scheme,
            "group_size": int(args.group_size),
            "ignore": list(args.ignore or DEFAULT_IGNORE),
            "samples": int(args.samples),
            "max_seq_length": int(args.max_seq_length),
            "calibration": str(args.calibration),
            "backend": "llmcompressor",
            "format": "compressed-tensors",
            "vllm_ready": True,
        }, handle, indent=2)
    return 0


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="kd quantize",
        description="Pack a distilled student to 4-bit weights for vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)   # the subprocess half
    parser.add_argument("-c", "--config", default=None, metavar="PATH",
                        help="Profile naming the adapter and the quantization "
                             "settings")
    parser.add_argument("--model", default=None, metavar="DIR",
                        help="Dense checkpoint to pack. Default: the distilled "
                             "student, merged from the adapter.")
    parser.add_argument("--adapter", default=None, metavar="DIR",
                        help="Adapter to merge and pack. Default: the newest "
                             "under project.runs_dir.")
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="Where the packed checkpoint goes")
    parser.add_argument("--calibration", default=None, metavar="PATH",
                        help="Rows to calibrate on. Default: the training file.")
    parser.add_argument("--scheme", default="W4A16")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dampening", type=float, default=0.01)
    parser.add_argument("--ignore", action="append", default=[],
                        help="A module never to quantise, repeatable "
                             f"(default: {', '.join(DEFAULT_IGNORE)})")
    args = parser.parse_args(argv)

    if args.worker:
        return run_worker(args)

    # ---- the by-hand front end ------------------------------------------- #
    import logging

    log = logging.getLogger("kd.quantize")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    from . import config as kd_config
    from . import merge, paths
    from .runlog import discover_adapters, is_adapter

    config = kd_config.load_config(args.config, use_env=False)
    settings = config.get("quantization") or {}

    model = args.model
    if not model:
        adapter = args.adapter
        if not adapter:
            found = discover_adapters(config["project"].get("runs_dir") or "./runs")
            adapter = found[0] if found else None
        if not adapter:
            log.error("xx  nothing to quantize: pass --model or --adapter, or "
                      "train something first")
            return 1
        adapter = paths.localise(paths.adapter_dir_of(adapter), config, log=log,
                                 label="adapter")
        if not is_adapter(adapter):
            log.error(f"xx  {adapter} has no adapter_config.json")
            return 1
        from transformers import AutoTokenizer

        base_id = paths.base_for_adapter(adapter, config["models"]["student"],
                                         log=log)
        tokenizer = AutoTokenizer.from_pretrained(
            adapter if os.path.isfile(
                os.path.join(str(adapter), "tokenizer_config.json")) else base_id)
        model = merge.materialise(paths.merged_cache(config, adapter), base_id,
                                  adapter, config=config, tokenizer=tokenizer,
                                  log=log)

    calibration = args.calibration or settings.get("calibration_file") \
        or (config.get("evaluation") or {}).get("arena_file")
    if not calibration:
        log.error("xx  nothing to calibrate on: pass --calibration, or set "
                  "quantization.calibration_file")
        return 1
    calibration = paths.localise(calibration, config, log=log,
                                 label="calibration set")

    out = args.out or os.path.join(str(model) + "-" + args.scheme.lower())
    try:
        written = quantize(
            model, out, calibration, scheme=args.scheme,
            group_size=args.group_size, ignore=args.ignore or None,
            samples=args.samples, max_seq_length=args.max_seq_length,
            dampening=args.dampening, log=log)
    except RuntimeError as exc:
        log.error(f"xx  {exc}")
        return 1

    facts = summarise(written, source=model)
    log.info("")
    log.info(f"  packed to {written}")
    if facts.get("compression"):
        log.info(f"    {facts['dense_bytes'] / 1e9:.2f} GB -> "
                 f"{facts['bytes'] / 1e9:.2f} GB  "
                 f"({facts['compression']:.1f}x smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
