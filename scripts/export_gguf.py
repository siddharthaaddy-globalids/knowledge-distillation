#!/usr/bin/env python3
# ===========================================================================
#  One stage of a GGUF build. Driven by scripts/gguf-pod.sh, runnable by hand.
#
#      python scripts/export_gguf.py --stage convert  --config <profile>
#      python scripts/export_gguf.py --stage imatrix  --config <profile>
#      python scripts/export_gguf.py --stage quantize --config <profile>
#
#  Every path comes from the profile - the same file `kd` reads, through the
#  same loader, `extends` chain and all. Nothing here is typed on a command
#  line, for the reason score-pod.sh gives about `--out`: a path spelled twice
#  is a path that drifts, and the failure is silent and hours late.
#
#  THE THREE STAGES, AND WHY THEY ARE THREE
#  ----------------------------------------
#      convert    adapter + base -> merged/ -> <run>-f16.gguf
#      imatrix    <run>-f16.gguf + calibration rows -> <run>.imatrix
#      quantize   <run>-f16.gguf + <run>.imatrix -> <run>-Q4_K_M.gguf
#
#  `convert_hf_to_gguf.py` does NOT quantize. It rewrites bf16 tensors into
#  GGUF's container and the file comes out BIGGER than the safetensors it read;
#  the 4-bit one is a second pass, `llama-quantize`, GGUF in and GGUF out. That
#  is also the only way to get a K-quant at all - the converter can emit
#  f32/f16/bf16/q8_0 and nothing else.
#
#  The split is not tidiness. `convert` merges a LoRA, so it wants torch,
#  transformers, peft and a card worth having. `quantize` is CPU-only and wants
#  none of them. Ship the f16 between the two and the expensive half can finish
#  on a rented pod that then gets destroyed, with the cheap half running days
#  later on a laptop. Hence gguf.keep_f16, and the `gguf-f16` upload group.
#
#  WHY IT MERGES RATHER THAN READING quantized/
#  --------------------------------------------
#  Because the destination is Q4_K_M. `quantized/` is already 4-bit - GPTQ, 16
#  levels per group of 128 - and Q4_K_M's grid is 16 levels per block of 32 with
#  quantized scales. The two do not align, so requantizing pays a second full
#  rounding error for no size saving. Converting from the dense merge pays one.
#
#  That argument is about the LOW end specifically. Converting `quantized/` to
#  Q6_K is fine and sometimes right - the finer grid nearly represents the
#  coarse one - and it is what you do when the dense weights genuinely no longer
#  exist. That is the standard fallback, not a hack: every GGUF of DeepSeek-R1
#  was made that way, because FP8 was all that was ever published. Here the
#  adapter exists and rebuilds the merge in CPU-minutes, so there is no reason
#  to take the lossier road.
#
#  WHAT IT DOES NOT DO
#  -------------------
#  Score the result. A Q4_K_M build is a different model from the W4A16 one the
#  arena measured - different algorithm, different rounding - so the report's
#  numbers do not describe it, and the stamp it writes says so in the artifact
#  itself. It enters the arena as its own player or it ships unmeasured; that is
#  a choice, and it should be made out loud.
# ===========================================================================

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Written beside the built GGUFs, naming what produced them. Deliberately the
# same shape as quantize.QUANT_STAMP: whoever finds this directory in the bucket
# six months from now should be able to say which adapter, which base and which
# calibration set it came from without opening a run bundle.
STAMP = "kd-gguf.json"

# What convert_hf_to_gguf.py can actually emit. Anything else has to come from
# llama-quantize, which is the whole reason this is two passes and not one.
CONVERTER_TYPES = {"f32", "f16", "bf16", "q8_0", "auto"}


def log(message=""):
    print(message, flush=True)


class _Log:
    """The .info(...) shape kd.paths and kd.merge expect from a run logger."""

    def info(self, message):
        log(message)


def die(message):
    raise SystemExit(f"xx  {message}")


# --------------------------------------------------------------------------- #
#  Where everything lives, worked out once
# --------------------------------------------------------------------------- #
class Plan:
    """Every path this build touches, derived from the profile and nothing else.

    Built before any stage runs so that all three agree without passing paths
    between processes: the pod script invokes this module once per stage, and
    two stages that disagreed about where the f16 is would fail by rebuilding
    it rather than by saying so.
    """

    def __init__(self, config, run_dir=None):
        from kd import paths

        self.config = config
        self.settings = config.get("gguf") or {}

        self.adapter = (config.get("evaluation") or {}).get("adapter")
        if not self.adapter:
            die("the profile sets no evaluation.adapter. This builds a GGUF of "
                "an adapter that is already in the bucket;\n    name it there, "
                "so that every stage refers to the same weights.")

        # The run id is embedded in the adapter URI, exactly as score-pod.sh
        # reads it: <bundle>/<run-id>/final_adapter.
        trimmed = str(self.adapter).replace("\\", "/").rstrip("/")
        self.run_id = os.path.basename(os.path.dirname(trimmed)) or "gguf-export"

        if run_dir:
            self.run_dir = os.path.abspath(os.path.expanduser(run_dir))
        else:
            runs = os.environ.get("KD_RUNS_DIR") or os.path.join(
                os.path.dirname(paths.cache_dir(config)), "runs")
            self.run_dir = os.path.join(os.path.abspath(
                os.path.expanduser(runs)), self.run_id)

        self.f16_dir = self._dir("f16_dir", "gguf-f16")
        self.out_dir = self._dir("output_dir", "gguf")
        self.merged_dir = os.path.join(self.run_dir, "merged")

        # Named for the run, not for the model, so that two builds of two
        # adapters cannot land on the same filename in the same bucket.
        self.f16 = os.path.join(self.f16_dir, f"{self.run_id}-f16.gguf")
        # Beside the f16 rather than beside the quants, because it is an INPUT
        # to quantizing: whoever downloads the gguf-f16 group to quantize
        # elsewhere needs this in the same download or the imatrix silently
        # does not get used.
        self.imatrix = os.path.join(self.f16_dir, f"{self.run_id}.imatrix")
        self.corpus = os.path.join(self.f16_dir,
                                   f"{self.run_id}-imatrix-calibration.txt")

        self.quants = list(self.settings.get("quants") or ["Q4_K_M"])
        self.want_imatrix = bool(self.settings.get("imatrix"))
        self.keep_f16 = bool(self.settings.get("keep_f16"))

        # Null means "the same rows GPTQ calibrates on", which is the point:
        # two packings of one adapter measured against different data are not
        # comparable and nothing would say so.
        self.calibration = (self.settings.get("calibration_file")
                            or (config.get("quantization") or {}).get(
                                "calibration_file"))
        self.samples = int(self.settings.get("calibration_samples") or 128)
        self.chunks = int(self.settings.get("imatrix_chunks") or 128)

    def _dir(self, key, default_name):
        configured = self.settings.get(key)
        if configured:
            return os.path.abspath(os.path.expanduser(str(configured)))
        return os.path.join(self.run_dir, default_name)

    def quant_file(self, quant):
        return os.path.join(self.out_dir, f"{self.run_id}-{quant}.gguf")

    def report(self):
        log(f"    run        : {self.run_id}")
        log(f"    adapter    : {self.adapter}")
        log(f"    f16        : {self.f16}")
        if self.want_imatrix:
            log(f"    imatrix    : {self.imatrix}")
            log(f"    calibrating: {self.calibration}")
        log(f"    quants     : {', '.join(self.quants)} -> {self.out_dir}")


# --------------------------------------------------------------------------- #
#  llama.cpp
# --------------------------------------------------------------------------- #
def _binary_names(stem):
    """Every place a llama.cpp build puts `stem`, newest layout first.

    cmake writes build/bin/ on Unix and build/bin/Release/ on Windows with the
    default multi-config generator; the pre-2024 Makefile build wrote the binary
    straight into the repo root. All three are still in the wild.
    """
    exe = ".exe" if platform.system() == "Windows" else ""
    return [os.path.join("build", "bin", stem + exe),
            os.path.join("build", "bin", "Release", stem + exe),
            os.path.join("build", stem + exe),
            stem + exe]


def find_llama_cpp(configured, need):
    """The convert script and whichever binaries `need` names, or a clear failure.

    Checked BEFORE the eight-gigabyte merge, always. A missing binary found
    forty minutes into a conversion is forty minutes of a rented card.
    """
    candidates = [configured, os.environ.get("LLAMA_CPP"),
                  "./llama.cpp", "../llama.cpp", "/workspace/llama.cpp"]
    roots, seen = [], set()
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(str(candidate)))
        if path not in seen:
            seen.add(path)
            roots.append(path)

    for root in roots:
        convert = os.path.join(root, "convert_hf_to_gguf.py")
        if not os.path.isfile(convert):
            continue
        found = {}
        for stem in ("llama-quantize", "llama-imatrix"):
            for relative in _binary_names(stem):
                path = os.path.join(root, relative)
                if os.path.isfile(path):
                    found[stem] = path
                    break
        missing = [stem for stem in need if stem not in found]
        if missing:
            die(f"{root} has convert_hf_to_gguf.py but not {', '.join(missing)}.\n"
                f"    The converter is a script and ships with the checkout; the "
                f"binaries are compiled and do not:\n"
                f"      cmake -B build -S . && cmake --build build --config "
                f"Release -j\n"
                f"    scripts/gguf-pod.sh setup does this for you.")
        return convert, found

    die("no llama.cpp checkout found (no convert_hf_to_gguf.py).\n"
        f"    looked in: {', '.join(roots)}\n"
        "    Set gguf.llama_cpp in the profile, export LLAMA_CPP, or run "
        "scripts/gguf-pod.sh setup.")


def run(command, label):
    """A subprocess whose output is INHERITED, not captured.

    Same reason quantize.quantize gives for the GPTQ worker: conversion and
    quantization walk the tensors one at a time printing as they go, and on an
    8 GB file that progress is the only sign the thing is alive.
    """
    log(f"\n>>  {label}")
    log(f"    {' '.join(str(part) for part in command)}\n")
    started = time.time()
    finished = subprocess.run([str(part) for part in command])
    if finished.returncode != 0:
        die(f"{label} exited {finished.returncode}. Its output is above.")
    log(f"\n    done in {time.time() - started:.0f}s")


def _gb(path):
    return os.path.getsize(path) / 1e9


# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
def stage_convert(plan, logger):
    """adapter + base -> dense merge -> f16 GGUF. The pass that needs the card."""
    from transformers import AutoTokenizer

    from kd import merge, paths

    convert_script, _ = find_llama_cpp(plan.settings.get("llama_cpp"), [])

    if os.path.isfile(plan.f16):
        log(f"    already converted: {plan.f16} ({_gb(plan.f16):.2f} GB)")
        log("    delete it to rebuild.")
        return 0

    adapter = paths.localise(paths.adapter_dir_of(plan.adapter), plan.config,
                             log=logger, label="adapter")
    base_id = paths.base_for_adapter(
        adapter, (plan.config.get("models") or {}).get("student"), log=logger)
    if not base_id:
        die("cannot tell which base this adapter was trained on: it records "
            "none and the profile names no models.student.")

    # The adapter's own tokenizer when it saved one, because a resized vocab
    # lives there and not in the base. Same test kd.pipeline makes at :463.
    source = adapter if os.path.isfile(
        os.path.join(str(adapter), "tokenizer_config.json")) else base_id
    tokenizer = AutoTokenizer.from_pretrained(source)

    # kd.merge.materialise, not a local reimplementation, because of the
    # vocabulary trap scripts/merge.py documents at length: a student trimmed to
    # a teacher's real vocab width needs its base trimmed the same way before
    # PEFT will load the adapter at all, and there must be exactly one answer to
    # how wide that is.
    log(f"\n>>  merging into {base_id}")
    merged = merge.materialise(plan.merged_dir, base_id, adapter,
                               config=plan.config, tokenizer=tokenizer,
                               log=logger)

    os.makedirs(plan.f16_dir, exist_ok=True)
    # f16 rather than bf16: llama.cpp's CPU and Metal paths are built around
    # f16, and this is a container change, not a precision decision - the
    # quantize pass is where bits actually get thrown away.
    run([sys.executable, convert_script, str(merged),
         "--outfile", plan.f16, "--outtype", "f16"],
        "convert to GGUF (container only - nothing is quantized here)")

    if not os.path.isfile(plan.f16):
        die(f"the converter finished but {plan.f16} does not exist.")
    log(f"    {os.path.basename(plan.f16)}  ({_gb(plan.f16):.2f} GB)")

    # transformers 5.x writes the chat template to its own file. The converter
    # embeds what it finds, but a GGUF that ended up without one is not obvious
    # until the model answers in the wrong shape - and this curriculum's answers
    # ARE shaped, they end in an <Answer> tag. Carrying the .jinja alongside
    # makes the fallback one flag away:
    #     llama-server --chat-template-file <run>-chat_template.jinja
    template = os.path.join(str(merged), "chat_template.jinja")
    if os.path.isfile(template):
        shutil.copy2(template, os.path.join(
            plan.f16_dir, f"{plan.run_id}-chat_template.jinja"))
    else:
        log(" !! no chat_template.jinja in the merge. If the model answers in "
            "the wrong shape,\n    that is the first thing to check.")
    return 0


def stage_imatrix(plan, logger):
    """The importance matrix, from the rows the student was trained on.

    quantize.calibration_texts does the choosing and the rendering - the SAME
    function GPTQ calls - so the matrix and a W4A16 packing of this adapter are
    measured against the same sample of the same file through the same chat
    template. Two calibration sets would make the two artifacts incomparable,
    quietly, which is the one kind of wrong nobody notices.
    """
    from transformers import AutoTokenizer

    from kd import paths, quantize

    if not plan.want_imatrix:
        log("    gguf.imatrix is false - skipping.")
        return 0
    if not os.path.isfile(plan.f16):
        die(f"no f16 GGUF at {plan.f16}. Run the convert stage first.")
    if os.path.isfile(plan.imatrix):
        log(f"    already built: {plan.imatrix}")
        return 0
    if not plan.calibration:
        die("gguf.imatrix is true but neither gguf.calibration_file nor "
            "quantization.calibration_file names a file.")

    _, binaries = find_llama_cpp(plan.settings.get("llama_cpp"),
                                 ["llama-imatrix"])

    calibration = paths.localise(plan.calibration, plan.config, log=logger,
                                 label="calibration set")
    # The merge is the tokenizer that matches the weights; the adapter is the
    # fallback when the merge has already been deleted.
    source = plan.merged_dir if os.path.isfile(
        os.path.join(plan.merged_dir, "tokenizer_config.json")) else \
        paths.localise(paths.adapter_dir_of(plan.adapter), plan.config,
                       log=logger, label="adapter")
    tokenizer = AutoTokenizer.from_pretrained(source)

    texts = quantize.calibration_texts(calibration, tokenizer,
                                       samples=plan.samples)
    os.makedirs(plan.f16_dir, exist_ok=True)
    with open(plan.corpus, "w", encoding="utf-8") as handle:
        # Blank line between rows: llama-imatrix chunks on token count and not
        # on rows, so without a separator the end of one answer runs into the
        # next question as though they were one sequence.
        handle.write("\n\n".join(texts))
    log(f"    {len(texts)} rows -> {os.path.basename(plan.corpus)} "
        f"({os.path.getsize(plan.corpus) / 1e6:.1f} MB)")

    run([binaries["llama-imatrix"], "-m", plan.f16, "-f", plan.corpus,
         "-o", plan.imatrix, "--chunks", str(plan.chunks)],
        "importance matrix")
    if not os.path.isfile(plan.imatrix):
        die(f"llama-imatrix finished but {plan.imatrix} does not exist.")
    return 0


def stage_quantize(plan, logger):
    """f16 GGUF -> the configured quants. CPU only: no torch, no adapter, no card."""
    if not os.path.isfile(plan.f16):
        die(f"no f16 GGUF at {plan.f16}.\n"
            f"    Run the convert stage, or - if it was built on a pod that is "
            f"gone - fetch the gguf-f16\n"
            f"    group of this run's bundle into {plan.f16_dir}.")

    _, binaries = find_llama_cpp(plan.settings.get("llama_cpp"),
                                 ["llama-quantize"])

    imatrix = plan.imatrix if os.path.isfile(plan.imatrix) else None
    if plan.want_imatrix and not imatrix:
        # A warning and not a failure: quantizing without it works and is
        # sometimes what you want. Silence would be wrong though, because the
        # result is a worse model that looks exactly like the intended one.
        log(" !! gguf.imatrix is true but no matrix was found at")
        log(f"    {plan.imatrix}")
        log("    Quantizing WITHOUT it. At Q4 that costs real accuracy, and "
            "the IQ quants require it.")

    os.makedirs(plan.out_dir, exist_ok=True)
    built = []
    for quant in plan.quants:
        if quant.lower() in CONVERTER_TYPES:
            log(f" !! {quant} is a converter output type, not a llama-quantize "
                f"type. Building it anyway;\n    --outtype would have been "
                f"cheaper.")
        target = plan.quant_file(quant)
        if os.path.isfile(target):
            log(f"    already built: {os.path.basename(target)} "
                f"({_gb(target):.2f} GB)")
        else:
            command = [binaries["llama-quantize"]]
            if imatrix:
                command += ["--imatrix", imatrix]
            command += [plan.f16, target, quant]
            run(command, f"quantize -> {quant}")
            if not os.path.isfile(target):
                die(f"llama-quantize finished but {target} does not exist.")
        built.append({"quant": quant, "file": os.path.basename(target),
                      "bytes": os.path.getsize(target)})

    # The chat template travels with the quants too, not only with the f16:
    # these are the files someone actually downloads to run the model.
    template = os.path.join(plan.f16_dir, f"{plan.run_id}-chat_template.jinja")
    if os.path.isfile(template):
        shutil.copy2(template, os.path.join(plan.out_dir,
                                            "chat_template.jinja"))

    with open(os.path.join(plan.out_dir, STAMP), "w", encoding="utf-8") as handle:
        json.dump({
            "run": plan.run_id,
            "adapter": str(plan.adapter),
            "quants": built,
            "imatrix": bool(imatrix),
            "calibration": str(plan.calibration) if imatrix else None,
            "calibration_samples": plan.samples if imatrix else None,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # Said out loud in the artifact itself, because the number someone
            # will reach for is the arena's and the arena did not measure this.
            "note": ("Q_K/IQ quants are llama.cpp's algorithm, not GPTQ. This "
                     "is a different model from the W4A16 checkpoint the arena "
                     "scored; the report's numbers do not describe it."),
        }, handle, indent=2)

    log("\n>>  built")
    for entry in built:
        log(f"      {entry['file']}  ({entry['bytes'] / 1e9:.2f} GB)")
    log(f"    in {plan.out_dir}")

    if not plan.keep_f16:
        os.remove(plan.f16)
        log("      removed the f16 intermediate (gguf.keep_f16 is false)")
    return 0


STAGES = {"convert": stage_convert, "imatrix": stage_imatrix,
          "quantize": stage_quantize}


# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="One stage of a GGUF build. See scripts/gguf-pod.sh.")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES),
                        help="which pass to run")
    parser.add_argument("--config", required=True,
                        help="a kd profile with a gguf block - see "
                             "configs/enlibra/enlibraQ3-14B-to-4B-gguf.yaml")
    parser.add_argument("--run-dir",
                        help="override where the build lands (default: the run "
                             "directory derived from evaluation.adapter)")
    parser.add_argument("--quants",
                        help="quantize only. Comma-separated, overriding "
                             "gguf.quants: Q4_K_M,Q6_K")
    parser.add_argument("--plan", action="store_true",
                        help="print every resolved path and exit, running nothing")
    args = parser.parse_args(argv)

    from kd.config import load_config

    config = load_config(args.config, use_env=True)
    if not (config.get("gguf") or {}).get("enabled"):
        die(f"gguf.enabled is false in {args.config}. Nothing here builds a "
            f"GGUF for a profile that did not ask for one.")

    plan = Plan(config, args.run_dir)
    if args.quants:
        # Overrides the profile for THIS invocation only, and deliberately does
        # not touch the stamp's idea of what the profile says - the stamp
        # records what was actually built, which is this list.
        plan.quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    log(f"\n>>  {args.stage}")
    plan.report()
    if args.plan:
        log("\n    --plan: nothing run.")
        return 0

    return STAGES[args.stage](plan, _Log())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        # kd.paths and kd.remote.s3 raise RuntimeError with a message already
        # written for a person - a missing extra, a denied bucket. A traceback
        # on top of one of those buries the sentence that says what to do.
        raise SystemExit(f"xx  {exc}")
    except KeyboardInterrupt:
        raise SystemExit("\nxx  interrupted. Partial files are in the run dir.")
