"""Measure how much of the teacher actually transferred into the distilled student.

Two questions, measured separately, because the distillation literature is explicit
that they do not track each other (Stanton et al., "Does Knowledge Distillation
Really Work?", NeurIPS 2021):

  FIDELITY    how closely does the student reproduce the TEACHER's predictions?
              -> top-1 agreement rate, KL(teacher || student)

  CAPABILITY  is the student actually better at the task than it was?
              -> held-out perplexity on the held-out split

Both are reported for the BASE student and the DISTILLED student against the same
teacher, on the same held-out split. The base student column is what makes the
numbers mean anything: without it there is no way to tell "distillation worked"
apart from "the small model could already do this".

A fourth column, the TEACHER'S OWN BASE - the stock model it was fine-tuned from
- is measured the same way when it is known (models.teacher_base, or a teacher
that is base + adapter) and `evaluation.players` names it. It says what the
fine-tune bought the teacher: how far the teacher moved from the stock model is
the most there was to distil, and a student that agrees with the teacher no more
than the stock base does has learned nothing the base did not already know. It
is loaded in a second pass after the student is freed, so peak memory is two
teachers rather than three models.

    kd evaluate --config configs/qwen/finance.yaml
    kd evaluate --config configs/qwen/finance.yaml --dtype bfloat16 --samples 100
    kd evaluate --config configs/enlibra/enlibraQ3-14B.yaml --quantized ./packed
    kd evaluate --config configs/smollm/mac.yaml --json results.json

The held-out split is rebuilt with the training seed, so it is exactly the split
the student never trained on.

Exit status is 0 when the distilled student improved on the base student, and 3
when it did not, so a run can be gated in a shell script.
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config, resolve_device
from .report import training_settings, write_report
from .runlog import discover_adapters, is_adapter

BAR = "=" * 78

DTYPES = {
    "auto": None,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Measure teacher->student transfer: fidelity and capability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("-c", "--config", default="configs/smollm/default.yaml",
                    help="Training config the adapter was produced from "
                         "(default: configs/smollm/default.yaml)")
    ap.add_argument("-a", "--adapter", default=None,
                    help="Adapter directory to evaluate "
                         "(default: the newest adapter under project.runs_dir)")
    ap.add_argument("-t", "--teacher", default=None, help="Override models.teacher")
    ap.add_argument("-s", "--student", default=None, help="Override models.student")
    ap.add_argument("--teacher-adapter", default=None,
                    help="LoRA adapter merged into the teacher, when the teacher is "
                         "a base model plus an adapter rather than a merged checkpoint")
    ap.add_argument("--teacher-base", default=None,
                    help="The stock model the teacher was fine-tuned from, scored as "
                         "a fourth column (default: models.teacher_base, or the "
                         "base of a teacher given as base + adapter)")
    ap.add_argument("--no-teacher-base", action="store_true",
                    help="Skip the fourth column even when the base is known")
    ap.add_argument("-n", "--samples", type=int, default=50,
                    help="Held-out samples to score (default: 50)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"],
                    help="Override hardware.device")
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES),
                    help="Override hardware.dtype (default: auto)")
    ap.add_argument("--no-generations", action="store_true",
                    help="Skip the qualitative side-by-side generations")
    ap.add_argument("--quantized", default=None, metavar="DIR",
                    help="Packed checkpoint to score alongside the dense "
                         "student, on the SAME tokens - which is what isolates "
                         "the cost of quantization from the cost of "
                         "distillation. Default: whatever `kd quantize` wrote.")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="Write all metrics to a JSON file (machine-readable)")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="Write a readable report. The extension picks the format: "
                         ".md for Markdown, .html for a self-contained page you can "
                         "open in a browser or send to someone.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Held-out split
# --------------------------------------------------------------------------- #
def build_eval_samples(config, hardware, tokenizer, limit):
    """Rebuild the exact held-out split the student was never trained on.

    kd.data.build_datasets() is reused rather than reimplemented: it is seeded
    from project.seed and applies the same length filter and dedup, so the split
    here is byte-identical to the one training held out. Reimplementing it would
    silently drift the moment either side changed.
    """
    from .data import build_datasets

    _, eval_dataset = build_datasets(tokenizer, config)
    rows = eval_dataset.select(range(min(limit, len(eval_dataset))))
    return [row["messages"] for row in rows]


def encode(tokenizer, messages, device):
    """Tokenize one sample and return (input_ids, prompt_token_count).

    The prompt/completion boundary is computed exactly as the GKD collator sees it:
    messages[:-1] rendered with a generation prompt is the prompt, the full turn
    list is prompt + completion. Only completion positions are scored.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)

    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    ids = tokenizer(full_text, add_special_tokens=False,
                    return_tensors="pt").input_ids.to(device)
    return ids, prompt_len


def completion_slice(logits, ids, prompt_len):
    """Return (logits_at_completion_positions, target_ids).

    Causal shift: logits[:, t] predicts token t+1, so the distribution over
    completion token `t` lives at logit index `t - 1`.
    """
    length = ids.shape[1]
    if prompt_len < 1 or prompt_len >= length:
        return None, None
    return logits[0, prompt_len - 1:length - 1, :], ids[0, prompt_len:length]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def top_k_overlap(teacher_logits, student_logits, k=5, chunk=32):
    """Mean size of the intersection of the two top-k sets, as a fraction of k.

    Top-1 agreement is brittle at exactly the positions that matter least: where
    the teacher is genuinely uncertain between two near-tied tokens, the student
    can be doing the right thing and still "disagree" on every one of them. The
    overlap of the top-5 sets moves smoothly where top-1 flips, so a student that
    tracks the teacher's shortlist without matching its argmax shows up here as
    close rather than as wrong.
    """
    positions = teacher_logits.shape[0]
    if not positions:
        return None
    total = 0.0
    for i in range(0, positions, chunk):
        t = teacher_logits[i:i + chunk].topk(k, dim=-1).indices
        s = student_logits[i:i + chunk].topk(k, dim=-1).indices
        # Set intersection per row, without building a Python set per position:
        # a [rows, k, 1] against a [rows, 1, k] comparison is the same question
        # and stays on the device the logits are already on.
        total += float((t.unsqueeze(-1) == s.unsqueeze(-2)).any(-1).sum())
    return total / (positions * k)


def compare_distributions(teacher_logits, student_logits, chunk=32):
    """Top-1 agreement count and summed KL(teacher || student) over positions.

    Chunked and cast to float32 per chunk: a full [positions, 248k] float32 tensor
    is hundreds of megabytes at these vocabulary sizes, which is enough to push a
    16 GB machine into swap during what is supposed to be the cheap step.
    """
    positions = teacher_logits.shape[0]
    kl_total, agree = 0.0, 0
    for i in range(0, positions, chunk):
        t = teacher_logits[i:i + chunk].float()
        s = student_logits[i:i + chunk].float()
        t_log = F.log_softmax(t, dim=-1)
        s_log = F.log_softmax(s, dim=-1)
        # F.kl_div(input, target) computes KL(target || input), i.e. the argument
        # order is the reverse of the mathematical convention - the same quirk TRL
        # works around in generalized_jsd_loss. Passing (student, teacher) here is
        # therefore KL(teacher || student), which is what fidelity means: how much
        # information is lost when the student stands in for the teacher.
        kl_total += float(F.kl_div(s_log, t_log, log_target=True, reduction="sum"))
        agree += int((t.argmax(-1) == s.argmax(-1)).sum())
    return agree, kl_total


def summed_nll(logits, targets, chunk=32):
    """Summed negative log-likelihood of the reference completion."""
    total = 0.0
    for i in range(0, logits.shape[0], chunk):
        total += float(F.cross_entropy(
            logits[i:i + chunk].float(), targets[i:i + chunk], reduction="sum"))
    return total


def measure_latency(model, tokenizer, prompt, device, new_tokens=32):
    """Indicative decode throughput, tokens/second. Single greedy run."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(device)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():  # warm up kernels/allocator so the timed run is representative
        model.generate(**ids, max_new_tokens=4, do_sample=False, pad_token_id=pad)
    start = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                             pad_token_id=pad)
    elapsed = time.time() - start
    produced = out.shape[1] - ids.input_ids.shape[1]
    return (produced / elapsed) if elapsed > 0 else float("nan")


def generate(model, tokenizer, prompt, device, new_tokens=64):
    """Greedy generation. Deterministic on purpose: sampled output is not evidence."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=new_tokens, do_sample=False,
                             repetition_penalty=1.1,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# The teacher's own base, scored against the teacher
# --------------------------------------------------------------------------- #
def score_teacher_base(base_id, teacher, teacher_id, tokenizer, samples, device,
                       dtype, config):
    """Agreement with, KL from, and perplexity beside the fine-tuned teacher.

    The same measurement the student gets, applied to the model the teacher
    was fine-tuned from. Returns {agreement_pct, kl, perplexity, params,
    generations}, or None when nothing could be scored.

    Loaded here rather than alongside the student because it is the teacher's
    size, and the caller frees the student first. The base is trimmed to the
    teacher's output width the way the student is - a stock checkpoint pads
    its vocabulary for alignment and the fine-tune may have trimmed it - so a
    genuine vocabulary mismatch is the only reason the comparison is refused.
    """
    from . import paths

    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=dtype, low_cpu_mem_usage=True)
    try:
        paths.fit_vocab(model, paths.matched_width(model, teacher, len(tokenizer)),
                        label="teacher base")
    except ValueError as exc:
        print(f" !! {exc}")
        return None
    model = model.to(device).eval()

    tokens, agree, kl, nll = 0, 0, 0.0, 0.0
    for messages in samples:
        ids, prompt_len = encode(tokenizer, messages, device)
        with torch.no_grad():
            t_logits, targets = completion_slice(teacher(ids).logits, ids, prompt_len)
            if t_logits is None or t_logits.shape[0] == 0:
                continue
            b_logits, _ = completion_slice(model(ids).logits, ids, prompt_len)
        if t_logits.shape[-1] != b_logits.shape[-1]:
            print(f" !! the teacher's base has a {b_logits.shape[-1]}-wide vocabulary "
                  f"against the teacher's {t_logits.shape[-1]}; not comparable")
            return None
        tokens += t_logits.shape[0]
        a, k = compare_distributions(t_logits, b_logits)
        agree += a
        kl += k
        nll += summed_nll(b_logits, targets)
        del t_logits, b_logits
    if not tokens:
        return None

    generations = {}
    for prompt in (config.get("benchmark_prompts") or [])[:3]:
        generations[prompt] = generate(model, tokenizer, prompt, device)
    params = sum(p.numel() for p in model.parameters())
    del model
    gc.collect()
    return {"agreement_pct": agree / tokens * 100, "kl": kl / tokens,
            "perplexity": math.exp(nll / tokens), "params": params,
            "generations": generations}


# --------------------------------------------------------------------------- #
# The packed student, on the same tokens
# --------------------------------------------------------------------------- #
def score_quantized(where, scored_ids, tokenizer, device, dtype, probe, config):
    """Perplexity and throughput for the packed student, or None.

    THE SAME TOKENS the dense student was scored on, replayed from `scored_ids`.
    That is what makes the difference a measurement of quantization rather than
    of two slightly different held-out splits - and it is why the ids are kept
    through the whole run instead of being rebuilt here.

    Absent rather than fatal: a run with nothing packed reports every other
    number unchanged.
    """
    from . import quantize

    if not (where and quantize.is_quantized(where)):
        return None

    print(f"\n  Scoring the packed student on the same {len(scored_ids)} samples...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(where), dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    except Exception as exc:  # noqa: BLE001 - a missing compressed-tensors, a bad dir
        print(f"  !! could not load {where}: {exc}")
        print("     compressed-tensors is what reads this format:  "
              "uv pip install compressed-tensors")
        return None

    nll, tokens = 0.0, 0
    with torch.no_grad():
        for ids, prompt_len in scored_ids:
            ids = ids.to(device)
            logits, targets = completion_slice(model(ids).logits, ids, prompt_len)
            if logits is None or logits.shape[0] == 0:
                continue
            nll += summed_nll(logits, targets)
            tokens += logits.shape[0]
    if not tokens:
        return None

    result = {
        "path": str(where),
        "tokens": tokens,
        "nll": nll / tokens,
        "perplexity": math.exp(nll / tokens),
        "tok_per_s": measure_latency(model, tokenizer, probe, device),
    }
    result.update({k: v for k, v in quantize.summarise(where).items()
                   if k in ("scheme", "group_size", "ignore", "format",
                            "calibration_samples", "bytes", "dense_bytes",
                            "compression")})
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def quantization_delta(packed, ppl_dense, nll_dense, tps_dense, tokens):
    """What 4-bit weights cost, as differences on identical tokens.

    Perplexity as a PERCENTAGE change and NLL as an absolute one, deliberately:
    perplexity is exponential so its absolute movement means nothing without the
    base, and NLL is the quantity that is actually linear in the error being
    added. Quoting only one of them hides that.
    """
    ppl_packed = packed["perplexity"]
    return {
        **packed,
        "dense_perplexity": ppl_dense,
        "dense_nll": nll_dense,
        "dense_tok_per_s": tps_dense,
        "scored_tokens": tokens,
        "perplexity_change_pct": ((ppl_packed - ppl_dense) / ppl_dense * 100)
        if ppl_dense else None,
        "nll_change": packed["nll"] - nll_dense,
        "throughput_change_pct": ((packed["tok_per_s"] - tps_dense) / tps_dense * 100)
        if tps_dense else None,
    }


def report_quantization(q):
    """The quantization block, for a terminal."""
    print("\n" + BAR)
    print(f"  QUANTIZATION - what {q.get('scheme') or '4-bit'} cost")
    print(BAR)
    print(f"  Both students on the identical {q['scored_tokens']} completion "
          f"tokens.\n")
    print(f"  {'measure':28} {'dense':>12} {'packed':>12} {'change':>12}")
    print("  " + "-" * 66)
    print(f"  {'perplexity':28} {q['dense_perplexity']:12.4f} "
          f"{q['perplexity']:12.4f} {q['perplexity_change_pct']:+11.2f}%")
    print(f"  {'negative log-likelihood':28} {q['dense_nll']:12.4f} "
          f"{q['nll']:12.4f} {q['nll_change']:+12.4f}")
    if q.get("dense_bytes") and q.get("bytes"):
        print(f"  {'on disk':28} {q['dense_bytes'] / 2**30:11.1f}G "
              f"{q['bytes'] / 2**30:11.1f}G {q['compression']:11.1f}x")
    if q.get("dense_tok_per_s") and q.get("tok_per_s"):
        print(f"  {'decode throughput (tok/s)':28} {q['dense_tok_per_s']:12.2f} "
              f"{q['tok_per_s']:12.2f} {q['throughput_change_pct']:+11.1f}%")
    print("\n  Single-stream decode is memory-bandwidth bound, so at this size")
    print("  4-bit weights are usually FASTER than bf16 despite the unpacking.")


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def main(args=None):
    # The pipeline calls this with a prepared Namespace rather than through argparse,
    # so the stage and the standalone command run exactly the same code.
    args = args or parse_args()
    config = load_config(args.config)
    if args.device:
        config.setdefault("hardware", {})["device"] = args.device
    if args.dtype != "auto":
        config.setdefault("hardware", {})["dtype"] = args.dtype

    hardware = resolve_device(config)
    device = hardware["device"]
    dtype = DTYPES[args.dtype] or hardware["dtype"]
    dtype_name = str(dtype).replace("torch.", "")

    from . import paths

    # s3:// parts fetched, and a models.teacher that is really a LoRA adapter
    # split into base + adapter - unless the command line named the teacher
    # itself, in which case it is taken exactly as given.
    if not (args.teacher or args.teacher_adapter):
        try:
            paths.resolve_teacher(config)
        except RuntimeError as exc:
            raise SystemExit(f"xx  {exc}") from exc
    teacher_id = args.teacher or config["models"]["teacher"]
    student_id = args.student or config["models"]["student"]
    teacher_adapter = args.teacher_adapter or config["models"].get("teacher_adapter")

    # The fourth column. Known from the command line, from the config, or from
    # the teacher being base + adapter; wanted when evaluation.players says so.
    players = list((config.get("evaluation") or {}).get("players")
                   or ["base", "distilled", "teacher-base", "teacher"])
    teacher_base_id = None
    if not getattr(args, "no_teacher_base", False) and (
            getattr(args, "teacher_base", None) or "teacher-base" in players):
        teacher_base_id = getattr(args, "teacher_base", None) or paths.teacher_base_of(config)
        if teacher_base_id and paths.is_remote(teacher_base_id):
            teacher_base_id = paths.localise(teacher_base_id, config, label="teacher base")

    # Explicit --adapter wins; then a pinned project.output_dir; otherwise the newest
    # adapter any run produced. Falling back to the newest run is what makes
    # "train, then evaluate" work without having to copy a run id between commands.
    adapter_dir = args.adapter
    if not adapter_dir and config["project"].get("output_dir"):
        adapter_dir = os.path.join(config["project"]["output_dir"], "final_adapter")
    if not adapter_dir:
        found = discover_adapters(config["project"].get("runs_dir") or "./runs")
        adapter_dir = found[0] if found else None
    if not adapter_dir or not is_adapter(adapter_dir):
        raise SystemExit(
            f"No adapter found at {adapter_dir or config['project'].get('runs_dir')}\n"
            f"Train one first (kd train --config ...), or point --adapter at the "
            f"directory holding adapter_config.json."
        )

    tokenizer_choice = str(config["models"].get("tokenizer", "teacher"))
    tokenizer_id = {"teacher": teacher_id, "student": student_id}.get(
        tokenizer_choice, tokenizer_choice)

    print(BAR)
    print("  Distillation evaluation")
    print(BAR)
    print(f"   teacher   : {teacher_id}" + (f"  + {teacher_adapter}" if teacher_adapter else ""))
    if teacher_base_id:
        print(f"   its base  : {teacher_base_id}  (scored as a fourth column)")
    print(f"   student   : {student_id}")
    print(f"   adapter   : {adapter_dir}")
    print(f"   device    : {device} ({dtype_name})")
    print(f"   samples   : {args.samples} held-out")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = build_eval_samples(config, hardware, tokenizer, args.samples)
    if not samples:
        raise SystemExit("Held-out split is empty; nothing to evaluate.")

    # --- models -------------------------------------------------------------- #
    # Teacher and student are both resident: agreement and KL need both
    # distributions for the same position at the same time.
    phases = 5 if teacher_base_id else 4
    print(f"\n[1/{phases}] Loading teacher ({teacher_id})...")
    from .teacher import load_teacher

    # The same loader training used, so an adapter that carries resized
    # embeddings is attached to a base trimmed to match, here as there.
    teacher, _info = load_teacher(teacher_id, teacher_adapter, dtype=dtype,
                                  device=device)
    teacher = teacher.eval()

    # One student instance serves as both columns: PEFT's disable_adapter() context
    # turns the LoRA branches off, which is the base student exactly. Loading a
    # second copy would double peak memory for no additional information.
    print(f"[2/{phases}] Loading student ({student_id}) + adapter...")
    from peft import PeftModel
    student = AutoModelForCausalLM.from_pretrained(
        student_id, dtype=dtype, low_cpu_mem_usage=True)
    # The same trim training applied: what the adapter recorded, else the
    # teacher's width as loaded. Without it PEFT meets an lm_head 271 rows
    # wider than the adapter was built against and refuses the state dict on
    # a shape mismatch.
    from . import paths
    target = paths.adapter_meta(adapter_dir).get("vocab_size") or \
        paths.matched_width(student, teacher, len(tokenizer))
    paths.fit_vocab(student, target, label="student")
    student = PeftModel.from_pretrained(student, adapter_dir).to(device).eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    adapter_params = sum(p.numel() for n, p in student.named_parameters() if "lora_" in n)

    # --- fidelity + perplexity ----------------------------------------------- #
    print(f"[3/{phases}] Scoring {len(samples)} held-out samples...")
    tokens = 0
    agree_base = agree_dist = 0
    top5_dist = 0.0
    # Every completion slice, kept so the packed student can be scored on the
    # IDENTICAL tokens after the dense one is freed. That identity is the whole
    # point: two perplexities from two different token sets do not subtract.
    scored_ids = []
    kl_base = kl_dist = 0.0
    nll_teacher = nll_base = nll_dist = 0.0
    skipped = 0

    for index, messages in enumerate(samples):
        ids, prompt_len = encode(tokenizer, messages, device)
        with torch.no_grad():
            t_logits_full = teacher(ids).logits
            t_logits, targets = completion_slice(t_logits_full, ids, prompt_len)
            if t_logits is None or t_logits.shape[0] == 0:
                skipped += 1
                continue

            d_logits, _ = completion_slice(student(ids).logits, ids, prompt_len)
            with student.disable_adapter():
                b_logits, _ = completion_slice(student(ids).logits, ids, prompt_len)

        # Standard GKD compares per-token distributions, which is only meaningful
        # over a shared vocabulary. Fail loudly rather than reporting a number that
        # cannot mean anything.
        if t_logits.shape[-1] != d_logits.shape[-1]:
            raise SystemExit(
                f"Vocabulary mismatch: teacher {t_logits.shape[-1]} vs student "
                f"{d_logits.shape[-1]}. Teacher and student must share a tokenizer "
                f"for token-level distillation metrics to mean anything."
            )

        n = t_logits.shape[0]
        tokens += n
        scored_ids.append((ids.detach().to("cpu"), prompt_len))
        a, k = compare_distributions(t_logits, b_logits)
        agree_base += a
        kl_base += k
        a, k = compare_distributions(t_logits, d_logits)
        agree_dist += a
        kl_dist += k
        overlap = top_k_overlap(t_logits, d_logits)
        if overlap is not None:
            top5_dist += overlap * n
        nll_teacher += summed_nll(t_logits, targets)
        nll_base += summed_nll(b_logits, targets)
        nll_dist += summed_nll(d_logits, targets)

        del t_logits_full, t_logits, d_logits, b_logits
        if (index + 1) % 10 == 0:
            print(f"   {index + 1}/{len(samples)} scored ({tokens} completion tokens)")

    if not tokens:
        raise SystemExit("No scorable completion tokens; try --samples with a larger value.")

    agreement_base = agree_base / tokens * 100
    agreement_dist = agree_dist / tokens * 100
    kl_base_avg = kl_base / tokens
    kl_dist_avg = kl_dist / tokens
    ppl_teacher = math.exp(nll_teacher / tokens)
    ppl_base = math.exp(nll_base / tokens)
    ppl_dist = math.exp(nll_dist / tokens)

    # --- efficiency + generations -------------------------------------------- #
    print(f"[4/{phases}] Measuring decode throughput...")
    probe = (config.get("benchmark_prompts") or ["Explain compound interest."])[0]
    tps_teacher = measure_latency(teacher, tokenizer, probe, device)
    tps_dist = measure_latency(student, tokenizer, probe, device)

    # ----------------------------------------------------------------------- #
    print("\n" + BAR)
    print("  FIDELITY - how much of the teacher transferred")
    print(BAR)
    print(f"  {len(samples) - skipped} held-out samples, {tokens} completion tokens, "
          f"teacher-forced\n")
    print(f"  {'metric':34} {'base':>10} {'distilled':>11} {'change':>12}")
    print("  " + "-" * 70)
    print(f"  {'top-1 agreement with teacher':34} {agreement_base:9.2f}% "
          f"{agreement_dist:10.2f}% {agreement_dist - agreement_base:+11.2f} pts")
    kl_change = ((kl_dist_avg - kl_base_avg) / kl_base_avg * 100) if kl_base_avg else 0.0
    print(f"  {'KL(teacher || student), per token':34} {kl_base_avg:10.4f} "
          f"{kl_dist_avg:11.4f} {kl_change:+11.1f}%")
    print("\n  Agreement is the fraction of positions where the student's top token")
    print("  matches the teacher's. Fidelity is routinely far below task accuracy -")
    print("  a student can score well while disagreeing with the teacher often.")

    print("\n" + BAR)
    print("  CAPABILITY - held-out perplexity (lower is better)")
    print(BAR)
    print(f"\n  {'teacher':34} {ppl_teacher:10.3f}")
    print(f"  {'base student':34} {ppl_base:10.3f}")
    print(f"  {'distilled student':34} {ppl_dist:10.3f}")
    gap = ppl_base - ppl_teacher
    recovered = ((ppl_base - ppl_dist) / gap * 100) if abs(gap) > 1e-9 else float("nan")
    if math.isfinite(recovered):
        print(f"\n  teacher-student gap recovered : {recovered:.1f}%")
        print("  (fraction of the base->teacher perplexity gap the adapter closed;")
        print("   negative means the distilled student is worse than the base)")

    # --- the single "how close is it" number ------------------------------- #
    # Two standard percentages, deliberately NOT blended into one score. There is
    # no accepted composite closeness metric, and averaging these would combine a
    # token-level agreement rate with a likelihood ratio - different units,
    # different questions. Raw perplexity is reported above, as a number, rather
    # than as a retention ratio a reader has to un-normalise before it can be
    # compared with anything.

    print("\n" + BAR)
    print("  CLOSENESS TO TEACHER")
    print(BAR)
    print(f"\n  {'':34} {'base':>10} {'distilled':>11} {'teacher':>10}")
    print("  " + "-" * 70)
    print(f"  {'prediction agreement':34} {agreement_base:9.2f}% "
          f"{agreement_dist:10.2f}% {100.0:9.2f}%")
    if top5_dist and tokens:
        print(f"  {'top-5 overlap':34} {'-':>10} "
              f"{top5_dist / tokens * 100:10.2f}% {100.0:9.2f}%")
    if math.isfinite(recovered):
        print(f"\n  Training closed {recovered:.1f}% of the base->teacher gap.")
    print("""
  Read these as "how close", not "how good". The base student already scores
  most of this before any training, because student and teacher share an
  architecture, a tokenizer and instruction tuning - so the absolute number is
  dominated by that head start, not by distillation. What distillation bought
  is the LIFT over the base column, and the gap-recovered figure.

  The two rows answer different questions (token agreement vs likelihood) and
  are not averaged: no standard composite closeness score exists.""")

    print("\n" + BAR)
    print("  EFFICIENCY - what the retention cost")
    print(BAR)
    print(f"\n  {'':34} {'teacher':>12} {'distilled':>12}")
    print("  " + "-" * 60)
    print(f"  {'parameters':34} {teacher_params / 1e9:11.3f}B {student_params / 1e9:11.3f}B")
    print(f"  {'decode throughput (tok/s)':34} {tps_teacher:12.1f} {tps_dist:12.1f}")
    print(f"  {'size ratio':34} {'1.00x':>12} "
          f"{student_params / teacher_params:11.2f}x")
    if tps_teacher > 0:
        print(f"  {'speed ratio':34} {'1.00x':>12} {tps_dist / tps_teacher:11.2f}x")
    print(f"\n  trainable adapter parameters : {adapter_params / 1e6:.2f}M "
          f"({adapter_params / student_params * 100:.2f}% of the student)")

    generations = {}
    if not args.no_generations:
        print("\n" + BAR)
        print("  GENERATIONS - greedy, deterministic")
        print(BAR)
        for prompt in (config.get("benchmark_prompts") or [])[:3]:
            with student.disable_adapter():
                base_text = generate(student, tokenizer, prompt, device)
            dist_text = generate(student, tokenizer, prompt, device)
            teach_text = generate(teacher, tokenizer, prompt, device)
            generations[prompt] = {"base": base_text, "distilled": dist_text,
                                   "teacher": teach_text}
            print(f"\n  Q: {prompt}")
            print(f"    base      : {base_text[:220]}")
            print(f"    distilled : {dist_text[:220]}")
            print(f"    teacher   : {teach_text[:220]}")

    # --- the fourth column: the teacher's own base ---------------------------- #
    # After everything that needs the student, which is freed first: the base
    # is the teacher's size, and two teachers plus a student is more than a
    # 16 GB card holds for a 3B pair. Same split, same positions, same teacher
    # logits, so the numbers sit in the same table as the student's.
    teacher_base = None
    if teacher_base_id:
        print(f"\n[5/{phases}] Loading the teacher's base ({teacher_base_id})...")
        del student
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        try:
            teacher_base = score_teacher_base(
                teacher_base_id, teacher, teacher_id, tokenizer, samples, device,
                dtype, config)
        except Exception as exc:  # noqa: BLE001 - a column, not the evaluation
            print(f" !! teacher base skipped: {type(exc).__name__}: {exc}")
            teacher_base = None
        if teacher_base:
            print("\n" + BAR)
            print("  THE TEACHER'S OWN BASE - what the fine-tune bought the teacher")
            print(BAR)
            print(f"\n  {'metric':34} {'teacher base':>14} {'distilled':>11}")
            print("  " + "-" * 70)
            print(f"  {'top-1 agreement with teacher':34} "
                  f"{teacher_base['agreement_pct']:13.2f}% {agreement_dist:10.2f}%")
            print(f"  {'KL(teacher || model), per token':34} "
                  f"{teacher_base['kl']:14.4f} {kl_dist_avg:11.4f}")
            print(f"  {'held-out perplexity':34} "
                  f"{teacher_base['perplexity']:14.3f} {ppl_dist:11.3f}")
            print("\n  How far the teacher moved from the stock model it was fine-tuned")
            print("  from is the most there was to distil. A distilled student that")
            print("  agrees with the teacher no more than the stock base does has")
            print("  learned nothing the base did not already know.")

    payload = {
        "teacher": teacher_id,
        "teacher_adapter": teacher_adapter,
        "teacher_base": teacher_base_id if teacher_base else None,
        "student": student_id,
        "adapter": adapter_dir,
        # Where that adapter lives, here and in the bucket, and the profile
        # that scored it: what the report's "The adapter" section prints, so a
        # `kd evaluate --report` page says it too. The pipeline's report stage
        # recomputes both with what it additionally knows about the run.
        "adapter_locations": paths.adapter_locations(adapter_dir, config),
        "profile": config["_meta"].get("source"),
        "training": training_settings(config),
        "device": device,
        "dtype": dtype_name,
        "samples": len(samples) - skipped,
        "completion_tokens": tokens,
        "fidelity": {
            "top1_agreement_base_pct": agreement_base,
            "top1_agreement_distilled_pct": agreement_dist,
            "agreement_lift_pts": agreement_dist - agreement_base,
            "kl_base": kl_base_avg,
            "kl_distilled": kl_dist_avg,
            # Where top-1 flips on a near-tie, this moves smoothly. See
            # top_k_overlap.
            "top5_overlap_distilled": (top5_dist / tokens) if tokens else None,
        },
        "capability": {
            "perplexity_teacher": ppl_teacher,
            "perplexity_base": ppl_base,
            "perplexity_distilled": ppl_dist,
            "gap_recovered_pct": recovered,
        },
        "closeness_to_teacher": {
            "prediction_agreement_base_pct": agreement_base,
            "prediction_agreement_distilled_pct": agreement_dist,
            "gap_recovered_pct": recovered,
        },
        "efficiency": {
            "teacher_params": teacher_params,
            "student_params": student_params,
            "adapter_params": adapter_params,
            "teacher_tok_per_s": tps_teacher,
            "distilled_tok_per_s": tps_dist,
        },
        "generations": generations,
    }

    if teacher_base:
        payload["fidelity"]["top1_agreement_teacher_base_pct"] = teacher_base["agreement_pct"]
        payload["fidelity"]["kl_teacher_base"] = teacher_base["kl"]
        payload["capability"]["perplexity_teacher_base"] = teacher_base["perplexity"]
        payload["closeness_to_teacher"]["prediction_agreement_teacher_base_pct"] = \
            teacher_base["agreement_pct"]
        payload["efficiency"]["teacher_base_params"] = teacher_base["params"]
        for prompt, text in (teacher_base.get("generations") or {}).items():
            generations.setdefault(prompt, {})["teacher-base"] = text

    # Everything resident is given back before the packed student is loaded. The
    # quantized pass measures ONE more model on the same tokens; it should not
    # also mean holding three at once on a card sized for two.
    del teacher
    if not teacher_base_id:
        del student
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    packed = score_quantized(getattr(args, "quantized", None), scored_ids,
                             tokenizer, device, dtype, probe, config)
    if packed:
        payload["quantization"] = quantization_delta(
            packed, ppl_dist, nll_dist / tokens, tps_dist, tokens)
        report_quantization(payload["quantization"])

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  metrics written to {args.json}")

    if args.report:
        written = write_report(payload, args.report)
        print(f"  report written to {written}")

    improved = (agreement_dist > agreement_base) and (ppl_dist < ppl_base)
    print("\n" + BAR)
    if improved:
        print("  ALL OK - the adapter moved the student toward the teacher")
        print(BAR)
        print(f"   agreement  : {agreement_base:.2f}% -> {agreement_dist:.2f}% "
              f"({agreement_dist - agreement_base:+.2f} pts)")
        print(f"   perplexity : {ppl_base:.3f} -> {ppl_dist:.3f} "
              f"(teacher {ppl_teacher:.3f})")
        print()
        return 0

    print("  NO IMPROVEMENT - the adapter did not move the student toward the teacher")
    print(BAR)
    print(f"   agreement  : {agreement_base:.2f}% -> {agreement_dist:.2f}%")
    print(f"   perplexity : {ppl_base:.3f} -> {ppl_dist:.3f}")
    print("""
  Things worth checking, in order:
    * did the teacher pass --check-teacher? distilling from a broken teacher
      trains the student toward noise;
    * were enough steps run? 100-300 steps on a 2B->0.8B pair is a small budget;
    * do the LoRA target_modules cover this architecture? a Llama-style target
      list misses linear attention in 18 of Qwen3.5's 24 layers;
    * is lmbda > 0? without on-policy rollouts this is plain off-policy KD.
""")
    return 3


if __name__ == "__main__":
    sys.exit(main())
