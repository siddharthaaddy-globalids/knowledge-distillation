"""Measure how much of the teacher actually transferred into the distilled student.

Two questions, measured separately, because the distillation literature is explicit
that they do not track each other (Stanton et al., "Does Knowledge Distillation
Really Work?", NeurIPS 2021):

  FIDELITY    how closely does the student reproduce the TEACHER's predictions?
              -> top-1 agreement rate, KL(teacher || student)

  CAPABILITY  is the student actually better at the task than it was?
              -> held-out perplexity, and optionally task benchmarks via
                 lm-evaluation-harness

Both are reported for the BASE student and the DISTILLED student against the same
teacher, on the same held-out split. The base student column is what makes the
numbers mean anything: without it there is no way to tell "distillation worked"
apart from "the small model could already do this".

    python evaluate.py --config configs/finance.yaml
    python evaluate.py --config configs/finance.yaml --dtype bfloat16 --samples 100
    python evaluate.py --config configs/finance.yaml --tasks ifeval,hellaswag
    python evaluate.py --config configs/mac.yaml --json results.json

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
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kd_config import load_config, resolve_device

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
    ap.add_argument("-c", "--config", default="configs/default.yaml",
                    help="Training config the adapter was produced from "
                         "(default: configs/default.yaml)")
    ap.add_argument("-a", "--adapter", default=None,
                    help="Adapter directory to evaluate "
                         "(default: <output_dir>/final_adapter from the config)")
    ap.add_argument("-t", "--teacher", default=None, help="Override models.teacher")
    ap.add_argument("-s", "--student", default=None, help="Override models.student")
    ap.add_argument("--teacher-adapter", default=None,
                    help="LoRA adapter merged into the teacher, when the teacher is "
                         "a base model plus an adapter rather than a merged checkpoint")
    ap.add_argument("-n", "--samples", type=int, default=50,
                    help="Held-out samples to score (default: 50)")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"],
                    help="Override hardware.device")
    ap.add_argument("--dtype", default="auto", choices=sorted(DTYPES),
                    help="Override hardware.dtype (default: auto)")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated lm-evaluation-harness tasks to run as well, "
                         "e.g. ifeval,hellaswag,arc_easy. Requires `lm-eval` "
                         "(uv sync --extra eval). Slow: it runs three models.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-task example cap passed to lm-eval (use for a quick look)")
    ap.add_argument("--no-generations", action="store_true",
                    help="Skip the qualitative side-by-side generations")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="Write all metrics to a JSON file")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Held-out split
# --------------------------------------------------------------------------- #
def build_eval_samples(config, hardware, tokenizer, limit):
    """Rebuild the exact held-out split the student was never trained on.

    train_scaled.build_datasets() is reused rather than reimplemented: it is seeded
    from project.seed and applies the same length filter and dedup, so the split
    here is byte-identical to the one training held out. Reimplementing it would
    silently drift the moment either side changed.
    """
    import train_scaled

    train_scaled.apply_config(config, hardware)
    _, eval_dataset = train_scaled.build_datasets(tokenizer)
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
# Optional: lm-evaluation-harness
# --------------------------------------------------------------------------- #
def run_lm_eval(tasks, student_id, adapter_dir, teacher_id, teacher_adapter,
                dtype_name, device, limit):
    """Score base student, distilled student and teacher on standard benchmarks.

    lm-evaluation-harness is the de facto standard harness (it is what the HF Open
    LLM Leaderboard runs), so numbers produced here are comparable with published
    ones rather than only with each other.
    """
    if importlib.util.find_spec("lm_eval") is None:
        print("\n !! lm-eval is not installed; skipping --tasks.")
        print("    Install the optional extra and re-run:")
        print("      uv sync --extra eval")
        return None

    runs = {
        "base_student": f"pretrained={student_id}",
        "distilled_student": f"pretrained={student_id},peft={adapter_dir}",
        "teacher": f"pretrained={teacher_id}"
                   + (f",peft={teacher_adapter}" if teacher_adapter else ""),
    }

    results = {}
    for label, model_args in runs.items():
        print(f"\n  [lm-eval] {label} on {tasks}")
        # Invoked as a module rather than by console-script name: the entry point is
        # spelled lm-eval in some releases and lm_eval in others, and neither is
        # guaranteed to be on PATH. sys.executable always resolves to this venv.
        #
        # --apply_chat_template is set for all three models, including the base
        # student. Every model here is instruct-tuned and the adapter was trained
        # against a chat template, so this keeps the three columns internally
        # consistent, which is what a base-vs-distilled comparison needs. It does
        # mean the absolute numbers are not directly comparable to leaderboard
        # entries that scored multiple-choice tasks without a template.
        cmd = [
            sys.executable, "-m", "lm_eval", "run",
            "--model", "hf",
            "--model_args", f"{model_args},dtype={dtype_name}",
            "--tasks", tasks,
            "--device", device,
            "--batch_size", "1",
            "--seed", "42",
            "--apply_chat_template",
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        out_dir = pathlib.Path("./evals") / label
        cmd += ["--output_path", str(out_dir)]

        # lm-eval renders its summary table with Unicode arrows. On a Windows
        # console that defaults to cp1252 the print raises UnicodeEncodeError and
        # the process exits 1 - AFTER the results file has been written. Force
        # UTF-8 so it does not happen, and treat the results file as the source of
        # truth below rather than the exit status.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        completed = subprocess.run(cmd, env=env)

        # lm-eval writes results_<timestamp>.json under a model-named subdirectory.
        files = sorted(out_dir.rglob("results_*.json"))
        if not files:
            print(f"  !! lm-eval produced no results for {label} "
                  f"(exit {completed.returncode})")
            continue
        if completed.returncode != 0:
            print(f"  -- lm-eval exited {completed.returncode} for {label}, but wrote "
                  f"results; using them")
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        results[label] = payload.get("results", {})
    return results or None


def report_tasks(results):
    """Print the retention table: distilled / teacher, the standard KD headline."""
    print("\n" + BAR)
    print("  TASK BENCHMARKS - lm-evaluation-harness")
    print(BAR)

    base, dist, teach = (results.get("base_student", {}),
                         results.get("distilled_student", {}),
                         results.get("teacher", {}))
    task_names = sorted(set(base) | set(dist) | set(teach))

    def stderr_key(metric):
        """lm-eval names metrics 'acc,none' and their error 'acc_stderr,none'."""
        name, _, suffix = metric.partition(",")
        return f"{name}_stderr,{suffix}" if suffix else f"{name}_stderr"

    def is_reportable(task, metric):
        # Skip the error bars themselves (they are attached to their metric below)
        # and lm-eval's bookkeeping entries, which have no meaningful retention.
        name = metric.partition(",")[0]
        if name.endswith("_stderr") or name in ("alias", "sample_len"):
            return False
        return isinstance((dist.get(task) or {}).get(metric), (int, float))

    def cell(value, err):
        if not isinstance(value, (int, float)):
            return f"{'-':>14}"
        return f"{value:8.4f}+-{err:<4.3f}" if isinstance(err, (int, float)) \
            else f"{value:8.4f}      "

    print(f"\n  {'task / metric':30} {'base':>14} {'distilled':>14} "
          f"{'teacher':>14} {'retention':>10}")
    print("  " + "-" * 86)
    noisy = []
    for task in task_names:
        for metric in [m for m in (dist.get(task) or {}) if is_reportable(task, m)]:
            sk = stderr_key(metric)
            b, d, t = ((base.get(task) or {}).get(metric),
                       (dist.get(task) or {}).get(metric),
                       (teach.get(task) or {}).get(metric))
            be, de, te = ((base.get(task) or {}).get(sk),
                          (dist.get(task) or {}).get(sk),
                          (teach.get(task) or {}).get(sk))
            retention = (f"{d / t * 100:9.1f}%"
                         if isinstance(d, (int, float)) and isinstance(t, (int, float)) and t
                         else "        -")
            print(f"  {task + ' / ' + metric.partition(',')[0]:30} "
                  f"{cell(b, be)} {cell(d, de)} {cell(t, te)} {retention}")

            # A difference smaller than the combined error bars is not a result.
            # Saying so here is the whole point of running a standard harness.
            if all(isinstance(v, (int, float)) for v in (b, d, be, de)):
                if abs(d - b) <= (be + de):
                    noisy.append(f"{task}/{metric.partition(',')[0]}")

    print("\n  retention = distilled / teacher. The comparison that matters is")
    print("  distilled vs base: if that lift is ~0, distillation changed nothing.")
    print("  +- values are lm-eval's standard error at the sample count you ran.")
    if noisy:
        print("\n  !! NOT SIGNIFICANT - distilled vs base is within the error bars for:")
        for item in noisy:
            print(f"       {item}")
        print("     Re-run with a larger --eval-limit (or none) before drawing")
        print("     any conclusion from these.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
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
    dtype_name = str(dtype).replace("torch.", "")

    teacher_id = args.teacher or config["models"]["teacher"]
    student_id = args.student or config["models"]["student"]
    teacher_adapter = args.teacher_adapter or config["models"].get("teacher_adapter")

    adapter_dir = args.adapter or os.path.join(
        config["project"]["output_dir"], "final_adapter")
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise SystemExit(
            f"No adapter_config.json in {adapter_dir}\n"
            f"Train one first, or point --adapter at the right directory."
        )

    tokenizer_choice = str(config["models"].get("tokenizer", "teacher"))
    tokenizer_id = {"teacher": teacher_id, "student": student_id}.get(
        tokenizer_choice, tokenizer_choice)

    print(BAR)
    print("  Distillation evaluation")
    print(BAR)
    print(f"   teacher   : {teacher_id}" + (f"  + {teacher_adapter}" if teacher_adapter else ""))
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
    print(f"\n[1/4] Loading teacher ({teacher_id})...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, dtype=dtype, low_cpu_mem_usage=True)
    if teacher_adapter:
        from peft import PeftModel as _Peft
        print(f" -> merging teacher adapter: {teacher_adapter}")
        teacher = _Peft.from_pretrained(teacher, teacher_adapter).merge_and_unload()
    teacher = teacher.to(device).eval()

    # One student instance serves as both columns: PEFT's disable_adapter() context
    # turns the LoRA branches off, which is the base student exactly. Loading a
    # second copy would double peak memory for no additional information.
    print(f"[2/4] Loading student ({student_id}) + adapter...")
    from peft import PeftModel
    student = AutoModelForCausalLM.from_pretrained(
        student_id, dtype=dtype, low_cpu_mem_usage=True)
    student = PeftModel.from_pretrained(student, adapter_dir).to(device).eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    adapter_params = sum(p.numel() for n, p in student.named_parameters() if "lora_" in n)

    # --- fidelity + perplexity ----------------------------------------------- #
    print(f"[3/4] Scoring {len(samples)} held-out samples...")
    tokens = 0
    agree_base = agree_dist = 0
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
        a, k = compare_distributions(t_logits, b_logits)
        agree_base += a
        kl_base += k
        a, k = compare_distributions(t_logits, d_logits)
        agree_dist += a
        kl_dist += k
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
    print("[4/4] Measuring decode throughput...")
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

    payload = {
        "teacher": teacher_id,
        "teacher_adapter": teacher_adapter,
        "student": student_id,
        "adapter": adapter_dir,
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
        },
        "capability": {
            "perplexity_teacher": ppl_teacher,
            "perplexity_base": ppl_base,
            "perplexity_distilled": ppl_dist,
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

    # Free the resident models before lm-eval spawns its own processes. Not
    # cosmetic: each lm-eval run loads its own copy of the model, so on a 16 GB
    # unified-memory Mac the parent still caching ~6 GB is the difference between
    # the benchmark running and the machine swapping itself to a halt.
    del teacher, student
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    if args.tasks:
        task_results = run_lm_eval(args.tasks, student_id, adapter_dir, teacher_id,
                                   teacher_adapter, dtype_name, device, args.limit)
        if task_results:
            report_tasks(task_results)
            payload["tasks"] = task_results

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  metrics written to {args.json}")

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
