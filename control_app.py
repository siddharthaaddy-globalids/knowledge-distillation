#!/usr/bin/env python3
"""
Control-plane UI for the knowledge-distillation runner.

`./distill.sh --ui` launches this. It drives the same pipeline the terminal does by
shelling out to the runner script instead of reimplementing it, so distill.sh stays the
single source of truth for profile resolution, KD_* precedence and exit-code meanings.
Every screen prints the exact command it is about to run, so the UI teaches the CLI
rather than hiding it.

  Train    - build a run from a form, launch it, watch the log stream live
  Evaluate - score an adapter against the teacher, then render the report it writes
  Compare  - the three-way generation comparison from app.py, loaded on demand

In a source checkout there is no built distill.sh (CI generates it from
scripts/distill.sh.template), so the same forms fall back to invoking train_scaled.py /
evaluate.py directly with the KD_* variables the script would have exported. The header
says which backend is live.

Run with:  python control_app.py   (then open http://127.0.0.1:7860)
"""

import argparse
import glob
import os
import platform
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time

import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = os.name == "nt"

PROFILES = ["auto", "default", "mac", "smoke", "finance", "qwen-poc"]
DEVICES = ["", "auto", "cpu", "mps", "cuda"]
DTYPES = ["", "auto", "float32", "bfloat16", "float16"]

# Keep the browser responsive: a 600-step run emits tens of thousands of lines, and
# re-sending the whole buffer on every tick is what makes streaming UIs crawl.
MAX_LOG_LINES = 1500

SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860

# Adapter search order, newest training layout first. Mirrors app.ADAPTER_CANDIDATES,
# duplicated here only so this module can list adapters without importing torch.
ADAPTER_CANDIDATES = [
    "./distilled_output/final_adapter",
    "./distilled_smollm_mac/final_adapter",
    "./distilled_smollm_scaled/final_adapter",
    "./distilled_smollm_poc/final_adapter",
]


# --------------------------------------------------------------------------- #
# Backend discovery
# --------------------------------------------------------------------------- #
def find_runner():
    """Absolute path to a built distill.sh, or None in a source checkout."""
    for candidate in (os.environ.get("KD_RUNNER"),
                      os.path.join(ROOT, "distill.sh"),
                      os.path.join(os.getcwd(), "distill.sh")):
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def find_bash():
    """A bash able to run distill.sh. On Windows that means Git Bash."""
    found = shutil.which("bash")
    if found:
        return found
    if IS_WINDOWS:
        for candidate in (r"C:\Program Files\Git\bin\bash.exe",
                          r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if os.path.isfile(candidate):
                return candidate
    return None


RUNNER = find_runner()
BASH = find_bash() if RUNNER else None
USE_RUNNER = bool(RUNNER and BASH)


def auto_profile():
    """The profile distill.sh would pick with no --profile: Apple Silicon -> mac."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mac"
    return "default"


def config_file(profile):
    name = auto_profile() if profile == "auto" else profile
    # Forward slashes on every platform: this string is both an argv element and the
    # command shown on screen for copy-pasting into a shell.
    return f"configs/{name}.yaml"


def backend_note():
    if USE_RUNNER:
        return f"runner &mdash; `{RUNNER}`"
    if RUNNER and not BASH:
        return ("direct &mdash; `distill.sh` found but no bash on PATH "
                "(install Git Bash to drive the runner)")
    return "direct &mdash; no built `distill.sh` in this checkout"


# --------------------------------------------------------------------------- #
# Command construction
#
# One table per mode, so the two backends cannot drift: the middle column is what
# distill.sh accepts, and the third is how the same value reaches Python without the
# script - a KD_* variable for training (train_scaled.py reads them through kd_config)
# and a real flag for evaluation (evaluate.py has its own argparse).
# --------------------------------------------------------------------------- #
TRAIN_OPTS = [
    # ui key,            runner flag,          KD_* variable
    ("teacher",          "--teacher",          "KD_TEACHER_MODEL"),
    ("student",          "--student",          "KD_STUDENT_MODEL"),
    ("teacher_adapter",  "--teacher-adapter",  "KD_TEACHER_ADAPTER"),
    ("dataset",          "--dataset",          "KD_DATASET"),
    ("device",           "--device",           "KD_DEVICE"),
    ("dtype",            "--dtype",            "KD_DTYPE"),
    ("steps",            "--steps",            "KD_MAX_STEPS"),
    ("batch_size",       "--batch-size",       "KD_BATCH_SIZE"),
    ("grad_accum",       "--grad-accum",       "KD_GRAD_ACCUM"),
    ("lr",               "--lr",               "KD_LEARNING_RATE"),
    ("lora_r",           "--lora-r",           "KD_LORA_R"),
    ("lora_alpha",       "--lora-alpha",       "KD_LORA_ALPHA"),
    ("lmbda",            "--lmbda",            "KD_LMBDA"),
    ("output",           "--output",           "KD_OUTPUT_DIR"),
]

EVAL_OPTS = [
    # ui key,            runner flag,          evaluate.py flag
    ("adapter",          "--adapter",          "--adapter"),
    ("device",           "--device",           "--device"),
    ("dtype",            "--dtype",            "--dtype"),
    ("samples",          "--eval-samples",     "--samples"),
    ("gen_similarity",   "--gen-similarity",   "--gen-similarity"),
    ("similarity_model", "--similarity-model", "--similarity-model"),
    ("tasks",            "--tasks",            "--tasks"),
    ("limit",            "--eval-limit",       "--limit"),
    ("report",           "--report",           "--report"),
    ("eval_json",        "--eval-json",        "--json"),
]

# Passed to every child. distill.sh refuses to open a UI when this is set, so a control
# UI can never spawn another one underneath itself.
GUARD_ENV = {"KD_UI_ACTIVE": "1"}


def _clean(value):
    """Blank fields mean 'not specified' - the profile's YAML value then wins."""
    if value is None:
        return ""
    return str(value).strip()


def build_train_command(profile, fields):
    """(argv, env, display) for a training run."""
    env = dict(GUARD_ENV)
    if USE_RUNNER:
        argv = [BASH, RUNNER.replace("\\", "/")]
        if profile != "auto":
            argv += ["--profile", profile]
        for key, flag, _ in TRAIN_OPTS:
            value = _clean(fields.get(key))
            if value:
                argv += [flag, value]
        return argv, env, "./distill.sh " + " ".join(shlex.quote(a) for a in argv[2:])

    argv = [sys.executable, "train_scaled.py", "--config", config_file(profile)]
    for key, _, env_var in TRAIN_OPTS:
        value = _clean(fields.get(key))
        if value:
            env[env_var] = value
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items())
                      if k != "KD_UI_ACTIVE")
    display = "python " + " ".join(shlex.quote(a) for a in argv[1:])
    return argv, env, (prefix + " " + display).strip()


def build_eval_command(profile, fields):
    """(argv, env, display) for an evaluation run."""
    env = dict(GUARD_ENV)
    if USE_RUNNER:
        argv = [BASH, RUNNER.replace("\\", "/"), "--evaluate"]
        if profile != "auto":
            argv += ["--profile", profile]
        for key, flag, _ in EVAL_OPTS:
            value = _clean(fields.get(key))
            if value:
                argv += [flag, value]
        return argv, env, "./distill.sh " + " ".join(shlex.quote(a) for a in argv[2:])

    argv = [sys.executable, "evaluate.py", "--config", config_file(profile)]
    for key, _, flag in EVAL_OPTS:
        value = _clean(fields.get(key))
        if value:
            argv += [flag, value]
    display = "python " + " ".join(shlex.quote(a) for a in argv[1:])
    return argv, env, display


# --------------------------------------------------------------------------- #
# Subprocess job
# --------------------------------------------------------------------------- #
class Job:
    """One child process, its captured output, and a stop that takes the tree with it."""

    def __init__(self, argv, env, cwd=None, label=""):
        self.argv = argv
        self.env = env
        self.cwd = cwd or os.getcwd()
        self.label = label
        self.proc = None
        self.lines = []
        self.returncode = None
        self.stopped = False
        self.started_at = None
        self.dropped = 0
        self._queue = queue.Queue()

    def start(self):
        environ = dict(os.environ)
        environ.update(self.env)
        # Without this, a child's stdout is block-buffered when it is a pipe and the
        # browser sees nothing for minutes at a time.
        environ["PYTHONUNBUFFERED"] = "1"
        kwargs = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.started_at = time.time()
        self.proc = subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self._queue.put(line.rstrip("\n"))
        finally:
            self.proc.wait()
            self.returncode = self.proc.returncode
            self._queue.put(None)

    def stop(self):
        """Kill the whole tree: uv and python spawn children that outlive a plain kill."""
        self.stopped = True
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, check=False)
            else:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    @property
    def elapsed(self):
        return time.time() - (self.started_at or time.time())

    def drain(self, timeout=0.4):
        """Collect whatever has arrived. Returns True once the process has finished."""
        deadline = time.time() + timeout
        done = False
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                break
            if item is None:
                done = True
                break
            self.lines.append(item)
            if len(self.lines) > MAX_LOG_LINES * 2:
                self.dropped += len(self.lines) - MAX_LOG_LINES
                self.lines = self.lines[-MAX_LOG_LINES:]
        return done

    def text(self):
        tail = self.lines[-MAX_LOG_LINES:]
        elided = self.dropped + len(self.lines) - len(tail)
        header = f"... {elided} earlier lines elided ...\n" if elided > 0 else ""
        return header + "\n".join(tail)


_JOB_LOCK = threading.Lock()
_CURRENT = None


def _claim(job):
    """One child at a time: two trainings on one laptop just OOM each other.

    A finished job stays registered until the next claim replaces it. That is
    deliberate: closing the browser tab cancels the streaming generator while the child
    keeps training, and forgetting the job there would orphan it - Stop could no longer
    reach the process, and the next run would start on top of the first.
    """
    global _CURRENT
    with _JOB_LOCK:
        if _CURRENT is not None and _CURRENT.proc and _CURRENT.proc.poll() is None:
            return _CURRENT.label
        _CURRENT = job
        return None


def stop_current():
    with _JOB_LOCK:
        job = _CURRENT
    if job is None or job.proc is None or job.proc.poll() is not None:
        return status_html("idle", "Nothing is running.")
    job.stop()
    return status_html("running", "Stop signal sent; waiting for the process to exit...")


def job_running():
    with _JOB_LOCK:
        job = _CURRENT
    return job is not None and job.proc is not None and job.proc.poll() is None


# --------------------------------------------------------------------------- #
# Status rendering
# --------------------------------------------------------------------------- #
_STATUS_COLORS = {
    "idle": ("#94a3b8", "IDLE"),
    "running": ("#38bdf8", "RUNNING"),
    "ok": ("#4ade80", "ALL OK"),
    "fail": ("#f87171", "FAILED"),
}


def status_html(kind, message):
    color, label = _STATUS_COLORS[kind]
    return (f"<div style='padding:0.6rem 0.8rem;border-radius:8px;"
            f"border:1px solid {color};color:{color};font-size:0.9rem;'>"
            f"<b>{label}</b> &nbsp; {message}</div>")


def _fmt_elapsed(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def exit_message(job, mode):
    """Translate an exit code the way the runner's banners do."""
    took = _fmt_elapsed(job.elapsed)
    if job.stopped:
        return "fail", f"Stopped after {took}."
    rc = job.returncode
    if rc == 0:
        return "ok", f"{mode} completed in {took}."
    if mode == "Evaluation" and rc == 3:
        return "fail", (f"The adapter is no closer to the teacher than the base student "
                        f"(exit 3, after {took}). Verify the teacher, then train longer.")
    return "fail", f"{mode} did not complete - exit {rc}, after {took}."


# --------------------------------------------------------------------------- #
# Adapters and paths
# --------------------------------------------------------------------------- #
def _is_adapter(path):
    return bool(path) and os.path.isfile(os.path.join(path, "adapter_config.json"))


def discover_adapters():
    found = {p.replace("\\", "/")
             for p in ADAPTER_CANDIDATES + glob.glob("./*/final_adapter")
             if _is_adapter(p)}
    return sorted(found, key=lambda p: -os.path.getmtime(p))


def default_output_dir():
    """Absolute, and under the UI's own working directory.

    The runner does its work inside its pinned checkout ($KD_WORKDIR/src), so a relative
    output path would drop the adapter somewhere the Evaluate and Compare tabs never
    look. Anchoring it here keeps train -> evaluate -> compare working in one sitting.
    """
    return os.path.join(os.getcwd(), "distilled_output").replace("\\", "/")


def rescan_adapters():
    found = discover_adapters()
    return gr.update(choices=found, value=found[0] if found else None)


def default_report_path():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(os.getcwd(), "reports", f"eval-{stamp}.md").replace("\\", "/")


# --------------------------------------------------------------------------- #
# Shared job driver
# --------------------------------------------------------------------------- #
def _stream_job(argv, env, display, mode, label, extra=None):
    """Drive one child process and stream its output into the page.

    Yields [command, log, status, start_btn, stop_btn] plus whatever `extra` appends -
    used by the Evaluate tab to render the report once the run succeeds.
    """
    extra = extra or (lambda job: [])
    idle_extra = extra(None)
    running = [gr.update(interactive=False), gr.update(interactive=True)]
    stopped = [gr.update(interactive=True), gr.update(interactive=False)]

    job = Job(argv, env, label=label)
    busy = _claim(job)
    if busy is not None:
        yield ([display, "", status_html("fail", f"A '{busy}' job is already running.")]
               + stopped + idle_extra)
        return

    try:
        job.start()
    except (OSError, ValueError) as exc:
        yield ([display, "", status_html("fail", f"Could not launch: {exc}")]
               + stopped + idle_extra)
        return

    yield [display, "", status_html("running", f"{mode} started.")] + running + idle_extra

    last_emit = 0.0
    while True:
        done = job.drain()
        now = time.time()
        if done or now - last_emit >= 0.5:
            last_emit = now
            message = f"{mode} running - {_fmt_elapsed(job.elapsed)} elapsed."
            yield ([display, job.text(), status_html("running", message)]
                   + running + idle_extra)
        if done:
            break

    kind, message = exit_message(job, mode)
    yield [display, job.text(), status_html(kind, message)] + stopped + extra(job)


# --------------------------------------------------------------------------- #
# Train tab
# --------------------------------------------------------------------------- #
def run_training(profile, teacher, student, teacher_adapter, dataset, device, dtype,
                 steps, batch_size, grad_accum, lr, lora_r, lora_alpha, lmbda, output):
    fields = {
        "teacher": teacher, "student": student, "teacher_adapter": teacher_adapter,
        "dataset": dataset, "device": device, "dtype": dtype, "steps": steps,
        "batch_size": batch_size, "grad_accum": grad_accum, "lr": lr,
        "lora_r": lora_r, "lora_alpha": lora_alpha, "lmbda": lmbda, "output": output,
    }
    argv, env, display = build_train_command(profile, fields)
    yield from _stream_job(argv, env, display, "Training", "train")


# --------------------------------------------------------------------------- #
# Evaluate tab
# --------------------------------------------------------------------------- #
def run_evaluation(profile, adapter, device, dtype, samples, gen_similarity,
                   similarity_model, tasks, limit, report, eval_json):
    # Absolute, for the same reason the output dir is: the runner's cwd is its own
    # checkout, and a relative path would write the report where nothing reads it.
    report = os.path.abspath(_clean(report) or default_report_path()).replace("\\", "/")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    eval_json = _clean(eval_json)
    if eval_json:
        eval_json = os.path.abspath(eval_json).replace("\\", "/")
        os.makedirs(os.path.dirname(eval_json), exist_ok=True)

    fields = {
        "adapter": adapter, "device": device, "dtype": dtype, "samples": samples,
        "gen_similarity": gen_similarity, "similarity_model": similarity_model,
        "tasks": tasks, "limit": limit, "report": report, "eval_json": eval_json,
    }
    argv, env, display = build_eval_command(profile, fields)

    def render_report(job):
        if job is None or job.returncode != 0 or not os.path.isfile(report):
            return [gr.update()]
        if report.endswith(".html"):
            return [gr.update(value=f"Report written to `{report}` - open it in a browser.")]
        with open(report, encoding="utf-8") as handle:
            return [gr.update(value=handle.read())]

    yield from _stream_job(argv, env, display, "Evaluation", "evaluate", extra=render_report)


# --------------------------------------------------------------------------- #
# Compare tab - app.py, loaded on demand
# --------------------------------------------------------------------------- #
_COMPARE = {"loaded": False, "app": None}
AWAITING = "_awaiting generation_"


def load_compare_models(profile, adapter, progress=gr.Progress()):
    """Import app.py and load its three models.

    Deliberately not done at startup: it costs a torch import plus a gigabyte of resident
    models, which would compete with a training run for the same RAM.
    """
    if job_running():
        return (status_html("fail", "A job is running - loading three models alongside it "
                                    "will exhaust memory. Stop it first."),
                gr.update(), gr.update(), gr.update())

    progress(0.1, desc="Importing torch and transformers...")
    import app as compare_app

    student = teacher = teacher_adapter = None
    path = config_file(profile)
    if os.path.isfile(path):
        # Read the ids from the training config so the UI never compares something
        # other than what was trained.
        import kd_config
        cfg = kd_config.load_config(path)
        student = cfg["models"]["student"]
        teacher = cfg["models"]["teacher"]
        teacher_adapter = cfg["models"].get("teacher_adapter")

    progress(0.3, desc="Loading models (the first run downloads them)...")
    compare_app.configure(adapter=_clean(adapter) or None, student=student,
                          teacher=teacher, teacher_adapter=teacher_adapter)
    compare_app.load_all_models()
    _COMPARE["app"] = compare_app
    _COMPARE["loaded"] = True

    headers = [compare_app._column_header(compare_app.MODELS[key])
               for key in ("original", "distilled", "teacher")]
    return (status_html("ok", f"Loaded. Adapter: <code>{compare_app.ADAPTER_PATH}</code>"),
            *headers)


def run_compare(prompt, temperature, max_new_tokens, repetition_penalty):
    if not _COMPARE["loaded"]:
        note = "**Load the models first.**"
        return "", note, "", note, "", note
    return _COMPARE["app"].compare(prompt, temperature, max_new_tokens, repetition_penalty)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
#kd-header h1 { margin-bottom: 0.15rem; }
.kd-cmd textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   font-size: 0.82rem; }
.kd-log textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   font-size: 0.78rem; line-height: 1.35; }
.kd-metrics { font-size: 0.82rem; opacity: 0.88;
              border-top: 1px solid var(--border-color-primary);
              padding-top: 0.5rem; margin-top: 0.25rem; }
.kd-col-head { font-weight: 600; margin-bottom: 0.2rem; }
.kd-col-sub { font-size: 0.8rem; opacity: 0.7; }
"""

EXAMPLE_PROMPTS = [
    "List three states of matter. Use a numbered list.",
    "Name three programming data types, formatted as a numbered list.",
    "What are three ways to reduce household energy use? Present them as a numbered list.",
    "List three renewable energy sources. Use a numbered list.",
]


def build_ui():
    with gr.Blocks(title="Knowledge Distillation Control", fill_width=True) as demo:
        gr.Markdown(
            "# Knowledge Distillation - Control Panel\n"
            f"Backend: {backend_note()} &nbsp;|&nbsp; every tab shows the exact command it "
            "runs, so nothing here is hidden from the terminal.",
            elem_id="kd-header",
        )

        with gr.Tabs():
            # ------------------------------------------------------------ Train
            with gr.Tab("Train"):
                with gr.Row():
                    t_profile = gr.Dropdown(PROFILES, value="auto", label="Profile",
                                            info="auto = mac on Apple Silicon, else default")
                    t_steps = gr.Textbox(label="Steps", placeholder="from profile")
                    t_device = gr.Dropdown(DEVICES, value="", label="Device",
                                           info="blank = from profile")
                    t_dtype = gr.Dropdown(DTYPES, value="", label="Dtype")
                with gr.Row():
                    t_teacher = gr.Textbox(
                        label="Teacher model",
                        placeholder="from profile, e.g. HuggingFaceTB/SmolLM2-1.7B-Instruct")
                    t_student = gr.Textbox(label="Student model", placeholder="from profile")
                with gr.Row():
                    t_teacher_adapter = gr.Textbox(
                        label="Teacher adapter",
                        placeholder="LoRA merged into the teacher at load time (optional)")
                    t_dataset = gr.Textbox(label="Dataset", placeholder="from profile")
                t_output = gr.Textbox(
                    label="Output directory", value=default_output_dir(),
                    info="Absolute on purpose: the runner works inside its own checkout, so "
                         "a relative path would hide the adapter from the other tabs.")

                with gr.Accordion("Hyperparameters (blank = profile default)", open=False):
                    with gr.Row():
                        t_batch = gr.Textbox(label="Batch size")
                        t_accum = gr.Textbox(label="Grad accumulation")
                        t_lr = gr.Textbox(label="Learning rate", placeholder="3e-4")
                    with gr.Row():
                        t_lora_r = gr.Textbox(label="LoRA r")
                        t_lora_alpha = gr.Textbox(label="LoRA alpha")
                        t_lmbda = gr.Textbox(label="lmbda (on-policy fraction 0-1)")

                with gr.Row():
                    t_start = gr.Button("Start training", variant="primary", scale=3)
                    t_stop = gr.Button("Stop", variant="stop", scale=1, interactive=False)

                t_cmd = gr.Textbox(label="Command", lines=2, interactive=False,
                                   elem_classes="kd-cmd", buttons=["copy"])
                t_status = gr.HTML(status_html("idle", "Not started."))
                t_log = gr.Textbox(label="Log", lines=24, max_lines=24, interactive=False,
                                   autoscroll=True, elem_classes="kd-log")

                t_inputs = [t_profile, t_teacher, t_student, t_teacher_adapter, t_dataset,
                            t_device, t_dtype, t_steps, t_batch, t_accum, t_lr,
                            t_lora_r, t_lora_alpha, t_lmbda, t_output]
                t_start.click(
                    fn=run_training, inputs=t_inputs,
                    outputs=[t_cmd, t_log, t_status, t_start, t_stop],
                    concurrency_id="job", concurrency_limit=1,
                )
                t_stop.click(fn=stop_current, inputs=None, outputs=t_status)

            # --------------------------------------------------------- Evaluate
            with gr.Tab("Evaluate"):
                adapters = discover_adapters()
                with gr.Row():
                    e_profile = gr.Dropdown(PROFILES, value="auto", label="Profile")
                    e_adapter = gr.Dropdown(
                        adapters, value=adapters[0] if adapters else None,
                        label="Adapter", allow_custom_value=True,
                        info="blank = <output>/final_adapter from the profile")
                    e_refresh = gr.Button("Rescan", scale=0)
                with gr.Row():
                    e_samples = gr.Textbox(label="Held-out samples", placeholder="50")
                    e_device = gr.Dropdown(DEVICES, value="", label="Device")
                    e_dtype = gr.Dropdown(DTYPES, value="", label="Dtype")

                with gr.Accordion("Slower, optional measurements", open=False):
                    gr.Markdown(
                        "Both need the `eval` extra (`lm-eval`, `bert-score`). The runner "
                        "installs it automatically; the direct backend does not - run "
                        "`uv sync --extra eval` once if these fail to import."
                    )
                    with gr.Row():
                        e_gen_sim = gr.Textbox(
                            label="Generation similarity (N prompts)", placeholder="0 = off",
                            info="Compares free-running generations, not next-token agreement")
                        e_sim_model = gr.Textbox(label="Similarity model",
                                                 placeholder="roberta-large")
                    with gr.Row():
                        e_tasks = gr.Textbox(label="Benchmark tasks",
                                             placeholder="e.g. ifeval,arc_easy")
                        e_limit = gr.Textbox(label="Per-task example cap", placeholder="e.g. 100")

                with gr.Row():
                    e_report = gr.Textbox(label="Report path", placeholder=default_report_path(),
                                          info="blank = timestamped .md under ./reports")
                    e_json = gr.Textbox(label="Metrics JSON path", placeholder="optional")

                with gr.Row():
                    e_start = gr.Button("Run evaluation", variant="primary", scale=3)
                    e_stop = gr.Button("Stop", variant="stop", scale=1, interactive=False)

                e_cmd = gr.Textbox(label="Command", lines=2, interactive=False,
                                   elem_classes="kd-cmd", buttons=["copy"])
                e_status = gr.HTML(status_html("idle", "Not started."))
                with gr.Tabs():
                    with gr.Tab("Report"):
                        e_report_md = gr.Markdown(
                            "_The report appears here once a run finishes._")
                    with gr.Tab("Log"):
                        e_log = gr.Textbox(label="Log", lines=24, max_lines=24,
                                           interactive=False, autoscroll=True,
                                           elem_classes="kd-log")

                e_inputs = [e_profile, e_adapter, e_device, e_dtype, e_samples, e_gen_sim,
                            e_sim_model, e_tasks, e_limit, e_report, e_json]
                e_start.click(
                    fn=run_evaluation, inputs=e_inputs,
                    outputs=[e_cmd, e_log, e_status, e_start, e_stop, e_report_md],
                    concurrency_id="job", concurrency_limit=1,
                )
                e_stop.click(fn=stop_current, inputs=None, outputs=e_status)
                e_refresh.click(fn=rescan_adapters, inputs=None, outputs=e_adapter)

            # ---------------------------------------------------------- Compare
            with gr.Tab("Compare"):
                gr.Markdown(
                    "Three-way generation comparison: the untrained student, the same "
                    "student plus the distilled LoRA, and the teacher. Models load on "
                    "demand - they are not held in memory while you train."
                )
                with gr.Row():
                    c_profile = gr.Dropdown(PROFILES, value="auto", label="Profile",
                                            info="Supplies the student and teacher ids")
                    c_adapter = gr.Dropdown(
                        discover_adapters(), value=None, label="Adapter",
                        allow_custom_value=True, info="blank = newest discoverable")
                    c_load = gr.Button("Load models", variant="secondary", scale=0)
                c_status = gr.HTML(status_html("idle", "Models not loaded."))

                with gr.Row():
                    c_prompt = gr.Textbox(label="Prompt", lines=3, scale=4,
                                          placeholder="Ask something all three can attempt...")
                    with gr.Column(scale=1, min_width=170):
                        c_run = gr.Button("Generate comparison", variant="primary", size="lg")
                        c_clear = gr.Button("Clear", size="sm")

                gr.Examples(
                    examples=[[p] for p in EXAMPLE_PROMPTS], inputs=[c_prompt],
                    label="Prompts where the distilled adapter measurably wins",
                    examples_per_page=4,
                )

                with gr.Accordion("Generation controls", open=False):
                    with gr.Row():
                        c_temp = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Temperature")
                        c_tokens = gr.Slider(16, 128, value=64, step=8, label="Max new tokens")
                        c_rep = gr.Slider(1.0, 1.3, value=1.15, step=0.01,
                                          label="Repetition penalty")

                c_headers, c_outputs = [], []
                titles = [("Original Student", "baseline, no adapter"),
                          ("Distilled Student", "base + GKD LoRA"),
                          ("Teacher Reference", "what was distilled from")]
                with gr.Row(equal_height=True):
                    for title, subtitle in titles:
                        with gr.Column():
                            header = gr.HTML(f"<div class='kd-col-head'>{title}</div>"
                                             f"<div class='kd-col-sub'>{subtitle}</div>")
                            response = gr.Textbox(lines=12, max_lines=20, buttons=["copy"],
                                                  interactive=False, show_label=False)
                            metrics = gr.Markdown(AWAITING, elem_classes="kd-metrics")
                            c_headers.append(header)
                            c_outputs.extend([response, metrics])

                c_load.click(fn=load_compare_models, inputs=[c_profile, c_adapter],
                             outputs=[c_status, *c_headers])
                c_controls = [c_prompt, c_temp, c_tokens, c_rep]
                c_run.click(fn=run_compare, inputs=c_controls, outputs=c_outputs)
                c_prompt.submit(fn=run_compare, inputs=c_controls, outputs=c_outputs)
                c_clear.click(fn=lambda: tuple([""] + ["", AWAITING] * 3),
                              inputs=None, outputs=[c_prompt] + c_outputs)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control-plane UI for the knowledge-distillation pipeline.")
    parser.add_argument("--host", default=None, help=f"Bind address (default {SERVER_NAME})")
    parser.add_argument("-p", "--port", type=int, default=None,
                        help=f"Port (default {SERVER_PORT})")
    parser.add_argument("--share", action="store_true", help="Create a public share link.")
    parser.add_argument("--open", action="store_true", help="Open a browser on launch.")
    return parser.parse_args()


def main():
    args = parse_args()
    host = args.host or os.environ.get("KD_UI_HOST") or SERVER_NAME
    port = int(args.port or os.environ.get("KD_UI_PORT") or SERVER_PORT)

    backend = f"runner {RUNNER}" if USE_RUNNER else f"direct ({sys.executable})"
    print("=" * 70)
    print(" Knowledge distillation control panel")
    print(f" backend  : {backend}")
    print(f" profile  : auto -> {auto_profile()}")
    print(f" adapters : {', '.join(discover_adapters()) or '(none found yet)'}")
    print("=" * 70)
    if RUNNER and not BASH:
        print(" !! distill.sh found but no bash on PATH - falling back to direct "
              "invocation.\n    Install Git Bash to drive the runner itself.")

    demo = build_ui()
    print(f"\nLaunching on http://{host}:{port}\n")
    # Stop and Rescan must stay responsive while a job streams, so the default limit is
    # above one; the job events themselves share a single-slot lane.
    demo.queue(default_concurrency_limit=8).launch(
        server_name=host,
        server_port=port,
        share=args.share,
        theme=gr.themes.Ocean(),
        css=CUSTOM_CSS,
        show_error=True,
        inbrowser=args.open,
    )


if __name__ == "__main__":
    main()
