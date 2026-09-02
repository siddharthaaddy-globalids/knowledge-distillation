"""
Side-by-side comparison UI for the SmolLM2 knowledge-distillation PoC.

Serves three CPU-resident models behind a Gradio interface:
  1. Original Student  - HuggingFaceTB/SmolLM2-135M-Instruct
  2. Distilled Student - the same base + the GKD-trained LoRA adapter
  3. Teacher Reference - HuggingFaceTB/SmolLM2-360M-Instruct

Run with:  python app.py   (then open http://127.0.0.1:7860)
"""

import os
import time
import traceback

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mirror the training script: use every core available on this CPU box.
NUM_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CORES)

STUDENT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEACHER_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
# Prefer the scaled 300-step adapter; fall back to the 50-step PoC if it is absent.
# Override with:  set KD_ADAPTER_PATH=./some/other/final_adapter
ADAPTER_CANDIDATES = [
    "./distilled_smollm_scaled/final_adapter",
    "./distilled_smollm_poc/final_adapter",
]


def _resolve_adapter_path():
    override = os.environ.get("KD_ADAPTER_PATH")
    if override:
        return override
    for candidate in ADAPTER_CANDIDATES:
        if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            return candidate
    return ADAPTER_CANDIDATES[0]


ADAPTER_PATH = _resolve_adapter_path()

DEVICE = "cpu"
COMPUTE_DTYPE = torch.float32
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860

# Prompts where the base model fails and the distilled model succeeds in 3/3 runs.
# The measured difference is instruction-shape compliance: the baseline cannot suppress
# a "Here is the list of..." preamble or honour an exact item count. See DEMO_PROMPTS.md
# for the recorded outputs and for the prompts that are NOT worth demoing.
EXAMPLE_PROMPTS = [
    "List three states of matter. Use a numbered list.",
    "Name three programming data types, formatted as a numbered list.",
    "What are three ways to reduce household energy use? Present them as a numbered list.",
    "List three renewable energy sources. Use a numbered list.",
]


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class ModelBundle:
    """A single loaded model plus the metadata the UI renders around it."""

    def __init__(self, key, title, subtitle, model=None, tokenizer=None, error=None):
        self.key = key
        self.title = title
        self.subtitle = subtitle
        self.model = model
        self.tokenizer = tokenizer
        self.error = error

    @property
    def available(self):
        return self.model is not None and self.tokenizer is not None

    @property
    def param_count(self):
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def param_summary(self):
        if not self.available:
            return "unavailable"
        summary = f"{self.param_count / 1e6:.1f}M params"
        if self.key == "distilled":
            adapter = sum(
                p.numel() for n, p in self.model.named_parameters() if "lora_" in n
            )
            if adapter:
                summary += f" (incl. {adapter / 1e6:.2f}M LoRA)"
        return summary


MODELS = {}


def _load_base_causal_lm(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=COMPUTE_DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return model


def _load_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_all_models():
    """Load and cache every model once, at startup, in eval() mode."""
    print("=" * 70)
    print(f" Loading comparison models on {DEVICE.upper()} "
          f"({COMPUTE_DTYPE}, {NUM_CORES} torch threads)")
    print("=" * 70)

    adapter_ok = os.path.isdir(ADAPTER_PATH) and os.path.isfile(
        os.path.join(ADAPTER_PATH, "adapter_config.json")
    )
    if not adapter_ok:
        print(f" !! Adapter not found at '{ADAPTER_PATH}'. The distilled column "
              f"will report the error; run main.py to train it.")

    # The adapter directory carries its own tokenizer copy - the one the student was
    # actually distilled against. Fall back to the hub tokenizer if it is missing.
    tokenizer_source = ADAPTER_PATH if adapter_ok else STUDENT_MODEL_ID
    print(f"\n[1/4] Loading student tokenizer from '{tokenizer_source}'...")
    try:
        student_tokenizer = _load_tokenizer(tokenizer_source)
        print(" -> ok")
    except Exception as exc:
        print(f" !! Tokenizer load failed: {exc}")
        student_tokenizer = None

    # --- 1. Original baseline student -------------------------------------- #
    print(f"[2/4] Loading original student ({STUDENT_MODEL_ID})...")
    try:
        baseline = _load_base_causal_lm(STUDENT_MODEL_ID)
        MODELS["original"] = ModelBundle(
            "original", "Original Student", "SmolLM2-135M-Instruct (baseline)",
            model=baseline, tokenizer=student_tokenizer,
        )
        print(" -> ok")
    except Exception as exc:
        traceback.print_exc()
        MODELS["original"] = ModelBundle(
            "original", "Original Student", "SmolLM2-135M-Instruct (baseline)",
            error=f"Failed to load base student: {exc}",
        )

    # --- 2. Distilled student (base + LoRA) --------------------------------- #
    # NOTE: PeftModel.from_pretrained injects LoRA modules into the object it is
    # handed, so a *second, independent* base instance is loaded here. Reusing the
    # baseline object would silently turn the "original" column into the distilled
    # one as well.
    print(f"[3/4] Loading distilled student ({STUDENT_MODEL_ID} + LoRA adapter)...")
    if not adapter_ok:
        MODELS["distilled"] = ModelBundle(
            "distilled", "Distilled Student", "SmolLM2-135M-Instruct + LoRA adapter",
            error=(f"Adapter directory '{ADAPTER_PATH}' is missing or has no "
                   f"adapter_config.json. Run `python main.py` to produce it."),
        )
        print(" -> skipped (adapter missing)")
    else:
        try:
            distilled_base = _load_base_causal_lm(STUDENT_MODEL_ID)
            distilled = PeftModel.from_pretrained(distilled_base, ADAPTER_PATH).to(DEVICE)
            distilled.eval()
            MODELS["distilled"] = ModelBundle(
                "distilled", "Distilled Student", "SmolLM2-135M-Instruct + LoRA adapter",
                model=distilled, tokenizer=student_tokenizer,
            )
            print(" -> ok")
        except Exception as exc:
            traceback.print_exc()
            MODELS["distilled"] = ModelBundle(
                "distilled", "Distilled Student", "SmolLM2-135M-Instruct + LoRA adapter",
                error=f"Failed to attach adapter: {exc}",
            )

    # --- 3. Teacher reference ----------------------------------------------- #
    print(f"[4/4] Loading teacher ({TEACHER_MODEL_ID})...")
    try:
        teacher = _load_base_causal_lm(TEACHER_MODEL_ID)
        MODELS["teacher"] = ModelBundle(
            "teacher", "Teacher Reference", "SmolLM2-360M-Instruct",
            model=teacher, tokenizer=_load_tokenizer(TEACHER_MODEL_ID),
        )
        print(" -> ok")
    except Exception as exc:
        traceback.print_exc()
        MODELS["teacher"] = ModelBundle(
            "teacher", "Teacher Reference", "SmolLM2-360M-Instruct",
            error=f"Failed to load teacher: {exc}",
        )

    print("\nModel registry ready:")
    for bundle in MODELS.values():
        state = bundle.param_summary() if bundle.available else f"ERROR - {bundle.error}"
        print(f"  - {bundle.title:<20} {state}")
    print()
    return MODELS


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def build_prompt(tokenizer, user_prompt):
    """Apply the model's own chat template; fall back to the raw prompt if absent."""
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        print(f" !! chat template unavailable ({exc}); using raw prompt")
        return user_prompt


def _metrics_error(message):
    return f"<span style='color:#f87171'>&#9888; {message}</span>"


def generate(bundle, user_prompt, temperature, max_new_tokens, repetition_penalty):
    """Run one model over one prompt; returns (response_text, metrics_markdown)."""
    if not bundle.available:
        return "", _metrics_error(bundle.error or "Model unavailable")

    tokenizer, model = bundle.tokenizer, bundle.model
    formatted = build_prompt(tokenizer, user_prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    prompt_tokens = inputs.input_ids.shape[1]

    started = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                do_sample=True,
                top_p=0.9,
                repetition_penalty=float(repetition_penalty),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        traceback.print_exc()
        return "", _metrics_error(f"Generation failed: {exc}")
    elapsed = time.perf_counter() - started

    completion_ids = outputs[0][prompt_tokens:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    new_tokens = int(completion_ids.shape[0])
    tps = new_tokens / elapsed if elapsed > 0 else 0.0
    metrics = (
        f"**{elapsed:.2f}s** latency &nbsp;|&nbsp; "
        f"**{tps:.1f}** tok/s &nbsp;|&nbsp; "
        f"{new_tokens} tokens &nbsp;|&nbsp; "
        f"{len(text.split())} words &nbsp;|&nbsp; "
        f"{len(text)} chars"
    )
    return (text or "(empty response)"), metrics


def compare(user_prompt, temperature, max_new_tokens, repetition_penalty):
    """Gradio callback: run all three models over the same prompt, in order."""
    if not user_prompt or not user_prompt.strip():
        warn = _metrics_error("Enter a prompt first.")
        return "", warn, "", warn, "", warn

    user_prompt = user_prompt.strip()
    print(f"\n[compare] prompt={user_prompt!r} temp={temperature} "
          f"max_new_tokens={max_new_tokens} rep_penalty={repetition_penalty}")

    results = []
    for key in ("original", "distilled", "teacher"):
        text, metrics = generate(
            MODELS[key], user_prompt, temperature, max_new_tokens, repetition_penalty
        )
        print(f"  {key:<10} -> {metrics}")
        results.extend([text, metrics])
    return tuple(results)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
#kd-header h1 { margin-bottom: 0.15rem; }
.kd-metrics {
    font-size: 0.82rem;
    opacity: 0.88;
    border-top: 1px solid var(--border-color-primary);
    padding-top: 0.5rem;
    margin-top: 0.25rem;
}
.kd-col-head { font-weight: 600; margin-bottom: 0.2rem; }
.kd-col-sub { font-size: 0.8rem; opacity: 0.7; }
"""


def _column_header(bundle):
    status = "" if bundle.available else " <span style='color:#f87171'>offline</span>"
    return (
        f"<div class='kd-col-head'>{bundle.title}{status}</div>"
        f"<div class='kd-col-sub'>{bundle.subtitle} &nbsp;|&nbsp; "
        f"{bundle.param_summary()}</div>"
    )


def build_ui():
    with gr.Blocks(title="SmolLM2 Distillation Comparison", fill_width=True) as demo:
        adapter_state = (
            f"`{ADAPTER_PATH}`" if MODELS["distilled"].available
            else f"`{ADAPTER_PATH}` (**not loaded**)"
        )
        gr.Markdown(
            "# SmolLM2 Knowledge Distillation - 3-Way Comparison\n"
            f"Generalized Knowledge Distillation (GKD + LoRA) served on "
            f"**CPU / float32**, {NUM_CORES} torch threads.\n\n"
            f"**Student base:** `{STUDENT_MODEL_ID}` &nbsp;|&nbsp; "
            f"**Teacher:** `{TEACHER_MODEL_ID}` &nbsp;|&nbsp; "
            f"**Adapter:** {adapter_state}",
            elem_id="kd-header",
        )

        if not MODELS["distilled"].available:
            gr.Markdown(
                f"<div style='padding:0.75rem;border-radius:8px;"
                f"border:1px solid #f87171;color:#f87171;'>"
                f"<b>Distilled model unavailable.</b> {MODELS['distilled'].error}</div>"
            )

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="Ask something the student and teacher can both attempt...",
                lines=3,
                scale=4,
            )
            with gr.Column(scale=1, min_width=170):
                run_btn = gr.Button("Generate Comparison", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="sm")

        gr.Examples(
            examples=[[p] for p in EXAMPLE_PROMPTS],
            inputs=[prompt_box],
            label="Example prompts - cases where the distilled adapter measurably wins",
            examples_per_page=4,
        )

        with gr.Accordion("Generation controls", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    0.1, 1.0, value=0.3, step=0.05, label="Temperature",
                    info="Lower = more deterministic",
                )
                max_new_tokens = gr.Slider(
                    16, 128, value=64, step=8, label="Max new tokens",
                    info="CPU latency scales roughly linearly with this",
                )
                repetition_penalty = gr.Slider(
                    1.0, 1.3, value=1.15, step=0.01, label="Repetition penalty",
                    info="Higher discourages loops in the small student",
                )

        outputs = []
        with gr.Row(equal_height=True):
            for key in ("original", "distilled", "teacher"):
                bundle = MODELS[key]
                with gr.Column():
                    gr.HTML(_column_header(bundle))
                    response = gr.Textbox(
                        label=bundle.title, lines=12, max_lines=20,
                        buttons=["copy"], interactive=False, show_label=False,
                    )
                    metrics = gr.Markdown(
                        "_awaiting generation_", elem_classes="kd-metrics"
                    )
                    outputs.extend([response, metrics])

        controls = [prompt_box, temperature, max_new_tokens, repetition_penalty]
        run_btn.click(fn=compare, inputs=controls, outputs=outputs)
        prompt_box.submit(fn=compare, inputs=controls, outputs=outputs)
        clear_btn.click(
            fn=lambda: tuple([""] + ["", "_awaiting generation_"] * 3),
            inputs=None,
            outputs=[prompt_box] + outputs,
        )

    return demo


def main():
    load_all_models()
    demo = build_ui()
    print(f"Launching Gradio on http://{SERVER_NAME}:{SERVER_PORT}\n")
    demo.queue(default_concurrency_limit=1).launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        theme=gr.themes.Ocean(),
        css=CUSTOM_CSS,
        show_error=True,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
