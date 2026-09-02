"""
Configuration loading and hardware resolution for the distillation pipeline.

Everything tunable lives in a YAML file (see `configs/`). Nothing in this module is
platform-specific at import time: the same config runs on Windows/Linux CPU, Apple
Silicon (MPS / unified memory) and CUDA, with the device layer resolved at runtime.

Precedence, lowest to highest:
    built-in DEFAULTS  <  --config file  <  KD_* environment variables  <  CLI flags
"""

import copy
import os

import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced as a clear message to the user
    raise SystemExit(
        "PyYAML is required for config loading. Install it with:  uv add pyyaml"
    ) from exc


# --------------------------------------------------------------------------- #
# Built-in defaults - a config file only needs to override what it changes.
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "project": {
        "name": "smollm2-distillation",
        "output_dir": "./distilled_smollm_scaled",
        "seed": 42,
    },
    "models": {
        "student": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "teacher": "HuggingFaceTB/SmolLM2-360M-Instruct",
        # Tokenizer source: "teacher", "student", or an explicit path/hub id.
        # Student and teacher MUST share a vocabulary for standard GKD.
        "tokenizer": "teacher",
    },
    "hardware": {
        # auto | cpu | mps | cuda
        "device": "auto",
        # auto | float32 | bfloat16 | float16
        #   auto -> float32 on CPU/MPS, bfloat16 on CUDA
        "dtype": "auto",
        # CPU thread count; 0 or null means "all cores". Ignored off CPU.
        "threads": 0,
        # Apple Silicon: allow the MPS allocator to spill to unified memory instead of
        # hard-failing when a graph exceeds the recommended working-set size.
        "mps_high_watermark_ratio": 0.0,
        # Route MPS-unsupported ops to CPU rather than raising.
        "mps_fallback": True,
    },
    "dataset": {
        "source": "HuggingFaceTB/smoltalk",
        "max_prompt_tokens": 128,
        "max_total_tokens": 384,
        "validation_size": 50,
        "include_synthetic": True,
        "domains": [
            {"name": "everyday-conversations", "config": "everyday-conversations", "quota": 320, "pool": 1800},
            {"name": "summarization", "config": "smol-summarize", "quota": 240, "pool": 14000},
            {"name": "rewriting", "config": "smol-rewrite", "quota": 240, "pool": 2500},
            {"name": "constraints", "config": "smol-constraints", "quota": 320, "pool": 900},
            {"name": "factual-qa", "config": "openhermes-100k", "quota": 380, "pool": 1500},
        ],
    },
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    },
    "training": {
        "max_steps": 300,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3.0e-4,
        "lr_scheduler_type": "cosine",
        # Float in [0,1) is a ratio of total steps; an int >= 1 is an exact step count.
        # transformers 5.x removed `warmup_ratio` in favour of this dual-typed field.
        "warmup": 0.05,
        "max_grad_norm": 1.0,
        "logging_steps": 1,
        "save_steps": 100,
        "save_total_limit": 2,
        "eval_enabled": True,
        "benchmark_every": 100,
    },
    "gkd": {
        "lmbda": 0.5,
        "beta": 0.5,
        "temperature": 0.7,
        "max_new_tokens": 40,
        "seq_kd": False,
    },
    "benchmark_prompts": [
        "List three states of matter. Use a numbered list.",
        "Solve for x: 3x + 12 = 27.",
        "Explain why the sky looks blue to human eyes in two sentences.",
    ],
}


# Environment overrides. These are what the generated distill.sh sets, so a downloaded
# runner can be retargeted without touching YAML at all.
ENV_OVERRIDES = {
    "KD_STUDENT_MODEL": ("models", "student", str),
    "KD_TEACHER_MODEL": ("models", "teacher", str),
    "KD_TOKENIZER": ("models", "tokenizer", str),
    "KD_DATASET": ("dataset", "source", str),
    "KD_OUTPUT_DIR": ("project", "output_dir", str),
    "KD_DEVICE": ("hardware", "device", str),
    "KD_DTYPE": ("hardware", "dtype", str),
    "KD_THREADS": ("hardware", "threads", int),
    "KD_MAX_STEPS": ("training", "max_steps", int),
    "KD_BATCH_SIZE": ("training", "batch_size", int),
    "KD_GRAD_ACCUM": ("training", "gradient_accumulation_steps", int),
    "KD_LEARNING_RATE": ("training", "learning_rate", float),
    "KD_LORA_R": ("lora", "r", int),
    "KD_LORA_ALPHA": ("lora", "alpha", int),
    "KD_LMBDA": ("gkd", "lmbda", float),
    "KD_BETA": ("gkd", "beta", float),
    "KD_MAX_NEW_TOKENS": ("gkd", "max_new_tokens", int),
    "KD_SEED": ("project", "seed", int),
}


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base`. Lists replace wholesale."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_env(config):
    """Overlay KD_* environment variables onto the config."""
    applied = []
    for env_name, (section, key, caster) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            config.setdefault(section, {})[key] = caster(raw)
            applied.append(f"{env_name}={raw}")
        except (TypeError, ValueError):
            print(f" !! ignoring {env_name}={raw!r} (not a valid {caster.__name__})")
    return applied


def load_config(path=None, overrides=None):
    """Build the effective config from defaults, an optional YAML file, env vars and CLI."""
    config = copy.deepcopy(DEFAULTS)
    source = "built-in defaults"

    if path:
        if not os.path.isfile(path):
            raise SystemExit(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"Config file {path} must contain a YAML mapping at the top level.")
        config = _deep_merge(config, loaded)
        source = path

    applied_env = _apply_env(config)
    config = _deep_merge(config, overrides or {})

    config["_meta"] = {"source": source, "env_overrides": applied_env}
    return config


# --------------------------------------------------------------------------- #
# Hardware resolution
# --------------------------------------------------------------------------- #
def _mps_available():
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def resolve_device(config):
    """Pick the device and dtype, and apply the platform-specific runtime tweaks.

    Returns a dict with:
        device      torch device string ("cpu" | "mps" | "cuda")
        dtype       torch dtype for model weights
        use_cpu     value for TrainingArguments.use_cpu
        bf16        value for TrainingArguments.bf16
        fp16        value for TrainingArguments.fp16
        pin_memory  value for TrainingArguments.dataloader_pin_memory
        notes       human-readable lines describing what was chosen and why
    """
    hardware = config.get("hardware", {})
    requested = str(hardware.get("device", "auto")).lower()
    requested_dtype = str(hardware.get("dtype", "auto")).lower()
    notes = []

    # --- device ------------------------------------------------------------ #
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif _mps_available():
            device = "mps"
        else:
            device = "cpu"
        notes.append(f"device=auto resolved to '{device}'")
    else:
        device = requested
        if device == "cuda" and not torch.cuda.is_available():
            notes.append("!! cuda requested but unavailable; falling back to cpu")
            device = "cpu"
        elif device == "mps" and not _mps_available():
            notes.append("!! mps requested but unavailable (needs Apple Silicon + macOS); "
                         "falling back to cpu")
            device = "cpu"

    # --- Apple Silicon unified memory --------------------------------------- #
    if device == "mps":
        if hardware.get("mps_fallback", True):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            notes.append("PYTORCH_ENABLE_MPS_FALLBACK=1 (unsupported ops run on CPU)")
        ratio = hardware.get("mps_high_watermark_ratio", 0.0)
        if ratio is not None:
            # 0.0 lifts the allocator ceiling so large graphs can use the full unified
            # memory pool instead of aborting at the default working-set limit.
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(ratio))
            notes.append(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO={ratio} (unified memory pool)")

    # --- dtype --------------------------------------------------------------- #
    bf16_ok = False
    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
    elif device == "mps":
        # transformers gates bf16 on macOS >= 14.0; below that it is unsupported, and on
        # M1/M2 it is emulated in software over fp32 rather than run natively.
        try:
            bf16_ok = torch.backends.mps.is_macos_or_newer(14, 0)
        except Exception:
            bf16_ok = False

    if requested_dtype == "auto":
        if device == "cuda":
            dtype, dtype_name = (torch.bfloat16, "bfloat16") if bf16_ok else (torch.float16, "float16")
        else:
            # float32 is the safe default on both CPU and MPS.
            dtype, dtype_name = torch.float32, "float32"
        notes.append(f"dtype=auto resolved to {dtype_name}")
    else:
        mapping = {"float32": torch.float32, "fp32": torch.float32,
                   "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                   "float16": torch.float16, "fp16": torch.float16}
        if requested_dtype not in mapping:
            raise SystemExit(f"Unknown hardware.dtype '{requested_dtype}'. "
                             f"Use one of: auto, float32, bfloat16, float16")
        dtype = mapping[requested_dtype]
        dtype_name = requested_dtype
        if dtype is torch.bfloat16 and device == "mps" and not bf16_ok:
            notes.append("!! bfloat16 requested on MPS but macOS < 14.0; using float32")
            dtype, dtype_name = torch.float32, "float32"
        if dtype is not torch.float32 and device == "cpu":
            notes.append(f"!! {dtype_name} on CPU is slow and often unstable; using float32")
            dtype, dtype_name = torch.float32, "float32"

    # --- trainer flags -------------------------------------------------------- #
    # Trainer mixed-precision flags are meaningful only on an accelerator. On MPS the
    # autocast/grad-scaler path is not the same as CUDA's, so they stay off and the
    # dtype is carried by the model weights instead.
    use_cpu = device == "cpu"
    bf16 = bool(device == "cuda" and dtype is torch.bfloat16)
    fp16 = bool(device == "cuda" and dtype is torch.float16)
    pin_memory = device == "cuda"

    # --- CPU threads ---------------------------------------------------------- #
    if device == "cpu":
        threads = hardware.get("threads") or (os.cpu_count() or 4)
        torch.set_num_threads(int(threads))
        notes.append(f"torch threads = {threads}")

    return {
        "device": device,
        "dtype": dtype,
        "dtype_name": dtype_name,
        "use_cpu": use_cpu,
        "bf16": bf16,
        "fp16": fp16,
        "pin_memory": pin_memory,
        "notes": notes,
    }


def describe(config, hardware):
    """Render a short startup banner summarising the effective configuration."""
    meta = config.get("_meta", {})
    lines = [
        "=" * 78,
        f" {config['project']['name']}",
        "=" * 78,
        f" config source : {meta.get('source', 'built-in defaults')}",
        f" student       : {config['models']['student']}",
        f" teacher       : {config['models']['teacher']}",
        f" dataset       : {config['dataset']['source']}",
        f" device        : {hardware['device']} ({hardware['dtype_name']})",
        f" steps         : {config['training']['max_steps']} "
        f"(effective batch {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']})",
        f" lora          : r={config['lora']['r']} alpha={config['lora']['alpha']}",
        f" output        : {config['project']['output_dir']}",
    ]
    if meta.get("env_overrides"):
        lines.append(f" env overrides : {', '.join(meta['env_overrides'])}")
    for note in hardware["notes"]:
        lines.append(f"   - {note}")
    lines.append("=" * 78)
    return "\n".join(lines)
