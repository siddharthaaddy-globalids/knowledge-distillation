"""
Configuration loading, validation and hardware resolution.

Every default lives in `configs/_base.yaml`. Nothing in this module carries a
training value: it only merges layers, checks the result, and resolves the
device. That is what makes the YAML you point at the whole truth about a run.

Precedence, lowest to highest:

    configs/_base.yaml  <  profile  <  KD_* env vars  <  --set  <  explicit flag

A profile opts into the base layer with `extends: _base.yaml` at the top; the
path is resolved relative to the profile, and the chain may be any depth as
long as it does not loop back on itself.

Unknown keys are a hard error rather than a silent no-op. A run that quietly
ignores `traning: {max_steps: 500}` wastes the whole training budget before
anyone notices, so a typo is worth failing on immediately.
"""

import copy
import difflib
import os
import re

import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced as a clear message to the user
    raise SystemExit(
        "PyYAML is required for config loading. Install it with:  uv add pyyaml"
    ) from exc


# The key a profile uses to name its parent. Stripped before validation, since it
# describes how the config was assembled rather than anything about the run.
EXTENDS_KEY = "extends"

# Where _base.yaml lives, so a profile can be loaded from anywhere on disk.
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")
BASE_CONFIG = os.path.join(CONFIG_DIR, "_base.yaml")


# --------------------------------------------------------------------------- #
# Environment overrides.
#
# These are what a generated runner sets, so a downloaded distill.sh can be
# retargeted without touching YAML. They sit below --set and explicit flags:
# something typed on the command line always beats something inherited from the
# shell.
# --------------------------------------------------------------------------- #
ENV_OVERRIDES = {
    "KD_STUDENT_MODEL": ("models.student", str),
    "KD_TEACHER_MODEL": ("models.teacher", str),
    "KD_TEACHER_ADAPTER": ("models.teacher_adapter", str),
    "KD_TOKENIZER": ("models.tokenizer", str),
    "KD_DATASET": ("dataset.source", str),
    "KD_OUTPUT_DIR": ("project.output_dir", str),
    "KD_RUNS_DIR": ("project.runs_dir", str),
    "KD_DEVICE": ("hardware.device", str),
    "KD_DTYPE": ("hardware.dtype", str),
    "KD_THREADS": ("hardware.threads", int),
    "KD_MAX_STEPS": ("training.max_steps", int),
    "KD_BATCH_SIZE": ("training.batch_size", int),
    "KD_GRAD_ACCUM": ("training.gradient_accumulation_steps", int),
    "KD_LEARNING_RATE": ("training.learning_rate", float),
    "KD_LORA_R": ("lora.r", int),
    "KD_LORA_ALPHA": ("lora.alpha", int),
    "KD_LMBDA": ("gkd.lmbda", float),
    "KD_BETA": ("gkd.beta", float),
    "KD_MAX_NEW_TOKENS": ("gkd.max_new_tokens", int),
    "KD_SEED": ("project.seed", int),
    "KD_EVAL_TASKS": ("evaluation.tasks", str),
    "KD_EVAL_SAMPLES": ("evaluation.samples", int),
    "KD_MAX_RUNTIME_MINUTES": ("limits.max_runtime_minutes", int),
    "KD_MAX_COST_USD": ("limits.max_cost_usd", float),
}


# --------------------------------------------------------------------------- #
# Lists of mappings that carry their own key vocabulary. The generic validator
# below cannot check these against _base.yaml position by position, because a
# profile is free to define a different number of entries than the base does.
# --------------------------------------------------------------------------- #
DOMAIN_KEYS = {
    "name", "config", "quota", "pool",
    # Alpaca-format datasets (instruction / input / output) rather than chat turns.
    "format", "instruction_column", "input_column", "output_column",
}
STAGE_KEYS = {"name", "gate"}

# Mappings whose KEYS are data rather than settings. Everywhere else an unknown
# key is a typo worth failing on; here the whole point is that the caller chooses
# the names, so validating them would reject every legitimate use.
OPAQUE_MAPPINGS = ["runpod.extra_env"]


class ConfigError(SystemExit):
    """A configuration problem stated plainly enough to act on without a traceback."""


# YAML 1.1, which PyYAML implements, only recognises a float in exponent form when
# it has both a decimal point and a signed exponent: 3.0e-4 is a number, 1e-5 is
# the string "1e-5". That is a trap for anyone typing a learning rate, so these
# forms are recognised explicitly rather than left to fail as a type error.
_SCIENTIFIC = re.compile(r"^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$")


def _maybe_number(text):
    """Return `text` as a float if it is scientific notation YAML 1.1 misses, else None."""
    if isinstance(text, str) and _SCIENTIFIC.match(text.strip()):
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _yaml_number_literal(number):
    """Format a float the way PyYAML will actually read back as a number."""
    text = repr(float(number))
    if "e" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        if exponent and exponent[0] not in "+-":
            exponent = "+" + exponent
        text = f"{mantissa}e{exponent}"
    return text


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base`. Lists replace wholesale.

    Lists are replaced rather than concatenated because every list in this config
    is a complete specification - target_modules, domains, benchmark_prompts - and
    appending to one would silently keep entries the profile meant to drop.
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _flatten(node, prefix=""):
    """Flatten nested mappings to dotted paths. Lists are leaves, not containers."""
    flat = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _set_path(config, path, value):
    """Assign a dotted path, creating intermediate mappings as needed."""
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _get_path(config, path, default=None):
    """Read a dotted path, returning `default` if any segment is missing."""
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------- #
# Loading a file and its extends chain
# --------------------------------------------------------------------------- #
def _read_yaml(path):
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level.")
    return loaded


def _load_chain(path, seen=None):
    """Resolve a config and everything it extends, parent first.

    Returns (merged_mapping, chain) where chain lists the files that contributed,
    base first. The chain goes into the run manifest so a run can be reproduced
    from the files it actually read rather than from the one that was named.
    """
    path = os.path.abspath(path)
    seen = seen or []
    if path in seen:
        loop = " -> ".join(os.path.basename(p) for p in seen + [path])
        raise ConfigError(f"Circular 'extends' in config chain: {loop}")

    raw = _read_yaml(path)
    parent_ref = raw.pop(EXTENDS_KEY, None)

    if parent_ref is None:
        return raw, [path]

    parent_path = parent_ref if os.path.isabs(parent_ref) else os.path.join(
        os.path.dirname(path), parent_ref)
    if not os.path.isfile(parent_path):
        raise ConfigError(
            f"{os.path.basename(path)} extends '{parent_ref}', which does not exist.\n"
            f"  looked for: {parent_path}")

    parent, chain = _load_chain(parent_path, seen + [path])
    return _deep_merge(parent, raw), chain + [path]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _kind(value):
    """Coarse type name used for comparison. int and float are one kind.

    Booleans are checked before numbers: in Python `True` is an int, and treating
    `eval_enabled: 1` as valid would let a genuine mistake through.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "mapping"
    return type(value).__name__


def _suggest(unknown, candidates):
    """'did you mean' for a mistyped key, or an empty string when nothing is close.

    The whole dotted path is compared first. Comparing only the last segment would
    match `traning.max_steps` against `limits.max_steps` just as readily as against
    `training.max_steps`, and point at the wrong section.
    """
    close = difflib.get_close_matches(unknown, sorted(candidates), n=1, cutoff=0.7)
    if close:
        return f"  did you mean '{close[0]}'?"

    # Nothing matched whole; the section may be right and only the leaf wrong, or
    # the key may have been written without its section at all.
    leaves = {}
    for candidate in sorted(candidates):
        leaves.setdefault(candidate.split(".")[-1], candidate)
    close = difflib.get_close_matches(unknown.split(".")[-1], list(leaves), n=1, cutoff=0.6)
    if close:
        return f"  did you mean '{leaves[close[0]]}'?"
    return ""


def _validate_list_of_mappings(items, allowed, path, errors):
    """Check a list whose entries are mappings with a fixed key vocabulary."""
    if not isinstance(items, list):
        errors.append(f"{path} must be a list, got {_kind(items)}")
        return
    for index, item in enumerate(items):
        where = f"{path}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be a mapping, got {_kind(item)}")
            continue
        for key in item:
            if key not in allowed:
                errors.append(f"unknown key '{where}.{key}'{_suggest(key, allowed)}")


def validate(config, base):
    """Check `config` against the shape of `base`, raising with every problem at once.

    Reporting all of them together matters: fixing one typo, re-running, and
    discovering the next one is a slow way to learn that three keys were wrong.
    """
    errors = []

    # The two lists of mappings are checked by their own vocabularies, then removed
    # from the generic comparison so their contents are not matched position by
    # position against the base file.
    working = copy.deepcopy(config)
    domains = _get_path(working, "dataset.domains")
    if domains is not None:
        _validate_list_of_mappings(domains, DOMAIN_KEYS, "dataset.domains", errors)
        working["dataset"].pop("domains", None)
    stages = _get_path(working, "pipeline.stages")
    if stages is not None:
        _validate_list_of_mappings(stages, STAGE_KEYS, "pipeline.stages", errors)
        working["pipeline"].pop("stages", None)

    for path in OPAQUE_MAPPINGS:
        value = _get_path(working, path)
        if value is None:
            continue
        if not isinstance(value, dict):
            errors.append(f"{path} must be a mapping of names to values, "
                          f"got {_kind(value)}")
        section, _, leaf = path.rpartition(".")
        _get_path(working, section).pop(leaf, None)

    base_flat = _flatten(base)
    for path, value in _flatten(working).items():
        if path.startswith("_meta"):
            continue
        if path not in base_flat:
            errors.append(f"unknown key '{path}'{_suggest(path, base_flat)}")
            continue
        expected = base_flat[path]
        # A base default of null means the key is optional and untyped - there is
        # nothing to compare against, so anything the profile puts there is fine.
        if expected is None:
            continue
        if _kind(value) != _kind(expected) and _kind(value) != "null":
            hint = ""
            number = _maybe_number(value)
            if _kind(expected) == "number" and number is not None:
                # The value is a number written in a form YAML 1.1 does not read as
                # one. Say which form does: "should be number, got string" is
                # baffling when you are looking at what is obviously a number.
                hint = (f"\n      YAML reads exponents only with a decimal point and "
                        f"a signed exponent - write {_yaml_number_literal(number)}")
            errors.append(
                f"{path} should be {_kind(expected)}, got {_kind(value)} ({value!r}){hint}")

    if errors:
        listed = "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(f"Configuration is not valid:\n{listed}")


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def parse_set(assignments):
    """Turn ['training.max_steps=500', ...] into {'training.max_steps': 500}.

    Values are parsed as YAML, so ints, floats, booleans, null and inline lists
    all work and mean the same thing they would in the config file:

        --set training.max_steps=500
        --set gkd.seq_kd=true
        --set limits.max_cost_usd=null
        --set lora.target_modules='[q_proj, v_proj]'
    """
    parsed = {}
    for item in assignments or []:
        if "=" not in item:
            raise ConfigError(
                f"--set needs KEY=VALUE, got '{item}'\n"
                f"  example: --set training.max_steps=500")
        path, _, raw = item.partition("=")
        path = path.strip()
        if not path:
            raise ConfigError(f"--set needs a key before '=', got '{item}'")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        # --set training.learning_rate=1e-5 is the obvious thing to type, and YAML
        # hands it back as a string. Treat it as the number it plainly is.
        number = _maybe_number(value)
        parsed[path] = value if number is None else number
    return parsed


def _apply_env(config):
    """Overlay KD_* environment variables. Returns the paths it set."""
    applied = {}
    for env_name, (path, caster) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            _set_path(config, path, caster(raw))
            applied[path] = env_name
        except (TypeError, ValueError):
            raise ConfigError(
                f"{env_name}={raw!r} is not a valid {caster.__name__} "
                f"(it sets {path})")
    return applied


def _apply_paths(config, values):
    """Overlay a flat {dotted path: value} mapping. Returns the paths it set.

    Whether the path is a real setting is not decided here - validate() checks the
    merged result against the base schema, so a mistyped --set is reported in the
    same list as a mistyped YAML key rather than through a separate code path.
    """
    applied = {}
    for path, value in (values or {}).items():
        _set_path(config, path, value)
        applied[path] = value
    return applied


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #
def load_config(path=None, set_overrides=None, flag_overrides=None, use_env=True):
    """Build the effective config from every layer, validate it, and describe it.

    path            profile to load; None loads _base.yaml alone
    set_overrides   {dotted path: value} from --set
    flag_overrides  {dotted path: value} from explicit flags (highest precedence)
    use_env         whether KD_* environment variables are consulted

    The returned mapping carries a `_meta` block recording where the values came
    from. That block is written into the run bundle, and is what lets the startup
    banner say which layer won each setting that differs from the base.
    """
    base, base_chain = _load_chain(BASE_CONFIG)

    if path:
        config, chain = _load_chain(path)
        profile = os.path.splitext(os.path.basename(path))[0]
    else:
        config, chain = copy.deepcopy(base), list(base_chain)
        profile = "_base"

    # The profile chain may or may not include _base.yaml. Merging onto the base
    # regardless means a profile that forgets `extends:` still resolves, rather
    # than failing on the first key it did not happen to spell out.
    config = _deep_merge(base, config)

    provenance = {}
    if use_env:
        for path_key, env_name in _apply_env(config).items():
            provenance[path_key] = env_name
    for path_key in _apply_paths(config, parse_set(set_overrides) if isinstance(
            set_overrides, (list, tuple)) else set_overrides):
        provenance[path_key] = "--set"
    for path_key in _apply_paths(config, flag_overrides):
        provenance[path_key] = "flag"

    validate(config, base)

    base_flat = _flatten(base)
    changed = {
        p: {"value": v, "base": base_flat.get(p), "by": provenance.get(p, "profile")}
        for p, v in _flatten(config).items()
        if p in base_flat and v != base_flat[p]
    }

    config["_meta"] = {
        "profile": profile,
        "source": path or BASE_CONFIG,
        # Forward slashes regardless of platform: this string is read by people and
        # written into manifests that are compared across machines.
        "chain": [os.path.relpath(p, os.path.dirname(CONFIG_DIR)).replace(os.sep, "/")
                  for p in chain],
        "changed": changed,
        "overridden": {p: by for p, by in provenance.items()},
    }
    return config


def strip_meta(config):
    """A copy without `_meta`, for writing config.resolved.yaml or comparing runs."""
    clean = copy.deepcopy(config)
    clean.pop("_meta", None)
    return clean


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
            raise ConfigError(f"Unknown hardware.dtype '{requested_dtype}'. "
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


def describe(config, hardware, run_id=None):
    """Render the startup banner: what will run, and which layer decided each value."""
    meta = config.get("_meta", {})
    training = config["training"]
    lines = [
        "=" * 78,
        f" {config['project']['name']}",
        "=" * 78,
    ]
    if run_id:
        lines.append(f" run           : {run_id}")
    lines += [
        f" config        : {meta.get('source', BASE_CONFIG)}",
        f" chain         : {' -> '.join(meta.get('chain', [])) or '(base only)'}",
        f" student       : {config['models']['student']}",
        f" teacher       : {config['models']['teacher']}",
    ]
    if config["models"].get("teacher_adapter"):
        lines.append(f" teacher lora  : {config['models']['teacher_adapter']}")
    lines += [
        f" dataset       : {config['dataset']['source']}",
        f" device        : {hardware['device']} ({hardware['dtype_name']})",
        f" steps         : {training['max_steps']} "
        f"(effective batch {training['batch_size'] * training['gradient_accumulation_steps']})",
        f" lora          : r={config['lora']['r']} alpha={config['lora']['alpha']}",
    ]

    # Only values that differ from the base are worth printing, and only those set
    # by something other than the profile need saying where they came from: the
    # profile is the file already named two lines above.
    overridden = {p: by for p, by in meta.get("overridden", {}).items()}
    if overridden:
        lines.append(" overrides     :")
        changed = meta.get("changed", {})
        for path in sorted(overridden):
            entry = changed.get(path, {})
            was = entry.get("base")
            now = entry.get("value", _get_path(config, path))
            lines.append(f"   {path} = {now!r}  <- {overridden[path]}  (base: {was!r})")

    for note in hardware["notes"]:
        lines.append(f"   - {note}")
    lines.append("=" * 78)
    return "\n".join(lines)
