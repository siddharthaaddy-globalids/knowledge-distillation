"""
Resolving the paths in a config, wherever they point.

Anywhere the config names a model, an adapter or a dataset, that value may be:

    a Hugging Face id     Qwen/Qwen3.5-2B          left alone; transformers fetches it
    a local path          ./teacher-fixed          left alone
    an s3:// URI          s3://bucket/kd/teacher   fetched here, into a local cache

Resolution happens once, in the preflight stage, before anything tries to load a
model. That ordering is the point: a missing bucket or a bad credential should
fail in the first two seconds of a run, not forty minutes in when the training
loop finally reaches the teacher.

Downloads are cached by bucket and key, and a completed download leaves a marker
behind. A second run against the same URI costs one HEAD request rather than
several gigabytes - which matters most on a rented GPU, where the download is
billed at the GPU's hourly rate.

Nothing here imports boto3 at module level. Object storage is an optional extra,
and a machine that never uses it must not need it installed.
"""

import os
import sys

S3_SCHEME = "s3://"

# Written into a completed cache directory. Its contents are what the download
# consisted of, so a directory that was interrupted halfway is not mistaken for a
# finished one on the next run.
COMPLETE_MARKER = ".kd-complete"

# Config paths that may carry an s3:// URI. Kept explicit rather than discovered
# by scanning every string in the config: `project.name` looking like a URI should
# stay a name, and a typo elsewhere should not silently start a download.
RESOLVABLE = [
    "models.teacher",
    "models.student",
    "models.teacher_adapter",
    "dataset.source",
]


def is_remote(value):
    """True when this value names an object-storage location rather than a path."""
    return isinstance(value, str) and value.startswith(S3_SCHEME)


def split_uri(uri):
    """s3://bucket/some/key -> ("bucket", "some/key"). The key may be empty."""
    if not is_remote(uri):
        raise ValueError(f"not an s3 URI: {uri!r}")
    remainder = uri[len(S3_SCHEME):]
    bucket, _, key = remainder.partition("/")
    if not bucket:
        raise ValueError(f"s3 URI has no bucket: {uri!r}")
    return bucket, key.strip("/")


def cache_dir(config):
    """Where fetched objects are kept between runs."""
    configured = (config.get("s3") or {}).get("cache_dir") or "~/.cache/kd/s3"
    return os.path.abspath(os.path.expanduser(configured))


def cache_path(config, uri):
    """The local directory a given URI caches to. Deterministic, so runs share it."""
    bucket, key = split_uri(uri)
    return os.path.join(cache_dir(config), bucket, *key.split("/")) if key \
        else os.path.join(cache_dir(config), bucket)


def is_cached(local):
    """True when a previous run finished downloading here."""
    return os.path.isfile(os.path.join(local, COMPLETE_MARKER))


def mark_complete(local, detail=""):
    with open(os.path.join(local, COMPLETE_MARKER), "w", encoding="utf-8") as handle:
        handle.write(detail + "\n")


BYTES_PER_PARAM = {"float32": 4, "bfloat16": 2, "float16": 2}

# Binary gigabytes, because that is what an operating system means by "16 GB" and
# what the machine was sold as. Dividing by 1e9 would call a 16 GB Mac a 17 GB one
# and quietly undermine the whole message.
GB = 1024 ** 3


def parameter_count(model_id):
    """Total parameters, from the Hub's metadata. No weights are downloaded.

    Returns None for anything not on the Hub, or when the field is absent - not
    knowing is common and is never a reason to block a run.
    """
    if not model_id or os.path.isdir(str(model_id)):
        return None
    try:
        from huggingface_hub import model_info
        total = (model_info(model_id).safetensors or {}).total
        return int(total) if total else None
    except Exception:
        return None


def memory_estimate(config, dtype_name):
    """Resident weight size for teacher + student, and what the machine has.

    Both models are held at once - the teacher frozen, the student training - so
    the sum is what has to fit. This is weights only: activations, gradients and
    the generation KV cache sit on top, which is why the warning threshold below
    is well under 100%.
    """
    models = config.get("models") or {}
    counts = {role: parameter_count(models.get(role))
              for role in ("teacher", "student")}
    if not any(counts.values()):
        return None

    per_param = BYTES_PER_PARAM.get(dtype_name, 4)
    weight_bytes = sum(n for n in counts.values() if n) * per_param

    return {"counts": counts, "dtype": dtype_name, "weight_bytes": weight_bytes,
            "total_ram_bytes": total_memory()}


def total_memory():
    """Physical RAM of this machine, in bytes, or None if it cannot be determined.

    Queried from the operating system every time - nothing here assumes a size.

    Total rather than currently-available, deliberately: available fluctuates with
    whatever else is open, so a warning keyed to it would appear and disappear
    between identical runs. The question being answered is whether a config is
    viable on this machine at all.
    """
    # POSIX first: Linux and macOS both implement these.
    try:
        if hasattr(os, "sysconf"):
            names = os.sysconf_names
            if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
                pages = os.sysconf("SC_PHYS_PAGES")
                size = os.sysconf("SC_PAGE_SIZE")
                if pages > 0 and size > 0:
                    return pages * size
    except (ValueError, OSError, AttributeError):
        pass

    # macOS fallback. hw.memsize is the canonical source there, and is worth
    # having because the sysconf route above is the one path this project has
    # never been able to exercise on a Mac.
    if sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass

    total_ram = None
    try:
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total_ram = int(status.ullTotalPhys)
    except Exception:
        total_ram = None

    return total_ram


def memory_warning(estimate, device):
    """A warning when the weights alone will not comfortably fit, else None.

    Only meaningful where the models share the machine's memory - CPU, and Apple
    unified memory. A discrete GPU has its own budget this cannot see.
    """
    if not estimate or not estimate["total_ram_bytes"] or device == "cuda":
        return None

    weights = estimate["weight_bytes"] / GB
    total = estimate["total_ram_bytes"] / GB
    # Weights past ~60% of RAM leaves too little for activations, gradients, the
    # KV cache and the operating system. Past that the allocator does not fail -
    # it spills to swap, and the run gets slow rather than stopping.
    if weights < total * 0.6:
        return None

    line = (f"the teacher and student together are {weights:.1f} GB of "
            f"{estimate['dtype']} weights, against a {total * 0.6:.1f} GB budget "
            f"(60% of this machine's {total:.0f} GB).\n"
            f"Activations, gradients and the generation cache sit on top of the "
            f"weights, so this will most likely swap - which shows up as a very "
            f"slow run rather than an error.")
    if estimate["dtype"] == "float32":
        line += (f"\n  Halve it:  --set hardware.dtype=bfloat16   "
                 f"(~{weights / 2:.1f} GB)")
    return line


def adapter_cache(config, source):
    """Where a converted copy of `source` is kept, so it is converted once.

    Keyed by the source name with the separators flattened, which keeps a Hub id
    like org/name readable in the path instead of hashing it into noise.
    """
    root = os.path.join(os.path.dirname(cache_dir(config)), "adapters")
    safe = str(source).replace("\\", "/").strip("./").replace("/", "__")
    return os.path.join(root, safe)


def ensure_peft_adapter(config, log=None):
    """Convert an MLX/unsloth teacher adapter to PEFT, once, and point at the copy.

    A published LoRA is often in MLX format, which PEFT cannot read. The
    conversion is deterministic, needs no GPU, and reads only the base model's
    config.json - so making the caller do it by hand buys nothing except a step
    to forget. It happens here, in preflight, and is announced.

    The config is rewritten to the converted directory, so everything downstream
    sees an ordinary PEFT adapter and knows nothing about MLX.
    """
    from .teacher import adapter_problem

    source = (config.get("models") or {}).get("teacher_adapter")
    if not source:
        return None

    problem = adapter_problem(source, config["models"].get("teacher"))
    if not problem:
        return None
    if "MLX" not in problem:
        # Missing, or not an adapter at all. Not something converting can fix.
        raise RuntimeError(problem)

    destination = adapter_cache(config, source)
    if is_cached(destination):
        if log:
            log.info(f"      teacher lora: converted copy cached at {destination}")
        config["models"]["teacher_adapter"] = destination
        return {"source": source, "local": destination, "converted": False}

    if log:
        log.info(f"      teacher lora: {source} is MLX format; converting once")
    import argparse

    from . import adapters

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    code = adapters.main(argparse.Namespace(
        adapter=source,
        base=config["models"].get("teacher"),
        out=destination,
        dry_run=False,
        force=True,
    ))
    if code not in (0, None):
        raise RuntimeError(
            f"could not convert {source} to PEFT format (exit {code}).\n"
            f"Convert it by hand to see the detail:\n"
            f"  kd convert-adapter --adapter {source} "
            f"--base {config['models'].get('teacher')} --out ./peft-adapter")

    mark_complete(destination, f"converted from {source}")
    config["models"]["teacher_adapter"] = destination
    if log:
        log.info(f"      teacher lora: converted to {destination}")
    return {"source": source, "local": destination, "converted": True}


def remote_values(config):
    """[(dotted path, uri)] for every resolvable key currently holding an s3 URI."""
    found = []
    for path in RESOLVABLE:
        section, _, key = path.partition(".")
        value = (config.get(section) or {}).get(key)
        if is_remote(value):
            found.append((path, value))
    return found


def resolve_inputs(config, log=None):
    """Fetch every s3:// input and rewrite the config to point at the local copies.

    Mutates `config`, deliberately: everything downstream - the trainer, the
    evaluator, the report - then reads ordinary local paths and needs to know
    nothing about object storage. What was substituted is returned so the run
    manifest can record where the inputs actually came from.
    """
    pending = remote_values(config)
    if not pending:
        return {}

    if not (config.get("s3") or {}).get("enabled"):
        raise RuntimeError(
            "the config names s3:// inputs but s3.enabled is false:\n  "
            + "\n  ".join(f"{path} = {uri}" for path, uri in pending)
            + "\nSet --set s3.enabled=true, or use local paths.")

    from .remote import s3

    resolved = {}
    for path, uri in pending:
        local = cache_path(config, uri)
        if is_cached(local):
            if log:
                log.info(f"      {path}: cached  {uri}")
        else:
            if log:
                log.info(f"      {path}: fetching {uri}")
            s3.download(config, uri, local)
        section, _, key = path.partition(".")
        config[section][key] = local
        resolved[path] = {"uri": uri, "local": local}
    return resolved
