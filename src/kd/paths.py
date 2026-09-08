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
