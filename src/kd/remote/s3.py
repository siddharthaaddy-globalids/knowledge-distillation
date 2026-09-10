"""
Object storage: fetching inputs, and shipping the finished run bundle.

Optional in every sense. Nothing in the core pipeline imports this module, boto3
lives in the `remote` extra, and a config with `s3.enabled: false` never reaches
any of it. A machine with no AWS credentials runs every local stage unaffected.

Credentials come from the environment - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
AWS_PROFILE, or an instance role - and never from the config file, which is
committed to a repository. The config names the bucket; the environment proves who
you are.

`endpoint_url` makes the same code work against anything speaking the S3 API:
MinIO, Cloudflare R2, RunPod's own volumes. That is why it is a setting rather
than a hardcoded AWS host.
"""

import os

from ..paths import mark_complete, split_uri

# What each name in `s3.upload` actually covers. Relative to the run directory.
#
# checkpoints is deliberately not in the default list: it is optimizer state,
# hundreds of megabytes, and nothing outside a resumed run reads it. It is worth
# naming when limits are tight, because a hard stop leaves the last checkpoint as
# the only artifact of the run.
UPLOAD_GROUPS = {
    "adapter": ["final_adapter/**"],
    "logs": ["run.log", "events.jsonl"],
    "report": ["report.html", "report.md"],
    "metrics": ["metrics.json", "evaluation.json"],
    "checkpoints": ["checkpoints/**"],
}

# Always uploaded, whatever `s3.upload` says. Both are small, and without them the
# bundle in the bucket cannot say what produced it - which is the entire reason
# for uploading a bundle rather than a bare adapter.
ALWAYS = ["manifest.json", "config.resolved.yaml"]

MISSING_BOTO3 = (
    "boto3 is needed for s3 support and is not installed.\n"
    "  uv sync --extra remote\n"
    "It is an optional extra so that a local run needs nothing from AWS.")


def client(config):
    """A configured S3 client, or a clear message about what is missing."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(MISSING_BOTO3) from exc

    settings = config.get("s3") or {}
    kwargs = {}
    if settings.get("endpoint_url"):
        kwargs["endpoint_url"] = settings["endpoint_url"]
    if settings.get("region"):
        kwargs["region_name"] = settings["region"]
    return boto3.client("s3", **kwargs)


def bucket_or_die(config):
    bucket = (config.get("s3") or {}).get("bucket")
    if not bucket:
        raise RuntimeError("s3.enabled is true but s3.bucket is not set")
    return bucket


def run_prefix(config, run_id):
    """Where a run bundle lives in the bucket: <prefix>/runs/<run-id>."""
    prefix = ((config.get("s3") or {}).get("prefix") or "").strip("/")
    return f"{prefix}/runs/{run_id}" if prefix else f"runs/{run_id}"


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def reachable(config, uri):
    """Can this URI actually be read? Returns (ok, detail). Downloads nothing.

    One ListObjectsV2 capped at a single key. The point is to turn the most
    expensive failure in this project into the cheapest one: a credential that
    cannot read the teacher is otherwise discovered by the preflight stage, on a
    rented GPU, after the pod has been paid for and started.

    Both halves of the permission are exercised, deliberately. `download` resolves
    a prefix by listing it and then fetching each key, so an identity with
    GetObject and no ListBucket fails there - and a probe that only did a
    HeadObject would have said everything was fine.
    """
    try:
        s3 = client(config)
    except RuntimeError as exc:
        return False, str(exc)

    try:
        bucket, key = split_uri(uri)
    except ValueError as exc:
        return False, str(exc)

    prefix = f"{key}/" if key else ""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    except Exception as exc:  # noqa: BLE001 - botocore raises many shapes here
        name = type(exc).__name__
        text = str(exc)
        if "AccessDenied" in text or "403" in text:
            return False, (
                "access denied. This is a permissions problem, not a key "
                "problem - new access keys for the same user will fail "
                "identically. The identity needs s3:ListBucket on the bucket "
                "and s3:GetObject on this prefix.")
        if "NoSuchBucket" in text:
            return False, f"no such bucket: {bucket}"
        return False, f"{name}: {text[:160]}"

    count = response.get("KeyCount", 0)
    if not count:
        return False, ("reachable, but nothing is stored there. Check the "
                       "prefix for a typo or a missing trailing path segment.")
    return True, f"reachable ({response['Contents'][0]['Key']} ...)"


def download(config, uri, destination):
    """Fetch everything under `uri` into `destination`.

    An S3 prefix is not a directory - it is a shared string across flat keys - so
    a "directory" download is a listing plus one GET per object. Both the
    single-object and prefix cases are handled, because s3://bucket/model.tar and
    s3://bucket/model/ are both reasonable things to write in a config.
    """
    s3 = client(config)
    bucket, key = split_uri(uri)
    os.makedirs(destination, exist_ok=True)

    prefix = f"{key}/" if key else ""
    paginator = s3.get_paginator("list_objects_v2")
    downloaded, total_bytes = 0, 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            relative = item["Key"][len(prefix):]
            if not relative or relative.endswith("/"):
                continue  # a zero-byte key standing in for a folder
            target = os.path.join(destination, *relative.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file(bucket, item["Key"], target)
            downloaded += 1
            total_bytes += item.get("Size", 0)

    if downloaded == 0:
        # Not a prefix, so try it as a single object before giving up. Reporting
        # "nothing there" when the key exists would send someone hunting for a
        # permissions problem that is really a trailing slash.
        try:
            target = os.path.join(destination, os.path.basename(key) or "object")
            s3.download_file(bucket, key, target)
            downloaded, total_bytes = 1, os.path.getsize(target)
        except Exception as exc:
            raise RuntimeError(
                f"nothing found at {uri}\n"
                f"  checked prefix {prefix!r} and the key itself\n"
                f"  {type(exc).__name__}: {exc}") from exc

    mark_complete(destination, f"{uri}\n{downloaded} objects, {total_bytes} bytes")
    return {"objects": downloaded, "bytes": total_bytes, "local": destination}


# --------------------------------------------------------------------------- #
# Shipping
# --------------------------------------------------------------------------- #
def _selected_files(run_dir, groups):
    """Absolute paths of everything the named groups cover, deduplicated."""
    import glob

    patterns = list(ALWAYS)
    for name in groups:
        patterns.extend(UPLOAD_GROUPS.get(name, []))

    found = []
    for pattern in patterns:
        for path in glob.glob(os.path.join(run_dir, pattern), recursive=True):
            if os.path.isfile(path):
                found.append(path)
    return sorted(set(found))


def upload_bundle(config, run_dir, run_id, groups=None, log=None):
    """Sync the parts of a run bundle named by `s3.upload` to the bucket.

    Returns a summary including the s3:// URI of the bundle, which goes into the
    run manifest so the local record says where the remote copy went.
    """
    settings = config.get("s3") or {}
    groups = list(groups if groups is not None else (settings.get("upload") or []))

    # A misspelled group would otherwise upload the two ALWAYS files and nothing
    # else, which looks like a successful upload of an almost-empty bundle.
    unknown = [name for name in groups if name not in UPLOAD_GROUPS]
    if unknown:
        raise RuntimeError(
            f"unknown s3.upload entries: {unknown}\n"
            f"  valid: {', '.join(sorted(UPLOAD_GROUPS))}")

    bucket = bucket_or_die(config)
    prefix = run_prefix(config, run_id)

    files = _selected_files(run_dir, groups)
    if not files:
        raise RuntimeError(
            f"nothing to upload: s3.upload={groups} matched no files in {run_dir}")

    s3 = client(config)
    sent_bytes = 0
    for path in files:
        relative = os.path.relpath(path, run_dir).replace(os.sep, "/")
        s3.upload_file(path, bucket, f"{prefix}/{relative}")
        sent_bytes += os.path.getsize(path)

    uri = f"s3://{bucket}/{prefix}"
    if log:
        log.info(f"      {len(files)} files, {sent_bytes / 1e6:.1f} MB -> {uri}")
    return {"uri": uri, "files": len(files), "bytes": sent_bytes, "groups": groups}
