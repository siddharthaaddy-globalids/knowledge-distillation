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
import time

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
    # arena.json belongs here, not with the report: it is the answer-key score -
    # accuracy, Elo, and the head-to-head record behind them - and it is the
    # number anyone asks about first. metrics.json carries a summary of it, but
    # only the summary, so a bundle without this cannot say which questions were
    # scored or how the players actually differed.
    "metrics": ["metrics.json", "evaluation.json", "arena.json",
                # Every question and every word each player said about it. Large
                # - a few MB - and the thing you want when a number is
                # surprising, which is exactly when the pod is already gone.
                "arena-transcript.jsonl"],
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


def uri_of(bucket, prefix):
    """The s3:// URI a bucket and prefix name. One spelling, used everywhere."""
    return f"s3://{bucket}/{prefix}"


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


def _human(num_bytes):
    """Bytes as something a person reads at a glance."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024


def _duration(seconds):
    seconds = int(max(0, seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


class _Progress:
    """Live progress for a multi-object transfer, in either direction.

    A teacher checkpoint is tens of gigabytes and, on a rented GPU, the download
    is billed at the GPU's hourly rate. Before this, `download` printed nothing
    between "fetching" and "done", so the one part of a run where you most want
    to know whether to wait or to kill it was the part that said least. Uploads
    are usually far smaller, with one exception that matters: a rescue upload
    after a limit stopped a run ships hundreds of megabytes of checkpoint off a
    pod that is about to be destroyed.

    Two output shapes, because the two places this runs want different things.
    A terminal gets one line rewritten in place. A log file - which is where a
    pod run actually ends up, through kd.runlog - gets an occasional new line
    instead, since several thousand carriage returns make a log unreadable.

    boto3 transfers each object in parallel parts and calls back from several
    threads, so the running total is taken under a lock.
    """

    # Redraw at most this often. A callback per 8 MB part on a fast link is
    # dozens a second, and rendering every one costs more than it tells you.
    TTY_INTERVAL = 0.25
    LOG_INTERVAL = 30.0

    def __init__(self, total_bytes, total_objects, log=None, stream=None):
        import sys
        import threading

        self.total = max(1, total_bytes)
        self.objects = total_objects
        self.log = log
        self.stream = stream or sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.done = 0
        self.finished_objects = 0
        self.started = time.time()
        self.last_render = 0.0
        self.lock = threading.Lock()

    def callback(self, chunk):
        with self.lock:
            self.done += chunk
            self._maybe_render()

    def object_done(self):
        with self.lock:
            self.finished_objects += 1

    def _maybe_render(self):
        now = time.time()
        interval = self.TTY_INTERVAL if self.tty else self.LOG_INTERVAL
        if now - self.last_render < interval:
            return
        self.last_render = now
        self._render(now)

    def _render(self, now, final=False):
        elapsed = max(1e-6, now - self.started)
        rate = self.done / elapsed
        share = min(1.0, self.done / self.total)
        # Remaining time from the rate so far. Honest about being an estimate:
        # a stalled connection makes it grow, which is itself the useful signal.
        eta = (self.total - self.done) / rate if rate > 0 else 0
        line = (f"{share * 100:5.1f}%  {_human(self.done)} / {_human(self.total)}"
                f"  {_human(rate)}/s"
                f"  {self.finished_objects}/{self.objects} files"
                + ("" if final else f"  eta {_duration(eta)}"))
        if self.tty and not final:
            self.stream.write(f"\r      {line}")
            self.stream.flush()
        elif self.tty:
            self.stream.write(f"\r      {line}\n")
            self.stream.flush()
        elif self.log:
            self.log.info(f"      {line}")

    def finish(self):
        with self.lock:
            self._render(time.time(), final=True)


def download(config, uri, destination, log=None):
    """Fetch everything under `uri` into `destination`, reporting progress.

    An S3 prefix is not a directory - it is a shared string across flat keys - so
    a "directory" download is a listing plus one GET per object. Both the
    single-object and prefix cases are handled, because s3://bucket/model.tar and
    s3://bucket/model/ are both reasonable things to write in a config.

    The listing is walked fully before anything is fetched. That costs one extra
    round trip and buys the total size, without which "47% done" is not a thing
    that can be said.
    """
    s3 = client(config)
    bucket, key = split_uri(uri)
    os.makedirs(destination, exist_ok=True)

    prefix = f"{key}/" if key else ""
    paginator = s3.get_paginator("list_objects_v2")

    wanted = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            relative = item["Key"][len(prefix):]
            if not relative or relative.endswith("/"):
                continue  # a zero-byte key standing in for a folder
            wanted.append((item["Key"], relative, item.get("Size", 0)))

    downloaded, total_bytes = 0, 0
    if wanted:
        expected = sum(size for _k, _r, size in wanted)
        if log:
            log.info(f"      {len(wanted)} files, {_human(expected)} to fetch")
        progress = _Progress(expected, len(wanted), log=log)
        for object_key, relative, size in wanted:
            target = os.path.join(destination, *relative.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3.download_file(bucket, object_key, target,
                             Callback=progress.callback)
            progress.object_done()
            downloaded += 1
            total_bytes += size
        progress.finish()

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
    # Sizes are known from the filesystem, so unlike a download this needs no
    # extra round trip to say how much there is.
    expected = sum(os.path.getsize(path) for path in files)
    if log:
        log.info(f"      {len(files)} files, {_human(expected)} -> {uri_of(bucket, prefix)}")

    # Usually tens of megabytes against a download's tens of gigabytes, so this
    # matters less - except in the one case it matters most. `checkpoints` is a
    # rescue upload after a limit stopped a run, which is hundreds of megabytes
    # of optimizer state on a pod that is about to be destroyed, and watching it
    # move is the difference between waiting and guessing.
    progress = _Progress(expected, len(files), log=log)
    sent_bytes = 0
    for path in files:
        relative = os.path.relpath(path, run_dir).replace(os.sep, "/")
        s3.upload_file(path, bucket, f"{prefix}/{relative}",
                       Callback=progress.callback)
        progress.object_done()
        sent_bytes += os.path.getsize(path)
    progress.finish()

    uri = uri_of(bucket, prefix)
    if log:
        log.info(f"      uploaded to {uri}")
    return {"uri": uri, "files": len(files), "bytes": sent_bytes, "groups": groups}
