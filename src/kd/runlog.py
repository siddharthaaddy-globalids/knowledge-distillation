"""
Run directories, logging and the run manifest.

Every invocation creates one directory holding everything that run produced:

    runs/finance-2026-09-08-1412/
        config.resolved.yaml   every value, after all overrides
        manifest.json          git sha, versions, timings, exit code
        run.log                full detail - everything the terminal showed
        events.jsonl           one JSON object per event, for machines
        metrics.json           the final numbers
        report.html
        final_adapter/
        checkpoints/

That directory is also the unit that gets uploaded to S3, so a run is either
entirely recoverable or entirely absent - never half of each.

Two log streams exist because they have different readers. The console gets a
short human line. run.log gets everything, including the output of the training
libraries, which print directly to stdout rather than through logging.
"""

import getpass
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import yaml

from . import config as kdconfig

RUN_LOG = "run.log"
EVENTS = "events.jsonl"
MANIFEST = "manifest.json"
RESOLVED_CONFIG = "config.resolved.yaml"
METRICS = "metrics.json"
ADAPTER_DIR = "final_adapter"
CHECKPOINT_DIR = "checkpoints"

# Packages whose version changes the numbers a run produces. Recorded so a result
# that cannot be reproduced can at least be explained.
TRACKED_PACKAGES = ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]


def git_sha(short=True):
    """Current commit, or 'nogit' outside a checkout. Never raises."""
    args = ["git", "rev-parse", "--short=7", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def git_dirty():
    """True when the working tree has uncommitted changes, so a sha alone is a lie."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5)
        return bool(out.returncode == 0 and out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def package_versions():
    """Installed versions of the packages that affect results."""
    from importlib.metadata import PackageNotFoundError, version
    found = {}
    for name in TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def make_run_id(profile, when=None):
    """<profile>-<YYYY-MM-DD>-<HHMM>, in UTC: enlibraQ25-3B-2026-09-10-1416.

    The profile leads, so a bucket listing groups a profile's runs together and
    the date sorts them within it. Nothing else: the git sha that used to end
    the name looked like noise to everyone who was not the person who ran it,
    and it lives in manifest.json regardless. UTC because the same run is
    named the same whether it was started from a laptop or a pod.

    Two runs of one profile in one minute would collide; Run appends -2, -3
    to the second and later ones rather than sharing a directory.
    """
    when = when or datetime.now(timezone.utc)
    return f"{profile}-{when.strftime('%Y-%m-%d-%H%M')}"


# --------------------------------------------------------------------------- #
# Finding what previous runs produced
# --------------------------------------------------------------------------- #
def is_adapter(path):
    """True when `path` is a directory PEFT can load an adapter from."""
    return bool(path) and os.path.isfile(os.path.join(str(path), "adapter_config.json"))


def discover_adapters(runs_dir="./runs", extra=()):
    """Every trained adapter this machine has, newest first.

    Looks inside run bundles, then at any additional directories the caller names.
    Ordering is by modification time rather than by run id, so an adapter copied in
    from elsewhere still sorts sensibly against locally trained ones.
    """
    import glob

    found = []
    runs_dir = os.path.expanduser(runs_dir or "./runs")
    for pattern in (os.path.join(runs_dir, "*", ADAPTER_DIR),
                    os.path.join(runs_dir, "*", CHECKPOINT_DIR, "checkpoint-*")):
        for path in glob.glob(pattern):
            # `latest` is a pointer at another run, not a run. Where the platform
            # allows a symlink it matches this glob too, so without skipping it
            # every adapter would be listed twice - once under its run id and once
            # under the alias. A run id says which run; the alias moves.
            if is_latest_alias(path, runs_dir):
                continue
            if is_adapter(path):
                found.append(path)

    for candidate in extra:
        if is_adapter(candidate):
            found.append(str(candidate))

    seen, unique = set(), []
    for path in sorted(found, key=os.path.getmtime, reverse=True):
        # realpath, not normpath: two different paths reaching one directory
        # through a link are the same adapter, and offering both is noise.
        key = os.path.realpath(path)
        if key not in seen:
            seen.add(key)
            unique.append(os.path.normpath(path).replace("\\", "/"))
    return unique


def recorded_upload(run_dir):
    """The s3:// URI a run bundle was uploaded to, or None.

    Read from events.jsonl rather than the manifest, because the upload stage
    is the last one and its result is recorded as an event - the manifest only
    knows that the stage ran. The last upload wins: a rescue upload after a
    hard stop lands in the same prefix, so there is only ever one answer.
    """
    path = os.path.join(str(run_dir or ""), EVENTS)
    if not os.path.isfile(path):
        return None
    found = None
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (record.get("stage") == "upload"
                        and record.get("event") == "bundle" and record.get("uri")):
                    found = record["uri"]
    except OSError:
        return None
    return found


def is_latest_alias(path, runs_dir):
    """True when `path` sits under the `latest` pointer rather than a real run."""
    relative = os.path.relpath(path, runs_dir)
    return relative.split(os.sep)[0] == "latest"


# --------------------------------------------------------------------------- #
# Teeing stdout
#
# The training libraries print to stdout directly - progress bars, warnings, the
# trainer's own tables - and none of that passes through `logging`. Without a tee
# the run.log would hold our messages and omit the ones most likely to explain a
# failure.
# --------------------------------------------------------------------------- #
class _Tee:
    """Write to the real stream and to the log file at once."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        try:
            self._handle.write(text)
        except (ValueError, OSError):
            # The log file is closed or gone; the console still matters more.
            pass
        return len(text)

    def flush(self):
        self._stream.flush()
        try:
            self._handle.flush()
        except (ValueError, OSError):
            pass

    # Progress bars and Gradio ask these before deciding how to render. Answering
    # from the real stream keeps their behaviour identical to an untee'd run.
    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


class Run:
    """One invocation: its directory, its logs, and its manifest.

    Use it as a context manager so the manifest is written whatever happens:

        with Run(config, argv=sys.argv) as run:
            run.log.info("...")
    """

    def __init__(self, config, argv=None, run_id=None, quiet=False):
        self.config = config
        self.meta = config.get("_meta", {})
        self.profile = self.meta.get("profile", "run")
        self.run_id = run_id or make_run_id(self.profile)
        self.argv = list(argv or sys.argv)
        self.quiet = quiet

        project = config.get("project", {})
        pinned = project.get("output_dir")
        if pinned:
            # An explicitly pinned directory is used exactly as given, so a caller
            # that needs a predictable path gets one.
            self.dir = os.path.abspath(os.path.expanduser(pinned))
        else:
            runs_dir = os.path.expanduser(project.get("runs_dir") or "./runs")
            if not run_id:
                # A generated id is only as fine-grained as a minute, so the
                # second run of a profile inside one gets a suffix rather than
                # the first run's directory.
                base, extra = self.run_id, 2
                while os.path.exists(os.path.join(runs_dir, self.run_id)):
                    self.run_id, extra = f"{base}-{extra}", extra + 1
            self.dir = os.path.abspath(os.path.join(runs_dir, self.run_id))
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, CHECKPOINT_DIR), exist_ok=True)

        self.started = time.time()
        self.stages = []
        self.status = "running"
        self.stopped_reason = None
        self._events = open(self.path(EVENTS), "a", encoding="utf-8")
        self._logfile = open(self.path(RUN_LOG), "a", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _Tee(self._stdout, self._logfile)
        sys.stderr = _Tee(self._stderr, self._logfile)

        self.log = self._make_logger()
        self._write_resolved_config()
        self.event("run", "start", run_id=self.run_id, profile=self.profile,
                   argv=" ".join(self.argv))

    # --- paths ------------------------------------------------------------- #
    def path(self, *parts):
        """Absolute path to something inside the run directory."""
        return os.path.join(self.dir, *parts)

    @property
    def adapter_dir(self):
        return self.path(ADAPTER_DIR)

    @property
    def checkpoint_dir(self):
        return self.path(CHECKPOINT_DIR)

    # --- logging ----------------------------------------------------------- #
    def _make_logger(self):
        logger = logging.getLogger(f"kd.{self.run_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        # Console. Deliberately plain: no colour, because ANSI escapes do not survive
        # every terminal, pipe, CI log or `tee`, and a banner rendered as raw escape
        # codes is worse than no colour at all.
        #
        # This writes to the real stdout, not the tee. If it went through the tee the
        # file would get every line twice - once from the tee and once from the
        # handler below.
        console = logging.StreamHandler(self._stdout)
        console.setLevel(logging.WARNING if self.quiet else logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)

        # File. The same open handle the tee writes to, so run.log has one writer and
        # its lines stay in the order they happened. This one carries the timestamps
        # and the DEBUG detail that would be noise on screen.
        detail = logging.StreamHandler(self._logfile)
        detail.setLevel(logging.DEBUG)
        detail.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(detail)
        return logger

    def banner(self, text):
        """A section heading in the terminal."""
        self.log.info("")
        self.log.info("=" * 78)
        self.log.info(f"  {text}")
        self.log.info("=" * 78)

    # --- events ------------------------------------------------------------ #
    def event(self, stage, kind, **fields):
        """Append one structured record to events.jsonl.

        This is the machine-readable stream: step metrics, stage transitions and
        cost ticks. Anything a dashboard, a cost graph or a CI check would want to
        read without parsing prose out of run.log.
        """
        record = {
            "ts": round(time.time(), 3),
            "elapsed_s": round(time.time() - self.started, 3),
            "stage": stage,
            "event": kind,
        }
        record.update(fields)
        try:
            self._events.write(json.dumps(record, default=str) + "\n")
            self._events.flush()
        except (ValueError, OSError):
            pass

    @contextmanager
    def stage(self, name):
        """Time a stage, record start and end, and re-raise whatever it raised.

        Gating - whether a failure here stops the pipeline - is the pipeline's
        decision, not this object's. All that happens here is timing and recording.
        """
        started = time.time()
        self.event(name, "stage_start")
        entry = {"name": name, "status": "running", "started": started}
        self.stages.append(entry)
        try:
            yield entry
        except BaseException as exc:
            entry.update(status="failed", seconds=round(time.time() - started, 2),
                         error=f"{type(exc).__name__}: {exc}")
            self.event(name, "stage_end", status="failed",
                       seconds=entry["seconds"], error=entry["error"])
            raise
        else:
            if entry["status"] == "running":
                entry["status"] = "ok"
            entry["seconds"] = round(time.time() - started, 2)
            self.event(name, "stage_end", status=entry["status"],
                       seconds=entry["seconds"])

    def skip_stage(self, name, why):
        """Record a stage that was deliberately not run, and say why."""
        self.stages.append({"name": name, "status": "skipped", "reason": why,
                            "seconds": 0.0})
        self.event(name, "stage_skipped", reason=why)

    # --- artifacts --------------------------------------------------------- #
    def _write_resolved_config(self):
        """The complete config after every override, as the run actually saw it."""
        with open(self.path(RESOLVED_CONFIG), "w", encoding="utf-8") as handle:
            handle.write(
                "# Resolved configuration for this run, after every override was\n"
                "# applied. This is what actually ran - re-run it exactly with:\n"
                f"#   kd pipeline --config {RESOLVED_CONFIG}\n\n")
            yaml.safe_dump(kdconfig.strip_meta(self.config), handle,
                           sort_keys=False, default_flow_style=False, allow_unicode=True)

    def write_metrics(self, metrics):
        """Write metrics.json. Merges with anything already there."""
        existing = {}
        if os.path.isfile(self.path(METRICS)):
            try:
                with open(self.path(METRICS), encoding="utf-8") as handle:
                    existing = json.load(handle)
            except (ValueError, OSError):
                existing = {}
        existing.update(metrics or {})
        with open(self.path(METRICS), "w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2, default=str)
        return self.path(METRICS)

    def manifest(self):
        """Everything needed to explain, reproduce or audit this run."""
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "status": self.status,
            "stopped_reason": self.stopped_reason,
            "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
            "seconds": round(time.time() - self.started, 2),
            "argv": self.argv,
            "config": {
                "source": self.meta.get("source"),
                "chain": self.meta.get("chain", []),
                "overridden": self.meta.get("overridden", {}),
            },
            "git": {"sha": git_sha(), "dirty": git_dirty()},
            "packages": package_versions(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "host": socket.gethostname(),
                "user": getpass.getuser(),
            },
            "stages": self.stages,
        }

    def write_manifest(self):
        with open(self.path(MANIFEST), "w", encoding="utf-8") as handle:
            json.dump(self.manifest(), handle, indent=2, default=str)
        return self.path(MANIFEST)

    # --- lifecycle --------------------------------------------------------- #
    def point_latest_here(self):
        """Make <runs_dir>/latest resolve to this run.

        A symlink where the platform allows one, and a text file naming the run
        where it does not: creating a directory symlink on Windows needs Developer
        Mode or an elevated shell, which is not worth requiring for a convenience.
        """
        parent = os.path.dirname(self.dir)
        link = os.path.join(parent, "latest")
        try:
            if os.path.islink(link) or os.path.isfile(link):
                os.unlink(link)
            elif os.path.isdir(link):
                return link  # a real directory named 'latest'; leave it alone
            os.symlink(self.dir, link, target_is_directory=True)
            return link
        except (OSError, NotImplementedError, AttributeError):
            pointer = os.path.join(parent, "latest.txt")
            try:
                with open(pointer, "w", encoding="utf-8") as handle:
                    handle.write(self.dir + "\n")
                return pointer
            except OSError:
                return None

    def finish(self, status="ok", stopped_reason=None):
        self.status = status
        self.stopped_reason = stopped_reason
        self.event("run", "end", status=status, stopped_reason=stopped_reason,
                   seconds=round(time.time() - self.started, 2))
        self.write_manifest()
        self.point_latest_here()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        for handler in list(self.log.handlers):
            handler.close()
            self.log.removeHandler(handler)
        for handle in (self._events, self._logfile):
            try:
                handle.close()
            except (ValueError, OSError):
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.status == "running":
            if exc_type is None:
                self.finish("ok")
            elif issubclass(exc_type, KeyboardInterrupt):
                self.finish("interrupted", "KeyboardInterrupt")
            else:
                self.finish("failed", f"{exc_type.__name__}: {exc}")
        else:
            self.write_manifest()
        self.close()
        return False
