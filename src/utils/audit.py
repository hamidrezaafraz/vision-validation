"""
Audit log: what ran on a dataset, with which settings, on which code, and what it changed.

Every audited step appends one entry to `<result_dir>/audits.json` (a JSON list, never rewritten
in place, so history is kept). A flag only means something together with the settings that produced
it ('blurry' at min_blur_score=40 is not 'blurry' at 100), so the report reads its thresholds from here.

Entries:
- check entries (written by @audited): check name, every argument except the dataset, start/end
  time, status ('ok' or 'failed' + error), the code and package versions, and what changed in the
  manifest (tags added per tag, rows newly marked per action) plus the check's own stats.
- reset events ('manifest_created', 'restart'): every flag is gone after these, so entries written
  before the last reset no longer describe the manifest. current_audits() drops them.

This also answers what the manifest alone cannot: a duplicate step that found no groups leaves the
manifest untouched, but its entry still says it ran and found 0 groups.
"""
import functools
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.utils.logger import Logger

logger = Logger(__name__)

AUDIT_SCHEMA_VERSION = 1
MANIFEST_CREATED, RESTART = "manifest_created", "restart"
RESET_EVENTS = (MANIFEST_CREATED, RESTART)
OK, FAILED = "ok", "failed"

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("numpy", "pandas", "pillow", "opencv-python-headless", "imagehash", "torch", "torchvision", "facenet-pytorch")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@functools.lru_cache(maxsize=1)
def _package_versions() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info() -> dict:
    """Code and machine the step ran on. Git is read on every call, since files can change between steps."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        # Untracked files count: new, uncommitted modules (e.g. a new check) change behavior too.
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _package_versions(),
    }


def paths_digest(paths) -> str:
    """blake2b over the sorted paths: changes when files are added, removed or renamed. No file I/O."""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(map(str, paths)):
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()


def dataset_fingerprint(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> dict:
    """
    Identifies a delivery: blake2b over the sorted (path relative to data_dir, size, content) of every row.

    Content is the manifest's 'hash' column when every row has one (method 'content'); otherwise
    the file's modification time stands in (method 'size_mtime', weaker: a copied folder gets new
    mtimes, so the same files can give a different fingerprint). Paths resolve against base_dir
    (default: the current directory). Files that no longer exist are counted, not hashed.
    """
    base_dir = base_dir if base_dir is not None else os.getcwd()
    root = os.path.abspath(data_dir)
    hashes = manifest["hash"] if "hash" in manifest.columns else None
    use_content = hashes is not None and not (hashes.isna() | (hashes.astype(str).str.strip() == "")).any()

    records, total_bytes, missing = [], 0, 0
    for i, path in manifest["path"].items():
        full = os.path.abspath(os.path.join(base_dir, path))
        try:
            st = os.stat(full)
        except OSError:
            missing += 1
            continue
        total_bytes += st.st_size
        content = str(hashes[i]) if use_content else str(st.st_mtime_ns)
        records.append(f"{os.path.relpath(full, root)}\0{st.st_size}\0{content}")

    h = hashlib.blake2b(digest_size=16)
    for record in sorted(records):
        h.update(record.encode())
        h.update(b"\n")
    return {
        "fingerprint": h.hexdigest(),
        "method": "content" if use_content else "size_mtime",
        "file_count": len(records),
        "total_bytes": total_bytes,
        "missing_files": missing,
    }


def _to_json(value):
    """json.dump fallback for numpy scalars, tuples inside objects, paths and anything else."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def load_audits(path: str) -> list[dict]:
    """All entries in the audit file, oldest first; [] if it doesn't exist."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        try:
            entries = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Audit file {path!r} is not valid JSON ({e}); fix or move it, it is never overwritten") from e
    if not isinstance(entries, list):
        raise ValueError(f"Audit file {path!r} must contain a JSON list, got {type(entries).__name__}")
    return entries


def append_audit(path: str, entry: dict) -> None:
    """Appends one entry. Writes to a temp file and renames, so a crash never leaves a half-written file."""
    entries = load_audits(path)
    entries.append({"schema_version": AUDIT_SCHEMA_VERSION, **entry})
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".audits-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=2, default=_to_json)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def record_event(dataset, event: str, **details) -> None:
    """Appends a non-check entry, e.g. a reset event."""
    manifest = dataset._manifest
    append_audit(dataset._audit_path, {
        "event": event,
        "at": _now(),
        "manifest_rows": len(manifest),
        "paths_digest": paths_digest(manifest["path"]),
        "environment": environment_info(),
        **details,
    })


def current_audits(entries: list[dict]) -> list[dict]:
    """Entries from the last reset event on (inclusive); everything before it describes flags that are gone."""
    last_reset = max((i for i, e in enumerate(entries) if e.get("event") in RESET_EVENTS), default=None)
    return entries if last_reset is None else entries[last_reset:]


def latest_by_check(entries: list[dict]) -> dict[str, dict]:
    """
    For each check in the current entries: {'latest': last entry, 'history': earlier entries, oldest first}.
    The latest entry is returned even if it failed, so the report can say the last run did not finish.
    """
    by_check: dict[str, list[dict]] = {}
    for entry in current_audits(entries):
        if "check" in entry:
            by_check.setdefault(entry["check"], []).append(entry)
    return {name: {"latest": runs[-1], "history": runs[:-1]} for name, runs in by_check.items()}


def _tags(reason) -> set[str]:
    return {t.strip() for t in reason.split(";") if t.strip()} if isinstance(reason, str) else set()


def manifest_changes(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Tags added per tag, and rows whose action changed counted by their new action."""
    common = before.index.intersection(after.index)
    old_reason, new_reason = before.loc[common, "reason"].fillna(""), after.loc[common, "reason"].fillna("")
    old_action, new_action = before.loc[common, "action"].fillna(""), after.loc[common, "action"].fillna("")

    tags_added = Counter()
    changed = old_reason != new_reason
    for old, new in zip(old_reason[changed], new_reason[changed]):
        tags_added.update(_tags(new) - _tags(old))

    moved = old_action != new_action
    return {
        "tags_added": dict(sorted(tags_added.items())),
        "rows_marked": {k: int(v) for k, v in sorted(new_action[moved].value_counts().items())},
    }


def audited(check: str):
    """
    Decorator for a step whose first argument is the dataset. Appends one entry per call to the
    dataset's audits.json with every other argument (defaults included) as `params`.

    Stats: a returned DataFrame gives rows_inspected = its length (check_decode, check_photometric);
    a returned dict is merged into stats as-is (the grouping steps). If the step raises, the entry is
    written with status 'failed' and the error, and the exception propagates.
    """
    def decorator(func):
        signature = inspect.signature(func)
        dataset_arg = next(iter(signature.parameters))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            dataset = bound.arguments[dataset_arg]
            params = {k: v for k, v in bound.arguments.items() if k != dataset_arg}

            before = dataset._manifest[["action", "reason"]].copy()
            started_at, t0 = _now(), time.perf_counter()
            status, error, stats, result = OK, None, {}, None
            try:
                result = func(*args, **kwargs)
                return result
            except BaseException as e:
                status, error = FAILED, f"{type(e).__name__}: {e}"
                raise
            finally:
                if isinstance(result, pd.DataFrame):
                    stats["rows_inspected"] = len(result)
                elif isinstance(result, dict):
                    stats.update(result)
                manifest = dataset._manifest
                entry = {
                    "check": check,
                    "status": status,
                    "error": error,
                    "started_at": started_at,
                    "finished_at": _now(),
                    "duration_s": round(time.perf_counter() - t0, 3),
                    "params": params,
                    "stats": stats,
                    "changes": manifest_changes(before, manifest),
                    "manifest_rows": len(manifest),
                    "paths_digest": paths_digest(manifest["path"]),
                    "environment": environment_info(),
                }
                try:
                    append_audit(dataset._audit_path, entry)
                except Exception as audit_error:
                    if status == OK:
                        raise
                    # Don't hide the step's own exception behind an audit-writing one.
                    logger.error(f"Could not write audit entry for failed {check}: {audit_error}")

        return wrapper
    return decorator
