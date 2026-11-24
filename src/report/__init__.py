"""
Read-only report generation from a dataset's manifest. Nothing in this package writes to the manifest.

build_report(dataset, config) assembles validation_report.json as a dict:
- `valid` / `issues`: validate_manifest's result. Errors stop the report unless config.force is set;
  then the report is written with valid = false, so the numbers can be inspected but not trusted.
- `flags`, `profile`, `duplicates`: the sections in src/report/sections/.
- `checks`: the settings, stats and code version of the last run of each check (from audits.json).
- `run`: the code and settings that generated this report. `dataset`: the delivery's fingerprint.

The JSON is the single source of every number: the HTML summary (todo 28) only renders it. Its
format is versioned (REPORT_SCHEMA_VERSION) and described by src/report/schema.json; change both together.
No absolute local paths go in: files are relative to data_dir, outputs relative to result_dir.
"""
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.report.exports import write_exports
from src.report.sections.duplicates import (
    LARGEST_GROUPS,
    MAX_EXPECTED_GROUP_SIZE,
    duplicate_statistics,
    load_cached_embeddings,
)
from src.report.sections.flags import TOP_COMBINATIONS, flag_statistics
from src.report.sections.profile import BINS, BORDERLINE_MARGIN, TOP_RESOLUTIONS, dataset_profile, relative_paths
from src.report.validate import ERROR, Issue, ManifestValidationError, errors, validate_manifest
from src.utils.audit import (
    RESET_EVENTS,
    current_audits,
    dataset_fingerprint,
    environment_info,
    latest_by_check,
    load_audits,
    paths_digest,
)
from src.utils.logger import Logger

logger = Logger(__name__)

REPORT_SCHEMA_VERSION = 1
REPORT_DIR = "report"
REPORT_FILE = "validation_report.json"
SCHEMA_PATH = Path(__file__).with_name("schema.json")
REPORT_DESCRIPTION = "Every number in this report, machine-readable. The other files are views of the manifest."


@dataclass(frozen=True)
class ReportConfig:
    # Root of the delivery. Default: the dataset's own (ImageDataset.data_dir / ImageClassificationDataset.root_dir).
    data_dir: str | None = None
    # Directory the manifest's paths are relative to. Default: the current directory.
    base_dir: str | None = None
    # Write the report even if the manifest has errors (valid = false) or a section fails.
    force: bool = False
    # Use cached embeddings (with saved row indices) for the similarity of duplicate groups.
    use_embeddings: bool = True
    default_model: str | None = None
    top_combinations: int = TOP_COMBINATIONS
    bins: int = BINS
    top_resolutions: int = TOP_RESOLUTIONS
    borderline_margin: float = BORDERLINE_MARGIN
    largest_groups: int = LARGEST_GROUPS
    max_expected_group_size: int = MAX_EXPECTED_GROUP_SIZE


# Config fields that hold local paths; left out of the report.
_LOCAL_PATH_FIELDS = ("data_dir", "base_dir")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dataset_data_dir(dataset) -> str:
    """The delivery root of either dataset class."""
    for attr in ("data_dir", "root_dir"):
        value = getattr(dataset, attr, None)
        if value:
            return value
    raise ValueError("The dataset has no data_dir/root_dir; pass ReportConfig(data_dir=...)")


def to_jsonable(value):
    """
    Converts a report to plain JSON types: numpy scalars and arrays to Python, NaN and +-inf to None
    (aspect_ratio is inf for corrupt files, and NaN breaks strict JSON parsers), tuples and sets to
    lists, dict keys to str, timestamps and paths to str.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def summarize_checks(checks: dict, manifest: pd.DataFrame) -> dict:
    """
    Per check, one entry per distinct set of params run since the last reset, oldest first. Every one
    of them still shapes the manifest: alexnet and facenet runs of group_by_embeddings are two
    configurations, not a run and its rerun, and a rerun with a new threshold adds its tags to the old
    ones (nothing is cleared without a reset).

    Each entry is the latest run with those params: status, error, timing, params, stats, changes, the
    code it ran on, `earlier_runs` (reruns with the same params) and `paths_digest_matches` (whether it
    ran on the same set of files as the current manifest).
    """
    digest = paths_digest(manifest["path"])
    summary = {}
    for name, runs in sorted(checks.items()):
        by_params: dict[str, list[dict]] = {}
        for entry in [*runs["history"], runs["latest"]]:
            key = json.dumps(to_jsonable(entry.get("params", {})), sort_keys=True)
            by_params.setdefault(key, []).append(entry)
        entries = []
        for same in by_params.values():
            latest = same[-1]
            environment = latest.get("environment") or {}
            entries.append({
                "status": latest.get("status"),
                "error": latest.get("error"),
                "started_at": latest.get("started_at"),
                "finished_at": latest.get("finished_at"),
                "duration_s": latest.get("duration_s"),
                "params": latest.get("params", {}),
                "stats": latest.get("stats", {}),
                "changes": latest.get("changes", {}),
                "git_commit": environment.get("git_commit"),
                "git_dirty": environment.get("git_dirty"),
                "paths_digest_matches": latest.get("paths_digest") == digest,
                "earlier_runs": len(same) - 1,
            })
        summary[name] = sorted(entries, key=lambda e: e["started_at"] or "")
    return summary


def _run_block(config: ReportConfig, entries: list[dict]) -> dict:
    current = current_audits(entries)
    reset = current[0] if current and current[0].get("event") in RESET_EVENTS else None
    return {
        "generated_by": environment_info(),
        "config": {k: v for k, v in asdict(config).items() if k not in _LOCAL_PATH_FIELDS},
        "audit_entries": len(entries),
        "current_audit_entries": len(current),
        "last_reset": {"event": reset["event"], "at": reset.get("at")} if reset else None,
    }


def relative_issues(issues: list[Issue], manifest: pd.DataFrame, data_dir: str, base_dir: str | None) -> list[dict]:
    """Issues as dicts, with examples that are manifest paths made relative to data_dir (others, e.g. column names, unchanged)."""
    rel = dict(zip(manifest["path"].astype(str), relative_paths(manifest, data_dir, base_dir))) if "path" in manifest else {}
    return [{**asdict(i), "examples": [rel.get(e, e) for e in i.examples]} for i in issues]


def _section_failed(name: str, exc: Exception) -> Issue:
    return Issue("section_failed", ERROR, f"Section {name!r} could not be built: {type(exc).__name__}: {exc}", 1, [name])


def build_report(dataset, config: ReportConfig = ReportConfig()) -> dict:
    """
    Builds the report dict (see the module docstring for the keys). Reads the manifest, audits.json and
    the cached embeddings, and writes nothing.

    Raises ManifestValidationError if validate_manifest finds errors, unless config.force is set. With
    force, a section that raises is set to None and reported as a 'section_failed' error issue.
    """
    manifest = dataset._manifest.copy()
    data_dir = config.data_dir or dataset_data_dir(dataset)
    entries = load_audits(dataset._audit_path)
    checks = latest_by_check(entries)

    issues = validate_manifest(manifest, data_dir=data_dir, base_dir=config.base_dir)
    if errors(issues) and not config.force:
        raise ManifestValidationError(errors(issues))

    embeddings, embeddings_unavailable = {}, []
    if config.use_embeddings:
        embeddings, embeddings_unavailable = load_cached_embeddings(manifest, dataset._embeddings_dir)

    builders = {
        "flags": lambda: flag_statistics(manifest, top_combinations=config.top_combinations),
        "profile": lambda: dataset_profile(
            manifest, data_dir, base_dir=config.base_dir, checks=checks, bins=config.bins,
            top=config.top_resolutions, borderline_margin=config.borderline_margin,
        ),
        "duplicates": lambda: {
            **duplicate_statistics(
                manifest, data_dir, base_dir=config.base_dir, embeddings=embeddings,
                default_model=config.default_model, largest=config.largest_groups,
                max_expected_group_size=config.max_expected_group_size,
            ),
            "similarity_unavailable": embeddings_unavailable,
        },
    }
    sections = {}
    for name, build in builders.items():
        try:
            sections[name] = build()
        except Exception as exc:
            if not config.force:
                raise
            logger.error(f"Section {name} failed: {exc}")
            sections[name] = None
            issues.append(_section_failed(name, exc))

    flags = sections["flags"]
    return to_jsonable({
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _now(),
        "run": _run_block(config, entries),
        "dataset": dataset_fingerprint(manifest, data_dir, config.base_dir),
        "valid": not errors(issues),
        "issues": relative_issues(issues, manifest, data_dir, config.base_dir),
        "status_counts": flags["status"] if flags else None,
        "flags": flags,
        "profile": sections["profile"],
        "duplicates": sections["duplicates"],
        "checks": summarize_checks(checks, manifest),
        "outputs": [],
    })


def write_report(report: dict, path: str) -> None:
    """
    Writes strict JSON (allow_nan=False) to a temp file in the same directory and renames it over
    `path`, so a crash never leaves a half-written report.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".report-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(to_jsonable(report), f, indent=2, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def report_dir(dataset) -> str:
    return os.path.join(dataset.result_dir, REPORT_DIR)


def generate_report(dataset, config: ReportConfig = ReportConfig()) -> str:
    """
    Builds the report, writes the CSV exports (src/report/exports.py) and then
    <result_dir>/report/validation_report.json, last, so the JSON only lists files that exist.
    Returns the JSON's path.

    `outputs` lists every file written: {path relative to result_dir, rows (None for the JSON), description}.
    If the exports fail, they are all removed; with config.force the JSON is still written with
    valid = false and an 'export_failed' issue, otherwise the exception propagates.
    """
    report = build_report(dataset, config)
    out_dir = report_dir(dataset)
    try:
        exports = write_exports(dataset._manifest.copy(), out_dir, config.data_dir or dataset_data_dir(dataset), config.base_dir)
    except Exception as exc:
        if not config.force:
            raise
        logger.error(f"Exports failed: {exc}")
        exports = []
        report["issues"].append(asdict(Issue("export_failed", ERROR, f"CSV exports could not be written: {type(exc).__name__}: {exc}", 1, [])))
        report["valid"] = False
    report["outputs"] = [
        *({"path": f"{REPORT_DIR}/{e['path']}", "rows": e["rows"], "description": e["description"]} for e in exports),
        {"path": f"{REPORT_DIR}/{REPORT_FILE}", "rows": None, "description": REPORT_DESCRIPTION},
    ]
    path = os.path.join(out_dir, REPORT_FILE)
    write_report(report, path)
    logger.info(f"Report written to {path} (valid: {report['valid']})")
    return path


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)
