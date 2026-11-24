"""
Flagged-subset exports: CSV views of the manifest that a person (or the client) can act on.

Written to <result_dir>/report/ on every report run and overwritten each time, since they are derived:
- ok_manifest.csv: rows whose status is 'ok', i.e. unflagged images, duplicate representatives included;
- review_manifest.csv: 'review' rows, worst cases first within each tag;
- drop_manifest.csv: rows recommended for exclusion, with their tags, severity and, for duplicate
  copies, the image they duplicate;
- duplicate_groups.csv: one row per member of every group with more than one member, for every grouping column.

ok, review and drop have one row per image, so their row counts add up to the manifest's.
Every file starts with the identity columns (path, status, reason, severity, plus label for labeled
datasets), then the metric columns relevant to that file. Paths are relative to the data root, so the
files still work when the folder is moved and carry no local paths. No comment lines, so any CSV
reader works; the meaning of each file is in EXPORTS and in `outputs` of validation_report.json.

Nothing here changes the manifest.
"""
import os
import tempfile

import pandas as pd

from src.report.reasons import CATALOG, SEVERITIES, get_reason, is_duplicate_tag, parse_tags
from src.report.sections.duplicates import group_labels, grouping_columns
from src.report.sections.flags import UNKNOWN, canonical_tag
from src.report.sections.profile import numeric_column, relative_paths
from src.report.status import DROP, OK, REVIEW, resolve_status

OK_FILE, REVIEW_FILE, DROP_FILE, GROUPS_FILE = "ok_manifest.csv", "review_manifest.csv", "drop_manifest.csv", "duplicate_groups.csv"

EXPORTS = {
    OK_FILE: "Images no check flagged (status ok), including the one image kept from each duplicate group.",
    REVIEW_FILE: "Images flagged for a person to look at, one row each, grouped by their most serious tag "
                 "and sorted worst first within it.",
    DROP_FILE: "Images recommended for exclusion, with their tags, severity and, for duplicate copies, the image they duplicate.",
    GROUPS_FILE: "One row per member of every duplicate group with more than one member, for every grouping column.",
}

SEVERITY_ORDER = (*SEVERITIES, UNKNOWN)

# How to rank the images of one tag, worst first: the metric (the min or max over `columns`) and its
# direction. Tags not listed here (corrupt, exif_rotated, color mode, duplicates, ...) have no
# ordering metric and are sorted by path.
SORT_KEYS = {
    "too_small": (("width", "height"), "min", True),
    "too_large": (("width", "height"), "max", False),
    "extreme_aspect_ratio": (("aspect_ratio",), "max", False),
    "blank_or_solid": (("pixel_std",), "max", True),
    "underexposed": (("dark_clip_ratio",), "max", False),
    "overexposed": (("bright_clip_ratio",), "max", False),
    "blurry": (("blur_score",), "max", True),
}


# ---- per-row helpers ---------------------------------------------------------------------------

def _severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.index(severity) if severity in SEVERITY_ORDER else len(SEVERITY_ORDER)


def _tag_order(tag: str) -> tuple:
    """Sort key for tags: most severe first, then catalog order, then name."""
    entry = get_reason(tag)
    position = next((i for i, r in enumerate(CATALOG) if r is entry), len(CATALOG))
    return (_severity_rank(entry.severity if entry else UNKNOWN), position, tag)


def row_tags(reason) -> list[str]:
    """A reason cell's distinct canonical tags, most serious first (see _tag_order)."""
    return sorted({canonical_tag(t) for t in parse_tags(reason)}, key=_tag_order)


def tag_severity(tag: str) -> str:
    entry = get_reason(tag)
    return entry.severity if entry else UNKNOWN


def primary_tag(reason) -> str:
    """The row's most serious tag ('' if it has none); a review row is listed under this one."""
    tags = row_tags(reason)
    return tags[0] if tags else ""


def row_severity(reason) -> str:
    """Severity of the row's most serious tag; '' for rows without tags."""
    tag = primary_tag(reason)
    return tag_severity(tag) if tag else ""


def metric_columns(tags, manifest: pd.DataFrame, exclude=("group_id", "split_group", "label")) -> list[str]:
    """The catalog `columns` of the given tags that exist in the manifest, in first-seen order."""
    columns = []
    for tag in sorted(set(tags), key=_tag_order):
        entry = get_reason(tag)
        for column in entry.columns if entry else ():
            if column in manifest.columns and column not in exclude and column not in columns:
                columns.append(column)
    return columns


def representatives(manifest: pd.DataFrame, status: pd.Series, rel: pd.Series, column: str) -> pd.Series:
    """
    Per row, the path of its group's representative in `column`: the first (by path) 'ok' member of a
    group with more than one member. '' for singletons and for groups with no 'ok' member.
    """
    labels = group_labels(manifest, column)
    sizes = labels.map(labels.value_counts())
    candidates = rel[(status == OK) & (sizes > 1)]
    first = candidates.groupby(labels[candidates.index]).min()
    return labels.map(first).fillna("").where(sizes > 1, "")


# Free-text metric columns that can contain the file's own path (Pillow puts it in the error message).
PATH_TEXT_COLUMNS = ("decode_error",)


def _metric_values(manifest: pd.DataFrame, rows: pd.DataFrame, column: str, rel: pd.Series) -> pd.Series:
    """The column's values for `rows`; in PATH_TEXT_COLUMNS, the stored and absolute path become the relative one."""
    values = rows[column]
    if column not in PATH_TEXT_COLUMNS:
        return values
    stored = manifest.loc[rows.index, "path"].astype(str)
    out = []
    for text, path, relative in zip(values.astype(str), stored, rel[rows.index]):
        out.append(text.replace(os.path.abspath(path), relative).replace(path, relative))
    return pd.Series(out, index=rows.index)


def identity(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.DataFrame:
    """path (relative to data_dir), status, reason, severity, and label when the manifest has one."""
    frame = pd.DataFrame({
        "path": relative_paths(manifest, data_dir, base_dir),
        "status": resolve_status(manifest),
        "reason": manifest["reason"].fillna("").astype(str),
        "severity": manifest["reason"].map(row_severity),
    }, index=manifest.index)
    if "label" in manifest.columns:
        frame["label"] = manifest["label"]
    return frame


# ---- the four files ----------------------------------------------------------------------------

def ok_manifest(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.DataFrame:
    """
    Status 'ok' rows, sorted by path, with `is_duplicate_representative` and `duplicates_removed`
    (the other members of its group_id, which this image stands for).
    """
    frame = identity(manifest, data_dir, base_dir)
    sizes = manifest.groupby("group_id")["group_id"].transform("size")
    ok = frame["status"] == OK
    frame["is_duplicate_representative"] = ok & (sizes > 1)
    frame["duplicates_removed"] = (sizes - 1).where(frame["is_duplicate_representative"], 0).astype(int)
    return frame[ok].sort_values("path").reset_index(drop=True)


def review_manifest(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.DataFrame:
    """
    Status 'review' rows, one per image, listed under their most serious tag (`tag`). Tags are in
    severity order; within a tag, the worst case first by `metric` / `metric_value` (see SORT_KEYS),
    then by path. The catalog metric columns of every review tag follow.
    """
    frame = identity(manifest, data_dir, base_dir)
    review = frame["status"] == REVIEW
    frame = frame[review].copy()
    rows = manifest.loc[frame.index]
    frame["tag"] = rows["reason"].map(primary_tag)

    frame["metric"], frame["metric_value"] = "", float("nan")
    for tag, (columns, how, _) in SORT_KEYS.items():
        mask = frame["tag"] == tag
        if not mask.any() or not all(c in manifest.columns for c in columns):
            continue
        values = pd.concat([numeric_column(rows, c) for c in columns], axis=1)
        frame.loc[mask, "metric"] = columns[0] if len(columns) == 1 else f"{how}({','.join(columns)})"
        frame.loc[mask, "metric_value"] = (values.min(axis=1) if how == "min" else values.max(axis=1))[mask]

    # One sort key that is "worse first" for every tag: negate the values of tags where high is worse.
    ascending = frame["tag"].map(lambda t: SORT_KEYS.get(t, ((), "", True))[2])
    frame["_tag_order"] = frame["tag"].map(lambda t: _tag_order(t) if t else (len(SEVERITY_ORDER), 0, ""))
    frame["_key"] = frame["metric_value"].where(ascending, -frame["metric_value"])
    frame = frame.sort_values(["_tag_order", "_key", "path"], na_position="last").drop(columns=["_tag_order", "_key"])

    all_tags = [t for reason in rows["reason"] for t in row_tags(reason)]
    for column in metric_columns(all_tags, manifest):
        frame[column] = _metric_values(manifest, rows.loc[frame.index], column, frame["path"])
    return frame.reset_index(drop=True)


def drop_manifest(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.DataFrame:
    """
    Status 'drop' rows, most severe first, then by path. `duplicate_of` is the group_id representative
    for rows that carry a duplicate tag (empty otherwise). The catalog metric columns of every drop tag follow.
    """
    frame = identity(manifest, data_dir, base_dir)
    status = frame["status"]
    duplicate_of = representatives(manifest, status, frame["path"], "group_id")
    frame = frame[status == DROP].copy()
    rows = manifest.loc[frame.index]
    has_duplicate_tag = rows["reason"].map(lambda r: any(is_duplicate_tag(t) for t in parse_tags(r)))
    frame["duplicate_of"] = duplicate_of[frame.index].where(has_duplicate_tag, "")

    frame["_rank"] = frame["severity"].map(_severity_rank)
    frame = frame.sort_values(["_rank", "path"]).drop(columns="_rank")
    all_tags = [t for reason in rows["reason"] for t in row_tags(reason)]
    for column in metric_columns(all_tags, manifest):
        frame[column] = _metric_values(manifest, rows.loc[frame.index], column, frame["path"])
    return frame.reset_index(drop=True)


def duplicate_groups(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.DataFrame:
    """
    One row per member of every group with more than one member, for every grouping column (hash,
    group_id, split_group, group_<model>_<thr>): group_column, group_id, group_size, path, status,
    reason, is_representative and representative_path (see representatives). Sorted by column, then
    largest group first, then path.
    """
    base = identity(manifest, data_dir, base_dir)
    parts = []
    for column in grouping_columns(manifest):
        labels = group_labels(manifest, column)
        sizes = labels.map(labels.value_counts())
        rep = representatives(manifest, base["status"], base["path"], column)
        multi = sizes > 1
        part = pd.DataFrame({
            "group_column": column,
            "group_id": labels[multi],
            "group_size": sizes[multi].astype(int),
            "path": base.loc[multi, "path"],
            "status": base.loc[multi, "status"],
            "reason": base.loc[multi, "reason"],
            "is_representative": (base.loc[multi, "path"] == rep[multi]) & (rep[multi] != ""),
            "representative_path": rep[multi],
        })
        part["_first"] = part.groupby("group_id")["path"].transform("min")
        parts.append(part.sort_values(["group_size", "_first", "path"], ascending=[False, True, True]).drop(columns="_first"))
    columns = ["group_column", "group_id", "group_size", "path", "status", "reason", "is_representative", "representative_path"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


BUILDERS = {OK_FILE: ok_manifest, REVIEW_FILE: review_manifest, DROP_FILE: drop_manifest, GROUPS_FILE: duplicate_groups}


# ---- writing -----------------------------------------------------------------------------------

def write_csv(frame: pd.DataFrame, path: str) -> None:
    """to_csv(index=False) to a temp file in the same directory, then rename, so a crash never leaves half a file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".export-", suffix=".csv")
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_exports(manifest: pd.DataFrame, out_dir: str, data_dir: str, base_dir: str | None = None) -> list[dict]:
    """
    Builds every file in EXPORTS, then writes them to out_dir. Returns one {path, rows, description}
    per file, `path` being the file name. All files are built before any is written, and if anything
    fails, every export file in out_dir is removed before the exception propagates: a half-updated
    set would mix files from two runs, and a stale file must never be mistaken for a current one.
    """
    try:
        frames = {name: build(manifest, data_dir, base_dir) for name, build in BUILDERS.items()}
        for name, frame in frames.items():
            write_csv(frame, os.path.join(out_dir, name))
    except Exception:
        remove_exports(out_dir)
        raise
    return [{"path": name, "rows": len(frame), "description": EXPORTS[name]} for name, frame in frames.items()]


def remove_exports(out_dir: str) -> None:
    """Deletes every file in EXPORTS from out_dir (missing ones are fine)."""
    for name in EXPORTS:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
