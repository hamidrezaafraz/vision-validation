"""
Consistency checks on a manifest, run before anything is reported.

The report states numbers to a client. If the manifest is internally inconsistent (a half-finished
run, an older run's manifest against a changed folder, a bug in a check), those numbers would be
wrong. So every invariant below is checked first, and errors stop the report (see raise_for_errors).

Severities:
- 'error': the numbers cannot be trusted (unknown actions, flags without a reason, tags missing from
  the catalog in src/report/reasons.py, tags without the metric that explains them, duplicate paths,
  missing files, files that were never scanned, missing required columns).
- 'warning': the numbers are right, but something is worth a look before they go out (a duplicate
  copy whose representative was later dropped, a duplicate group with several unflagged members,
  an unflagged row that still carries a reason).

Nothing here changes the manifest.
"""
import os
from dataclasses import dataclass, field

import pandas as pd

from src.report.reasons import get_reason, is_duplicate_tag, parse_tags
from src.report.status import ACTION_TO_STATUS, DROP, OK
from src.utils.images import find_image_paths

ERROR, WARNING = "error", "warning"
REQUIRED_COLUMNS = ("path", "group_id", "action", "reason")
MAX_EXAMPLES = 20

@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str
    count: int
    examples: list[str] = field(default_factory=list)


class ManifestValidationError(ValueError):
    """Raised by raise_for_errors when validate_manifest found at least one error."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        lines = [f"[{i.code}] {i.message} (examples: {i.examples[:3]})" for i in issues]
        super().__init__(f"{len(issues)} manifest error(s):\n" + "\n".join(lines))


def _issue(code: str, severity: str, message: str, paths: pd.Series | list) -> Issue:
    paths = list(paths)
    return Issue(code, severity, message, count=len(paths), examples=[str(p) for p in paths[:MAX_EXAMPLES]])


def _resolve(path: str, base_dir: str) -> str:
    return os.path.normpath(os.path.join(base_dir, path))


def validate_manifest(manifest: pd.DataFrame, data_dir: str | None = None, base_dir: str | None = None) -> list[Issue]:
    """
    Checks the manifest's invariants and returns one Issue per broken invariant (empty list = valid).

    - `base_dir`: directory the manifest's relative paths are relative to (default: the current
      directory). Manifests written from the repo root store 'emotion/Data/...', so a notebook in
      notebooks/ must pass base_dir='..'.
    - `data_dir`: if given, every file under it must be in the manifest (no unscanned files). Uses the
      same file listing as the dataset classes, and is resolved relative to the current directory.

    If required columns are missing, only that error is returned, since the other checks need them.
    """
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]
    if missing_columns:
        return [Issue("missing_columns", ERROR, f"Manifest is missing required columns {missing_columns}", len(missing_columns), missing_columns)]

    base_dir = base_dir if base_dir is not None else os.getcwd()
    issues: list[Issue] = []
    path = manifest["path"]
    action = manifest["action"]
    reason = manifest["reason"].fillna("").astype(str)
    status = action.map(ACTION_TO_STATUS)  # NaN for unknown actions

    unknown_action = status.isna()
    if unknown_action.any():
        values = sorted(map(repr, action[unknown_action].unique()))
        issues.append(_issue("unknown_action", ERROR, f"Rows with an unknown action {', '.join(values)}; expected one of {sorted(ACTION_TO_STATUS)}", path[unknown_action]))

    flagged_without_reason = status.isin(["review", DROP]) & (reason.str.strip() == "")
    if flagged_without_reason.any():
        issues.append(_issue("flag_without_reason", ERROR, "Rows flagged 'review' or 'drop' without a reason", path[flagged_without_reason]))

    tags = reason.map(parse_tags)
    issues.extend(_tag_issues(manifest, tags))

    duplicated_path = path.duplicated(keep=False)
    if duplicated_path.any():
        issues.append(_issue("duplicate_path", ERROR, "Paths that appear in more than one row", path[duplicated_path].drop_duplicates()))

    resolved = path.map(lambda p: _resolve(p, base_dir))
    missing_file = ~resolved.map(os.path.isfile)
    if missing_file.any():
        issues.append(_issue("missing_file", ERROR, f"Manifest paths that are not files (resolved against base_dir={base_dir!r})", path[missing_file]))

    if data_dir is not None:
        on_disk = {os.path.normpath(os.path.abspath(p)) for p in find_image_paths(data_dir)}
        unscanned = sorted(on_disk - set(resolved.map(os.path.abspath)))
        if unscanned:
            issues.append(_issue("unscanned_file", ERROR, f"Files under {data_dir!r} that are not in the manifest", unscanned))

    issues.extend(_group_issues(manifest, status, tags))

    unflagged_with_reason = (status == OK) & (reason.str.strip() != "")
    if unflagged_with_reason.any():
        issues.append(_issue("ok_with_reason", WARNING, "Unflagged rows ('' or 'keep') that still carry a reason; a flag may have been lost", path[unflagged_with_reason]))

    return issues


def _is_blank(values: pd.Series) -> pd.Series:
    return values.isna() | (values.astype(str).str.strip() == "")


def _tag_issues(manifest: pd.DataFrame, tags: pd.Series) -> list[Issue]:
    """Every tag is in the catalog, and every row carrying a tag has a value in the columns that explain it."""
    issues = []
    path = manifest["path"]
    exploded = tags.explode().dropna()

    unknown = exploded[~exploded.map(lambda t: get_reason(t) is not None)]
    if not unknown.empty:
        values = sorted(map(repr, unknown.unique()))
        issues.append(_issue("unknown_tag", ERROR, f"Rows with reason tags missing from the catalog (src/report/reasons.py): {', '.join(values)}", path[unknown.index.unique()]))

    # One issue per column, naming the tags that need it. A missing column and an empty cell are the
    # same problem: the report cannot show why the row was flagged.
    needs: dict[str, set] = {}
    for idx, tag in exploded.items():
        entry = get_reason(tag)
        for col in entry.columns if entry else ():
            needs.setdefault(col, set()).add(idx)
    for col, rows in sorted(needs.items()):
        rows = manifest.index[manifest.index.isin(rows)]
        lacking = rows if col not in manifest.columns else rows[_is_blank(manifest.loc[rows, col]).to_numpy()]
        if len(lacking):
            tag_names = sorted({get_reason(t).tag for t in exploded.loc[lacking] if get_reason(t) and col in get_reason(t).columns})
            where = "column is missing" if col not in manifest.columns else "value is empty"
            issues.append(_issue("missing_metric_column", ERROR, f"Rows tagged {tag_names} need {col!r}, but the {where}", path[lacking]))
    return issues


def _group_issues(manifest: pd.DataFrame, status: pd.Series, tags: pd.Series) -> list[Issue]:
    """Duplicate-group invariants on group_id."""
    issues = []
    path, group_id = manifest["path"], manifest["group_id"]
    not_dropped = status.notna() & (status != DROP)
    has_duplicate_tag = tags.map(lambda ts: any(is_duplicate_tag(t) for t in ts))

    # A copy dropped as a duplicate must have a surviving member in its group to stand for it.
    # This breaks when a check that runs after deduplication (e.g. check_decode) drops the representative.
    survivors_per_group = not_dropped.groupby(group_id).transform("sum")
    orphaned = (status == DROP) & has_duplicate_tag & (survivors_per_group == 0)
    if orphaned.any():
        issues.append(_issue("orphaned_duplicate", WARNING, "Rows dropped as duplicates whose group has no member left that isn't dropped; run deduplication after the integrity/photometric checks", path[orphaned]))

    # After keep_one_from_group, a duplicate group has exactly one unflagged member. More than one means
    # the group was formed but no keep step ran for it (or a later merge was not followed by one).
    ok_per_group = (status == OK).groupby(group_id).transform("sum")
    unresolved = (status == OK) & (ok_per_group > 1)
    if unresolved.any():
        n_groups = group_id[unresolved].nunique()
        issues.append(_issue("unresolved_group", WARNING, f"{n_groups} duplicate groups have more than one unflagged member; run keep_one_from_group after grouping", path[unresolved]))

    return issues


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == ERROR]


def raise_for_errors(issues: list[Issue]) -> None:
    """Raises ManifestValidationError if any issue is an error; warnings never raise."""
    found = errors(issues)
    if found:
        raise ManifestValidationError(found)
