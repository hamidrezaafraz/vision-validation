"""
Dataset profile section: what is actually in the delivery, independent of how it was flagged.

With no labels, this is most of the analysis. It tells the client what they sent, and it justifies the
thresholds: the threshold table shows how many images sit just on either side of each one, so a
reader can see how much the flag counts would move if a threshold moved a little.

Data sources:
- metric columns written by check_decode (format, color mode, size, ...) and check_photometric
  (brightness, contrast, clipping, blur), read from the manifest;
- file sizes from os.stat;
- thresholds from audits.json (the settings of the last successful run of each check), never from
  the checks' current defaults, which may differ from what produced the flags.

Image metrics are summarized over decodable rows only. Undecodable files have placeholder values
(0 x 0 pixels, infinite aspect ratio) that would distort every distribution, so they are counted
separately. A metric column that is missing, or empty for a row, is reported as such rather than guessed.

Nothing here changes the manifest.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.report.sections.flags import row_tags, share

BINS = 20
TOP_RESOLUTIONS = 10
BORDERLINE_MARGIN = 0.10
MAX_EXAMPLES = 20
ROOT_FOLDER = "."

FILE_SIZE = "file_size_bytes"
NUMERIC_METRICS = (
    FILE_SIZE, "width", "height", "megapixels", "aspect_ratio",
    "mean_brightness", "pixel_std", "dark_clip_ratio", "bright_clip_ratio", "blur_score",
)
CATEGORICAL_METRICS = ("format", "color_mode", "num_channels", "dtype", "bit_depth", "has_non_trivial_alpha", "exif_orientation")

# Extensions that are correct for each Pillow format. Formats missing here are not checked.
EXTENSIONS_BY_FORMAT = {
    "JPEG": {"jpg", "jpeg", "jpe", "jfif"},
    "MPO": {"jpg", "jpeg"},  # multi-picture JPEG, as written by many phone cameras
    "PNG": {"png"},
    "GIF": {"gif"},
    "BMP": {"bmp", "dib"},
    "TIFF": {"tif", "tiff"},
    "WEBP": {"webp"},
    "ICO": {"ico"},
    "PPM": {"ppm", "pgm", "pbm", "pnm"},
}


@dataclass(frozen=True)
class Threshold:
    """
    A check setting that flags a metric: 'min' flags values below it, 'max' flags values above it.
    `excluded_by` names another threshold (by parameter) whose flagged rows the check doesn't apply this
    one to, so the counts here match the tags.
    """
    column: str
    check: str
    parameter: str
    kind: str
    excluded_by: str | None = None


# The comparisons match the checks: too_small is width < min_width, blurry is blur_score < min_blur_score, ...
THRESHOLDS = (
    Threshold("width", "check_decode", "min_width", "min"),
    Threshold("height", "check_decode", "min_height", "min"),
    Threshold("width", "check_decode", "max_width", "max"),
    Threshold("height", "check_decode", "max_height", "max"),
    Threshold("aspect_ratio", "check_decode", "max_aspect_ratio", "max"),
    Threshold("pixel_std", "check_photometric", "min_std", "min"),
    # check_photometric gives blank frames no other photometric tag.
    Threshold("dark_clip_ratio", "check_photometric", "max_clip_ratio", "max", excluded_by="min_std"),
    Threshold("bright_clip_ratio", "check_photometric", "max_clip_ratio", "max", excluded_by="min_std"),
    Threshold("blur_score", "check_photometric", "min_blur_score", "min", excluded_by="min_std"),
)
_BY_PARAMETER = {t.parameter: t for t in THRESHOLDS}


# ---- helpers -----------------------------------------------------------------------------------

def decodable_mask(manifest: pd.DataFrame) -> pd.Series:
    """False only for rows explicitly marked undecodable; True when check_decode didn't run or left it empty."""
    if "is_decodable" not in manifest.columns:
        return pd.Series(True, index=manifest.index)
    values = manifest["is_decodable"].astype(str).str.strip().str.lower()
    return ~values.isin({"false", "0"})


def numeric_column(manifest: pd.DataFrame, column: str) -> pd.Series:
    """The column as floats; empty, non-numeric and infinite values become NaN."""
    values = pd.to_numeric(manifest[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def _has_value(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def _abs_paths(manifest: pd.DataFrame, base_dir: str | None) -> pd.Series:
    base_dir = base_dir if base_dir is not None else os.getcwd()
    return manifest["path"].map(lambda p: os.path.abspath(os.path.join(base_dir, p)))


def file_sizes(manifest: pd.DataFrame, base_dir: str | None = None) -> pd.Series:
    """File size in bytes per row, NaN for files that no longer exist. Paths resolve against base_dir."""
    def size(path):
        try:
            return float(os.stat(path).st_size)
        except OSError:
            return np.nan
    return _abs_paths(manifest, base_dir).map(size).rename(FILE_SIZE)


def relative_paths(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.Series:
    """Each row's path relative to data_dir, with '/' separators, so reports carry no local paths."""
    root = os.path.abspath(data_dir)
    paths = _abs_paths(manifest, base_dir).map(lambda p: os.path.relpath(p, root))
    return paths.astype(str).str.replace(os.sep, "/", regex=False)


def relative_folders(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> pd.Series:
    """Each row's folder relative to data_dir, with '/' separators; '.' for files directly in data_dir."""
    root = os.path.abspath(data_dir)
    folders = _abs_paths(manifest, base_dir).map(lambda p: os.path.relpath(os.path.dirname(p), root))
    return folders.astype(str).str.replace(os.sep, "/", regex=False)


def _float(x) -> float:
    return round(float(x), 6)


def last_ok_params(checks: dict | None, check: str) -> tuple[dict | None, str | None]:
    """Settings and finish time of the most recent successful run of `check` in latest_by_check output."""
    if not checks or check not in checks:
        return None, None
    runs = [checks[check]["latest"], *reversed(checks[check]["history"])]
    for entry in runs:
        if entry.get("status") == "ok":
            return entry.get("params", {}), entry.get("finished_at")
    return None, None


# ---- section parts -----------------------------------------------------------------------------

def distribution(values: pd.Series, bins: int = BINS) -> dict:
    """count, min, p1, p5, median, p95, p99, max, mean and a `bins`-bin linear histogram of the non-NaN values."""
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"count": 0, "min": None, "p1": None, "p5": None, "median": None, "p95": None,
                "p99": None, "max": None, "mean": None, "histogram": {"edges": [], "counts": []}}
    p1, p5, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
    counts, edges = np.histogram(values, bins=bins)
    return {
        "count": int(values.size),
        "min": _float(values.min()), "p1": _float(p1), "p5": _float(p5), "median": _float(p50),
        "p95": _float(p95), "p99": _float(p99), "max": _float(values.max()), "mean": _float(values.mean()),
        "histogram": {"edges": [_float(e) for e in edges], "counts": [int(c) for c in counts]},
    }


def numeric_distributions(manifest: pd.DataFrame, sizes: pd.Series, bins: int = BINS) -> dict:
    """
    {metric: distribution} over decodable rows, plus `unavailable` (metrics whose column is missing).
    File sizes cover every file that exists, decodable or not.
    """
    decodable = decodable_mask(manifest)
    result, unavailable = {FILE_SIZE: distribution(sizes, bins)}, []
    for column in NUMERIC_METRICS:
        if column == FILE_SIZE:
            continue
        if column not in manifest.columns:
            unavailable.append(column)
            continue
        result[column] = distribution(numeric_column(manifest, column)[decodable], bins)
    return {"metrics": result, "unavailable": unavailable}


def categorical_counts(manifest: pd.DataFrame) -> dict:
    """
    For each categorical metric, the share of decodable rows per value, most common first. Rows with an
    empty value are counted as `missing`. `unavailable` lists metrics whose column is missing.
    """
    decodable = decodable_mask(manifest)
    population = int(decodable.sum())
    result, unavailable = {}, []
    for column in CATEGORICAL_METRICS:
        if column not in manifest.columns:
            unavailable.append(column)
            continue
        values = manifest.loc[decodable, column]
        present = values[_has_value(values)].astype(str).str.strip()
        counts = present.value_counts()
        result[column] = {
            "values": [{"value": v, **share(counts[v], population)} for v in sorted(counts.index, key=lambda v: (-counts[v], v))],
            "missing": share(population - len(present), population),
        }
    return {"metrics": result, "unavailable": unavailable}


def extension_formats(manifest: pd.DataFrame) -> dict:
    """
    File extension against the format the decoder found. A file is `mismatched` when its format is in
    EXTENSIONS_BY_FORMAT and its extension is not one of that format's (e.g. a PNG named .jpg).
    Formats not in the table are listed in `unchecked_formats` instead of being guessed.
    """
    if "format" not in manifest.columns:
        return {"available": False}
    rows = manifest[decodable_mask(manifest) & _has_value(manifest["format"])]
    ext = rows["path"].map(lambda p: os.path.splitext(p)[1].lstrip(".").lower())
    fmt = rows["format"].astype(str).str.strip()
    pairs = pd.DataFrame({"extension": ext, "format": fmt}).value_counts()
    table = [{"extension": e, "format": f, "count": int(c)} for (e, f), c in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))]

    checked = fmt.isin(EXTENSIONS_BY_FORMAT.keys())
    mismatched = checked & ~pd.Series([e in EXTENSIONS_BY_FORMAT.get(f, set()) for e, f in zip(ext, fmt)], index=rows.index)
    return {
        "available": True,
        "table": table,
        "mismatched": share(mismatched.sum(), checked.sum()),
        "mismatched_examples": rows.loc[mismatched, "path"].head(MAX_EXAMPLES).tolist(),
        "unchecked_formats": sorted(set(fmt[~checked])),
    }


def top_resolutions(manifest: pd.DataFrame, top: int = TOP_RESOLUTIONS) -> dict:
    """The most common exact width x height among decodable rows, with `distinct` resolutions overall."""
    if not {"width", "height"} <= set(manifest.columns):
        return {"available": False}
    w, h = numeric_column(manifest, "width"), numeric_column(manifest, "height")
    valid = decodable_mask(manifest) & w.notna() & h.notna()
    sizes = pd.DataFrame({"width": w[valid].astype(int), "height": h[valid].astype(int)}).value_counts()
    ordered = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "available": True,
        "distinct": len(ordered),
        "top": [{"resolution": f"{w_}x{h_}", "width": w_, "height": h_, **share(c, valid.sum())} for (w_, h_), c in ordered[:top]],
    }


def threshold_sensitivity(manifest: pd.DataFrame, checks: dict | None, margin: float = BORDERLINE_MARGIN) -> dict:
    """
    For each threshold in THRESHOLDS whose check has a successful audit entry: how many decodable images
    are on its flagged side, and how many are `borderline` (within `margin` x |threshold| of it), split
    into the borderline ones that are flagged and the ones that pass. Computed from the metric values
    with the same comparisons and exclusions as the checks (see Threshold.excluded_by), over the rows
    that have a value. Thresholds that can't be evaluated are listed in `unavailable` with the reason.
    """
    decodable = decodable_mask(manifest)
    result, unavailable = [], []
    for t in THRESHOLDS:
        params, finished_at = last_ok_params(checks, t.check)
        if params is None:
            unavailable.append({"parameter": t.parameter, "column": t.column, "reason": f"no successful {t.check} run in audits.json"})
            continue
        if params.get(t.parameter) is None:
            unavailable.append({"parameter": t.parameter, "column": t.column, "reason": f"{t.parameter} not in the {t.check} audit entry"})
            continue
        if t.column not in manifest.columns:
            unavailable.append({"parameter": t.parameter, "column": t.column, "reason": f"column {t.column} missing from the manifest"})
            continue
        threshold = float(params[t.parameter])
        population = decodable.copy()
        if t.excluded_by is not None:
            other = _BY_PARAMETER[t.excluded_by]
            other_values = numeric_column(manifest, other.column)
            other_threshold = float(params[other.parameter])
            population &= ~(other_values < other_threshold if other.kind == "min" else other_values > other_threshold)
        values = numeric_column(manifest, t.column)[population].dropna()
        flagged = values < threshold if t.kind == "min" else values > threshold
        band = margin * abs(threshold)
        borderline = (values - threshold).abs() <= band
        result.append({
            "column": t.column,
            "check": t.check,
            "parameter": t.parameter,
            "kind": t.kind,
            "threshold": threshold,
            "audited_at": finished_at,
            "excluded_by": t.excluded_by,
            "flagged": share(flagged.sum(), len(values)),
            "borderline_range": [_float(threshold - band), _float(threshold + band)],
            "borderline": share(borderline.sum(), len(values)),
            "borderline_flagged": int((borderline & flagged).sum()),
            "borderline_passing": int((borderline & ~flagged).sum()),
        })
    return {"margin": margin, "thresholds": result, "unavailable": unavailable}


def folder_breakdown(manifest: pd.DataFrame, data_dir: str, base_dir: str | None = None) -> dict:
    """
    Subfolders are often the only metadata in an unlabeled delivery (camera, date, source), so this shows
    whether a problem is concentrated in one of them:
    - `folders`: per folder (relative to data_dir), its share of all images, its share of images that
      carry at least one tag, and its most common tag;
    - `depths`: images per folder depth (0 = directly in data_dir);
    - `tag_concentration`: per tag, the folder holding most of that tag's images and its share of them.
    Flags use the same tag counting as the flags section.
    """
    total = len(manifest)
    folders = relative_folders(manifest, data_dir, base_dir)
    tags = row_tags(manifest)
    tag_folders = pd.DataFrame({"tag": tags.values, "folder": folders.loc[tags.index].values})
    flagged_rows = folders.loc[tags.index.unique()]

    folder_counts, flagged_counts = folders.value_counts(), flagged_rows.value_counts()
    rows = []
    for folder in sorted(folder_counts.index, key=lambda f: (-folder_counts[f], f)):
        in_folder = tag_folders.loc[tag_folders["folder"] == folder, "tag"].value_counts()
        top = sorted(in_folder.index, key=lambda t: (-in_folder[t], t))[:1]
        rows.append({
            "folder": folder,
            **share(folder_counts[folder], total),
            "flagged": share(flagged_counts.get(folder, 0), folder_counts[folder]),
            "top_tag": {"tag": top[0], "count": int(in_folder[top[0]])} if top else None,
        })

    depth = folders.map(lambda f: 0 if f == ROOT_FOLDER else f.count("/") + 1)
    depth_counts = depth.value_counts().sort_index()

    concentration = []
    for tag, group in tag_folders.groupby("tag"):
        counts = group["folder"].value_counts()
        folder = sorted(counts.index, key=lambda f: (-counts[f], f))[0]
        concentration.append({"tag": tag, "folder": folder, **share(counts[folder], len(group)), "folders_with_tag": len(counts)})
    concentration.sort(key=lambda c: (-c["of"], c["tag"]))

    return {
        "folders": rows,
        "depths": [{"depth": int(d), **share(c, total)} for d, c in depth_counts.items()],
        "tag_concentration": concentration,
    }


def dataset_profile(
    manifest: pd.DataFrame,
    data_dir: str,
    base_dir: str | None = None,
    checks: dict | None = None,
    bins: int = BINS,
    top: int = TOP_RESOLUTIONS,
    borderline_margin: float = BORDERLINE_MARGIN,
) -> dict:
    """
    The whole profile section as a JSON-ready dict.

    - `data_dir`: root of the delivery; folders are reported relative to it (resolved from the current directory).
    - `base_dir`: directory the manifest's paths are relative to (default: the current directory).
    - `checks`: latest_by_check(load_audits(...)); without it, the threshold table is empty.
    """
    sizes = file_sizes(manifest, base_dir)
    decodable = decodable_mask(manifest)
    total = len(manifest)
    return {
        "total_images": total,
        "files": {
            "file_count": total,
            "total_bytes": int(sizes.sum()),
            "missing_files": int(sizes.isna().sum()),
        },
        "decodable": share(decodable.sum(), total),
        "undecodable": share((~decodable).sum(), total),
        "extensions": extension_formats(manifest),
        "categorical": categorical_counts(manifest),
        "distributions": numeric_distributions(manifest, sizes, bins),
        "resolutions": top_resolutions(manifest, top),
        "thresholds": threshold_sensitivity(manifest, checks, borderline_margin),
        "folders": folder_breakdown(manifest, data_dir, base_dir),
    }
