"""
Duplicate-group section: how much of the delivery is redundant, at which level, and which groups to check by eye.

Redundancy is usually the biggest finding in a scraped or merged delivery ("you sent 59k images, only
28k are unique"). The counts are split by level to show what kind of redundancy it is:
- exact: byte-identical files (the `hash` column), i.e. re-uploads;
- + perceptual: visually identical (group_id, which also holds the pHash groups), i.e. re-encodes and resizes;
- + each embedding model: only similar (group_<model>_<thr>), e.g. bursts and near-frames.
Each level is the union of all columns before it, computed here on a copy. The manifest's split_group
is not used or changed for the levels; the result only reports whether it matches the last level.

Groups are built with union-find (A~B and B~C put A and C together), so a loose threshold can chain
unrelated images into one large group. Large groups are flagged as `chaining_suspect` so they are
looked at instead of trusted. With cached embeddings, each shown group also gets the mean pairwise
cosine similarity of its members: a low value means the group is a chain, not a set of copies.

Counts use every manifest row, dropped ones included, because the duplicate copies are the dropped rows.
Paths are relative to data_dir, so no local paths leak into the report. Nothing here changes the manifest.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

from src.report.sections.flags import share
from src.report.sections.profile import relative_folders, relative_paths
from src.report.status import STATUSES, resolve_status

HASH_COLUMN = "hash"
DUPLICATE_COLUMN = "group_id"
SPLIT_COLUMN = "split_group"
EMBEDDING_PREFIX = "group_"
EMBEDDINGS_SUFFIX = "_embeddings.npy"
EMBEDDINGS_INDICES_SUFFIX = "_embeddings_indices.npy"

LARGEST_GROUPS = 20
MAX_MEMBERS_SHOWN = 50
MAX_KEPT_SHOWN = 5
MAX_EXPECTED_GROUP_SIZE = 50
# The p99.9 rule needs enough groups to mean anything: with 10 groups, p99.9 is just below the largest
# one, so the largest would always be flagged.
MIN_GROUPS_FOR_PERCENTILE = 1000
SUSPECT_PERCENTILE = 99.9
TOP_FOLDER_PAIRS = 10


# ---- columns and levels ------------------------------------------------------------------------

def embedding_columns(manifest: pd.DataFrame) -> list[str]:
    """Per-model group columns (group_<model>_<thr>), in manifest order."""
    return [c for c in manifest.columns if c.startswith(EMBEDDING_PREFIX) and c != DUPLICATE_COLUMN]


def grouping_columns(manifest: pd.DataFrame) -> list[str]:
    """Every column that defines groups: hash, group_id, split_group, then the embedding columns (those present)."""
    fixed = [c for c in (HASH_COLUMN, DUPLICATE_COLUMN, SPLIT_COLUMN) if c in manifest.columns]
    return fixed + embedding_columns(manifest)


def column_kind(column: str) -> str:
    return {HASH_COLUMN: "exact", DUPLICATE_COLUMN: "duplicate", SPLIT_COLUMN: "split"}.get(column, "embedding")


def group_labels(manifest: pd.DataFrame, column: str) -> pd.Series:
    """
    The column's group of each row. In the hash column, an empty hash means "not hashed", so those rows
    get a group of their own instead of all being grouped together.
    """
    values = manifest[column].astype(str)
    if column == HASH_COLUMN:
        empty = manifest[column].isna() | (values.str.strip() == "")
        values = values.where(~empty, "__unhashed__" + pd.Series(range(len(manifest)), index=manifest.index).astype(str))
    return values


def union_groups(manifest: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Group label per row for the union of `columns`: two rows share a label when they share a value in
    any of the columns, directly or through a chain. Same rule as rebuild_split_group, without touching
    the manifest. Labels are the position of each group's first row.
    """
    parent = np.arange(len(manifest))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for column in columns:
        first_of = {}
        for pos, value in enumerate(group_labels(manifest, column)):
            if value in first_of:
                a, b = find(pos), find(first_of[value])
                if a != b:
                    parent[a] = b
            else:
                first_of[value] = pos
    return pd.Series([find(i) for i in range(len(manifest))], index=manifest.index, name="group")


def redundancy(labels: pd.Series) -> dict:
    """Groups with more than one member, the images in them, redundant copies (members - groups) and the unique count."""
    total = len(labels)
    sizes = labels.value_counts()
    multi = sizes[sizes > 1]
    redundant = int(multi.sum() - len(multi))
    return {
        "groups": int(len(multi)),
        "images_in_groups": share(multi.sum(), total),
        "redundant": share(redundant, total),
        "unique": share(total - redundant, total),
    }


def same_partition(a: pd.Series, b: pd.Series) -> bool:
    """True when two label series group the rows the same way, whatever the labels are."""
    pairs = pd.DataFrame({"a": a.values, "b": b.values}).drop_duplicates()
    return len(pairs) == a.nunique() == b.nunique()


def levels(manifest: pd.DataFrame) -> dict:
    """
    Redundancy per level, each level being the union of every column up to it: exact (hash), then
    + perceptual (group_id), then + each embedding column. `added_redundant` is what the level adds over
    the previous one. `matches_split_group` says whether the last level groups rows like the manifest's
    split_group (False means split_group was built from other columns or not rebuilt).
    """
    steps = []
    if HASH_COLUMN in manifest.columns:
        steps.append(("exact", [HASH_COLUMN]))
    if DUPLICATE_COLUMN in manifest.columns:
        steps.append(("perceptual", [DUPLICATE_COLUMN]))
    steps += [(c[len(EMBEDDING_PREFIX):], [c]) for c in embedding_columns(manifest)]

    result, used, previous, last = [], [], 0, None
    for name, columns in steps:
        used = used + columns
        last = union_groups(manifest, used)
        stats = redundancy(last)
        result.append({"level": name, "columns": list(used), **stats, "added_redundant": stats["redundant"]["count"] - previous})
        previous = stats["redundant"]["count"]
    matches = None
    if last is not None and SPLIT_COLUMN in manifest.columns:
        matches = same_partition(last, manifest[SPLIT_COLUMN].astype(str))
    return {"levels": result, "matches_split_group": matches}


# ---- embeddings --------------------------------------------------------------------------------

def load_cached_embeddings(manifest: pd.DataFrame, embeddings_dir: str) -> tuple[dict, list[dict]]:
    """
    Memory-maps every <model>_embeddings.npy in embeddings_dir together with <model>_embeddings_indices.npy,
    the manifest row index of each embedding row. Returns ({model: (matrix, manifest index)}, unavailable).

    A model without an indices file is listed in `unavailable`, never paired by position. Row i of the
    matrix is the i-th row that was not dropped *when the embeddings were made*, which is not
    recoverable later: with group_keep_strategy='random' a rerun keeps different rows, so the count
    still matches while the rows don't. (On emotion/results, pairing by position gave identical files
    a cosine similarity of 0.17-0.36.) Indices that aren't in the manifest, or a length mismatch, also
    make a model unavailable.
    """
    if not os.path.isdir(embeddings_dir):
        return {}, []
    found, unavailable = {}, []
    for name in sorted(os.listdir(embeddings_dir)):
        if not name.endswith(EMBEDDINGS_SUFFIX):
            continue
        model = name[: -len(EMBEDDINGS_SUFFIX)]
        indices_path = os.path.join(embeddings_dir, f"{model}{EMBEDDINGS_INDICES_SUFFIX}")
        if not os.path.exists(indices_path):
            unavailable.append({"model": model, "reason": f"no {model}{EMBEDDINGS_INDICES_SUFFIX}; rows can't be matched to the manifest"})
            continue
        matrix = np.load(os.path.join(embeddings_dir, name), mmap_mode="r")
        indices = pd.Index(np.load(indices_path))
        if matrix.ndim != 2 or len(matrix) != len(indices):
            unavailable.append({"model": model, "reason": f"{len(matrix)} embedding rows but {len(indices)} indices"})
            continue
        if not indices.isin(manifest.index).all():
            unavailable.append({"model": model, "reason": "indices refer to rows that are not in the manifest"})
            continue
        found[model] = (matrix, indices)
    return found, unavailable


def model_for_column(column: str, embeddings: dict, default_model: str | None) -> str | None:
    """The column's own model for group_<model>_<thr>; `default_model` for the other columns."""
    for model in sorted(embeddings, key=len, reverse=True):
        if column.startswith(f"{EMBEDDING_PREFIX}{model}_"):
            return model
    return default_model if default_model in embeddings else None


def mean_pairwise_similarity(matrix, row_positions: list[int]) -> float | None:
    """Mean cosine similarity over all pairs of the given rows (None for fewer than 2). O(n x dim)."""
    n = len(row_positions)
    if n < 2:
        return None
    vectors = np.asarray(matrix[sorted(row_positions)], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1, norms)
    total = unit.sum(axis=0)
    # sum over all ordered pairs i != j of u_i . u_j = |sum u|^2 - sum |u_i|^2
    return round(float((total @ total - (unit * unit).sum()) / (n * (n - 1))), 4)


# ---- per-column statistics ---------------------------------------------------------------------

def suspect_threshold(sizes: pd.Series, max_expected: int) -> float:
    """Group size above which a group is a chaining suspect: max_expected, or p99.9 when there are enough groups."""
    if len(sizes) >= MIN_GROUPS_FOR_PERCENTILE:
        return min(float(max_expected), float(np.percentile(sizes, SUSPECT_PERCENTILE)))
    return float(max_expected)


def describe_group(members: pd.DataFrame, suspect: bool, similarity: dict | None) -> dict:
    """One group: size, folders, status counts, kept (ok) members and member paths, both capped."""
    status_counts = members["status"].value_counts()
    folder_counts = members["folder"].value_counts()
    kept = members.loc[members["status"] == "ok", "rel_path"]
    return {
        "size": len(members),
        "folders": {f: int(c) for f, c in sorted(folder_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
        "status": {s: int(status_counts.get(s, 0)) for s in STATUSES},
        "kept": sorted(kept)[:MAX_KEPT_SHOWN],
        "members": sorted(members["rel_path"])[:MAX_MEMBERS_SHOWN],
        "members_truncated": len(members) > MAX_MEMBERS_SHOWN,
        "chaining_suspect": suspect,
        "similarity": similarity,
    }


def column_statistics(
    rows: pd.DataFrame,
    labels: pd.Series,
    column: str,
    embeddings: dict,
    default_model: str | None,
    largest: int = LARGEST_GROUPS,
    max_expected_group_size: int = MAX_EXPECTED_GROUP_SIZE,
) -> dict:
    """
    Statistics for one grouping column. `rows` has rel_path, folder and status per manifest row.
    Returns redundancy, a size histogram, the `largest` groups, chaining suspects, and groups that span
    more than one folder (with the most common folder pairs).
    """
    sizes = labels.value_counts()
    multi = sizes[sizes > 1]
    limit = suspect_threshold(multi, max_expected_group_size)
    suspects = multi[multi > limit]

    model = model_for_column(column, embeddings, default_model)
    position = None
    if model is not None:
        _, index = embeddings[model]
        position = pd.Series(np.arange(len(index)), index=index)

    def similarity(group_index):
        if position is None:
            return None
        found = position.reindex(group_index).dropna().astype(int).tolist()
        return {"model": model, "members_with_embeddings": len(found),
                "mean_pairwise_cosine": mean_pairwise_similarity(embeddings[model][0], found)}

    in_multi = labels.isin(multi.index)
    grouped = rows[in_multi].groupby(labels[in_multi])
    folders_per_group = grouped["folder"].nunique()
    cross = folders_per_group[folders_per_group > 1]

    pair_counts = {}
    for label in cross.index:
        for pair in combinations(sorted(grouped.get_group(label)["folder"].unique()), 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    top_pairs = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_FOLDER_PAIRS]

    first_path = grouped["rel_path"].min()

    def ordered(labels_):
        """Largest first; ties broken by the group's first path, so the order is deterministic."""
        return sorted(labels_, key=lambda g: (-multi[g], first_path[g]))

    def describe(label):
        members = grouped.get_group(label)
        return describe_group(members, bool(multi[label] > limit), similarity(members.index))

    size_hist = multi.value_counts().sort_index()
    return {
        "column": column,
        "kind": column_kind(column),
        **redundancy(labels),
        "size_histogram": [{"size": int(s), "groups": int(n)} for s, n in size_hist.items()],
        "largest": [describe(g) for g in ordered(multi.index)[:largest]],
        "chaining": {
            "size_limit": limit,
            "suspects": len(suspects),
            "groups": [describe(g) for g in ordered(suspects.index)[:largest]],
        },
        "cross_folder": {
            **share(len(cross), len(multi)),
            "folder_pairs": [{"folders": list(p), "groups": c} for p, c in top_pairs],
        },
    }


def duplicate_statistics(
    manifest: pd.DataFrame,
    data_dir: str,
    base_dir: str | None = None,
    embeddings: dict | None = None,
    default_model: str | None = None,
    largest: int = LARGEST_GROUPS,
    max_expected_group_size: int = MAX_EXPECTED_GROUP_SIZE,
) -> dict:
    """
    The whole duplicates section as a JSON-ready dict.

    - `data_dir`: root of the delivery; paths and folders are reported relative to it.
    - `base_dir`: directory the manifest's paths are relative to (default: the current directory).
    - `embeddings`: {model: (matrix, manifest index)}, e.g. from load_cached_embeddings. Embedding columns
      use their own model; hash, group_id and split_group use `default_model` (default: the first model
      by name). Without embeddings, `similarity` is None everywhere.
    """
    embeddings = embeddings or {}
    if default_model is None and embeddings:
        default_model = sorted(embeddings)[0]
    rows = pd.DataFrame({
        "rel_path": relative_paths(manifest, data_dir, base_dir),
        "folder": relative_folders(manifest, data_dir, base_dir),
        "status": resolve_status(manifest),
    }, index=manifest.index)
    columns = grouping_columns(manifest)
    return {
        "total_images": len(manifest),
        "columns": {
            c: column_statistics(rows, group_labels(manifest, c), c, embeddings, default_model, largest, max_expected_group_size)
            for c in columns
        },
        **levels(manifest),
        "similarity_models": sorted(embeddings),
    }
