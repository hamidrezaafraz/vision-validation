"""
Flag statistics section: how many images were flagged, by which tags, and how the tags overlap.

One image can carry several tags, so per-tag counts add up to more than the number of flagged images.
`images_flagged` (rows with at least one tag) is the real total. Never sum the per-tag counts.

Every count comes as {'count', 'of', 'pct'}, where `of` is the number it is a share of, so a reader
never has to guess the denominator. Shares of the whole delivery use every manifest row (total
scanned), dropped rows included.

Tags are counted under one name per finding:
- aliases (old spellings from before todo 17) resolve to their catalog tag;
- prefix tags keep their written name, because it carries settings: near_duplicate_phash_4 and
  near_duplicate_phash_8 are different findings;
- tags missing from the catalog are counted as written, with severity and category 'unknown', so a
  report built with --force still shows them (validate_manifest reports them as errors).
A tag written twice in one row counts once.

Nothing here changes the manifest.
"""
import pandas as pd

from src.report.reasons import SEVERITIES, get_reason, parse_tags
from src.report.status import STATUSES, resolve_status

UNKNOWN = "unknown"
# Worst first. An unknown tag has no severity, so it only decides an image's worst severity when
# the image has no known tag.
SEVERITY_ORDER = (*SEVERITIES, UNKNOWN)
TOP_COMBINATIONS = 10

OVERLAP_NOTE = (
    "An image can carry several tags, so per-tag counts add up to more than the number of flagged "
    "images. Use images_flagged as the total."
)


def share(count, of) -> dict:
    """{'count', 'of', 'pct'}: `count` as a percentage of `of`, rounded to 2 decimals (0.0 when `of` is 0)."""
    count, of = int(count), int(of)
    return {"count": count, "of": of, "pct": round(100 * count / of, 2) if of else 0.0}


def canonical_tag(tag: str) -> str:
    """The name a written tag is counted under (see the module docstring)."""
    entry = get_reason(tag)
    if entry is None or (entry.prefix and tag not in entry.aliases):
        return tag
    return entry.tag


def tag_info(tag: str) -> dict:
    """Category and severity of a canonical tag, 'unknown' for both if it isn't in the catalog."""
    entry = get_reason(tag)
    return {"category": entry.category, "severity": entry.severity} if entry else {"category": UNKNOWN, "severity": UNKNOWN}


def row_tags(manifest: pd.DataFrame) -> pd.Series:
    """
    Long form of the `reason` column: one entry per (row, distinct canonical tag), indexed by the
    manifest's index and named 'tag'. Rows without tags don't appear.
    """
    tags = manifest["reason"].map(lambda r: sorted({canonical_tag(t) for t in parse_tags(r)}))
    return tags.explode().dropna().astype(str).rename("tag")


def status_counts(manifest: pd.DataFrame) -> dict:
    """Share of all rows per final status ('ok', 'review', 'drop'); the three counts add up to the total."""
    counts = resolve_status(manifest).value_counts()
    return {s: share(counts.get(s, 0), len(manifest)) for s in STATUSES}


def tag_counts(manifest: pd.DataFrame) -> list[dict]:
    """
    One dict per tag, most frequent first: tag, category, severity, its share of all rows, and
    `by_status` (how many of its rows ended up ok / review / drop, which shows the policy that was used).
    """
    tags = row_tags(manifest)
    status = resolve_status(manifest)
    by_status = pd.crosstab(tags.values, status.loc[tags.index].values) if len(tags) else pd.DataFrame()
    counts = tags.value_counts()
    rows = []
    for tag in sorted(counts.index, key=lambda t: (-counts[t], t)):
        rows.append({
            "tag": tag,
            **tag_info(tag),
            **share(counts[tag], len(manifest)),
            "by_status": {s: int(by_status.at[tag, s]) if s in by_status.columns else 0 for s in STATUSES},
        })
    return rows


def severity_counts(manifest: pd.DataFrame) -> dict:
    """
    Images per flag severity, as shares of all rows:
    - `any`: images with at least one tag of that severity. These overlap (an image can be both
      'error' and 'info'), so they don't add up to images_flagged.
    - `worst`: images by their most serious tag (error > warning > info > unknown). These partition
      the flagged images and add up to images_flagged.
    """
    tags = row_tags(manifest)
    severity = tags.map(lambda t: tag_info(t)["severity"])
    total = len(manifest)
    any_counts = severity.groupby(level=0).unique().explode().value_counts() if len(tags) else pd.Series(dtype=int)
    rank = severity.map(SEVERITY_ORDER.index)
    worst_counts = rank.groupby(level=0).min().map(lambda r: SEVERITY_ORDER[r]).value_counts() if len(tags) else pd.Series(dtype=int)
    return {
        "any": {s: share(any_counts.get(s, 0), total) for s in SEVERITY_ORDER},
        "worst": {s: share(worst_counts.get(s, 0), total) for s in SEVERITY_ORDER},
    }


def cooccurrence_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Tag × tag counts of images carrying both tags (M.T @ M of the one-hot tag matrix). The diagonal is
    each tag's own count. Rows and columns are the tags in sorted order; empty if no row has a tag.
    """
    tags = row_tags(manifest)
    if tags.empty:
        return pd.DataFrame(dtype=int)
    onehot = pd.crosstab(tags.index, tags.values).clip(upper=1)
    matrix = onehot.T @ onehot
    matrix.index.name = matrix.columns.name = None
    return matrix.sort_index().sort_index(axis=1)


def cooccurrence_pairs(manifest: pd.DataFrame) -> list[dict]:
    """
    Every pair of different tags that share at least one image, most shared first. Each has its share
    of all rows and `given`: per tag of the pair, the share of that tag's images that also carry the
    other one ("80% of too_small images are also blurry").
    """
    matrix = cooccurrence_matrix(manifest)
    tags = list(matrix.index)
    pairs = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            both = matrix.at[a, b]
            if both:
                pairs.append({
                    "tags": [a, b],
                    **share(both, len(manifest)),
                    "given": {a: share(both, matrix.at[a, a]), b: share(both, matrix.at[b, b])},
                })
    return sorted(pairs, key=lambda p: (-p["count"], p["tags"]))


def tag_combinations(manifest: pd.DataFrame, top: int = TOP_COMBINATIONS) -> dict:
    """
    The exact tag set of each flagged image, counted. A single-tag combination means the image has
    only that tag. Returns `distinct` (number of different combinations) and the `top` most common,
    each with its share of all rows.
    """
    tags = row_tags(manifest)
    combos = tags.groupby(level=0).agg(lambda t: tuple(sorted(t))).value_counts() if len(tags) else pd.Series(dtype=int)
    ordered = sorted(combos.index, key=lambda c: (-combos[c], c))
    return {
        "distinct": len(ordered),
        "top": [{"tags": list(c), **share(combos[c], len(manifest))} for c in ordered[:top]],
    }


def flag_statistics(manifest: pd.DataFrame, top_combinations: int = TOP_COMBINATIONS) -> dict:
    """The whole flags section as a JSON-ready dict. Raises ValueError on unknown actions (see resolve_status)."""
    total = len(manifest)
    flagged = row_tags(manifest).index.nunique()
    return {
        "total_images": total,
        "status": status_counts(manifest),
        "images_flagged": share(flagged, total),
        "note": OVERLAP_NOTE,
        "tags": tag_counts(manifest),
        "severity": severity_counts(manifest),
        "cooccurrence": cooccurrence_pairs(manifest),
        "combinations": tag_combinations(manifest, top=top_combinations),
    }
