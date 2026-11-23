"""
Catalog of every reason tag a check can write into the manifest's `reason` column.

It is the single source of truth for what a flag means: the check that writes it, how serious it is,
a plain-language description for the client report, and the manifest columns and check parameters
that explain it. A tag missing from here is reported as an error by validate_manifest, so new checks
must add their tags. tests/report/test_reasons.py scans src/ for tags to enforce this.

Flag severity (how bad the finding is for the image, unrelated to validate.Issue severity):
- 'error':   the file is unusable (cannot be decoded, or has no content).
- 'warning': a quality concern a person should judge.
- 'info':    redundancy; the image is fine, but another image already stands for it.
"""
from dataclasses import dataclass

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITIES = (ERROR, WARNING, INFO)

INTEGRITY, PHOTOMETRIC, DUPLICATE, LABEL = "integrity", "photometric", "duplicate", "label"

# Same separator as src.utils.dataset.append_reason.
TAG_SEPARATOR = ";"


@dataclass(frozen=True)
class ReasonTag:
    tag: str
    category: str
    check: str
    severity: str
    description: str
    columns: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    # When True, `tag` is a prefix and the rest of the written tag carries settings,
    # e.g. 'near_duplicate_' matches 'near_duplicate_phash_4' (hash and Hamming threshold).
    prefix: bool = False
    # Older spellings written before the tags were standardized (todo 17), or still written by the
    # classification CLI. They resolve to this entry so old manifests stay readable.
    aliases: tuple[str, ...] = ()


CATALOG: tuple[ReasonTag, ...] = (
    # check_decode (src/integrity)
    ReasonTag("corrupt", INTEGRITY, "check_decode", ERROR,
              "The file could not be decoded as an image (truncated, corrupted or not an image).",
              columns=("is_decodable", "decode_error")),
    ReasonTag("exif_rotated", INTEGRITY, "check_decode", WARNING,
              "The file has an EXIF orientation tag, so viewers rotate or flip it while the raw pixels are not.",
              columns=("exif_orientation",)),
    ReasonTag("unexpected_color_mode", INTEGRITY, "check_decode", WARNING,
              "The color mode (e.g. grayscale, CMYK, with alpha) is not one of the expected modes.",
              columns=("color_mode",), parameters=("expected_modes",)),
    ReasonTag("unexpected_bit_depth", INTEGRITY, "check_decode", WARNING,
              "The pixel bit depth (e.g. 16-bit) is not one of the expected depths.",
              columns=("dtype", "bit_depth"), parameters=("expected_dtypes",)),
    ReasonTag("too_small", INTEGRITY, "check_decode", WARNING,
              "The image is smaller than the minimum width or height.",
              columns=("width", "height"), parameters=("min_width", "min_height")),
    ReasonTag("too_large", INTEGRITY, "check_decode", WARNING,
              "The image is larger than the maximum width or height.",
              columns=("width", "height"), parameters=("max_width", "max_height")),
    ReasonTag("extreme_aspect_ratio", INTEGRITY, "check_decode", WARNING,
              "The image is much wider than tall, or much taller than wide.",
              columns=("aspect_ratio",), parameters=("max_aspect_ratio",)),
    # check_photometric (src/photometric)
    ReasonTag("blank_or_solid", PHOTOMETRIC, "check_photometric", ERROR,
              "The image is blank or a single solid color, with almost no variation between pixels.",
              columns=("pixel_std",), parameters=("min_std",)),
    ReasonTag("underexposed", PHOTOMETRIC, "check_photometric", WARNING,
              "Most of the image is pure black, so detail in the dark areas is lost.",
              columns=("dark_clip_ratio",), parameters=("max_clip_ratio", "clip_margin")),
    ReasonTag("overexposed", PHOTOMETRIC, "check_photometric", WARNING,
              "Most of the image is pure white, so detail in the bright areas is lost.",
              columns=("bright_clip_ratio",), parameters=("max_clip_ratio", "clip_margin")),
    ReasonTag("blurry", PHOTOMETRIC, "check_photometric", WARNING,
              "The image is out of focus or motion-blurred (low sharpness score).",
              columns=("blur_score",), parameters=("min_blur_score",)),
    # Duplicate steps (src/cli/lib)
    ReasonTag("exact_duplicate", DUPLICATE, "group_by_hash", INFO,
              "A byte-identical copy of another image that was kept.",
              columns=("group_id",),
              aliases=("exact hash duplicates", "exact byte-level duplicates")),
    ReasonTag("near_duplicate_", DUPLICATE, "group_by_perceptual_hash", INFO,
              "Visually almost identical to another image that was kept (e.g. re-encoded, resized or lightly edited).",
              columns=("group_id",), parameters=("hash_name", "threshold"), prefix=True,
              aliases=("near duplicates", "near-perceptual-hash duplicates")),
    # Labeled datasets only (ImageClassificationDataset)
    ReasonTag("label_conflict", LABEL, "keep_one_from_group", WARNING,
              "The same or a near-identical image appears with different labels.",
              columns=("label", "split_group"),
              aliases=("Conflicting labels in group",)),
)

_BY_TAG = {r.tag: r for r in CATALOG if not r.prefix}
_BY_ALIAS = {alias: r for r in CATALOG for alias in r.aliases}
_PREFIXES = tuple(r for r in CATALOG if r.prefix)


def parse_tags(reason) -> list[str]:
    """Splits a `reason` cell into its tags; empty, None and NaN give []."""
    if not isinstance(reason, str) or not reason:
        return []
    return [t.strip() for t in reason.split(TAG_SEPARATOR) if t.strip()]


def get_reason(tag: str) -> ReasonTag | None:
    """The catalog entry for a written tag (exact tag, alias, or prefix match), or None if unknown."""
    if tag in _BY_TAG:
        return _BY_TAG[tag]
    if tag in _BY_ALIAS:
        return _BY_ALIAS[tag]
    for entry in _PREFIXES:
        if tag.startswith(entry.tag) and len(tag) > len(entry.tag):
            return entry
    return None


def is_known_tag(tag: str) -> bool:
    return get_reason(tag) is not None


def is_duplicate_tag(tag: str) -> bool:
    entry = get_reason(tag)
    return entry is not None and entry.category == DUPLICATE
