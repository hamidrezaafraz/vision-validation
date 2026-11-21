from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from src.utils.dataset import ImageClassificationDataset, ImageDataset, apply_policy, write_back
from src.utils.logger import Logger

logger = Logger(__name__)

# Make Pillow raise OSError on truncated pixel streams instead of silently padding with gray.
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXIF_ORIENTATION_TAG = 0x0112
DEFAULT_ORIENTATION = 1
MISMATCH_POLICIES = ("review", "drop")

FORMAT_COLUMNS = ["color_mode", "num_channels", "dtype", "bit_depth", "has_non_trivial_alpha"]
DIMENSION_COLUMNS = ["width", "height", "aspect_ratio", "megapixels"]

# Pillow raises different exception types depending on the format plugin.
DECODE_COLUMNS = ["is_decodable", "exif_orientation", "decode_error"]
DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, TypeError, Image.DecompressionBombError)




@dataclass(frozen=True)
class DecodeResult:
    path: str
    is_decodable: bool
    exif_orientation: int
    decode_error: str

    color_mode: str
    num_channels: int
    dtype: str
    bit_depth: int
    has_non_trivial_alpha: bool

    width: int
    height: int
    aspect_ratio: float
    megapixels: float

    @classmethod
    def failed(cls, path: str, exc: Exception) -> "DecodeResult":
        return cls(
            path=path,
            is_decodable=False,
            exif_orientation=-1,
            decode_error=f"{type(exc).__name__}: {exc}",
            color_mode="invalid",
            num_channels=0,
            dtype="invalid",
            bit_depth=0,
            has_non_trivial_alpha=False,
            width=0,
            height=0,
            aspect_ratio=float("inf"),
            megapixels=0.0,
        )


def _has_non_trivial_alpha(arr: np.ndarray, bands: tuple[str, ...]) -> bool:
    """True when the image has an alpha band with at least one non-opaque pixel."""
    if "A" not in bands or arr.ndim != 3:
        return False
    fully_opaque = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
    return bool(arr[..., bands.index("A")].min() < fully_opaque)


def _dimensions(w: int, h: int) -> dict:
    """Symmetric aspect ratio max(w/h, h/w) and megapixels; a zero side gives inf / 0."""
    if w <= 0 or h <= 0:
        return {"width": w, "height": h, "aspect_ratio": float("inf"), "megapixels": 0.0}
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(max(w / h, h / w), 4),
        "megapixels": round(w * h / 1_000_000, 4),
    }


def _inspect_single_image(path: str | Path) -> DecodeResult:
    """Opens the image once: decodes the full pixel buffer and reads EXIF orientation, pixel format and size."""
    path = str(path)
    try:
        with Image.open(path) as img:
            raw_orientation = img.getexif().get(EXIF_ORIENTATION_TAG)
            orientation = int(raw_orientation) if raw_orientation is not None else DEFAULT_ORIENTATION
            img.load()  # Image.open is lazy; this forces full decompression.

            bands = img.getbands()
            arr = np.asarray(img)
            return DecodeResult(
                path=path,
                is_decodable=True,
                exif_orientation=orientation,
                decode_error="",
                color_mode=img.mode,
                num_channels=len(bands),
                dtype=str(arr.dtype),
                bit_depth=1 if img.mode == "1" else arr.dtype.itemsize * 8,
                has_non_trivial_alpha=_has_non_trivial_alpha(arr, bands),
                **_dimensions(*img.size),
            )
    except DECODE_ERRORS as exc:
        return DecodeResult.failed(path, exc)


def check_decode(
    dataset: ImageDataset | ImageClassificationDataset,
    path_col: str = "path",
    expected_modes: tuple[str, ...] = ("RGB",),
    expected_dtypes: tuple[str, ...] = ("uint8",),
    on_mismatch: str = "review",
    min_width: int = 32,
    min_height: int = 32,
    max_width: int = 8192,
    max_height: int = 8192,
    max_aspect_ratio: float = 5.0,
    on_dimension_violation: str = "review",
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Opens every non-dropped image once, fully decodes it, and records the results in the manifest.

    - Undecodable files are dropped with reason 'corrupt'.
    - Files with a non-default EXIF orientation are marked 'review' with reason 'exif_rotated'.
    - Files whose color mode or dtype is not in the expected lists get the
      'on_mismatch' action ('review' or 'drop').
    - Files outside the size bounds get the 'on_dimension_violation' action, with reason
      'too_small' (below min_width/min_height), 'too_large' (above max_width/max_height)
      or 'extreme_aspect_ratio' (max(w/h, h/w) above max_aspect_ratio).

    Existing 'drop' actions are never downgraded and existing reasons are appended to,
    not overwritten. The manifest is saved, and the checked rows are returned, including any
    this call dropped.
    """
    for name, policy in (("on_mismatch", on_mismatch), ("on_dimension_violation", on_dimension_violation)):
        if policy not in MISMATCH_POLICIES:
            raise ValueError(f"{name} must be one of {MISMATCH_POLICIES}, got {policy!r}")
    if min_width > max_width or min_height > max_height:
        raise ValueError(
            f"min dimensions must not exceed max dimensions, got width [{min_width}, {max_width}], "
            f"height [{min_height}, {max_height}]"
        )

    df = dataset.df
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(_inspect_single_image, df[path_col]))
    inspected = pd.DataFrame([asdict(r) for r in results], index=df.index)

    new_columns = DECODE_COLUMNS + FORMAT_COLUMNS + DIMENSION_COLUMNS
    df[new_columns] = inspected[new_columns]

    decodable = inspected["is_decodable"]
    apply_policy(df, ~decodable, "drop", "corrupt")
    apply_policy(df, decodable & (inspected["exif_orientation"] != DEFAULT_ORIENTATION), "review", "exif_rotated")

    # Undecodable rows are already dropped, so only check the decodable ones.
    apply_policy(df, decodable & ~inspected["color_mode"].isin(expected_modes), on_mismatch, "unexpected_color_mode")
    apply_policy(df, decodable & ~inspected["dtype"].isin(expected_dtypes), on_mismatch, "unexpected_bit_depth")

    w, h = inspected["width"], inspected["height"]
    too_small = decodable & ((w < min_width) | (h < min_height))
    too_large = decodable & ((w > max_width) | (h > max_height))
    extreme_ratio = decodable & (inspected["aspect_ratio"] > max_aspect_ratio)
    apply_policy(df, too_small, on_dimension_violation, "too_small")
    apply_policy(df, too_large, on_dimension_violation, "too_large")
    apply_policy(df, extreme_ratio, on_dimension_violation, "extreme_aspect_ratio")

    write_back(dataset, df, new_columns)
    return df



