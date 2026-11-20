from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from src.utils.dataset import ImageClassificationDataset, ImageDataset, apply_policy
from src.utils.logger import Logger

logger = Logger(__name__)

# Make Pillow raise OSError on truncated pixel streams instead of silently padding with gray.
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXIF_ORIENTATION_TAG = 0x0112
DEFAULT_ORIENTATION = 1
MISMATCH_POLICIES = ("review", "drop")

DECODE_COLUMNS = ["is_decodable", "exif_orientation"]
FORMAT_COLUMNS = ["color_mode", "num_channels", "dtype", "bit_depth", "has_non_trivial_alpha"]

# Pillow raises different exception types depending on the format plugin.
DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError)


@dataclass(frozen=True)
class DecodeResult:
    path: str
    is_decodable: bool
    exif_orientation: int
    error: str

    color_mode: str
    num_channels: int
    dtype: str
    bit_depth: int
    has_non_trivial_alpha: bool

    @classmethod
    def failed(cls, path: str, exc: Exception) -> "DecodeResult":
        return cls(
            path=path,
            is_decodable=False,
            exif_orientation=-1,
            error=f"{type(exc).__name__}: {exc}",
            color_mode="invalid",
            num_channels=0,
            dtype="invalid",
            bit_depth=0,
            has_non_trivial_alpha=False,
        )


def _has_non_trivial_alpha(arr: np.ndarray, bands: tuple[str, ...]) -> bool:
    """True when the image has an alpha band with at least one non-opaque pixel."""
    if "A" not in bands or arr.ndim != 3:
        return False
    fully_opaque = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
    return bool(arr[..., bands.index("A")].min() < fully_opaque)


def _inspect_single_image(path: str | Path) -> DecodeResult:
    """Decodes the full pixel buffer and reads the EXIF orientation and pixel format."""
    path = str(path)
    try:
        with Image.open(path) as img:
            orientation = int(img.getexif().get(EXIF_ORIENTATION_TAG, DEFAULT_ORIENTATION))
            img.load()  # Image.open is lazy; this forces full decompression.

            bands = img.getbands()
            arr = np.asarray(img)
            return DecodeResult(
                path=path,
                is_decodable=True,
                exif_orientation=orientation,
                error="",
                color_mode=img.mode,
                num_channels=len(bands),
                dtype=str(arr.dtype),
                bit_depth=1 if img.mode == "1" else arr.dtype.itemsize * 8,
                has_non_trivial_alpha=_has_non_trivial_alpha(arr, bands),
            )
    except DECODE_ERRORS as exc:
        return DecodeResult.failed(path, exc)


def check_decode(
    dataset: ImageDataset | ImageClassificationDataset,
    path_col: str = "path",
    check_format: bool = True,
    expected_modes: tuple[str, ...] = ("RGB",),
    expected_dtypes: tuple[str, ...] = ("uint8",),
    on_mismatch: str = "review",
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Fully decodes every non-dropped image and records the results in the manifest.

    - Undecodable files are dropped with reason 'corrupt'.
    - Files with a non-default EXIF orientation are marked 'review' with reason 'exif_rotated'.
    - If check_format is set, files whose color mode or dtype is not in the expected
      lists get the 'on_mismatch' action ('review' or 'drop').

    Existing 'drop' actions are never downgraded and existing reasons are appended to,
    not overwritten. The manifest is saved, and the updated non-dropped rows are returned.
    """
    if on_mismatch not in MISMATCH_POLICIES:
        raise ValueError(f"on_mismatch must be one of {MISMATCH_POLICIES}, got {on_mismatch!r}")

    df = dataset.df
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(_inspect_single_image, df[path_col]))
    inspected = pd.DataFrame([asdict(r) for r in results], index=df.index)

    new_columns = DECODE_COLUMNS + (FORMAT_COLUMNS if check_format else [])
    df[new_columns] = inspected[new_columns]

    decodable = inspected["is_decodable"]
    apply_policy(df, ~decodable, "drop", "corrupt")
    apply_policy(df, decodable & (inspected["exif_orientation"] != DEFAULT_ORIENTATION), "review", "exif_rotated")

    if check_format:
        # Undecodable rows are already dropped, so only check the decodable ones.
        apply_policy(df, decodable & ~inspected["color_mode"].isin(expected_modes), on_mismatch, "unexpected_color_mode")
        apply_policy(df, decodable & ~inspected["dtype"].isin(expected_dtypes), on_mismatch, "unexpected_bit_depth")

    for row in inspected.loc[~decodable].itertuples():
        logger.warning(f"Cannot decode {row.path}: {row.error}")
    logger.info(
        f"Decoded {len(df)} images: {(~decodable).sum()} corrupt, "
        f"{(df['action'] == 'review').sum()} marked for review"
    )

    _write_back(dataset, df, new_columns)
    return df


def _write_back(dataset: ImageDataset | ImageClassificationDataset, df: pd.DataFrame, new_columns: list[str]) -> None:
    """Copies the checked rows back into the full manifest and saves it.

    New columns are created first: assigning through .loc aligns on column names,
    so columns missing from the manifest would otherwise be silently discarded.
    """
    manifest = dataset._manifest
    for col in new_columns:
        if col not in manifest.columns:
            manifest[col] = ""
    columns = ["action", "reason", *new_columns]
    manifest.loc[df.index, columns] = df[columns]
    dataset.save_manifest()
