from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from src.utils.dataset import ImageClassificationDataset, ImageDataset, apply_policy, write_back
from src.utils.logger import Logger

logger = Logger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = False

MISMATCH_POLICIES = ("review", "drop")
DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, TypeError, Image.DecompressionBombError)
PHOTOMETRIC_COLUMNS = [
    "mean_brightness",
    "pixel_std",
    "dark_clip_ratio",
    "bright_clip_ratio",
    "blur_score",
]


BLUR_REF_MAX_SIDE = 256
# 4-neighbour discrete Laplacian. It is symmetric, so filter2D's correlation equals convolution.
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


@dataclass(frozen=True)
class PhotometricResult:
    mean_brightness: float
    pixel_std: float
    dark_clip_ratio: float
    bright_clip_ratio: float
    blur_score: float

    @classmethod
    def invalid(cls) -> "PhotometricResult":
        return cls(
            mean_brightness=float("nan"),
            pixel_std=float("nan"),
            dark_clip_ratio=float("nan"),
            bright_clip_ratio=float("nan"),
            blur_score=float("nan"),
        )


def _extract_normalized_rgb_or_gray(img: Image.Image) -> np.ndarray:
    """Returns a float32 array in [0.0, 1.0] with alpha stripped and CMYK converted to RGB."""
    if img.mode == "1":
        return np.asarray(img, dtype=np.float32)

    if img.mode == "RGBA":
        # Drop alpha channel so transparent regions don't bias brightness/contrast
        arr = np.asarray(img)[..., :-1]
    elif img.mode == "LA":
        # Index (not slice) the luminance band so the result is 2D (H, W), not (H, W, 1).
        arr = np.asarray(img)[..., 0]
    elif img.mode in ("P", "PA", "CMYK", "YCbCr", "HSV"):
        arr = np.asarray(img.convert("RGB"))
    else:
        arr = np.asarray(img)

    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        return arr.astype(np.float32) / max_val
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _compute_laplacian_variance(norm: np.ndarray) -> float:
    """Computes discrete 2D Laplacian variance on a <=512px luminance plane."""
    if norm.ndim == 2:
        gray_01 = norm
    else:
        gray_01 = 0.299 * norm[..., 0] + 0.587 * norm[..., 1] + 0.114 * norm[..., 2]

    h, w = gray_01.shape
    longest = max(h, w)
    if longest > BLUR_REF_MAX_SIDE:
        scale = BLUR_REF_MAX_SIDE / longest
        new_size = (max(3, int(round(w * scale))), max(3, int(round(h * scale))))
        gray_pil = Image.fromarray((gray_01 * 255.0).astype(np.uint8), mode="L")
        gray = np.asarray(gray_pil.resize(new_size, Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        gray = gray_01 * 255.0

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    # Border pixels are cropped so only full neighbourhoods count, rather than ones padded
    # by OpenCV's border reflection.
    lap = cv2.filter2D(np.ascontiguousarray(gray, dtype=np.float32), cv2.CV_32F, LAPLACIAN_KERNEL)
    return float(lap[1:-1, 1:-1].var())


def _inspect_single_photometric(
    path: str | Path,
    is_decodable: bool = True,
    clip_margin: float = 0.02,
) -> PhotometricResult:
    """Computes brightness, contrast std, shadow/highlight clipping ratios, and blur score."""
    if not is_decodable:
        return PhotometricResult.invalid()

    try:
        with Image.open(str(path)) as img:
            img.load()
            norm = _extract_normalized_rgb_or_gray(img)
    except DECODE_ERRORS:
        return PhotometricResult.invalid()

    if norm.size == 0:
        return PhotometricResult.invalid()

    mean_brightness = float(norm.mean())
    pixel_std = float(norm.std())
    dark_clip_ratio = float((norm <= clip_margin).mean())
    bright_clip_ratio = float((norm >= (1.0 - clip_margin)).mean())
    blur_score = _compute_laplacian_variance(norm)

    return PhotometricResult(
        mean_brightness=round(mean_brightness, 4),
        pixel_std=round(pixel_std, 4),
        dark_clip_ratio=round(dark_clip_ratio, 4),
        bright_clip_ratio=round(bright_clip_ratio, 4),
        blur_score=round(blur_score, 2),
    )


def check_photometric(
    dataset: ImageDataset | ImageClassificationDataset,
    path_col: str = "path",
    min_std: float = 0.02,
    max_clip_ratio: float = 0.90,
    clip_margin: float = 0.02,
    min_blur_score: float = 40.0,
    on_violation: str = "review",
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Evaluates photometric image quality across non-dropped images.

    - Blank / solid-color frames (pixel_std < min_std) get tag 'blank_or_solid'.
    - Severely underexposed frames (dark_clip_ratio > max_clip_ratio) get tag 'underexposed'.
    - Severely overexposed frames (bright_clip_ratio > max_clip_ratio) get tag 'overexposed'.
    - Out-of-focus / blurry frames (blur_score < min_blur_score) get tag 'blurry'.
    """
    if on_violation not in MISMATCH_POLICIES:
        raise ValueError(f"on_violation must be one of {MISMATCH_POLICIES}, got {on_violation!r}")
    if not (0.0 <= min_std <= 1.0):
        raise ValueError(f"min_std must be in [0.0, 1.0], got {min_std}")
    if not (0.0 < max_clip_ratio <= 1.0):
        raise ValueError(f"max_clip_ratio must be in (0.0, 1.0], got {max_clip_ratio}")

    df = dataset.df.copy()
    decodable_flags = (
        df["is_decodable"].fillna(True).astype(bool).tolist()
        if "is_decodable" in df.columns
        else [True] * len(df)
    )
    paths = df[path_col].tolist()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(
            pool.map(
                lambda args: _inspect_single_photometric(args[0], args[1], clip_margin),
                zip(paths, decodable_flags),
            )
        )

    inspected = pd.DataFrame([asdict(r) for r in results], index=df.index)
    df[PHOTOMETRIC_COLUMNS] = inspected[PHOTOMETRIC_COLUMNS]

    valid = inspected["pixel_std"].notna()
    is_blank = valid & (inspected["pixel_std"] < min_std)
    is_underexposed = valid & ~is_blank & (inspected["dark_clip_ratio"] > max_clip_ratio)
    is_overexposed = valid & ~is_blank & (inspected["bright_clip_ratio"] > max_clip_ratio)
    is_blurry = valid & ~is_blank & (inspected["blur_score"] < min_blur_score)

    apply_policy(df, is_blank, on_violation, "blank_or_solid")
    apply_policy(df, is_underexposed, on_violation, "underexposed")
    apply_policy(df, is_overexposed, on_violation, "overexposed")
    apply_policy(df, is_blurry, on_violation, "blurry")

    write_back(dataset, df, PHOTOMETRIC_COLUMNS)

    return df

