from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from src.utils.audit import audited
from src.utils.dataset import ImageClassificationDataset, ImageDataset, apply_policy, write_back
from src.utils.logger import Logger

logger = Logger(__name__)

# Make Pillow raise OSError on truncated files instead of silently padding the missing part
# with gray, which would distort brightness and blur stats without anyone noticing.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Allowed values for on_violation. 'review' only marks a row for a human to look at; 'drop'
# removes it from the dataset. There is no 'keep', because a check that flags nothing need not run.
MISMATCH_POLICIES = ("review", "drop")
# Pillow raises different exception types depending on the format plugin, so all of them mean
# "this file could not be read" here. Such rows get NaN stats and are never flagged by this check;
# reporting corrupt files is check_decode's job.
DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, TypeError, Image.DecompressionBombError)
# Manifest columns written by check_photometric, one per PhotometricResult field.
PHOTOMETRIC_COLUMNS = [
    "mean_brightness",
    "pixel_std",
    "dark_clip_ratio",
    "bright_clip_ratio",
    "blur_score",
]


# Longest side, in pixels, that the luminance plane is shrunk to before measuring blur.
# Why: Laplacian variance depends on resolution. In a 24 MP photo a sharp edge spans several
# pixels, so neighbouring pixels differ little and the score is low even when the photo is in
# focus. Shrinking large images to a common size puts their scores on one scale, so a single
# min_blur_score works for all of them, and judges sharpness at roughly the scale a model sees.
# Why 256: close to common training input sizes (224 for ImageNet-style models), so blur that
# disappears at this size would not matter in training anyway.
# Caveat: images smaller than this are NOT enlarged, so the 48x48 crops (most of emotion/Data)
# are measured at their own size and are not on exactly the same scale as shrunk photos.
# Changing this value changes every blur score, so re-pick min_blur_score afterwards.
BLUR_REF_MAX_SIDE = 256
# Longest side, in pixels, above which an image is shrunk while decoding (see _load_downscaled).
# After shrinking, the longest side is between this and about 2x this.
# Why: brightness, contrast and clipping stats were computed on full-resolution float arrays.
# For a 24 MP photo that costs ~1 GB of memory and ~0.5 s, and with 8 worker threads on an
# 8 GB machine the whole check took 443 s instead of 18 s.
# Why 1024: on the 247 photos over 10 MP in emotion/Data, shrinking to this size moved
# mean_brightness by at most 0.0007, pixel_std by 0.013 and the clip ratios by 0.04, and no image
# changed flags. It must stay well above BLUR_REF_MAX_SIDE, so the blur plane is still made
# by shrinking from a sharper image rather than from one already near its final size.
STATS_MAX_SIDE = 1024
# 4-neighbour discrete Laplacian: 4 x (average of the up/down/left/right neighbours - the pixel). Flat and smoothly changing areas give ~0 and sharp edges give large values,
# so the variance of the result is high for sharp images and low for blurry ones.
# Why this kernel: it is the standard one for the "variance of the Laplacian" blur measure, and
# it is the smallest, so it reacts to the finest detail. It is symmetric, so filter2D's
# correlation equals convolution.
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


def _load_downscaled(img: Image.Image, max_side: int = STATS_MAX_SIDE) -> Image.Image:
    """Loads the image shrunk so its longest side is between max_side and 2 * max_side.

    Smaller images are loaded unchanged. For JPEGs, draft() makes the decoder itself output a
    1/2, 1/4 or 1/8 scale image, which also skips most of the decoding work. For other formats,
    or if the draft is still too large, reduce() averages integer-sized pixel blocks.
    Must be called before the image is loaded.
    """
    if max(img.size) <= max_side:
        img.load()
        return img

    img.draft(img.mode, (max_side, max_side))  # no-op for non-JPEG formats
    img.load()

    factor = max(img.size) // max_side
    if factor > 1:
        try:
            img = img.reduce(factor)
        except ValueError:  # modes like "1" and "P" can't be block-averaged; keep full size
            pass
    return img


def _luminance(norm: np.ndarray) -> np.ndarray:
    """The (H, W) luminance plane in [0, 1] (ITU-R BT.601 weights); grayscale arrays are returned as they are."""
    if norm.ndim == 2:
        return norm
    return 0.299 * norm[..., 0] + 0.587 * norm[..., 1] + 0.114 * norm[..., 2]


def _max_channel_std(norm: np.ndarray) -> float:
    """
    The largest per-channel standard deviation over the pixels. Computing one std over every value of
    an (H, W, 3) array would also count the differences *between* channels: a solid pure red frame,
    (1, 0, 0) everywhere, would get ~0.47 although no pixel differs from another. Per channel it is 0.
    """
    if norm.ndim == 2:
        return float(norm.std())
    return float(norm.reshape(-1, norm.shape[-1]).std(axis=0).max())


def _compute_laplacian_variance(norm: np.ndarray) -> float:
    """Computes discrete 2D Laplacian variance on a luminance plane shrunk to <= BLUR_REF_MAX_SIDE px."""
    gray_01 = _luminance(norm)

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
            norm = _extract_normalized_rgb_or_gray(_load_downscaled(img))
    except DECODE_ERRORS:
        return PhotometricResult.invalid()

    if norm.size == 0:
        return PhotometricResult.invalid()

    # Brightness and clipping are per pixel, on luminance: a pixel is clipped when it is (near) pure
    # black or white, not when one of its channels is 0 or 1 (pure red has two channels at 0 but is not
    # dark). Contrast is the largest per-channel std, so a solid color frame of any hue scores ~0.
    # For grayscale images all three equal the old whole-array values.
    luminance = _luminance(norm)
    mean_brightness = float(luminance.mean())
    pixel_std = _max_channel_std(norm)
    dark_clip_ratio = float((luminance <= clip_margin).mean())
    bright_clip_ratio = float((luminance >= (1.0 - clip_margin)).mean())
    blur_score = _compute_laplacian_variance(norm)

    return PhotometricResult(
        mean_brightness=round(mean_brightness, 4),
        pixel_std=round(pixel_std, 4),
        dark_clip_ratio=round(dark_clip_ratio, 4),
        bright_clip_ratio=round(bright_clip_ratio, 4),
        blur_score=round(blur_score, 2),
    )


@audited("check_photometric")
def check_photometric(
    dataset: ImageDataset | ImageClassificationDataset,
    # Manifest column that holds the image file paths.
    path_col: str = "path",
    # Pixel std (values in [0, 1]) below which an image counts as blank or a solid color.
    # 0.02 is about 5 gray levels out of 255: only near-uniform frames like a black screen or
    # a placeholder fall below it, while even a low-contrast face is far above it.
    # In emotion/Data the 1st percentile is 0.10, and 24 images fall below 0.02.
    min_std: float = 0.02,
    # Fraction of pixels allowed to be clipped (within clip_margin of pure black or pure white)
    # before an image counts as under- or overexposed. 0.90 means almost the whole frame has
    # lost its detail. Ordinary dark or bright photos, or a face on a white background, stay below it.
    # In emotion/Data the 99th percentile is ~0.25 for both, and 4 images are flagged underexposed.
    max_clip_ratio: float = 0.90,
    # How close to 0 or 1 a value must be to count as clipped. 0.02 means values <= 5 or >= 250
    # out of 255: a small margin so JPEG noise around pure black/white still counts as clipped.
    clip_margin: float = 0.02,
    # Laplacian variance (on 0-255 values, at BLUR_REF_MAX_SIDE) below which an image counts as blurry.
    # 40 is a rough rule of thumb, not calibrated on this dataset. In emotion/Data the median is
    # ~1950 and the 1st percentile ~165, so 40 only catches extreme cases (19 images).
    # The score depends on BLUR_REF_MAX_SIDE, so re-pick this whenever that changes.
    min_blur_score: float = 40.0,
    # 'review' by default: a low-quality image is not necessarily a wrong one, so flagged images
    # are kept for a human to look at rather than removed automatically.
    on_violation: str = "review",
    # Threads that read images in parallel. Decoding releases the GIL, so threads help, and 8
    # matches the core count of the machine this was tuned on.
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

