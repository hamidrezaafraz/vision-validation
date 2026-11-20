from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
import numpy as np
from src.utils.dataset import ImageClassificationDataset, ImageDataset, apply_policy

# Force Pillow to raise OSError on truncated pixel streams instead of silently padding gray
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXIF_ORIENTATION_TAG = 0x0112

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
    
def _inspect_single_image(path: str | Path) -> DecodeResult:
    """Decodes the full pixel buffer and reads the EXIF orientation tag."""
    path_str = str(path)
    try:
        with Image.open(path_str) as img:
            exif = img.getexif()
            orientation = int(exif.get(EXIF_ORIENTATION_TAG, 1)) if exif else 1
            img.load()  # Forces full pixel decompression
            
            mode = img.mode
            num_channels = len(img.getbands())
            arr = np.asarray(img)
            dtype_str = str(arr.dtype)
            bit_depth = 1 if mode == "1" else arr.dtype.itemsize * 8
            
            has_non_trivial_alpha = False
            if "A" in img.getbands() and arr.ndim == 3:
                alpha_idx = img.getbands().index("A")
                max_alpha = 65535 if arr.dtype == np.uint16 else 255
                has_non_trivial_alpha = bool(arr[..., alpha_idx].min() < max_alpha)
            
        return DecodeResult(
            path=path_str,
            is_decodable=True,
            exif_orientation=orientation,
            error="",
            color_mode=mode,
            num_channels=num_channels,
            dtype=dtype_str,
            bit_depth=bit_depth,
            has_non_trivial_alpha=has_non_trivial_alpha,
        )
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        return DecodeResult(
            path=path_str,
            is_decodable=False,
            exif_orientation=-1,
            error=f"{type(exc).__name__}: {exc}",
            color_mode="invalid",
            num_channels=0,
            dtype="invalid",
            bit_depth=0,
            has_non_trivial_alpha=False
        )
        
def check_decode(
    dataset: ImageDataset | ImageClassificationDataset,
    path_col: str = "path",
    check_format: bool = True,
    expected_modes: tuple[str] = ("RGB",),
    expected_dtypes: tuple[str] = ("uint8",),
    on_mismatch: str = "review",
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Runs full pixel decoding and EXIF orientation checks across dataset.df.
    Updates action ('keep' | 'drop' | 'review') and reason columns in-place.
    """
    df = dataset.df
    
    paths = df[path_col].tolist()
    
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(_inspect_single_image, paths))
        
    res_df = pd.DataFrame(results)
    
    df["is_decodable"] = res_df["is_decodable"].values
    df["exif_orientation"] = res_df["exif_orientation"].values
    
    corrupt_mask = ~df["is_decodable"]
    df.loc[corrupt_mask, "action"] = "drop"
    df.loc[corrupt_mask, "reason"] = "corrupt"
    
    rotated_mask = df["is_decodable"] & (df["exif_orientation"] != 1)
    review_target = rotated_mask & (df["action"] == "keep")
    df.loc[review_target, "action"] = "review"
    
    if check_format:
        df["color_mode"] = res_df["color_mode"].values
        df["num_channels"] = res_df["num_channels"].values
        df["dtype"] = res_df["dtype"].values
        df["bit_depth"] = res_df["bit_depth"].values
        df["has_non_trivial_alpha"] = res_df["has_non_trivial_alpha"].values

        valid_mask = df["color_mode"] != "invalid"
        
        # 1. Color mode mismatch (e.g., L, RGBA, CMYK, or P when expecting RGB)
        bad_mode = valid_mask & ~df["color_mode"].isin(expected_modes)
        apply_policy(df, bad_mode, on_mismatch, "unexpected_color_mode")

        # 2. Bit-depth / dtype mismatch (e.g., uint16 or float32 when expecting uint8)
        bad_dtype = valid_mask & ~df["dtype"].isin(expected_dtypes)
        apply_policy(df, bad_dtype, on_mismatch, "unexpected_bit_depth")

    
    dataset._manifest.loc[df.index] = df
    dataset.save_manifest()
    