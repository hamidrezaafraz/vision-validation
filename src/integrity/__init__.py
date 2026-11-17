from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

from src.utils.dataset import ImageClassificationDataset, ImageDataset

# Force Pillow to raise OSError on truncated pixel streams instead of silently padding gray
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXIF_ORIENTATION_TAG = 0x0112

@dataclass(frozen=True)
class DecodeResult:
    path: str
    is_decodable: bool
    exif_orientation: int
    error: str
    
def _inspect_single_image(path: str | Path) -> DecodeResult:
    """Decodes the full pixel buffer and reads the EXIF orientation tag."""
    path_str = str(path)
    try:
        with Image.open(path_str) as img:
            exif = img.getexif()
            orientation = int(exif.get(EXIF_ORIENTATION_TAG, 1)) if exif else 1
            img.load()  # Forces full pixel decompression
        return DecodeResult(
            path=path_str,
            is_decodable=True,
            exif_orientation=orientation,
            error="",
        )
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        return DecodeResult(
            path=path_str,
            is_decodable=False,
            exif_orientation=-1,
            error=f"{type(exc).__name__}: {exc}",
        )
        
def check_decode_and_exif(
    dataset: ImageDataset | ImageClassificationDataset,
    path_col: str = "path",
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Runs full pixel decoding and EXIF orientation checks across dataset.df.
    Updates status ('keep' | 'drop' | 'review') and reason columns in-place.
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
    df.loc[review_target, "status"] = "review"
    
    dataset._manifest.loc[df.index] = df
    dataset.save_manifest()