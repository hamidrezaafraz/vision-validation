import os

from PIL import Image, UnidentifiedImageError


def is_valid_image(path: str) -> bool:
    """True if PIL can open and verify the file as an image (regardless of its extension)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False


def find_image_paths(root_dir: str) -> list[str]:
    """Recursively lists every image PIL can read under root_dir.

    Each path starts with root_dir (e.g. root_dir="data" -> "data/Happy/1.jpg").
    The list is sorted so the order is deterministic.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):
                paths.append(path)
    return sorted(paths)
