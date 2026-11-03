"""Shared fixtures for all tests. Fixtures that only one module needs belong in that module."""
import os
import shutil

import pytest
from PIL import Image

EXAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "example_data")


@pytest.fixture
def example_duplicate_images():
    """Paths of the two byte-identical PNGs in src/example_data."""
    folder = os.path.join(EXAMPLE_DATA_DIR, "image", "classification")
    return [os.path.join(folder, "duplicate_1.png"), os.path.join(folder, "duplicate_2.png")]


@pytest.fixture
def make_file(tmp_path):
    """Factory that writes bytes to tmp_path/<name> and returns the path as a str."""
    def _make_file(name, content: bytes):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)
    return _make_file


@pytest.fixture
def make_image_folder(tmp_path):
    """
    Factory that builds an ImageFolder layout (root/<class>/<file>.png) and returns the root.
    example:

        root = make_image_folder({"Happy": [(255, 0, 0), (255, 0, 0)], "Sad": [(0, 0, 255)]})

    Each color becomes one 8x8 PNG, so equal colors in a class give byte-identical files.
    """
    def _make_image_folder(classes: dict[str, list[tuple[int, int, int]]]):
        root = tmp_path / "Data"
        for class_name, colors in classes.items():
            class_dir = root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i, color in enumerate(colors):
                Image.new("RGB", (8, 8), color).save(class_dir / f"{i:04d}.png")
        return str(root)
    return _make_image_folder


@pytest.fixture
def copy_example_images(tmp_path, example_duplicate_images):
    """Copies the example duplicates into tmp_path so tests can modify them freely."""
    dst = []
    for src in example_duplicate_images:
        dst.append(shutil.copy(src, tmp_path / os.path.basename(src)))
    return [str(p) for p in dst]
