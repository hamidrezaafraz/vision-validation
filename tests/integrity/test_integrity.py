import pandas as pd
import pytest
from PIL import Image

from src.integrity import _inspect_single_image, check_decode
from src.utils.dataset import ImageDataset


@pytest.fixture
def make_sized_dataset(tmp_path):
    """Builds tmp_path/Data/Happy/<name>.png RGB images of the given sizes and returns an ImageDataset."""
    def _make(sizes: dict[str, tuple[int, int]]):
        class_dir = tmp_path / "Data" / "Happy"
        class_dir.mkdir(parents=True, exist_ok=True)
        for name, size in sizes.items():
            Image.new("RGB", size, (10, 20, 30)).save(class_dir / f"{name}.png")
        return ImageDataset(str(tmp_path / "Data"), str(tmp_path / "results"))
    return _make


def _row(df: pd.DataFrame, name: str) -> pd.Series:
    return df[df["path"].str.endswith(f"{name}.png")].iloc[0]


class TestInspectSingleImage:
    def test_reads_size_and_symmetric_ratio(self, tmp_path):
        path = tmp_path / "tall.png"
        Image.new("RGB", (100, 400)).save(path)
        result = _inspect_single_image(path)
        assert result.is_decodable
        assert (result.width, result.height) == (100, 400)
        assert result.aspect_ratio == 4.0
        assert result.megapixels == 0.04

    def test_reads_format_from_content_not_extension(self, tmp_path):
        path = tmp_path / "really_a_png.jpg"
        Image.new("RGB", (8, 8)).save(path, format="PNG")
        assert _inspect_single_image(path).format == "PNG"

    def test_undecodable_file_gets_placeholder_dimensions(self, tmp_path):
        path = tmp_path / "junk.png"
        path.write_bytes(b"not an image")
        result = _inspect_single_image(path)
        assert not result.is_decodable
        assert (result.width, result.height, result.megapixels) == (0, 0, 0.0)
        assert result.aspect_ratio == float("inf")
        assert result.format == "invalid"


class TestCheckDecode:
    def test_opens_each_image_once(self, make_sized_dataset, monkeypatch):
        dataset = make_sized_dataset({"a": (64, 64), "b": (64, 64)})
        calls = []
        real_open = Image.open
        monkeypatch.setattr(Image, "open", lambda *a, **k: calls.append(a[0]) or real_open(*a, **k))
        check_decode(dataset)
        assert len(calls) == 2

    def test_flags_each_dimension_violation(self, make_sized_dataset):
        dataset = make_sized_dataset({
            "ok": (64, 64),
            "small": (16, 64),
            "large": (200, 100),
            "wide": (300, 50),
        })
        df = check_decode(dataset, max_width=150, max_height=150, max_aspect_ratio=5.0)

        assert _row(df, "ok")["reason"] == ""
        assert _row(df, "small")["reason"] == "too_small"
        assert _row(df, "large")["reason"] == "too_large"
        assert _row(df, "wide")["reason"] == "too_large;extreme_aspect_ratio"
        assert set(df.loc[df["reason"] != "", "action"]) == {"review"}

    def test_drop_policy_for_dimensions(self, make_sized_dataset):
        dataset = make_sized_dataset({"ok": (64, 64), "small": (8, 8)})
        check_decode(dataset, on_dimension_violation="drop")
        assert _row(dataset._manifest, "small")["action"] == "drop"
        assert len(dataset.df) == 1

    def test_corrupt_file_is_dropped_but_not_tagged_too_small(self, make_sized_dataset, tmp_path):
        dataset = make_sized_dataset({"ok": (64, 64), "bad": (64, 64)})
        (tmp_path / "Data" / "Happy" / "bad.png").write_bytes(b"not an image")
        df = check_decode(dataset)
        assert _row(df, "bad")["action"] == "drop"
        assert _row(df, "bad")["reason"] == "corrupt"

    def test_writes_all_columns_to_saved_manifest(self, make_sized_dataset, tmp_path):
        dataset = make_sized_dataset({"ok": (64, 32)})
        check_decode(dataset)
        saved = pd.read_csv(tmp_path / "results" / "manifest.csv", dtype=str, keep_default_na=False)
        row = _row(saved, "ok")
        assert (row["is_decodable"], row["format"], row["color_mode"]) == ("True", "PNG", "RGB")
        assert (row["width"], row["height"], row["aspect_ratio"]) == ("64", "32", "2.0")

    @pytest.mark.parametrize("kwarg", ["on_mismatch", "on_dimension_violation"])
    def test_rejects_unknown_policy(self, make_sized_dataset, kwarg):
        dataset = make_sized_dataset({"ok": (64, 64)})
        with pytest.raises(ValueError, match=kwarg):
            check_decode(dataset, **{kwarg: "ignore"})

    def test_rejects_inverted_bounds(self, make_sized_dataset):
        dataset = make_sized_dataset({"ok": (64, 64)})
        with pytest.raises(ValueError, match="min dimensions"):
            check_decode(dataset, min_width=100, max_width=50)


class TestDecodeErrorColumn:
    def test_corrupt_file_records_decode_error(self, make_sized_dataset, tmp_path):
        dataset = make_sized_dataset({"ok": (64, 64), "bad": (64, 64)})
        (tmp_path / "Data" / "Happy" / "bad.png").write_bytes(b"not an image")
        df = check_decode(dataset)
        assert _row(df, "ok")["decode_error"] == ""
        assert _row(df, "bad")["decode_error"].startswith("UnidentifiedImageError")
