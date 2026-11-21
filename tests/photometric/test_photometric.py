import numpy as np
import pytest
from PIL import Image

from src.photometric import STATS_MAX_SIDE, _compute_laplacian_variance, _inspect_single_photometric, _load_downscaled


class TestInspectSinglePhotometric:
    def test_la_image_does_not_crash(self, tmp_path):
        rng = np.random.default_rng(0)
        luminance = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8))
        path = tmp_path / "la.png"
        Image.merge("LA", [luminance, Image.new("L", (64, 64), 255)]).save(path)

        result = _inspect_single_photometric(path)

        assert result.pixel_std > 0.2
        assert result.blur_score > 0

    def test_la_matches_its_luminance_band(self, tmp_path):
        rng = np.random.default_rng(1)
        luminance = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8))
        la_path, l_path = tmp_path / "la.png", tmp_path / "l.png"
        Image.merge("LA", [luminance, Image.new("L", (64, 64), 128)]).save(la_path)
        luminance.save(l_path)

        assert _inspect_single_photometric(la_path) == _inspect_single_photometric(l_path)


class TestComputeLaplacianVariance:
    def test_matches_reference_stencil(self):
        rng = np.random.default_rng(2)
        norm = rng.random((40, 30), dtype=np.float32)
        gray = norm * 255.0
        ref = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
        ).var()
        assert _compute_laplacian_variance(norm) == pytest.approx(float(ref), rel=1e-4)

    def test_flat_image_scores_zero(self):
        assert _compute_laplacian_variance(np.full((32, 32), 0.5, dtype=np.float32)) == 0.0

    def test_too_small_scores_zero(self):
        assert _compute_laplacian_variance(np.zeros((2, 10), dtype=np.float32)) == 0.0


class TestLoadDownscaled:
    @pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
    def test_large_image_is_shrunk(self, tmp_path, fmt):
        path = tmp_path / f"big.{fmt.lower()}"
        Image.new("RGB", (4000, 3000), (120, 80, 40)).save(path, fmt)
        with Image.open(path) as img:
            out = _load_downscaled(img)
            assert STATS_MAX_SIDE <= max(out.size) < 2 * STATS_MAX_SIDE

    def test_small_image_is_unchanged(self, tmp_path):
        path = tmp_path / "small.png"
        Image.new("RGB", (48, 48)).save(path)
        with Image.open(path) as img:
            assert _load_downscaled(img).size == (48, 48)

    def test_mode_that_cannot_be_reduced_keeps_full_size(self, tmp_path):
        path = tmp_path / "bilevel.png"
        Image.new("1", (3000, 100), 1).save(path)
        with Image.open(path) as img:
            assert _load_downscaled(img).size == (3000, 100)

    def test_stats_close_to_full_resolution(self, tmp_path):
        rng = np.random.default_rng(3)
        # Smooth gradient plus mild noise, so block averaging shouldn't move the global stats much.
        y, x = np.mgrid[0:3000, 0:4000]
        arr = (x / 4000 * 200 + rng.normal(0, 10, x.shape)).clip(0, 255).astype(np.uint8)
        path = tmp_path / "gradient.png"
        Image.fromarray(arr).save(path)

        result = _inspect_single_photometric(path)
        full = arr.astype(np.float32) / 255
        assert result.mean_brightness == pytest.approx(full.mean(), abs=0.005)
        assert result.pixel_std == pytest.approx(full.std(), rel=0.05)
