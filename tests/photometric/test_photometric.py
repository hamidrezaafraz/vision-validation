import numpy as np
import pytest
from PIL import Image

from src.photometric import _compute_laplacian_variance, _inspect_single_photometric


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
