import numpy as np
import imagehash
from src.near_duplicates.perceptual_hashes import get_hash_func, get_perceptual_hashes, hamming_pairs
import pytest

class TestGetHashFunc:
    def test_supported_names_match_imagehash(self):
        for hash_name in ['phash', 'average_hash', 'dhash', 'whash']:
            assert callable(get_hash_func(hash_name))

    def test_unsupported_name_raises(self):
        with pytest.raises(ValueError, match=f"Unknown hash name: unsupported"):
            get_hash_func("unsupported")


class TestGetPerceptualHashes:
    def test_output_type_and_length(self, example_duplicate_images):
        H = get_perceptual_hashes(example_duplicate_images)
        assert isinstance(H, np.ndarray)
        assert len(H) == len(example_duplicate_images)
        assert H.shape == (2, 64)
        assert H.dtype == bool

    def test_all_hash_names(self, example_duplicate_images):
        for hash_name in ['phash', 'average_hash', 'dhash', 'whash']:
            H = get_perceptual_hashes(example_duplicate_images, hash_name)
            assert isinstance(H, np.ndarray)
            assert len(H) == len(example_duplicate_images)

    def test_empty_list(self):
        H = get_perceptual_hashes([])
        assert isinstance(H, np.ndarray)
        assert len(H) == 0


class TestHammingPairs:
    def test_output_type_and_length(self):
        hashes = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)
        pairs = hamming_pairs(hashes, threshold=2)
        assert isinstance(pairs, np.ndarray)
        assert pairs.shape == (1, 2)
        assert pairs.tolist() == [[0, 1]]

    def test_pairs_are_ordered_and_unique(self):
        hashes = np.zeros((4, 8), dtype=bool)
        pairs = hamming_pairs(hashes, threshold=1)
        assert len(pairs) == 6  # 4 choose 2
        assert all(i < j for i, j in pairs)

    def test_chunking_matches_single_chunk(self):
        rng = np.random.default_rng(0)
        hashes = rng.integers(0, 2, size=(20, 16)).astype(bool)
        full = hamming_pairs(hashes, threshold=6, chunk_size=100)
        chunked = hamming_pairs(hashes, threshold=6, chunk_size=3)
        assert sorted(map(tuple, full)) == sorted(map(tuple, chunked))

    def test_no_pairs_below_threshold(self):
        hashes = np.array([[0, 0], [1, 1]], dtype=bool)
        pairs = hamming_pairs(hashes, threshold=1)
        assert isinstance(pairs, np.ndarray)
        assert len(pairs) == 0

    def test_empty_input(self):
        pairs = hamming_pairs(np.empty((0, 64), dtype=bool), threshold=5)
        assert isinstance(pairs, np.ndarray)
        assert len(pairs) == 0
