import hashlib

import pytest

from src.exact_duplicates.hashing import get_hash_function, get_hash_hexdigest, get_paths_by_hash


class TestGetHashFunction:
    @pytest.mark.parametrize("hash_name", ["blake2b", "sha256", "md5", "sha1", "sha384"])
    def test_supported_names_match_hashlib(self, hash_name):
        assert get_hash_function(hash_name)().name == hashlib.new(hash_name).name

    def test_unsupported_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            get_hash_function("crc32")


class TestGetHashHexdigest:
    def test_matches_hashlib(self, make_file):
        path = make_file("a.bin", b"hello world")
        assert get_hash_hexdigest(path) == hashlib.blake2b(b"hello world").hexdigest()

    def test_chunk_size_does_not_change_result(self, make_file):
        path = make_file("a.bin", b"x" * 1000)
        assert get_hash_hexdigest(path, chunk_size=7) == get_hash_hexdigest(path, chunk_size=65536)

    def test_hash_name_is_used(self, make_file):
        path = make_file("a.bin", b"abc")
        assert get_hash_hexdigest(path, hash_name="md5") == hashlib.md5(b"abc").hexdigest()

    def test_empty_file(self, make_file):
        path = make_file("empty.bin", b"")
        assert get_hash_hexdigest(path) == hashlib.blake2b(b"").hexdigest()

    def test_identical_example_images_have_same_hash(self, example_duplicate_images):
        a, b = example_duplicate_images
        assert get_hash_hexdigest(a) == get_hash_hexdigest(b)


class TestGetPathsByHash:
    def test_groups_identical_files(self, make_file):
        paths = [
            (0, make_file("a.bin", b"same")),
            (1, make_file("b.bin", b"different")),
            (2, make_file("c.bin", b"same")),
        ]
        groups = sorted(get_paths_by_hash(paths).values())
        assert groups == [[0, 2], [1]]

    def test_keys_are_hexdigests(self, make_file):
        paths = [(5, make_file("a.bin", b"same"))]
        assert dict(get_paths_by_hash(paths)) == {hashlib.blake2b(b"same").hexdigest(): [5]}

    def test_accepts_any_iterable_of_pairs(self, make_file):
        indices = [10, 20]
        files = [make_file("a.bin", b"x"), make_file("b.bin", b"x")]
        groups = list(get_paths_by_hash(zip(indices, files)).values())
        assert groups == [[10, 20]]

    def test_empty_input(self):
        assert dict(get_paths_by_hash([])) == {}
