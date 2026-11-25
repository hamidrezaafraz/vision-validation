import numpy as np
import pytest

import src.cli.lib as lib
from src.cli.lib import cached_embeddings, group_by_perceptual_hash, make_and_save_embeddings, make_perceptual_hashes
from src.utils.dataset import ImageClassificationDataset, ImageDataset


@pytest.fixture
def dataset(make_image_folder, tmp_path):
    root = make_image_folder({"Happy": [(i * 40, 0, 0) for i in range(4)]})
    return ImageDataset(root, str(tmp_path / "results"))


@pytest.fixture
def fake_model(monkeypatch):
    """make_embeddings without a model: row for manifest index i is [i, i], and every call is counted."""
    calls = []

    def fake(dataset, model_name="alexnet", batch_size=128):
        calls.append(list(dataset.df.index))
        return np.array([[i, i] for i in dataset.df.index], dtype=np.float32)
    monkeypatch.setattr(lib, "make_embeddings", fake)
    return calls


class TestMakeAndSaveEmbeddings:
    def test_recomputes_even_when_the_cache_covers_the_rows(self, dataset, fake_model):
        cached_embeddings(dataset, "m")
        emb = make_and_save_embeddings(dataset, "m")
        assert emb.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]
        assert len(fake_model) == 2

    def test_saves_the_rows_not_dropped(self, dataset, fake_model):
        cached_embeddings(dataset, "m")  # cache for rows 0-3
        dataset._manifest.loc[[0, 2], "action"] = "drop"
        assert make_and_save_embeddings(dataset, "m").tolist() == [[1, 1], [3, 3]]
        assert fake_model[-1] == [1, 3]
        matrix_path, indices_path = dataset._embedding_paths("m")
        assert np.load(indices_path).tolist() == [1, 3]
        assert np.load(matrix_path).tolist() == [[1, 1], [3, 3]]


class TestCachedEmbeddings:
    def test_computes_and_saves_with_indices(self, dataset, fake_model):
        emb = cached_embeddings(dataset, "m")
        assert emb.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]
        assert np.load(dataset._embedding_paths("m")[1]).tolist() == [0, 1, 2, 3]

    def test_uses_the_cache_when_it_covers_the_rows(self, dataset, fake_model):
        cached_embeddings(dataset, "m")
        dataset._manifest.loc[1, "action"] = "drop"
        assert cached_embeddings(dataset, "m").tolist() == [[0, 0], [2, 2], [3, 3]]
        assert len(fake_model) == 1

    def test_recomputes_when_other_rows_are_kept(self, dataset, fake_model):
        """The stale-cache bug: same number of kept rows, different rows."""
        dataset._manifest.loc[3, "action"] = "drop"
        cached_embeddings(dataset, "m")  # cache for rows 0, 1, 2
        dataset._manifest.loc[3, "action"] = ""
        dataset._manifest.loc[0, "action"] = "drop"  # now rows 1, 2, 3
        assert cached_embeddings(dataset, "m").tolist() == [[1, 1], [2, 2], [3, 3]]
        assert fake_model == [[0, 1, 2], [1, 2, 3]]

    def test_old_cache_without_indices_is_recomputed(self, dataset, fake_model):
        np.save(dataset._embedding_paths("m")[0], np.zeros((4, 2)))
        assert cached_embeddings(dataset, "m").tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]
        assert len(fake_model) == 1


@pytest.fixture
def fake_phash(monkeypatch):
    """get_perceptual_hashes without images: hash of the i-th path given has bits 0..i set, so row i and
    row j differ in |i - j| bits. Every call is counted."""
    calls = []

    def fake(paths, hash_name="phash"):
        paths = list(paths)
        calls.append(paths)
        return np.array([[b <= i for b in range(8)] for i in range(len(paths))], dtype=bool)
    monkeypatch.setattr(lib, "get_perceptual_hashes", fake)
    return calls


@pytest.fixture(params=["unlabeled", "labeled"])
def any_dataset(request, make_image_folder, tmp_path):
    root = make_image_folder({"Happy": [(i * 40, 0, 0) for i in range(4)]})
    cls = ImageDataset if request.param == "unlabeled" else ImageClassificationDataset
    return cls(root, str(tmp_path / "results"))


class TestMakePerceptualHashes:
    def test_saves_hashes_with_row_indices(self, dataset, fake_phash):
        H = make_perceptual_hashes(dataset)
        assert H.shape == (4, 8)
        assert np.array_equal(dataset.load_perceptual_hashes("phash"), H)


class TestGroupByPerceptualHash:
    def test_uses_the_cache_on_the_second_run(self, dataset, fake_phash):
        group_by_perceptual_hash(dataset, threshold=0, seed=0)
        group_by_perceptual_hash(dataset, threshold=0, seed=0)
        assert len(fake_phash) == 1

    def test_threshold_is_inclusive_and_named_in_the_tag(self, dataset, fake_phash):
        stats = group_by_perceptual_hash(dataset, threshold=1, seed=0)
        assert stats["pairs_found"] == 3  # (0,1), (1,2), (2,3): one bit apart each
        assert stats["groups_found"] == 1  # chained into one group
        reasons = dataset._manifest["reason"]
        assert (reasons == "near_duplicate_phash_1").sum() == 3

    def test_works_for_both_dataset_classes(self, any_dataset, fake_phash):
        """ImageClassificationDataset used to crash: its cache methods had another signature."""
        stats = group_by_perceptual_hash(any_dataset, threshold=1, seed=0)
        assert stats["images_in_groups"] == 4
        group_by_perceptual_hash(any_dataset, threshold=1, seed=0)
        assert len(fake_phash) == 1
