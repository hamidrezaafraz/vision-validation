import numpy as np
import pytest

import src.cli.lib as lib
from src.cli.lib import cached_embeddings
from src.utils.dataset import ImageDataset


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
