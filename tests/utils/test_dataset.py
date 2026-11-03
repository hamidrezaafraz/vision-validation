import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.utils.dataset import ImageClassificationDataset, similar_pairs

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


@pytest.fixture
def dataset(make_image_folder, tmp_path):
    """4 images: Happy/0000 and Happy/0001 are identical, the rest are unique."""
    root = make_image_folder({"Happy": [RED, RED, GREEN], "Sad": [BLUE]})
    return ImageClassificationDataset(root, result_dir=str(tmp_path))


class TestManifest:
    def test_created_with_one_row_per_image(self, dataset, tmp_path):
        m = dataset._manifest
        assert list(m.columns) == ["path", "label", "group_id", "action", "reason"]
        assert len(m) == len(dataset) == 4
        assert list(m["label"]) == [0, 0, 0, 1]
        assert os.path.exists(tmp_path / "manifest.csv")

    def test_each_image_starts_in_its_own_group(self, dataset):
        assert dataset._manifest["group_id"].is_unique

    def test_reload_keeps_empty_strings(self, dataset, tmp_path):
        """Test that reloading a dataset keeps empty strings in the action and reason columns."""
        reloaded = ImageClassificationDataset(dataset.root_dir, result_dir=str(tmp_path))
        assert (reloaded._manifest["action"] == "").all()
        assert (reloaded._manifest["reason"] == "").all()
        assert list(reloaded._manifest["group_id"]) == list(dataset._manifest["group_id"])

    def test_delete_manifest(self, dataset, tmp_path):
        dataset.delete_manifest()
        assert not os.path.exists(tmp_path / "manifest.csv")


class TestGrouping:
    def test_group_merges_rows(self, dataset):
        dataset.group([[0, 1]])
        gids = dataset._manifest["group_id"]
        assert gids[0] == gids[1]
        assert gids.nunique() == 3

    def test_group_pulls_in_existing_group_members(self, dataset):
        dataset.group([[0, 1]])
        dataset.group([[1, 2]])
        assert dataset._manifest.loc[[0, 1, 2], "group_id"].nunique() == 1

    def test_keep_first_in_group(self, dataset):
        dataset.group([[0, 1]])
        dataset.keep_first_in_group(reason="exact dup")
        m = dataset._manifest
        assert list(m["action"]) == ["keep", "drop", "keep", "keep"]
        assert list(m["reason"]) == ["", "exact dup", "", ""]

    def test_keep_first_in_group_preserves_existing_reason(self, dataset):
        dataset.group([[0, 1]])
        dataset.keep_first_in_group(reason="exact dup")
        dataset.group([[0, 1, 2]])
        dataset.keep_first_in_group(reason="perceptual dup")
        assert list(dataset._manifest["reason"]) == ["", "exact dup", "perceptual dup", ""]

    def test_df_excludes_dropped_rows(self, dataset):
        dataset.group([[0, 1]])
        dataset.keep_first_in_group()
        assert list(dataset.df.index) == [0, 2, 3]

    def test_update_group_ids(self, dataset):
        dataset.update_group_ids([0, 3], ["a", "b"])
        assert dataset._manifest.loc[0, "group_id"] == "a"
        assert dataset._manifest.loc[3, "group_id"] == "b"


class TestSimilarPairs:
    def test_finds_pairs_above_threshold(self):
        X = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]]), dim=1)
        pairs = similar_pairs(X, thr=0.99)
        assert pairs.tolist() == [[0, 1]]

    def test_pairs_are_ordered_and_span_chunks(self):
        X = F.normalize(torch.ones(5, 3), dim=1)
        pairs = similar_pairs(X, thr=0.99, chunk=2)
        assert len(pairs) == 10  # 5 choose 2
        assert (pairs[:, 0] < pairs[:, 1]).all()

    def test_empty_input(self):
        assert similar_pairs(torch.empty(0, 4), thr=0.5).shape == (0, 2)
