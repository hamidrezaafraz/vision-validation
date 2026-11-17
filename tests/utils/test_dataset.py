import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.utils.dataset import ImageClassificationDataset, embedding_group_column, similar_pairs

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


@pytest.fixture
def dataset(make_image_folder, tmp_path):
    """4 images: Happy/0000 and Happy/0001 are identical, the rest are unique."""
    root = make_image_folder({"Happy": [RED, RED, GREEN], "Sad": [BLUE]})
    return ImageClassificationDataset(root, result_dir=str(tmp_path))


class TestManifest:
    def test_created_with_one_row_per_image(self, dataset, tmp_path):
        m = dataset._manifest
        assert list(m.columns) == ["path", "label", "group_id", "split_group", "label_conflict", "action", "reason"]
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

    def test_group_id_merge_mirrors_into_split_group(self, dataset):
        dataset.group([[0, 1]])
        m = dataset._manifest
        assert m.loc[0, "split_group"] == m.loc[1, "split_group"]

    def test_split_group_merge_leaves_group_id_and_drops_alone(self, dataset):
        dataset.group([[0, 1]], column="split_group")
        dataset.keep_first_in_group(reason="dup")
        m = dataset._manifest
        assert m.loc[0, "split_group"] == m.loc[1, "split_group"]
        assert m["group_id"].is_unique
        assert list(m["action"]) == ["keep"] * 4

    def test_embedding_group_column_name(self):
        assert embedding_group_column("alexnet", 0.95) == "group_alexnet_0_95"
        assert embedding_group_column("facenet-vggface2", 0.9) == "group_facenet-vggface2_0_9"

    def test_model_column_is_created_and_leaves_other_columns_alone(self, dataset):
        col = embedding_group_column("alexnet", 0.95)
        dataset.group([[0, 1]], column=col)
        m = dataset._manifest
        assert m.loc[0, col] == m.loc[1, col]
        assert m[col].nunique() == 3
        assert m["group_id"].is_unique
        assert m["split_group"].is_unique

    def test_rebuild_split_group_uses_only_listed_columns(self, dataset):
        a, b = embedding_group_column("alexnet", 0.95), embedding_group_column("facenet", 0.9)
        dataset.group([[0, 1]], column=a)
        dataset.group([[1, 2]], column=b)
        dataset.rebuild_split_group([a])
        s = dataset._manifest["split_group"]
        assert s[0] == s[1] and s[2] != s[0] and s.nunique() == 3
        dataset.rebuild_split_group([a, b])  # chain 0-1-2 through both models
        s = dataset._manifest["split_group"]
        assert s[0] == s[1] == s[2] and s[3] != s[0]

    def test_rebuild_split_group_includes_group_id_and_updates_label_conflict(self, dataset):
        col = embedding_group_column("alexnet", 0.95)
        dataset.group([[0, 1]])  # exact dup
        dataset.group([[2, 3]], column=col)  # Happy + Sad
        dataset.rebuild_split_group([col])
        m = dataset._manifest
        assert m.loc[0, "split_group"] == m.loc[1, "split_group"]
        assert list(m["label_conflict"]) == [False, False, True, True]

    def test_rebuild_split_group_unknown_column(self, dataset):
        with pytest.raises(KeyError):
            dataset.rebuild_split_group(["group_nope_0_5"])

    def test_model_column_survives_reload(self, dataset, tmp_path):
        col = embedding_group_column("alexnet", 0.95)
        dataset.group([[0, 1]], column=col)
        dataset.save_manifest()
        reloaded = ImageClassificationDataset(dataset.root_dir, result_dir=str(tmp_path))
        assert reloaded._manifest.loc[0, col] == reloaded._manifest.loc[1, col]
        assert reloaded.group_columns() == [col]

    def test_label_conflict_flags_groups_with_mixed_labels(self, dataset):
        assert not dataset._manifest["label_conflict"].any()
        dataset.group([[0, 1]])  # both Happy: no conflict
        assert not dataset._manifest["label_conflict"].any()
        dataset.group([[2, 3]], column="split_group")  # Happy + Sad
        assert list(dataset._manifest["label_conflict"]) == [False, False, True, True]

    def test_keep_first_in_split_group_skips_label_conflicts(self, dataset):
        dataset.group([[0, 1]], column="split_group")  # both Happy: no conflict
        dataset.group([[2, 3]], column="split_group")  # Happy + Sad: conflict
        dataset.keep_first_in_split_group(reason="emb")
        m = dataset._manifest
        assert list(m["action"]) == ["keep", "drop", "", ""]
        assert list(m["reason"]) == ["", "emb", "", ""]

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

    def test_keep_random_in_group_keeps_exactly_one_per_group(self, dataset):
        dataset.group([[0, 1, 2]])
        dataset.keep_random_in_group(reason="dup", seed=0)
        m = dataset._manifest
        assert list(m["action"]).count("keep") == 2  # one of the group + the singleton
        assert m.loc[3, "action"] == "keep"
        assert list(m.loc[:2, "action"]).count("keep") == 1
        assert set(m.loc[m["action"] == "drop", "reason"]) == {"dup"}

    def test_keep_random_in_group_is_reproducible_and_not_always_first(self, dataset):
        winners = set()
        for seed in range(20):
            dataset.restart()
            dataset.group([[0, 1, 2]])
            dataset.keep_random_in_group(seed=seed)
            winners.add(tuple(dataset._manifest["action"][:3]))
        assert len(winners) > 1
        dataset.restart(); dataset.group([[0, 1, 2]]); dataset.keep_random_in_group(seed=3)
        first = list(dataset._manifest["action"])
        dataset.restart(); dataset.group([[0, 1, 2]]); dataset.keep_random_in_group(seed=3)
        assert list(dataset._manifest["action"]) == first

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
