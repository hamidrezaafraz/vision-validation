import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.near_duplicates.embeddings import similar_pairs
from src.utils.dataset import ImageClassificationDataset, ImageDataset, apply_policy, embedding_group_column

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

    def test_keep_one_from_group_keeps_exactly_one_per_group(self, dataset):
        dataset.group([[0, 1, 2]])
        dataset.keep_one_from_group(reason="dup", seed=0)
        m = dataset._manifest
        assert list(m["action"]).count("keep") == 2  # one of the group + the singleton
        assert m.loc[3, "action"] == "keep"
        assert list(m.loc[:2, "action"]).count("keep") == 1
        assert set(m.loc[m["action"] == "drop", "reason"]) == {"dup"}

    def test_keep_one_from_group_is_reproducible_and_not_always_first(self, dataset):
        winners = set()
        for seed in range(20):
            dataset.restart()
            dataset.group([[0, 1, 2]])
            dataset.keep_one_from_group("exact_duplicate", seed=seed)
            winners.add(tuple(dataset._manifest["action"][:3]))
        assert len(winners) > 1
        dataset.restart(); dataset.group([[0, 1, 2]]); dataset.keep_one_from_group("exact_duplicate", seed=3)
        first = list(dataset._manifest["action"])
        dataset.restart(); dataset.group([[0, 1, 2]]); dataset.keep_one_from_group("exact_duplicate", seed=3)
        assert list(dataset._manifest["action"]) == first

    def test_df_excludes_dropped_rows(self, dataset):
        dataset.group([[0, 1]])
        dataset.keep_first_in_group("exact_duplicate")
        assert list(dataset.df.index) == [0, 2, 3]

    def test_update_group_ids(self, dataset):
        dataset.update_group_ids([0, 3], ["a", "b"])
        assert dataset._manifest.loc[0, "group_id"] == "a"
        assert dataset._manifest.loc[3, "group_id"] == "b"


@pytest.fixture
def unlabeled_dataset(make_image_folder, tmp_path):
    """4 unlabeled images: rows 0 and 1 are identical, the rest are unique."""
    root = make_image_folder({"Happy": [RED, RED, GREEN], "Sad": [BLUE]})
    return ImageDataset(root, result_dir=str(tmp_path))


class TestKeepOneFromGroup:
    def test_keeps_exactly_one_per_group(self, unlabeled_dataset):
        unlabeled_dataset.group([[0, 1, 2]])
        unlabeled_dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        m = unlabeled_dataset._manifest
        assert list(m.loc[:2, "action"]).count("keep") == 1
        assert list(m.loc[:2, "action"]).count("drop") == 2
        assert m.loc[3, "action"] == "keep"
        assert set(m.loc[m["action"] == "drop", "reason"]) == {"exact_duplicate"}

    def test_appends_to_existing_reason(self, unlabeled_dataset):
        m = unlabeled_dataset._manifest
        m.loc[[0, 1], "reason"] = "blurry"
        unlabeled_dataset.group([[0, 1]])
        unlabeled_dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        dropped = m.loc[[0, 1]][m.loc[[0, 1], "action"] == "drop"]
        kept = m.loc[[0, 1]][m.loc[[0, 1], "action"] == "keep"]
        assert list(dropped["reason"]) == ["blurry;exact_duplicate"]
        assert list(kept["reason"]) == ["blurry"]

    def test_tag_is_not_added_twice(self, unlabeled_dataset):
        unlabeled_dataset._manifest.loc[1, "reason"] = "exact_duplicate"
        unlabeled_dataset.group([[0, 1]])
        unlabeled_dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        assert "exact_duplicate;exact_duplicate" not in set(unlabeled_dataset._manifest["reason"])

    def test_leaves_dropped_and_review_rows_untouched(self, unlabeled_dataset):
        m = unlabeled_dataset._manifest
        apply_policy(m, m.index == 0, "drop", "corrupt")
        apply_policy(m, m.index == 1, "review", "exif_rotated")
        unlabeled_dataset.group([[0, 1, 2]])
        unlabeled_dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        assert list(m.loc[:2, "action"]) == ["drop", "review", "keep"]
        assert list(m.loc[:2, "reason"]) == ["corrupt", "exif_rotated", ""]

    def test_same_seed_gives_same_choice(self, unlabeled_dataset):
        actions = []
        for _ in range(2):
            unlabeled_dataset.restart()
            unlabeled_dataset.group([[0, 1, 2]])
            unlabeled_dataset.keep_one_from_group("exact_duplicate", seed=3)
            actions.append(list(unlabeled_dataset._manifest["action"]))
        assert actions[0] == actions[1]

    def test_saves_manifest(self, unlabeled_dataset, tmp_path):
        unlabeled_dataset.group([[0, 1]])
        unlabeled_dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        reloaded = ImageDataset(unlabeled_dataset_root(tmp_path), result_dir=str(tmp_path))
        assert list(reloaded._manifest["action"]) == list(unlabeled_dataset._manifest["action"])


class TestSelectOnePerGroup:
    def test_unknown_strategy_raises(self, unlabeled_dataset):
        eligible = unlabeled_dataset._manifest["action"] != "drop"
        with pytest.raises(ValueError, match="group_keep_strategy"):
            unlabeled_dataset._select_one_per_group(eligible, group_keep_strategy="sharpest")

    def test_returns_one_index_per_group(self, unlabeled_dataset):
        unlabeled_dataset.group([[0, 1, 2]])
        eligible = unlabeled_dataset._manifest["action"] != "drop"
        chosen = unlabeled_dataset._select_one_per_group(eligible, "random", seed=0)
        assert len(chosen) == 2
        assert 3 in chosen

    def test_ignores_ineligible_rows(self, unlabeled_dataset):
        unlabeled_dataset.group([[0, 1]])
        eligible = unlabeled_dataset._manifest.index != 0
        chosen = unlabeled_dataset._select_one_per_group(eligible, "random", seed=0)
        assert 0 not in chosen and 1 in chosen


def unlabeled_dataset_root(tmp_path):
    return str(tmp_path / "Data")


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


@pytest.fixture(params=["unlabeled", "labeled"])
def any_dataset(request, make_image_folder, tmp_path):
    """Both dataset classes over 4 images; the embedding and perceptual-hash cache methods are shared."""
    root = make_image_folder({"Happy": [(i * 40, 0, 0) for i in range(2)], "Sad": [(0, i * 40, 0) for i in range(2)]})
    cls = ImageDataset if request.param == "unlabeled" else ImageClassificationDataset
    return cls(root, str(tmp_path / "results"))


class TestSaveEmbeddings:
    def test_writes_matrix_and_row_indices(self, any_dataset):
        any_dataset.save_embeddings(np.eye(4), "m", indices=[0, 1, 2, 3])
        matrix_path, indices_path = any_dataset._embedding_paths("m")
        assert np.load(matrix_path).shape == (4, 4)
        assert np.load(indices_path).tolist() == [0, 1, 2, 3]

    def test_accepts_tensors(self, any_dataset):
        any_dataset.save_embeddings(torch.ones(2, 3), "m", indices=[1, 3])
        assert np.load(any_dataset._embedding_paths("m")[0]).shape == (2, 3)

    def test_length_mismatch(self, any_dataset):
        with pytest.raises(ValueError, match="3 embeddings but 2 indices"):
            any_dataset.save_embeddings(np.eye(3), "m", indices=[0, 1])


class TestLoadEmbeddings:
    def matrix(self):
        return np.arange(8, dtype=np.float32).reshape(4, 2)  # row i = [2i, 2i+1]

    def test_round_trip_for_non_dropped_rows(self, any_dataset):
        any_dataset.save_embeddings(self.matrix(), "m", indices=[0, 1, 2, 3])
        assert np.array_equal(any_dataset.load_embeddings("m"), self.matrix())

    def test_rows_follow_the_manifest_not_the_file_position(self, any_dataset):
        """The stale-cache bug: the cache was made when other rows were kept."""
        any_dataset.save_embeddings(self.matrix()[[3, 0, 1]], "m", indices=[3, 0, 1])
        any_dataset._manifest.loc[2, "action"] = "drop"
        loaded = any_dataset.load_embeddings("m")  # rows 0, 1, 3 now
        assert loaded.tolist() == [[0, 1], [2, 3], [6, 7]]

    def test_cache_missing_a_row_is_recomputed(self, any_dataset):
        any_dataset.save_embeddings(self.matrix()[:3], "m", indices=[0, 1, 2])
        assert any_dataset.load_embeddings("m") is None  # row 3 is not dropped but not cached

    def test_cache_without_indices_is_not_used(self, any_dataset):
        np.save(any_dataset._embedding_paths("m")[0], self.matrix())
        assert any_dataset.load_embeddings("m") is None

    def test_explicit_indices(self, any_dataset):
        any_dataset.save_embeddings(self.matrix(), "m", indices=[0, 1, 2, 3])
        assert any_dataset.load_embeddings("m", indices=[2, 0]).tolist() == [[4, 5], [0, 1]]

    def test_no_cache(self, any_dataset):
        assert any_dataset.load_embeddings("m") is None


class TestSavePerceptualHashes:
    def test_writes_hashes_and_row_indices_to_the_right_files(self, any_dataset):
        hashes = np.eye(4, 8, dtype=bool)
        any_dataset.save_perceptual_hashes(hashes, "phash", indices=[0, 1, 2, 3])
        hashes_path, indices_path = any_dataset._perceptual_hash_paths("phash")
        assert np.array_equal(np.load(hashes_path), hashes)
        assert np.load(indices_path).tolist() == [0, 1, 2, 3]

    def test_length_mismatch(self, any_dataset):
        with pytest.raises(ValueError, match="3 hashes but 2 indices"):
            any_dataset.save_perceptual_hashes(np.zeros((3, 8), bool), "phash", indices=[0, 1])


class TestLoadPerceptualHashes:
    def hashes(self):
        return np.arange(4)[:, None] == np.arange(8)[None, :]  # row i has bit i set

    def test_round_trip_for_non_dropped_rows(self, any_dataset):
        any_dataset.save_perceptual_hashes(self.hashes(), "phash", indices=[0, 1, 2, 3])
        assert np.array_equal(any_dataset.load_perceptual_hashes("phash"), self.hashes())

    def test_rows_follow_the_manifest(self, any_dataset):
        any_dataset.save_perceptual_hashes(self.hashes()[[3, 0, 1]], "phash", indices=[3, 0, 1])
        any_dataset._manifest.loc[2, "action"] = "drop"
        assert np.array_equal(any_dataset.load_perceptual_hashes("phash"), self.hashes()[[0, 1, 3]])

    def test_cache_missing_a_row_is_not_used(self, any_dataset):
        any_dataset.save_perceptual_hashes(self.hashes()[:3], "phash", indices=[0, 1, 2])
        assert any_dataset.load_perceptual_hashes("phash") is None

    def test_swapped_files_are_not_used(self, any_dataset):
        """What the old callers wrote: row ids in the hashes file, hashes in the indices file."""
        hashes_path, indices_path = any_dataset._perceptual_hash_paths("phash")
        np.save(hashes_path, np.arange(4))
        np.save(indices_path, self.hashes())
        assert any_dataset.load_perceptual_hashes("phash") is None

    def test_no_cache(self, any_dataset):
        assert any_dataset.load_perceptual_hashes("phash") is None


class TestHasPerceptualHash:
    def test_true_only_when_every_kept_row_is_cached(self, any_dataset):
        assert not any_dataset.has_perceptual_hash("phash")
        any_dataset.save_perceptual_hashes(np.zeros((3, 8), bool), "phash", indices=[0, 1, 2])
        assert not any_dataset.has_perceptual_hash("phash")
        any_dataset._manifest.loc[3, "action"] = "drop"
        assert any_dataset.has_perceptual_hash("phash")


class TestRestart:
    def test_resets_per_model_group_columns(self, make_image_folder, tmp_path):
        root = make_image_folder({"Happy": [(10, 0, 0), (20, 0, 0), (30, 0, 0)]})
        ds = ImageDataset(root, str(tmp_path / "results"))
        column = embedding_group_column("alexnet", 0.95)
        ds.group([[0, 1, 2]], column=column)
        assert ds._manifest[column].nunique() == 1
        ds.restart()
        assert ds._manifest[column].nunique() == 3
        assert ds._manifest["split_group"].nunique() == 3

    def test_new_links_are_not_merged_into_old_ones(self, make_image_folder, tmp_path):
        root = make_image_folder({"Happy": [(10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)]})
        ds = ImageDataset(root, str(tmp_path / "results"))
        column = embedding_group_column("alexnet", 0.95)
        ds.group([[0, 1]], column=column)
        ds.restart()
        ds.group([[1, 2]], column=column)
        sizes = ds._manifest.groupby(column).size()
        assert sorted(sizes.tolist()) == [1, 1, 2]  # before the fix: one group of 3
