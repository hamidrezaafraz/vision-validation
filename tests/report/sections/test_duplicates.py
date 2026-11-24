import json

import numpy as np
import pandas as pd
import pytest

from src.report.sections.duplicates import (
    column_kind,
    column_statistics,
    describe_group,
    duplicate_statistics,
    embedding_columns,
    group_labels,
    grouping_columns,
    levels,
    load_cached_embeddings,
    mean_pairwise_similarity,
    model_for_column,
    redundancy,
    same_partition,
    suspect_threshold,
    union_groups,
)

DATA = "/data"


def make_manifest(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with at least 'path' (relative to DATA); missing action/reason are ''."""
    df = pd.DataFrame([{"action": "", "reason": "", **r} for r in rows])
    df["path"] = DATA + "/" + df["path"]
    return df.astype(str)


@pytest.fixture
def example():
    """
    6 rows. hash: a1/a2 identical. group_id adds a3 (perceptual). split_group = group_id + alexnet,
    and alexnet links b1 (folder b) to a1, so one group spans folders a and b. c1 is alone.
    """
    return make_manifest([
        {"path": "a/1.png", "hash": "h1", "group_id": "G1", "split_group": "S1", "group_alexnet_0_95": "E1", "action": "keep"},
        {"path": "a/2.png", "hash": "h1", "group_id": "G1", "split_group": "S1", "group_alexnet_0_95": "x2", "action": "drop", "reason": "exact_duplicate"},
        {"path": "a/3.png", "hash": "h3", "group_id": "G1", "split_group": "S1", "group_alexnet_0_95": "x3", "action": "drop", "reason": "near_duplicate_phash_4"},
        {"path": "b/1.png", "hash": "h4", "group_id": "G4", "split_group": "S1", "group_alexnet_0_95": "E1", "action": "keep"},
        {"path": "b/2.png", "hash": "h5", "group_id": "G5", "split_group": "S5", "group_alexnet_0_95": "x5", "action": "keep"},
        {"path": "c/1.png", "hash": "", "group_id": "G6", "split_group": "S6", "group_alexnet_0_95": "x6", "action": "keep"},
    ])


class TestEmbeddingColumns:
    def test_finds_per_model_columns_only(self, example):
        assert embedding_columns(example) == ["group_alexnet_0_95"]


class TestGroupingColumns:
    def test_order(self, example):
        assert grouping_columns(example) == ["hash", "group_id", "split_group", "group_alexnet_0_95"]

    def test_missing_columns_are_skipped(self):
        assert grouping_columns(make_manifest([{"path": "a.png", "group_id": "g"}])) == ["group_id"]


class TestGroupLabels:
    def test_empty_hashes_are_not_grouped_together(self):
        manifest = make_manifest([{"path": f"{i}.png", "hash": ""} for i in range(3)])
        assert group_labels(manifest, "hash").nunique() == 3

    def test_other_columns_are_used_as_is(self, example):
        assert group_labels(example, "group_id").tolist() == ["G1", "G1", "G1", "G4", "G5", "G6"]


class TestUnionGroups:
    def test_chains_across_columns(self, example):
        labels = union_groups(example, ["group_id", "group_alexnet_0_95"])
        assert labels.iloc[0] == labels.iloc[1] == labels.iloc[2] == labels.iloc[3]
        assert labels.nunique() == 3

    def test_chain_within_one_column_through_another(self):
        # A~B in x, B~C in y, C~D in x: one group.
        manifest = make_manifest([
            {"path": "A", "x": "1", "y": "a"}, {"path": "B", "x": "1", "y": "b"},
            {"path": "C", "x": "2", "y": "b"}, {"path": "D", "x": "2", "y": "d"},
        ])
        assert union_groups(manifest, ["x", "y"]).nunique() == 1

    def test_keeps_index(self, example):
        example.index = example.index + 10
        assert union_groups(example, ["group_id"]).index.tolist() == list(range(10, 16))


class TestRedundancy:
    def test_counts(self):
        stats = redundancy(pd.Series(["a", "a", "a", "b", "b", "c"]))
        assert stats["groups"] == 2
        assert stats["images_in_groups"]["count"] == 5
        assert stats["redundant"] == {"count": 3, "of": 6, "pct": 50.0}
        assert stats["unique"]["count"] == 3

    def test_no_duplicates(self):
        stats = redundancy(pd.Series(["a", "b"]))
        assert (stats["groups"], stats["redundant"]["count"], stats["unique"]["count"]) == (0, 0, 2)


class TestSamePartition:
    def test_relabelled_is_same(self):
        assert same_partition(pd.Series([1, 1, 2]), pd.Series(["x", "x", "y"]))

    def test_split_or_merged_is_different(self):
        assert not same_partition(pd.Series([1, 1, 2]), pd.Series(["x", "y", "y"]))
        assert not same_partition(pd.Series([1, 1, 2]), pd.Series(["x", "x", "x"]))


class TestLevels:
    def test_cumulative_levels(self, example):
        result = levels(example)
        by_level = {l["level"]: l for l in result["levels"]}
        assert [l["level"] for l in result["levels"]] == ["exact", "perceptual", "alexnet_0_95"]
        assert (by_level["exact"]["redundant"]["count"], by_level["exact"]["added_redundant"]) == (1, 1)
        assert (by_level["perceptual"]["redundant"]["count"], by_level["perceptual"]["added_redundant"]) == (2, 1)
        assert by_level["alexnet_0_95"]["columns"] == ["hash", "group_id", "group_alexnet_0_95"]
        assert (by_level["alexnet_0_95"]["redundant"]["count"], by_level["alexnet_0_95"]["added_redundant"]) == (3, 1)
        assert by_level["alexnet_0_95"]["unique"]["count"] == 3

    def test_matches_split_group(self, example):
        assert levels(example)["matches_split_group"] is True
        example.loc[4, "split_group"] = "S1"
        assert levels(example)["matches_split_group"] is False


class TestLoadCachedEmbeddings:
    def test_loads_with_indices(self, example, tmp_path):
        np.save(tmp_path / "alexnet_embeddings.npy", np.eye(3, dtype=np.float32))
        np.save(tmp_path / "alexnet_embeddings_indices.npy", np.array([0, 3, 4]))
        found, unavailable = load_cached_embeddings(example, str(tmp_path))
        assert unavailable == []
        matrix, index = found["alexnet"]
        assert matrix.shape == (3, 3) and index.tolist() == [0, 3, 4]

    def test_never_pairs_by_position(self, example, tmp_path):
        """Without saved indices the rows can't be matched, even when the counts agree."""
        np.save(tmp_path / "alexnet_embeddings.npy", np.eye(6, dtype=np.float32))
        found, unavailable = load_cached_embeddings(example, str(tmp_path))
        assert found == {}
        assert unavailable[0]["model"] == "alexnet" and "indices" in unavailable[0]["reason"]

    def test_length_mismatch(self, example, tmp_path):
        np.save(tmp_path / "m_embeddings.npy", np.eye(3, dtype=np.float32))
        np.save(tmp_path / "m_embeddings_indices.npy", np.array([0, 1]))
        assert "but 2 indices" in load_cached_embeddings(example, str(tmp_path))[1][0]["reason"]

    def test_indices_not_in_manifest(self, example, tmp_path):
        np.save(tmp_path / "m_embeddings.npy", np.eye(2, dtype=np.float32))
        np.save(tmp_path / "m_embeddings_indices.npy", np.array([0, 99]))
        assert "not in the manifest" in load_cached_embeddings(example, str(tmp_path))[1][0]["reason"]

    def test_ignores_other_files_and_missing_dir(self, example, tmp_path):
        np.save(tmp_path / "phash_hashes.npy", np.zeros(3))
        assert load_cached_embeddings(example, str(tmp_path)) == ({}, [])
        assert load_cached_embeddings(example, str(tmp_path / "nope")) == ({}, [])


class TestModelForColumn:
    def test_own_model_longest_match(self):
        embeddings = {"face": None, "facenet-vggface2": None}
        assert model_for_column("group_facenet-vggface2_0_95", embeddings, "face") == "facenet-vggface2"
        assert model_for_column("group_face_0_9", embeddings, None) == "face"

    def test_default_for_other_columns(self):
        assert model_for_column("group_id", {"alexnet": None}, "alexnet") == "alexnet"
        assert model_for_column("group_id", {"alexnet": None}, "missing") is None


class TestMeanPairwiseSimilarity:
    def test_identical_and_orthogonal(self):
        matrix = np.array([[1.0, 0], [2.0, 0], [0, 3.0]])
        assert mean_pairwise_similarity(matrix, [0, 1]) == 1.0
        assert mean_pairwise_similarity(matrix, [0, 2]) == 0.0
        assert mean_pairwise_similarity(matrix, [0, 1, 2]) == round(1 / 3, 4)

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(7, 5))
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        sims = unit @ unit.T
        expected = sims[np.triu_indices(7, 1)].mean()
        assert mean_pairwise_similarity(matrix, list(range(7))) == round(expected, 4)

    def test_fewer_than_two(self):
        assert mean_pairwise_similarity(np.eye(2), [0]) is None


class TestSuspectThreshold:
    def test_few_groups_use_max_expected(self):
        assert suspect_threshold(pd.Series([2, 3, 40]), 50) == 50.0

    def test_many_groups_use_percentile_when_lower(self):
        sizes = pd.Series([2] * 1999 + [30])
        assert suspect_threshold(sizes, 50) == pytest.approx(np.percentile(sizes, 99.9))
        assert suspect_threshold(sizes, 50) < 30


class TestColumnStatistics:
    def rows(self, manifest):
        return pd.DataFrame({
            "rel_path": manifest["path"].str.removeprefix(DATA + "/"),
            "folder": manifest["path"].str.removeprefix(DATA + "/").str.split("/").str[0],
            "status": manifest["action"].map({"": "ok", "keep": "ok", "review": "review", "drop": "drop"}),
        }, index=manifest.index)

    def test_split_group_column(self, example):
        stats = column_statistics(self.rows(example), group_labels(example, "split_group"), "split_group", {}, None)
        assert stats["kind"] == "split"
        assert stats["size_histogram"] == [{"size": 4, "groups": 1}]
        top = stats["largest"][0]
        assert top["size"] == 4
        assert top["folders"] == {"a": 3, "b": 1}
        assert top["status"] == {"ok": 2, "review": 0, "drop": 2}
        assert top["kept"] == ["a/1.png", "b/1.png"]
        assert top["similarity"] is None and top["chaining_suspect"] is False
        assert stats["cross_folder"]["count"] == 1
        assert stats["cross_folder"]["folder_pairs"] == [{"folders": ["a", "b"], "groups": 1}]

    def test_chained_group_is_suspect_with_low_similarity(self):
        manifest = make_manifest([{"path": f"a/{i}.png", "g": "one"} for i in range(4)] + [{"path": "a/x.png", "g": "two"}])
        matrix = np.eye(5)
        embeddings = {"m": (matrix, manifest.index)}
        stats = column_statistics(self.rows(manifest), group_labels(manifest, "g"), "g", embeddings, "m",
                                  max_expected_group_size=3)
        assert stats["chaining"]["suspects"] == 1
        suspect = stats["chaining"]["groups"][0]
        assert suspect["chaining_suspect"] is True
        assert suspect["similarity"] == {"model": "m", "members_with_embeddings": 4, "mean_pairwise_cosine": 0.0}

    def test_members_are_capped(self):
        manifest = make_manifest([{"path": f"a/{i:03d}.png", "g": "one"} for i in range(60)])
        stats = column_statistics(self.rows(manifest), group_labels(manifest, "g"), "g", {}, None)
        top = stats["largest"][0]
        assert len(top["members"]) == 50 and top["members_truncated"] is True
        assert len(top["kept"]) == 5

    def test_largest_order_is_deterministic(self):
        manifest = make_manifest([{"path": p, "g": g} for p, g in
                                  [("b/1", "x"), ("b/2", "x"), ("a/1", "y"), ("a/2", "y"), ("c/1", "z"), ("c/2", "z"), ("c/3", "z")]])
        stats = column_statistics(self.rows(manifest), group_labels(manifest, "g"), "g", {}, None)
        assert [g["members"][0] for g in stats["largest"]] == ["c/1", "a/1", "b/1"]


class TestDuplicateStatistics:
    def test_section(self, example):
        section = duplicate_statistics(example, DATA)
        assert section["total_images"] == 6
        assert list(section["columns"]) == ["hash", "group_id", "split_group", "group_alexnet_0_95"]
        assert section["columns"]["hash"]["redundant"]["count"] == 1
        assert section["matches_split_group"] is True
        assert section["similarity_models"] == []

    def test_paths_are_relative_to_data_dir(self, example):
        top = duplicate_statistics(example, DATA)["columns"]["group_id"]["largest"][0]
        assert top["members"] == ["a/1.png", "a/2.png", "a/3.png"]

    def test_with_embeddings(self, example):
        matrix = np.array([[1.0, 0], [1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0], [0, 1.0]])
        section = duplicate_statistics(example, DATA, embeddings={"alexnet": (matrix, example.index)})
        top = section["columns"]["group_id"]["largest"][0]
        assert top["similarity"]["model"] == "alexnet"
        assert top["similarity"]["mean_pairwise_cosine"] == round(1 / 3, 4)

    def test_is_strict_json(self, example):
        matrix = np.ones((6, 2))
        json.dumps(duplicate_statistics(example, DATA, embeddings={"alexnet": (matrix, example.index)}), allow_nan=False)

    def test_does_not_modify_manifest(self, example):
        before = example.copy()
        duplicate_statistics(example, DATA)
        pd.testing.assert_frame_equal(example, before)

    def test_unknown_action_raises(self, example):
        example.loc[0, "action"] = "delete"
        with pytest.raises(ValueError, match="unknown action"):
            duplicate_statistics(example, DATA)

    def test_empty_manifest(self):
        manifest = pd.DataFrame(columns=["path", "group_id", "split_group", "action", "reason"], dtype=str)
        section = duplicate_statistics(manifest, DATA)
        assert section["columns"]["group_id"]["groups"] == 0
        json.dumps(section, allow_nan=False)


class TestColumnKind:
    def test_kinds(self):
        assert [column_kind(c) for c in ("hash", "group_id", "split_group", "group_alexnet_0_95")] == [
            "exact", "duplicate", "split", "embedding",
        ]


class TestDescribeGroup:
    def members(self, n, kept=()):
        return pd.DataFrame({
            "rel_path": [f"a/{i:03d}.png" for i in range(n)],
            "folder": ["a"] * (n - 1) + ["b"],
            "status": ["ok" if i in kept else "drop" for i in range(n)],
        })

    def test_fields(self):
        group = describe_group(self.members(3, kept=(1,)), suspect=False, similarity=None)
        assert group["size"] == 3 and group["folders"] == {"a": 2, "b": 1}
        assert group["status"] == {"ok": 1, "review": 0, "drop": 2}
        assert group["kept"] == ["a/001.png"] and group["members_truncated"] is False
        assert group["chaining_suspect"] is False and group["similarity"] is None

    def test_caps(self):
        group = describe_group(self.members(60, kept=range(10)), suspect=True, similarity={"x": 1})
        assert len(group["members"]) == 50 and group["members_truncated"] is True
        assert len(group["kept"]) == 5 and group["chaining_suspect"] is True
