import pandas as pd
import pytest

from src.report.status import DROP, OK, REVIEW, is_duplicate_representative, resolve_status


def make_manifest(actions, group_ids=None):
    n = len(actions)
    return pd.DataFrame({
        "path": [f"Data/{i:04d}.png" for i in range(n)],
        "group_id": group_ids if group_ids is not None else [f"g{i}" for i in range(n)],
        "action": actions,
        "reason": [""] * n,
    })


class TestResolveStatus:
    @pytest.mark.parametrize("action, expected", [("", OK), ("keep", OK), ("review", REVIEW), ("drop", DROP)])
    def test_maps_each_action(self, action, expected):
        assert resolve_status(make_manifest([action])).tolist() == [expected]

    def test_empty_and_keep_resolve_to_the_same_status(self):
        """A checks-only run ('') and a run with duplicate steps ('keep') must give the same clean set."""
        assert resolve_status(make_manifest(["", "keep"])).tolist() == [OK, OK]

    def test_keeps_index_and_is_named_status(self):
        manifest = make_manifest(["", "drop", "review"]).set_index(pd.Index([10, 20, 30]))
        status = resolve_status(manifest)
        assert status.name == "status"
        assert status.index.tolist() == [10, 20, 30]

    @pytest.mark.parametrize("bad", ["Keep", "delete", float("nan")])
    def test_unknown_action_raises_with_value_and_path(self, bad):
        manifest = make_manifest(["", bad])
        with pytest.raises(ValueError, match="unknown action") as exc:
            resolve_status(manifest)
        assert "Data/0001.png" in str(exc.value)

    def test_does_not_modify_manifest(self):
        manifest = make_manifest(["", "keep", "review", "drop"])
        before = manifest.copy()
        resolve_status(manifest)
        pd.testing.assert_frame_equal(manifest, before)

    def test_empty_manifest(self):
        assert resolve_status(make_manifest([])).tolist() == []


class TestIsDuplicateRepresentative:
    def test_kept_member_of_duplicate_group(self):
        manifest = make_manifest(["keep", "drop", "drop"], group_ids=["a", "a", "a"])
        assert is_duplicate_representative(manifest).tolist() == [True, False, False]

    def test_singleton_keep_is_not_a_representative(self):
        manifest = make_manifest(["keep", "keep", "drop"], group_ids=["a", "b", "b"])
        assert is_duplicate_representative(manifest).tolist() == [False, True, False]

    def test_empty_action_in_group_counts(self):
        """Groups can exist without a keep step having run (e.g. embedding grouping only)."""
        manifest = make_manifest(["", ""], group_ids=["a", "a"])
        assert is_duplicate_representative(manifest).tolist() == [True, True]

    def test_review_row_is_not_a_representative(self):
        manifest = make_manifest(["review", "drop"], group_ids=["a", "a"])
        assert is_duplicate_representative(manifest).tolist() == [False, False]

    def test_group_size_counts_rows_dropped_by_other_checks(self):
        manifest = make_manifest(["keep", "drop"], group_ids=["a", "a"])
        manifest.loc[1, "reason"] = "corrupt"
        assert is_duplicate_representative(manifest).tolist() == [True, False]

    def test_other_group_column(self):
        manifest = make_manifest(["keep", "keep"], group_ids=["a", "b"])
        manifest["group_alexnet_0_95"] = ["x", "x"]
        assert is_duplicate_representative(manifest, column="group_alexnet_0_95").tolist() == [True, True]
        assert is_duplicate_representative(manifest).tolist() == [False, False]

    def test_named_and_does_not_modify_manifest(self):
        manifest = make_manifest(["keep", "drop"], group_ids=["a", "a"])
        before = manifest.copy()
        result = is_duplicate_representative(manifest)
        assert result.name == "is_duplicate_representative"
        pd.testing.assert_frame_equal(manifest, before)


class TestWithRealManifest:
    def test_checks_only_run_is_all_ok(self, make_image_folder, tmp_path):
        from src.utils.dataset import ImageDataset

        root = make_image_folder({"A": [(255, 0, 0), (0, 255, 0)]})
        dataset = ImageDataset(root, result_dir=str(tmp_path))
        assert resolve_status(dataset._manifest).tolist() == [OK, OK]

    def test_after_keep_one_from_group(self, make_image_folder, tmp_path):
        from src.utils.dataset import ImageDataset

        root = make_image_folder({"A": [(255, 0, 0), (255, 0, 0), (0, 255, 0)]})
        dataset = ImageDataset(root, result_dir=str(tmp_path))
        dataset.group([[0, 1]])
        dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        m = dataset._manifest
        assert sorted(resolve_status(m)) == [DROP, OK, OK]
        assert is_duplicate_representative(m).sum() == 1
        assert not is_duplicate_representative(m)[2]
