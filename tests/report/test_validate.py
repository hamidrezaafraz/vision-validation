import os

import pandas as pd
import pytest

from src.report.validate import (
    ERROR,
    WARNING,
    ManifestValidationError,
    is_duplicate_tag,
    raise_for_errors,
    validate_manifest,
)
from src.utils.dataset import ImageDataset

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


@pytest.fixture
def folder(make_image_folder):
    """3 files on disk: Data/A/0000.png, 0001.png (identical) and 0002.png."""
    return make_image_folder({"A": [RED, RED, GREEN]})


@pytest.fixture
def manifest(folder):
    """A valid manifest for `folder`: rows 0 and 1 are duplicates, row 0 kept."""
    paths = sorted(os.path.join(folder, "A", f) for f in os.listdir(os.path.join(folder, "A")))
    return pd.DataFrame({
        "path": paths,
        "group_id": ["g0", "g0", "g2"],
        "action": ["keep", "drop", "keep"],
        "reason": ["", "exact_duplicate", ""],
    })


def codes(issues):
    return {i.code for i in issues}


def only(issues, code):
    [issue] = [i for i in issues if i.code == code]
    return issue


class TestValidateManifest:
    def test_valid_manifest_has_no_issues(self, manifest, folder):
        assert validate_manifest(manifest, data_dir=folder) == []

    def test_missing_required_columns_returns_only_that_error(self, manifest):
        issues = validate_manifest(manifest.drop(columns=["group_id"]))
        assert [(i.code, i.severity) for i in issues] == [("missing_columns", ERROR)]
        assert issues[0].examples == ["group_id"]

    @pytest.mark.parametrize("bad", ["Keep", "delete", float("nan")])
    def test_unknown_action(self, manifest, bad):
        manifest.loc[2, "action"] = bad
        issue = only(validate_manifest(manifest), "unknown_action")
        assert issue.severity == ERROR and issue.count == 1
        assert issue.examples == [manifest.loc[2, "path"]]

    @pytest.mark.parametrize("action", ["drop", "review"])
    def test_flag_without_reason(self, manifest, action):
        manifest.loc[2, ["action", "reason"]] = [action, ""]
        issue = only(validate_manifest(manifest), "flag_without_reason")
        assert issue.severity == ERROR and issue.count == 1

    def test_whitespace_reason_counts_as_missing(self, manifest):
        manifest.loc[2, ["action", "reason"]] = ["review", "  "]
        assert "flag_without_reason" in codes(validate_manifest(manifest))

    def test_duplicate_path_is_listed_once(self, manifest):
        manifest.loc[2, "path"] = manifest.loc[0, "path"]
        issue = only(validate_manifest(manifest), "duplicate_path")
        assert issue.severity == ERROR
        assert issue.examples == [manifest.loc[0, "path"]]

    def test_missing_file(self, manifest):
        os.remove(manifest.loc[2, "path"])
        issue = only(validate_manifest(manifest), "missing_file")
        assert issue.severity == ERROR and issue.examples == [manifest.loc[2, "path"]]

    def test_relative_paths_resolve_against_base_dir(self, manifest, folder, tmp_path):
        manifest["path"] = manifest["path"].map(lambda p: os.path.relpath(p, tmp_path))
        assert "missing_file" not in codes(validate_manifest(manifest, base_dir=str(tmp_path)))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        assert only(validate_manifest(manifest, base_dir=str(elsewhere)), "missing_file").count == 3

    def test_unscanned_file(self, manifest, folder):
        extra = os.path.join(folder, "A", "0003.png")
        open(extra, "wb").close()
        issue = only(validate_manifest(manifest, data_dir=folder), "unscanned_file")
        assert issue.severity == ERROR and issue.examples == [os.path.abspath(extra)]

    def test_unscanned_check_is_skipped_without_data_dir(self, manifest, folder):
        open(os.path.join(folder, "A", "0003.png"), "wb").close()
        assert validate_manifest(manifest) == []

    def test_orphaned_duplicate(self, manifest):
        """The representative was dropped by a later check, so the copy stands for nothing."""
        manifest.loc[0, ["action", "reason"]] = ["drop", "corrupt"]
        issue = only(validate_manifest(manifest), "orphaned_duplicate")
        assert issue.severity == WARNING and issue.examples == [manifest.loc[1, "path"]]

    def test_review_representative_is_not_orphaning(self, manifest):
        manifest.loc[0, ["action", "reason"]] = ["review", "blurry"]
        assert "orphaned_duplicate" not in codes(validate_manifest(manifest))

    def test_group_dropped_for_other_reasons_is_not_orphaned(self, manifest):
        manifest.loc[[0, 1], ["action", "reason"]] = ["drop", "corrupt"]
        assert "orphaned_duplicate" not in codes(validate_manifest(manifest))

    def test_unresolved_group(self, manifest):
        manifest.loc[1, ["action", "reason"]] = ["", ""]
        issue = only(validate_manifest(manifest), "unresolved_group")
        assert issue.severity == WARNING and issue.count == 2
        assert "1 duplicate groups" in issue.message

    def test_ok_with_reason(self, manifest):
        manifest.loc[2, "reason"] = "exif_rotated"
        issue = only(validate_manifest(manifest), "ok_with_reason")
        assert issue.severity == WARNING and issue.count == 1

    def test_examples_are_capped(self, manifest):
        big = pd.concat([manifest.iloc[[2]]] * 30, ignore_index=True)
        big["action"] = "Keep"
        big["path"] = [f"x/{i}.png" for i in range(30)]
        big["group_id"] = [f"g{i}" for i in range(30)]
        issue = only(validate_manifest(big), "unknown_action")
        assert issue.count == 30 and len(issue.examples) == 20

    def test_does_not_modify_manifest(self, manifest, folder):
        manifest.loc[1, "action"] = "Keep"
        before = manifest.copy()
        validate_manifest(manifest, data_dir=folder)
        pd.testing.assert_frame_equal(manifest, before)

    def test_real_dataset_after_dedup_is_valid(self, folder, tmp_path):
        dataset = ImageDataset(folder, result_dir=str(tmp_path / "results"))
        dataset.group([[0, 1]])
        dataset.keep_one_from_group(reason="exact_duplicate", seed=0)
        assert validate_manifest(dataset._manifest, data_dir=folder) == []


class TestIsDuplicateTag:
    @pytest.mark.parametrize("tag", ["exact_duplicate", "byte_duplicate", "near_duplicate_phash_4", "near_duplicate_alexnet_0_95", "exact hash duplicates"])
    def test_duplicate_tags(self, tag):
        assert is_duplicate_tag(tag)

    @pytest.mark.parametrize("tag", ["corrupt", "blurry", "", "duplicate"])
    def test_other_tags(self, tag):
        assert not is_duplicate_tag(tag)


class TestRaiseForErrors:
    def test_raises_on_errors_with_all_codes(self, manifest):
        manifest.loc[2, "action"] = "Keep"
        manifest.loc[1, "reason"] = ""
        with pytest.raises(ManifestValidationError) as exc:
            raise_for_errors(validate_manifest(manifest))
        assert {i.code for i in exc.value.issues} == {"unknown_action", "flag_without_reason"}
        assert "unknown_action" in str(exc.value)

    def test_warnings_do_not_raise(self, manifest):
        manifest.loc[2, "reason"] = "exif_rotated"
        issues = validate_manifest(manifest)
        assert issues and all(i.severity == WARNING for i in issues)
        raise_for_errors(issues)

    def test_no_issues_does_not_raise(self):
        raise_for_errors([])
