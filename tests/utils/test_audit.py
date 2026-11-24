import json
import os

import pandas as pd
import pytest

from src.cli.lib import group_by_hash
from src.integrity import check_decode
from src.photometric import check_photometric
from src.utils.audit import (
    FAILED,
    MANIFEST_CREATED,
    OK,
    RESTART,
    append_audit,
    audited,
    current_audits,
    dataset_fingerprint,
    environment_info,
    latest_by_check,
    load_audits,
    manifest_changes,
    paths_digest,
)
from src.utils.dataset import ImageDataset

# Gray, so every channel is equal: check_photometric computes pixel_std over all channels together,
# so a solid saturated color such as pure red is not seen as blank.
DARK, MID, LIGHT = (60, 60, 60), (128, 128, 128), (200, 200, 200)


@pytest.fixture
def dataset(make_image_folder, tmp_path):
    """3 solid gray 8x8 images: rows 0 and 1 are byte-identical."""
    root = make_image_folder({"A": [MID, MID, LIGHT]})
    return ImageDataset(root, result_dir=str(tmp_path / "results"))


def audits(dataset):
    return load_audits(dataset._audit_path)


def checks(dataset):
    return [e for e in audits(dataset) if "check" in e]


class TestRecordEvent:
    def test_new_manifest_records_manifest_created(self, dataset):
        [entry] = audits(dataset)
        assert entry["event"] == MANIFEST_CREATED
        assert entry["manifest_rows"] == 3
        assert entry["schema_version"] == 1

    def test_reloading_an_existing_manifest_records_nothing(self, dataset):
        ImageDataset(os.path.dirname(os.path.dirname(dataset._manifest["path"][0])), result_dir=dataset.result_dir)
        assert len(audits(dataset)) == 1

    def test_restart_records_restart(self, dataset):
        dataset.restart()
        assert [e["event"] for e in audits(dataset)] == [MANIFEST_CREATED, RESTART]


class TestAudited:
    def test_check_photometric_records_params_stats_and_changes(self, dataset):
        check_photometric(dataset, min_blur_score=12.5, num_workers=1)
        [entry] = checks(dataset)
        assert entry["check"] == "check_photometric" and entry["status"] == OK and entry["error"] is None
        # Explicit and default arguments are both recorded; the dataset is not.
        assert entry["params"]["min_blur_score"] == 12.5
        assert entry["params"]["min_std"] == 0.02
        assert "dataset" not in entry["params"]
        assert entry["stats"] == {"rows_inspected": 3}
        # Solid gray images are blank, and on_violation defaults to 'review'.
        assert entry["changes"] == {"tags_added": {"blank_or_solid": 3}, "rows_marked": {"review": 3}}
        assert entry["duration_s"] >= 0 and entry["started_at"] <= entry["finished_at"]

    def test_tuple_params_are_saved_as_lists(self, dataset):
        check_decode(dataset, expected_modes=("RGB", "L"), num_workers=1)
        [entry] = checks(dataset)
        assert entry["params"]["expected_modes"] == ["RGB", "L"]

    def test_only_new_tags_are_counted(self, dataset):
        check_photometric(dataset, num_workers=1)
        dataset._manifest.loc[:, "action"] = ""  # make the rows eligible again, keeping their tags
        check_photometric(dataset, num_workers=1)
        second = checks(dataset)[-1]
        assert second["changes"]["tags_added"] == {}
        assert second["changes"]["rows_marked"] == {"review": 3}

    def test_grouping_step_records_its_stats(self, dataset):
        group_by_hash(dataset, seed=0)
        [entry] = checks(dataset)
        assert entry["params"] == {"group_keep_strategy": "random", "seed": 0}
        assert entry["stats"] == {"groups_found": 1, "images_in_groups": 2}
        assert entry["changes"]["tags_added"] == {"exact_duplicate": 1}
        assert entry["changes"]["rows_marked"] == {"drop": 1, "keep": 2}

    def test_grouping_that_finds_nothing_is_still_recorded(self, make_image_folder, tmp_path):
        """The manifest can't tell 'not run' from 'found nothing'; the audit log can."""
        root = make_image_folder({"A": [DARK, MID, LIGHT]})
        ds = ImageDataset(root, result_dir=str(tmp_path / "results"))
        group_by_hash(ds)
        [entry] = checks(ds)
        assert entry["stats"] == {"groups_found": 0, "images_in_groups": 0}
        assert entry["changes"] == {"tags_added": {}, "rows_marked": {}}

    def test_failure_is_recorded_and_reraised(self, dataset):
        @audited("exploding_check")
        def exploding_check(ds, level=3):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            exploding_check(dataset)
        [entry] = checks(dataset)
        assert entry["status"] == FAILED
        assert entry["error"] == "RuntimeError: boom"
        assert entry["params"] == {"level": 3}

    def test_records_environment_and_paths_digest(self, dataset):
        check_decode(dataset, num_workers=1)
        [entry] = checks(dataset)
        assert entry["paths_digest"] == paths_digest(dataset._manifest["path"])
        assert set(entry["environment"]) == {"git_commit", "git_dirty", "python", "platform", "machine", "packages"}


class TestAppendAndLoad:
    def test_missing_file_is_empty(self, tmp_path):
        assert load_audits(str(tmp_path / "audits.json")) == []

    def test_append_keeps_existing_entries(self, tmp_path):
        path = str(tmp_path / "audits.json")
        append_audit(path, {"check": "a"})
        append_audit(path, {"check": "b"})
        assert [e["check"] for e in load_audits(path)] == ["a", "b"]
        assert not [f for f in os.listdir(tmp_path) if f.startswith(".audits-")]

    def test_invalid_json_raises_and_is_not_overwritten(self, tmp_path):
        path = tmp_path / "audits.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            append_audit(str(path), {"check": "a"})
        assert path.read_text() == "{not json"

    def test_non_list_raises(self, tmp_path):
        path = tmp_path / "audits.json"
        path.write_text(json.dumps({"check": "a"}))
        with pytest.raises(ValueError, match="JSON list"):
            load_audits(str(path))


class TestCurrentAudits:
    def test_drops_entries_before_last_reset(self):
        entries = [{"check": "a"}, {"event": RESTART}, {"check": "b"}, {"event": MANIFEST_CREATED}, {"check": "c"}]
        assert current_audits(entries) == [{"event": MANIFEST_CREATED}, {"check": "c"}]

    def test_without_reset_keeps_everything(self):
        entries = [{"check": "a"}, {"check": "b"}]
        assert current_audits(entries) == entries


class TestLatestByCheck:
    def test_latest_and_history(self):
        entries = [{"check": "a", "n": 1}, {"check": "b", "n": 2}, {"check": "a", "n": 3}]
        result = latest_by_check(entries)
        assert result["a"] == {"latest": {"check": "a", "n": 3}, "history": [{"check": "a", "n": 1}]}
        assert result["b"] == {"latest": {"check": "b", "n": 2}, "history": []}

    def test_ignores_runs_before_reset(self):
        entries = [{"check": "a", "n": 1}, {"event": RESTART}, {"check": "b"}]
        assert set(latest_by_check(entries)) == {"b"}

    def test_failed_latest_is_kept(self):
        entries = [{"check": "a", "status": OK}, {"check": "a", "status": FAILED}]
        assert latest_by_check(entries)["a"]["latest"]["status"] == FAILED


class TestManifestChanges:
    def test_counts_added_tags_and_new_actions(self):
        before = pd.DataFrame({"action": ["", "review", "", ""], "reason": ["", "blurry", "", ""]})
        after = pd.DataFrame({"action": ["drop", "drop", "", "keep"], "reason": ["corrupt;blurry", "blurry;corrupt", "", ""]})
        assert manifest_changes(before, after) == {
            "tags_added": {"blurry": 1, "corrupt": 2},
            "rows_marked": {"drop": 2, "keep": 1},
        }


class TestPathsDigest:
    def test_order_independent(self):
        assert paths_digest(["a", "b"]) == paths_digest(["b", "a"])

    def test_changes_with_paths(self):
        assert paths_digest(["a", "b"]) != paths_digest(["a", "c"])
        assert paths_digest(["ab"]) != paths_digest(["a", "b"])


class TestEnvironmentInfo:
    def test_reports_git_and_packages(self):
        info = environment_info()
        assert info["git_commit"] is None or len(info["git_commit"]) == 40
        assert "pandas" in info["packages"] and info["packages"]["pandas"]


class TestDatasetFingerprint:
    def make(self, tmp_path, files):
        root = tmp_path / "Data"
        root.mkdir(exist_ok=True)
        for name, data in files.items():
            (root / name).write_bytes(data)
        manifest = pd.DataFrame({"path": [str(root / n) for n in files]})
        return manifest, str(root)

    def test_counts_and_method(self, tmp_path):
        manifest, root = self.make(tmp_path, {"a.png": b"aa", "b.png": b"bbb"})
        fp = dataset_fingerprint(manifest, root)
        assert (fp["method"], fp["file_count"], fp["total_bytes"], fp["missing_files"]) == ("size_mtime", 2, 5, 0)

    def test_uses_content_hash_when_every_row_has_one(self, tmp_path):
        manifest, root = self.make(tmp_path, {"a.png": b"aa"})
        manifest["hash"] = ["h1"]
        first = dataset_fingerprint(manifest, root)
        manifest["hash"] = ["h2"]
        assert first["method"] == "content"
        assert dataset_fingerprint(manifest, root)["fingerprint"] != first["fingerprint"]

    def test_partial_hash_column_falls_back(self, tmp_path):
        manifest, root = self.make(tmp_path, {"a.png": b"aa", "b.png": b"b"})
        manifest["hash"] = ["h1", ""]
        assert dataset_fingerprint(manifest, root)["method"] == "size_mtime"

    def test_independent_of_row_order_and_folder_location(self, tmp_path):
        manifest, root = self.make(tmp_path, {"a.png": b"aa", "b.png": b"b"})
        manifest["hash"] = ["h1", "h2"]
        moved = tmp_path / "moved"
        os.rename(root, moved)
        moved_manifest = pd.DataFrame({"path": [str(moved / "b.png"), str(moved / "a.png")], "hash": ["h2", "h1"]})
        # Same content and relative paths in another folder give the same fingerprint.
        fp_moved = dataset_fingerprint(moved_manifest, str(moved))
        os.rename(moved, root)
        assert fp_moved["fingerprint"] == dataset_fingerprint(manifest, root)["fingerprint"]

    def test_detects_changed_content(self, tmp_path):
        manifest, root = self.make(tmp_path, {"a.png": b"aa"})
        before = dataset_fingerprint(manifest, root)
        (tmp_path / "Data" / "a.png").write_bytes(b"aaaa")
        assert dataset_fingerprint(manifest, root)["fingerprint"] != before["fingerprint"]

    def test_missing_files_are_counted(self, tmp_path):
        manifest, root = self.make(tmp_path, {"a.png": b"aa"})
        os.remove(tmp_path / "Data" / "a.png")
        fp = dataset_fingerprint(manifest, root)
        assert (fp["file_count"], fp["missing_files"]) == (0, 1)
