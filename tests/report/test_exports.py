import os

import pandas as pd
import pytest

import src.report.exports as exports_module
from src.report.exports import (
    EXPORTS,
    drop_manifest,
    duplicate_groups,
    identity,
    metric_columns,
    ok_manifest,
    primary_tag,
    remove_exports,
    tag_metric,
    tag_severity,
    worst_first_key,
    representatives,
    review_manifest,
    row_severity,
    row_tags,
    write_csv,
    write_exports,
)
from src.report.status import resolve_status

DATA = "/data"


def make_manifest(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"action": "", "reason": "", **r} for r in rows])
    df["path"] = DATA + "/" + df["path"]
    if "group_id" not in df:
        df["group_id"] = [f"g{i}" for i in range(len(df))]
    return df.astype(str)


@pytest.fixture
def manifest():
    """
    a/1 is kept for a/2 (exact copy) and a/3 (near copy). Three blurry review rows, one of them also
    too_small. One blank review row. A corrupt drop. b/ok is unflagged.
    """
    return make_manifest([
        {"path": "a/1.png", "group_id": "G", "hash": "h", "action": "keep"},
        {"path": "a/2.png", "group_id": "G", "hash": "h", "action": "drop", "reason": "exact_duplicate"},
        {"path": "a/3.png", "group_id": "G", "hash": "h3", "action": "drop", "reason": "near_duplicate_phash_4"},
        {"path": "b/blur50.png", "group_id": "g3", "hash": "x3", "action": "review", "reason": "blurry", "blur_score": "38", "width": "64", "height": "64", "pixel_std": "0.2"},
        {"path": "b/blur10.png", "group_id": "g4", "hash": "x4", "action": "review", "reason": "blurry", "blur_score": "10", "width": "64", "height": "64", "pixel_std": "0.2"},
        {"path": "b/small.png", "group_id": "g5", "hash": "x5", "action": "review", "reason": "blurry;too_small", "blur_score": "5", "width": "20", "height": "30", "pixel_std": "0.2"},
        {"path": "b/blank.png", "group_id": "g6", "hash": "x6", "action": "review", "reason": "blank_or_solid", "blur_score": "0", "width": "64", "height": "64", "pixel_std": "0.0"},
        {"path": "c/bad.png", "group_id": "g7", "hash": "x7", "action": "drop", "reason": "corrupt", "is_decodable": "False", "decode_error": "OSError: truncated"},
        {"path": "b/ok.png", "group_id": "g8", "hash": "x8", "action": ""},
    ])


class TestRowTags:
    def test_most_serious_first_and_canonical(self):
        assert row_tags("exact_duplicate;blurry;corrupt") == ["corrupt", "blurry", "exact_duplicate"]
        assert row_tags("exact hash duplicates;exact_duplicate") == ["exact_duplicate"]

    def test_empty(self):
        assert row_tags("") == [] and row_tags(float("nan")) == []


class TestTagSeverity:
    def test_catalog_and_unknown(self):
        assert tag_severity("corrupt") == "error" and tag_severity("blurry") == "warning"
        assert tag_severity("near_duplicate_phash_4") == "info" and tag_severity("made_up") == "unknown"


class TestPrimaryTag:
    def test_most_serious_tag(self):
        assert primary_tag("blurry;too_small") == "too_small"  # same severity: catalog order
        assert primary_tag("exact_duplicate;corrupt") == "corrupt"
        assert primary_tag("") == ""


class TestRowSeverity:
    def test_severity_of_primary_tag(self):
        assert row_severity("exact_duplicate;blurry") == "warning"
        assert row_severity("made_up") == "unknown" and row_severity("") == ""


class TestTagMetric:
    def test_single_and_combined_columns(self, manifest):
        rows = manifest.loc[[3, 5]]
        assert tag_metric(rows, "blurry")[0] == "blur_score"
        assert tag_metric(rows, "blurry")[1].tolist() == [38.0, 5.0]
        name, values = tag_metric(rows, "too_small")
        assert name == "min(width,height)" and values.tolist() == [64.0, 20.0]

    def test_no_metric(self, manifest):
        name, values = tag_metric(manifest, "corrupt")
        assert name == "" and values.isna().all()

    def test_missing_column(self, manifest):
        name, values = tag_metric(manifest.drop(columns="blur_score"), "blurry")
        assert name == "" and values.isna().all()


class TestWorstFirstKey:
    def test_low_is_worse_kept_high_is_worse_negated(self):
        tags = pd.Series(["blurry", "underexposed", "corrupt"])
        values = pd.Series([5.0, 0.95, float("nan")])
        key = worst_first_key(tags, values)
        assert key.iloc[0] == 5.0 and key.iloc[1] == -0.95 and pd.isna(key.iloc[2])


class TestMetricColumns:
    def test_existing_catalog_columns_only(self, manifest):
        assert metric_columns(["blurry", "too_small", "exact_duplicate"], manifest) == ["width", "height", "blur_score"]


class TestRepresentatives:
    def test_first_ok_member_of_each_group(self, manifest):
        frame = identity(manifest, DATA)
        rep = representatives(manifest, frame["status"], frame["path"], "group_id")
        assert rep.iloc[:3].tolist() == ["a/1.png"] * 3
        assert (rep.iloc[3:] == "").all()

    def test_group_without_ok_member(self):
        manifest = make_manifest([{"path": "x", "group_id": "G", "action": "review", "reason": "blurry"},
                                  {"path": "y", "group_id": "G", "action": "drop", "reason": "exact_duplicate"}])
        frame = identity(manifest, DATA)
        assert representatives(manifest, frame["status"], frame["path"], "group_id").tolist() == ["", ""]


class TestIdentity:
    def test_columns_and_relative_paths(self, manifest):
        frame = identity(manifest, DATA)
        assert list(frame.columns) == ["path", "status", "reason", "severity"]
        assert frame["path"].iloc[0] == "a/1.png"
        assert frame.loc[7, "severity"] == "error"

    def test_label_when_present(self, manifest):
        manifest["label"] = "Happy"
        assert "label" in identity(manifest, DATA).columns


class TestOkManifest:
    def test_ok_rows_and_representatives(self, manifest):
        ok = ok_manifest(manifest, DATA)
        assert ok["path"].tolist() == ["a/1.png", "b/ok.png"]
        assert ok["is_duplicate_representative"].tolist() == [True, False]
        assert ok["duplicates_removed"].tolist() == [2, 0]


class TestReviewManifest:
    def test_grouped_by_primary_tag_and_worst_first(self, manifest):
        review = review_manifest(manifest, DATA)
        # error (blank) first, then too_small (catalog order before blurry), then blurry, lowest blur first.
        assert review["path"].tolist() == ["b/blank.png", "b/small.png", "b/blur10.png", "b/blur50.png"]
        assert review["tag"].tolist() == ["blank_or_solid", "too_small", "blurry", "blurry"]
        assert review.loc[1, "metric"] == "min(width,height)" and review.loc[1, "metric_value"] == 20
        assert review.loc[2, "metric"] == "blur_score" and review.loc[2, "metric_value"] == 10

    def test_high_is_worse_metric_sorts_descending(self):
        manifest = make_manifest([
            {"path": f"{i}.png", "action": "review", "reason": "underexposed", "dark_clip_ratio": v}
            for i, v in enumerate(["0.91", "0.99", "0.95"])
        ])
        assert review_manifest(manifest, DATA)["metric_value"].tolist() == [0.99, 0.95, 0.91]

    def test_includes_metric_columns_of_every_tag(self, manifest):
        review = review_manifest(manifest, DATA)
        assert {"pixel_std", "width", "height", "blur_score"} <= set(review.columns)

    def test_tag_without_sort_metric_sorts_by_path(self):
        manifest = make_manifest([{"path": p, "action": "review", "reason": "exif_rotated", "exif_orientation": "6"} for p in ["b.png", "a.png"]])
        review = review_manifest(manifest, DATA)
        assert review["path"].tolist() == ["a.png", "b.png"] and (review["metric"] == "").all()


class TestDropManifest:
    def test_severity_order_and_duplicate_of(self, manifest):
        drop = drop_manifest(manifest, DATA)
        assert drop["path"].tolist() == ["c/bad.png", "a/2.png", "a/3.png"]
        assert drop["severity"].tolist() == ["error", "info", "info"]
        assert drop["duplicate_of"].tolist() == ["", "a/1.png", "a/1.png"]
        assert drop.loc[0, "decode_error"] == "OSError: truncated"

    def test_paths_in_decode_errors_are_relative(self, manifest):
        manifest.loc[7, "decode_error"] = f"UnidentifiedImageError: cannot identify image file '{DATA}/c/bad.png'"
        drop = drop_manifest(manifest, DATA)
        assert drop.loc[0, "decode_error"] == "UnidentifiedImageError: cannot identify image file 'c/bad.png'"


class TestDuplicateGroups:
    def test_rows_per_member_and_column(self, manifest):
        groups = duplicate_groups(manifest, DATA)
        by_column = {c: g for c, g in groups.groupby("group_column")}
        assert set(by_column) == {"hash", "group_id"}
        assert by_column["group_id"]["path"].tolist() == ["a/1.png", "a/2.png", "a/3.png"]
        assert by_column["hash"]["group_size"].tolist() == [2, 2]
        assert by_column["group_id"]["is_representative"].tolist() == [True, False, False]
        assert set(by_column["group_id"]["representative_path"]) == {"a/1.png"}

    def test_largest_group_first(self):
        manifest = make_manifest([{"path": p, "group_id": g, "action": "keep"} for p, g in
                                  [("z1", "small"), ("z2", "small"), ("a1", "big"), ("a2", "big"), ("a3", "big")]])
        assert duplicate_groups(manifest, DATA)["group_size"].tolist() == [3, 3, 3, 2, 2]

    def test_no_groups(self):
        manifest = make_manifest([{"path": "a"}, {"path": "b"}])
        assert len(duplicate_groups(manifest, DATA)) == 0


class TestWriteCsv:
    def test_round_trip_no_index(self, tmp_path):
        path = tmp_path / "out" / "x.csv"
        write_csv(pd.DataFrame({"a": [1, 2]}), str(path))
        assert path.read_text().splitlines() == ["a", "1", "2"]


class TestWriteExports:
    def test_writes_every_file_and_counts_add_up(self, manifest, tmp_path):
        written = write_exports(manifest, str(tmp_path), DATA)
        assert [w["path"] for w in written] == list(EXPORTS)
        rows = {w["path"]: w["rows"] for w in written}
        assert rows["ok_manifest.csv"] + rows["review_manifest.csv"] + rows["drop_manifest.csv"] == len(manifest)
        for name, count in rows.items():
            saved = pd.read_csv(tmp_path / name, dtype=str, keep_default_na=False)
            assert len(saved) == count
        assert all(w["description"] for w in written)

    def test_csv_paths_are_relative(self, manifest, tmp_path):
        write_exports(manifest, str(tmp_path), DATA)
        for name in EXPORTS:
            assert DATA not in (tmp_path / name).read_text()

    def test_failure_removes_all_exports(self, manifest, tmp_path, monkeypatch):
        write_exports(manifest, str(tmp_path), DATA)

        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setitem(exports_module.BUILDERS, "drop_manifest.csv", broken)
        with pytest.raises(RuntimeError):
            write_exports(manifest, str(tmp_path), DATA)
        assert os.listdir(tmp_path) == []

    def test_does_not_modify_manifest(self, manifest, tmp_path):
        before = manifest.copy()
        write_exports(manifest, str(tmp_path), DATA)
        pd.testing.assert_frame_equal(manifest, before)

    def test_unknown_action_raises(self, manifest, tmp_path):
        manifest.loc[0, "action"] = "delete"
        with pytest.raises(ValueError, match="unknown action"):
            write_exports(manifest, str(tmp_path), DATA)


class TestRemoveExports:
    def test_removes_only_export_files(self, tmp_path):
        for name in [*EXPORTS, "validation_report.json"]:
            (tmp_path / name).write_text("x")
        remove_exports(str(tmp_path))
        assert os.listdir(tmp_path) == ["validation_report.json"]
