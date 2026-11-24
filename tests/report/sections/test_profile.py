import json

import numpy as np
import pandas as pd
import pytest

from src.report.sections.profile import (
    FILE_SIZE,
    categorical_counts,
    dataset_profile,
    decodable_mask,
    distribution,
    extension_formats,
    file_sizes,
    folder_breakdown,
    last_ok_params,
    numeric_column,
    numeric_distributions,
    relative_folders,
    threshold_sensitivity,
    top_resolutions,
)


def make_manifest(rows: list[dict]) -> pd.DataFrame:
    """Every value as a string, like a manifest read with dtype=str, keep_default_na=False."""
    defaults = {"group_id": "", "action": "", "reason": ""}
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    df["group_id"] = [f"g{i}" for i in range(len(df))]
    return df.astype(str)


def checks(decode=None, photometric=None, status="ok"):
    """latest_by_check-shaped dict with one run per given check."""
    result = {}
    for name, params in (("check_decode", decode), ("check_photometric", photometric)):
        if params is not None:
            result[name] = {"latest": {"check": name, "status": status, "params": params, "finished_at": "t1"}, "history": []}
    return result


@pytest.fixture
def delivery(tmp_path):
    """data/a/1.png (10 B), data/a/2.jpg (20 B, really a PNG), data/b/c/3.jpg (30 B), data/4.png (corrupt, 5 B)."""
    data = tmp_path / "data"
    files = {"a/1.png": 10, "a/2.jpg": 20, "b/c/3.jpg": 30, "4.png": 5}
    for rel, size in files.items():
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(b"x" * size)
    manifest = make_manifest([
        {"path": str(data / "a/1.png"), "is_decodable": "True", "format": "PNG", "color_mode": "RGB",
         "width": "100", "height": "50", "aspect_ratio": "2.0", "blur_score": "38", "pixel_std": "0.3",
         "action": "review", "reason": "blurry"},
        {"path": str(data / "a/2.jpg"), "is_decodable": "True", "format": "PNG", "color_mode": "L",
         "width": "100", "height": "50", "aspect_ratio": "2.0", "blur_score": "45", "pixel_std": "0.3"},
        {"path": str(data / "b/c/3.jpg"), "is_decodable": "True", "format": "JPEG", "color_mode": "RGB",
         "width": "30", "height": "30", "aspect_ratio": "1.0", "blur_score": "1000", "pixel_std": "0.01",
         "action": "review", "reason": "too_small;blank_or_solid"},
        {"path": str(data / "4.png"), "is_decodable": "False", "format": "invalid", "color_mode": "invalid",
         "width": "0", "height": "0", "aspect_ratio": "inf", "blur_score": "", "pixel_std": "",
         "action": "drop", "reason": "corrupt"},
    ])
    return manifest, str(data)


class TestDecodableMask:
    def test_only_explicit_false_is_undecodable(self):
        manifest = make_manifest([{"path": p, "is_decodable": v} for p, v in
                                  [("a", "True"), ("b", "False"), ("c", ""), ("d", "0")]])
        assert decodable_mask(manifest).tolist() == [True, False, True, False]

    def test_missing_column_means_all_decodable(self):
        assert decodable_mask(make_manifest([{"path": "a"}])).tolist() == [True]


class TestNumericColumn:
    def test_empty_text_and_inf_become_nan(self):
        manifest = make_manifest([{"path": str(i), "x": v} for i, v in enumerate(["1.5", "", "abc", "inf"])])
        values = numeric_column(manifest, "x")
        assert values.iloc[0] == 1.5 and values.iloc[1:].isna().all()


class TestFileSizes:
    def test_sizes_and_missing(self, delivery):
        manifest, _ = delivery
        manifest = pd.concat([manifest, make_manifest([{"path": "nowhere.png"}])], ignore_index=True)
        sizes = file_sizes(manifest)
        assert sizes.iloc[:4].tolist() == [10, 20, 30, 5]
        assert np.isnan(sizes.iloc[4])

    def test_resolves_relative_paths_against_base_dir(self, delivery, tmp_path):
        manifest = make_manifest([{"path": "data/a/1.png"}])
        assert file_sizes(manifest, base_dir=str(tmp_path)).tolist() == [10]


class TestRelativeFolders:
    def test_folders_relative_to_data_dir(self, delivery):
        manifest, data = delivery
        assert relative_folders(manifest, data).tolist() == ["a", "a", "b/c", "."]


class TestLastOkParams:
    def test_skips_failed_latest_run(self):
        entries = {"check_decode": {
            "latest": {"status": "failed", "params": {"min_width": 99}, "finished_at": "t2"},
            "history": [{"status": "ok", "params": {"min_width": 10}, "finished_at": "t0"},
                        {"status": "ok", "params": {"min_width": 32}, "finished_at": "t1"}],
        }}
        assert last_ok_params(entries, "check_decode") == ({"min_width": 32}, "t1")

    def test_missing_check(self):
        assert last_ok_params({}, "check_decode") == (None, None)
        assert last_ok_params(None, "check_decode") == (None, None)


class TestDistribution:
    def test_summary_and_histogram(self):
        result = distribution(pd.Series([float(x) for x in range(1, 101)]), bins=4)
        assert (result["count"], result["min"], result["max"], result["mean"]) == (100, 1.0, 100.0, 50.5)
        assert result["median"] == 50.5
        assert result["histogram"]["counts"] == [25, 25, 25, 25]
        assert len(result["histogram"]["edges"]) == 5

    def test_ignores_nan(self):
        assert distribution(pd.Series([1.0, np.nan, 3.0]))["count"] == 2

    def test_empty(self):
        result = distribution(pd.Series([], dtype=float))
        assert result["count"] == 0 and result["median"] is None
        assert result["histogram"] == {"edges": [], "counts": []}

    def test_single_value(self):
        result = distribution(pd.Series([7.0]), bins=3)
        assert result["min"] == result["max"] == 7.0
        assert sum(result["histogram"]["counts"]) == 1


class TestNumericDistributions:
    def test_excludes_undecodable_rows_but_not_their_file_size(self, delivery):
        manifest, _ = delivery
        result = numeric_distributions(manifest, file_sizes(manifest))
        metrics = result["metrics"]
        assert metrics["width"]["count"] == 3 and metrics["width"]["min"] == 30.0
        assert metrics["aspect_ratio"]["max"] == 2.0  # the corrupt row's inf is not in it
        assert metrics[FILE_SIZE]["count"] == 4
        assert "megapixels" in result["unavailable"]


class TestCategoricalCounts:
    def test_counts_over_decodable_rows(self, delivery):
        manifest, _ = delivery
        result = categorical_counts(manifest)
        modes = result["metrics"]["color_mode"]
        assert [(v["value"], v["count"], v["of"]) for v in modes["values"]] == [("RGB", 2, 3), ("L", 1, 3)]
        assert modes["missing"]["count"] == 0
        assert "bit_depth" in result["unavailable"]

    def test_empty_values_are_missing(self):
        manifest = make_manifest([{"path": "a", "color_mode": "RGB"}, {"path": "b", "color_mode": ""}])
        assert categorical_counts(manifest)["metrics"]["color_mode"]["missing"] == {"count": 1, "of": 2, "pct": 50.0}


class TestExtensionFormats:
    def test_finds_png_named_jpg(self, delivery):
        manifest, _ = delivery
        result = extension_formats(manifest)
        assert result["mismatched"] == {"count": 1, "of": 3, "pct": 33.33}
        assert result["mismatched_examples"] == [manifest.loc[1, "path"]]
        assert {"extension": "png", "format": "PNG", "count": 1} in result["table"]

    def test_unknown_format_is_not_guessed(self):
        manifest = make_manifest([{"path": "x.weird", "format": "XYZ"}])
        result = extension_formats(manifest)
        assert result["mismatched"]["of"] == 0
        assert result["unchecked_formats"] == ["XYZ"]

    def test_extension_case_is_ignored(self):
        manifest = make_manifest([{"path": "x.JPG", "format": "JPEG"}])
        assert extension_formats(manifest)["mismatched"]["count"] == 0

    def test_older_manifest_without_format(self):
        assert extension_formats(make_manifest([{"path": "a.png"}])) == {"available": False}


class TestTopResolutions:
    def test_most_common_first(self, delivery):
        manifest, _ = delivery
        result = top_resolutions(manifest)
        assert result["distinct"] == 2
        assert [(r["resolution"], r["count"], r["of"]) for r in result["top"]] == [("100x50", 2, 3), ("30x30", 1, 3)]

    def test_top_limits_list(self, delivery):
        manifest, _ = delivery
        assert len(top_resolutions(manifest, top=1)["top"]) == 1


class TestThresholdSensitivity:
    def test_borderline_split(self, delivery):
        manifest, _ = delivery
        result = threshold_sensitivity(manifest, checks(photometric={"min_blur_score": 40.0, "min_std": 0.02}))
        blur = next(t for t in result["thresholds"] if t["parameter"] == "min_blur_score")
        # 38 is flagged and borderline, 45 passes and is outside [36, 44], 1000 is excluded (blank frame).
        assert blur["flagged"] == {"count": 1, "of": 2, "pct": 50.0}
        assert blur["borderline_range"] == [36.0, 44.0]
        assert (blur["borderline"]["count"], blur["borderline_flagged"], blur["borderline_passing"]) == (1, 1, 0)
        assert blur["excluded_by"] == "min_std"

    def test_blank_frames_are_flagged_by_min_std(self, delivery):
        manifest, _ = delivery
        result = threshold_sensitivity(manifest, checks(photometric={"min_blur_score": 40.0, "min_std": 0.02}))
        std = next(t for t in result["thresholds"] if t["parameter"] == "min_std")
        assert std["flagged"]["count"] == 1 and std["excluded_by"] is None

    def test_min_and_max_directions(self, delivery):
        manifest, _ = delivery
        result = threshold_sensitivity(manifest, checks(decode={"min_width": 32, "max_width": 99}))
        by_param = {t["parameter"]: t for t in result["thresholds"]}
        assert by_param["min_width"]["flagged"]["count"] == 1   # 30 < 32
        assert by_param["max_width"]["flagged"]["count"] == 2   # 100 > 99
        assert by_param["max_width"]["borderline_flagged"] == 2

    def test_unavailable_without_audit(self, delivery):
        manifest, _ = delivery
        result = threshold_sensitivity(manifest, None)
        assert result["thresholds"] == []
        assert all("no successful" in u["reason"] for u in result["unavailable"])

    def test_failed_run_is_not_used(self, delivery):
        manifest, _ = delivery
        result = threshold_sensitivity(manifest, checks(decode={"min_width": 32}, status="failed"))
        assert not any(t["check"] == "check_decode" for t in result["thresholds"])

    def test_missing_parameter_and_column(self, delivery):
        manifest, _ = delivery
        result = threshold_sensitivity(manifest, checks(decode={"min_width": 32}, photometric={"min_std": 0.02, "max_clip_ratio": 0.9}))
        reasons = {(u["parameter"], u["column"]): u["reason"] for u in result["unavailable"]}
        assert "not in the check_decode audit entry" in reasons[("min_height", "height")]
        assert "missing from the manifest" in reasons[("max_clip_ratio", "dark_clip_ratio")]


class TestFolderBreakdown:
    def test_folders_depths_and_concentration(self, delivery):
        manifest, data = delivery
        result = folder_breakdown(manifest, data)
        folders = {f["folder"]: f for f in result["folders"]}
        assert result["folders"][0]["folder"] == "a"
        assert folders["a"]["count"] == 2
        assert folders["a"]["flagged"] == {"count": 1, "of": 2, "pct": 50.0}
        assert folders["a"]["top_tag"] == {"tag": "blurry", "count": 1}
        assert [(d["depth"], d["count"]) for d in result["depths"]] == [(0, 1), (1, 2), (2, 1)]
        concentration = {c["tag"]: c for c in result["tag_concentration"]}
        assert concentration["too_small"]["folder"] == "b/c"
        assert concentration["too_small"]["pct"] == 100.0

    def test_folder_without_flags(self, delivery):
        manifest, data = delivery
        manifest.loc[:, "reason"] = ""
        manifest.loc[:, "action"] = ""
        folders = folder_breakdown(manifest, data)["folders"]
        assert all(f["top_tag"] is None and f["flagged"]["count"] == 0 for f in folders)


class TestDatasetProfile:
    def test_section(self, delivery):
        manifest, data = delivery
        section = dataset_profile(manifest, data, checks=checks(decode={"min_width": 32}))
        assert section["total_images"] == 4
        assert section["files"] == {"file_count": 4, "total_bytes": 65, "missing_files": 0}
        assert section["undecodable"]["count"] == 1
        assert {"extensions", "categorical", "distributions", "resolutions", "thresholds", "folders"} <= section.keys()

    def test_is_strict_json(self, delivery):
        manifest, data = delivery
        json.dumps(dataset_profile(manifest, data, checks=checks(decode={"min_width": 32})), allow_nan=False)

    def test_does_not_modify_manifest(self, delivery):
        manifest, data = delivery
        before = manifest.copy()
        dataset_profile(manifest, data)
        pd.testing.assert_frame_equal(manifest, before)

    def test_empty_manifest(self, tmp_path):
        manifest = make_manifest([]).reindex(columns=["path", "group_id", "action", "reason"])
        section = dataset_profile(manifest, str(tmp_path))
        assert section["total_images"] == 0
        json.dumps(section, allow_nan=False)
