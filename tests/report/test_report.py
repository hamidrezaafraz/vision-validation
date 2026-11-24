import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

import src.report as report_module
import src.report.exports as exports_module
from src.report.exports import EXPORTS
from src.cli.lib import group_by_hash
from src.integrity import check_decode
from src.report import (
    REPORT_FILE,
    REPORT_SCHEMA_VERSION,
    ReportConfig,
    build_report,
    dataset_data_dir,
    generate_report,
    load_schema,
    relative_issues,
    report_dir,
    summarize_checks,
    to_jsonable,
    write_report,
)
from src.report.validate import ERROR, Issue, ManifestValidationError
from src.utils.dataset import ImageDataset

TOP_LEVEL_KEYS = {"schema_version", "generated_at", "run", "dataset", "valid", "issues", "status_counts",
                  "flags", "profile", "duplicates", "checks", "outputs"}


def strict_loads(text: str):
    """json.loads that rejects NaN / Infinity, like strict parsers do."""
    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")
    return json.loads(text, parse_constant=reject)


def schema_errors(report: dict) -> list[str]:
    return [f"{list(e.path)}: {e.message}" for e in Draft202012Validator(load_schema()).iter_errors(report)]


def digest(path) -> str:
    return hashlib.blake2b(open(path, "rb").read()).hexdigest()


@pytest.fixture
def dataset(make_image_folder, tmp_path):
    """Happy: two identical red images and a blue one; Sad: a red one (identical to Happy's) and a green one."""
    root = make_image_folder({"Happy": [(255, 0, 0), (255, 0, 0), (0, 0, 255)], "Sad": [(255, 0, 0), (0, 255, 0)]})
    ds = ImageDataset(root, str(tmp_path / "results"))
    check_decode(ds, min_width=4, min_height=4)
    group_by_hash(ds, seed=0)
    return ds


class TestToJsonable:
    def test_non_finite_floats_become_null(self):
        assert to_jsonable({"a": float("nan"), "b": float("inf"), "c": -np.inf, "d": 1.5}) == {"a": None, "b": None, "c": None, "d": 1.5}

    def test_numpy_types(self):
        result = to_jsonable({"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True), "a": np.array([1, 2]), "n": np.float64("nan")})
        assert result == {"i": 3, "f": 0.5, "b": True, "a": [1, 2], "n": None}
        assert type(result["i"]) is int and type(result["b"]) is bool

    def test_containers_keys_and_timestamps(self):
        when = datetime(2026, 1, 2, tzinfo=timezone.utc)
        result = to_jsonable({1: (1, 2), "s": {"b", "a"}, "t": pd.Timestamp(when), "d": when})
        assert result == {"1": [1, 2], "s": ["a", "b"], "t": when.isoformat(), "d": when.isoformat()}

    def test_result_is_strict_json(self):
        json.dumps(to_jsonable({"x": [np.nan, {"y": np.inf}]}), allow_nan=False)


class TestDatasetDataDir:
    def test_both_dataset_classes(self):
        class Unlabeled:
            data_dir = "a"
        class Labeled:
            root_dir = "b"
        assert dataset_data_dir(Unlabeled()) == "a"
        assert dataset_data_dir(Labeled()) == "b"

    def test_missing(self):
        with pytest.raises(ValueError, match="data_dir"):
            dataset_data_dir(object())


class TestSummarizeChecks:
    def entry(self, params, started, digest="d", status="ok"):
        return {"check": "c", "status": status, "params": params, "started_at": started, "paths_digest": digest,
                "environment": {"git_commit": "abc", "git_dirty": False}}

    def test_one_entry_per_distinct_params(self):
        manifest = pd.DataFrame({"path": ["a"]})
        checks = {"group_by_embeddings": {
            "latest": self.entry({"model_name": "facenet"}, "t3"),
            "history": [self.entry({"model_name": "alexnet"}, "t1"), self.entry({"model_name": "alexnet"}, "t2")],
        }}
        runs = summarize_checks(checks, manifest)["group_by_embeddings"]
        assert [(r["params"]["model_name"], r["started_at"], r["earlier_runs"]) for r in runs] == [
            ("alexnet", "t2", 1), ("facenet", "t3", 0),
        ]
        assert runs[0]["git_commit"] == "abc"

    def test_paths_digest_match(self):
        from src.utils.audit import paths_digest
        manifest = pd.DataFrame({"path": ["a", "b"]})
        checks = {"c": {"latest": self.entry({}, "t1", digest=paths_digest(["b", "a"])), "history": []},
                  "d": {"latest": self.entry({}, "t1", digest="other"), "history": []}}
        summary = summarize_checks(checks, manifest)
        assert summary["c"][0]["paths_digest_matches"] is True
        assert summary["d"][0]["paths_digest_matches"] is False


class TestRelativeIssues:
    def test_paths_become_relative_other_examples_stay(self, tmp_path):
        data = tmp_path / "data"
        manifest = pd.DataFrame({"path": [str(data / "a" / "1.png")]})
        issues = [Issue("missing_file", ERROR, "m", 1, [str(data / "a" / "1.png")]),
                  Issue("missing_columns", ERROR, "m", 1, ["label"])]
        result = relative_issues(issues, manifest, str(data), None)
        assert result[0]["examples"] == ["a/1.png"]
        assert result[1]["examples"] == ["label"]


class TestBuildReport:
    def test_top_level_keys_and_schema(self, dataset):
        report = build_report(dataset)
        assert set(report) == TOP_LEVEL_KEYS
        assert report["schema_version"] == REPORT_SCHEMA_VERSION
        assert schema_errors(report) == []

    def test_numbers(self, dataset):
        report = build_report(dataset)
        assert report["valid"] is True and report["issues"] == []
        assert report["dataset"]["file_count"] == 5 and report["dataset"]["method"] == "content"
        # 3 identical red images: 1 kept, 2 dropped as exact duplicates.
        assert report["status_counts"]["drop"]["count"] == 2
        assert report["flags"]["images_flagged"]["count"] == 2
        assert report["duplicates"]["columns"]["hash"]["redundant"]["count"] == 2
        assert report["duplicates"]["columns"]["hash"]["cross_folder"]["count"] == 1
        assert report["outputs"] == []

    def test_checks_and_run_block(self, dataset):
        report = build_report(dataset)
        assert set(report["checks"]) == {"check_decode", "group_by_hash"}
        decode = report["checks"]["check_decode"][0]
        assert decode["params"]["min_width"] == 4 and decode["paths_digest_matches"] is True
        assert report["checks"]["group_by_hash"][0]["stats"] == {"groups_found": 1, "images_in_groups": 3}
        assert report["run"]["last_reset"]["event"] == "manifest_created"
        assert "data_dir" not in report["run"]["config"]
        assert report["profile"]["thresholds"]["thresholds"][0]["parameter"] == "min_width"

    def test_strict_json_without_local_paths(self, dataset, tmp_path):
        text = json.dumps(build_report(dataset), allow_nan=False)
        strict_loads(text)
        assert str(tmp_path) not in text

    def test_errors_stop_the_report(self, dataset):
        dataset._manifest.loc[0, "action"] = "delete"
        with pytest.raises(ManifestValidationError, match="unknown_action"):
            build_report(dataset)

    def test_force_writes_invalid_report_with_failed_sections(self, dataset):
        dataset._manifest.loc[0, "action"] = "delete"
        report = build_report(dataset, ReportConfig(force=True))
        codes = {i["code"] for i in report["issues"]}
        assert report["valid"] is False
        assert {"unknown_action", "section_failed"} <= codes
        assert report["flags"] is None and report["status_counts"] is None and report["duplicates"] is None
        assert report["profile"] is not None
        assert schema_errors(report) == []
        json.dumps(report, allow_nan=False)

    def test_section_error_propagates_without_force(self, dataset, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(report_module, "dataset_profile", broken)
        with pytest.raises(RuntimeError, match="boom"):
            build_report(dataset)

    def test_is_read_only(self, dataset):
        manifest, audits = dataset._manifest_path, dataset._audit_path
        before = (digest(manifest), digest(audits))
        in_memory = dataset._manifest.copy()
        build_report(dataset)
        assert (digest(manifest), digest(audits)) == before
        pd.testing.assert_frame_equal(dataset._manifest, in_memory)

    def test_without_audits(self, dataset):
        os.remove(dataset._audit_path)
        report = build_report(dataset)
        assert report["checks"] == {} and report["run"]["last_reset"] is None
        assert report["profile"]["thresholds"]["thresholds"] == []
        assert schema_errors(report) == []

    def test_embeddings_without_indices_are_unavailable(self, dataset):
        np.save(os.path.join(dataset._embeddings_dir, "m_embeddings.npy"), np.eye(3, dtype=np.float32))
        unavailable = build_report(dataset)["duplicates"]["similarity_unavailable"]
        assert [u["model"] for u in unavailable] == ["m"]

    def test_embeddings_with_indices_give_similarity(self, dataset):
        kept = dataset._manifest.index[dataset._manifest["action"] != "drop"]
        np.save(os.path.join(dataset._embeddings_dir, "m_embeddings.npy"), np.ones((len(kept), 2), dtype=np.float32))
        np.save(os.path.join(dataset._embeddings_dir, "m_embeddings_indices.npy"), kept.to_numpy())
        duplicates = build_report(dataset)["duplicates"]
        assert duplicates["similarity_models"] == ["m"]
        top = duplicates["columns"]["hash"]["largest"][0]
        # Only the kept member of the red group has an embedding.
        assert top["similarity"] == {"model": "m", "members_with_embeddings": 1, "mean_pairwise_cosine": None}


class TestWriteReport:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "out" / "r.json"
        write_report({"x": float("nan"), "y": np.int64(1)}, str(path))
        assert strict_loads(path.read_text()) == {"x": None, "y": 1}

    def test_failed_write_keeps_old_file_and_no_temp(self, tmp_path, monkeypatch):
        path = tmp_path / "r.json"
        write_report({"old": True}, str(path))

        def broken(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(report_module.json, "dump", broken)
        with pytest.raises(OSError):
            write_report({"new": True}, str(path))
        assert json.loads(path.read_text()) == {"old": True}
        assert os.listdir(tmp_path) == ["r.json"]


class TestGenerateReport:
    def test_writes_report_under_result_dir(self, dataset):
        path = generate_report(dataset)
        assert path == os.path.join(dataset.result_dir, "report", REPORT_FILE)
        report = strict_loads(open(path).read())
        assert [o["path"] for o in report["outputs"]] == [
            "report/ok_manifest.csv", "report/review_manifest.csv", "report/drop_manifest.csv",
            "report/duplicate_groups.csv", "report/report.html", "report/report.md", f"report/{REPORT_FILE}",
        ]
        rows = {o["path"]: o["rows"] for o in report["outputs"]}
        assert rows["report/ok_manifest.csv"] + rows["report/review_manifest.csv"] + rows["report/drop_manifest.csv"] == 5
        assert rows[f"report/{REPORT_FILE}"] is None
        for output in report["outputs"]:
            assert os.path.exists(os.path.join(dataset.result_dir, output["path"]))
        assert schema_errors(report) == []

    def test_regenerating_overwrites(self, dataset):
        first = generate_report(dataset)
        second = generate_report(dataset)
        assert first == second
        assert sorted(os.listdir(os.path.dirname(first))) == sorted([REPORT_FILE, *EXPORTS, "report.html", "report.md"])

    def test_failed_exports_with_force(self, dataset, monkeypatch):
        generate_report(dataset)

        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setitem(exports_module.BUILDERS, "drop_manifest.csv", broken)
        path = generate_report(dataset, ReportConfig(force=True))
        report = strict_loads(open(path).read())
        assert report["valid"] is False
        assert "export_failed" in {i["code"] for i in report["issues"]}
        assert [o["path"] for o in report["outputs"]] == ["report/report.html", "report/report.md", f"report/{REPORT_FILE}"]
        # No stale CSVs from the previous run are left next to the new JSON.
        assert sorted(os.listdir(os.path.dirname(path))) == sorted([REPORT_FILE, "report.html", "report.md"])

    def test_failed_exports_without_force(self, dataset, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setitem(exports_module.BUILDERS, "drop_manifest.csv", broken)
        with pytest.raises(RuntimeError, match="boom"):
            generate_report(dataset)

    def test_is_read_only(self, dataset):
        before = (digest(dataset._manifest_path), digest(dataset._audit_path))
        generate_report(dataset)
        assert (digest(dataset._manifest_path), digest(dataset._audit_path)) == before


@pytest.fixture
def known(make_image_folder):
    """
    A manifest with known flags, built with apply_policy and group directly (no real checks):
    10 images, Happy/0000-0005 = rows 0-5, Sad/0000-0003 = rows 6-9.
    - rows 0, 1: blurry (review); row 1 also too_small; row 2: too_small (review)
    - row 9: corrupt (drop)
    - rows 3, 4, 6: one exact-duplicate group across Happy and Sad; one is kept, two are dropped
    - rows 5 ~ 7 ~ 8: an embedding chain (group_alexnet_0_95), so one group of 3 that drops nothing
    Expected: ok 4, review 3, drop 3; 6 images flagged; blurry 2, too_small 2, exact_duplicate 2, corrupt 1.
    """
    from src.utils.dataset import apply_policy
    root = make_image_folder({"Happy": [(i * 20, 0, 0) for i in range(6)], "Sad": [(0, i * 20, 0) for i in range(4)]})
    ds = ImageDataset(root, os.path.join(os.path.dirname(root), "results"))
    df = ds._manifest
    df["blur_score"], df["width"], df["height"], df["pixel_std"] = 100.0, 64, 64, 0.2
    df["decode_error"] = ""
    df.loc[0, "blur_score"], df.loc[1, "blur_score"] = 5.0, 10.0
    df.loc[1, "width"], df.loc[2, "width"] = 20, 10
    df.loc[9, "is_decodable"], df.loc[9, "decode_error"] = False, "OSError: truncated"

    apply_policy(df, df.index.isin([0, 1]), "review", "blurry")
    apply_policy(df, df.index.isin([1, 2]), "review", "too_small")
    apply_policy(df, df.index == 9, "drop", "corrupt")
    ds.group([[3, 4, 6]])
    ds.keep_one_from_group(reason="exact_duplicate", seed=0)
    chain = [[5, 7], [7, 8]]
    ds.group(chain, column="group_alexnet_0_95")
    ds.group(chain, column="split_group")
    ds.save_manifest()
    return ds


class TestGenerateReportKnownAnswers:
    def generate(self, known):
        path = generate_report(known, ReportConfig(max_expected_group_size=2))
        return json.load(open(path)), os.path.dirname(path)

    def test_valid(self, known):
        report, _ = self.generate(known)
        assert report["valid"] is True and report["issues"] == []
        assert schema_errors(report) == []

    def test_status_counts(self, known):
        report, _ = self.generate(known)
        assert {s: c["count"] for s, c in report["status_counts"].items()} == {"ok": 4, "review": 3, "drop": 3}

    def test_tag_counts_with_multi_tag_rows(self, known):
        flags = self.generate(known)[0]["flags"]
        assert {t["tag"]: t["count"] for t in flags["tags"]} == {"blurry": 2, "too_small": 2, "exact_duplicate": 2, "corrupt": 1}
        assert flags["images_flagged"]["count"] == 6  # less than the 7 tag counts: row 1 has two tags

    def test_cooccurrence(self, known):
        pairs = self.generate(known)[0]["flags"]["cooccurrence"]
        assert len(pairs) == 1 and pairs[0]["tags"] == ["blurry", "too_small"] and pairs[0]["count"] == 1
        assert pairs[0]["given"]["blurry"]["pct"] == 50.0

    def test_groups_and_chained_group(self, known):
        duplicates = self.generate(known)[0]["duplicates"]
        exact = duplicates["columns"]["group_id"]
        assert (exact["groups"], exact["redundant"]["count"]) == (1, 2)
        assert exact["cross_folder"]["count"] == 1
        chain = duplicates["columns"]["group_alexnet_0_95"]
        assert chain["size_histogram"] == [{"size": 3, "groups": 1}]
        assert chain["chaining"]["suspects"] == 1 and chain["largest"][0]["chaining_suspect"] is True
        assert chain["largest"][0]["status"] == {"ok": 3, "review": 0, "drop": 0}
        levels = {l["level"]: l for l in duplicates["levels"]}
        assert levels["perceptual"]["unique"]["count"] == 8
        assert (levels["alexnet_0_95"]["unique"]["count"], levels["alexnet_0_95"]["added_redundant"]) == (6, 2)
        assert duplicates["matches_split_group"] is True

    def test_csvs_add_up_and_match_the_json(self, known):
        report, out = self.generate(known)
        read = lambda name: pd.read_csv(os.path.join(out, name), dtype=str, keep_default_na=False)
        ok, review, drop = read("ok_manifest.csv"), read("review_manifest.csv"), read("drop_manifest.csv")
        assert len(ok) + len(review) + len(drop) == report["dataset"]["file_count"] == 10
        assert (len(ok), len(review), len(drop)) == tuple(report["status_counts"][s]["count"] for s in ("ok", "review", "drop"))
        rows = {o["path"]: o["rows"] for o in report["outputs"]}
        assert rows["report/review_manifest.csv"] == len(review)
        # too_small before blurry (catalog order), worst (narrowest) first
        assert review["path"].tolist() == ["Happy/0002.png", "Happy/0001.png", "Happy/0000.png"]
        assert drop["path"].iloc[0] == "Sad/0003.png" and drop["severity"].iloc[0] == "error"
        groups = read("duplicate_groups.csv")
        assert groups.groupby("group_column").size().to_dict() == {"group_alexnet_0_95": 3, "group_id": 3, "split_group": 6}

    def test_summary_states_the_json_numbers(self, known):
        _, out = self.generate(known)
        text = open(os.path.join(out, "report.md"), encoding="utf-8").read()
        assert "We received 10 files" in text
        assert "4 (40.0%) are usable as delivered" in text and "excluding 3 (30.0%)" in text
        assert "6 (60.0%) images carry at least one flag" in text

    def test_read_only(self, known):
        before = (digest(known._manifest_path), digest(known._audit_path))
        self.generate(known)
        assert (digest(known._manifest_path), digest(known._audit_path)) == before


class TestReportDir:
    def test_under_result_dir(self, dataset):
        assert report_dir(dataset) == os.path.join(dataset.result_dir, "report")


class TestLoadSchema:
    def test_schema_is_valid_and_versioned(self):
        schema = load_schema()
        Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema_version"]["const"] == REPORT_SCHEMA_VERSION
