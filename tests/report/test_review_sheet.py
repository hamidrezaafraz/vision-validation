import base64
import io
import json
import os
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.report as report_module
from src.cli.lib import group_by_hash
from src.integrity import check_decode
from src.photometric import check_photometric
from src.report import REPORT_FILE, ReportConfig, build_report, generate_report
from src.report.review_sheet import (
    REVIEW_SHEET_FILE,
    THUMB_SIZE,
    _rows,
    baseline_section,
    build_review_sheet,
    group_sections,
    make_thumbnails,
    render_html,
    tag_sections,
    thumbnail_uri,
    write_review_sheet,
)
from src.utils.dataset import ImageDataset

DATA = "/data"


def decode_uri(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def make_manifest(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"action": "", "reason": "", **r} for r in rows])
    df["path"] = DATA + "/" + df["path"]
    if "group_id" not in df:
        df["group_id"] = [f"g{i}" for i in range(len(df))]
    return df.astype(str)


@pytest.fixture
def dataset(tmp_path):
    """
    Data/Happy: n0 and its copy n0_copy (exact duplicates), n1, and a gray solid frame (blank_or_solid);
    Data/Sad: another copy of n0 and n2. The noise images are sharp and unflagged.
    """
    rng = np.random.default_rng(0)
    noise = [Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for _ in range(3)]
    files = {"Happy/n0.png": noise[0], "Happy/n0_copy.png": noise[0], "Happy/n1.png": noise[1],
             "Happy/gray.png": Image.new("RGB", (64, 64), (128, 128, 128)), "Sad/n0.png": noise[0], "Sad/n2.png": noise[2]}
    root = tmp_path / "Data"
    for rel, img in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(root / rel)
    ds = ImageDataset(str(root), str(tmp_path / "results"))
    check_decode(ds, min_width=4, min_height=4)
    check_photometric(ds)
    group_by_hash(ds, seed=0)
    return ds


class TestThumbnailUri:
    def test_shrinks_large_images(self, tmp_path):
        path = tmp_path / "big.png"
        Image.new("RGB", (400, 200), (10, 20, 30)).save(path)
        thumb = decode_uri(thumbnail_uri(str(path)))
        assert thumb.format == "JPEG" and thumb.size == (THUMB_SIZE, THUMB_SIZE // 2)

    def test_never_enlarges(self, tmp_path):
        path = tmp_path / "small.png"
        Image.new("L", (48, 48), 100).save(path)
        assert decode_uri(thumbnail_uri(str(path))).size == (48, 48)

    @pytest.mark.parametrize("mode, value", [("RGBA", (255, 0, 0, 0)), ("LA", (10, 128)), ("I;16", 40000), ("P", 3)])
    def test_any_mode_gives_rgb_jpeg(self, tmp_path, mode, value):
        path = tmp_path / f"img_{mode.replace(';', '')}.png"
        Image.new(mode, (8, 8), value).save(path)
        thumb = decode_uri(thumbnail_uri(str(path)))
        assert thumb.mode == "RGB"

    def test_transparent_pixels_become_white(self, tmp_path):
        path = tmp_path / "clear.png"
        Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(path)
        assert np.asarray(decode_uri(thumbnail_uri(str(path)))).min() > 240

    def test_undecodable_is_none(self, tmp_path):
        path = tmp_path / "junk.png"
        path.write_bytes(b"not an image")
        assert thumbnail_uri(str(path)) is None
        assert thumbnail_uri(str(tmp_path / "missing.png")) is None


class TestMakeThumbnails:
    def test_one_entry_per_distinct_path(self, tmp_path):
        path = tmp_path / "a.png"
        Image.new("RGB", (8, 8)).save(path)
        thumbs = make_thumbnails([str(path), str(path), str(tmp_path / "missing.png")])
        assert len(thumbs) == 2 and thumbs[str(tmp_path / "missing.png")] is None


class TestTagSections:
    def manifest(self):
        return make_manifest([
            {"path": "b40.png", "action": "review", "reason": "blurry", "blur_score": "40"},
            {"path": "b5.png", "action": "review", "reason": "blurry;exact_duplicate", "blur_score": "5"},
            {"path": "b20.png", "action": "drop", "reason": "blurry", "blur_score": "20"},
            {"path": "bad.png", "action": "drop", "reason": "corrupt"},
            {"path": "dup.png", "action": "drop", "reason": "exact_duplicate"},
        ])

    def test_one_section_per_non_duplicate_tag_most_severe_first(self):
        manifest = self.manifest()
        sections = tag_sections(manifest, _rows(manifest, DATA, None))
        assert [s["tag"] for s in sections] == ["corrupt", "blurry"]
        assert sections[0]["severity"] == "error" and sections[0]["description"]

    def test_worst_first_any_status(self):
        manifest = self.manifest()
        blurry = tag_sections(manifest, _rows(manifest, DATA, None))[1]
        assert [i["rel_path"] for i in blurry["items"]] == ["b5.png", "b20.png", "b40.png"]
        assert blurry["items"][0]["caption"] == "blur_score = 5"
        assert blurry["items"][1]["status"] == "drop"

    def test_cap_keeps_total(self):
        manifest = self.manifest()
        blurry = tag_sections(manifest, _rows(manifest, DATA, None), max_per_section=1)[1]
        assert len(blurry["items"]) == 1 and blurry["total"] == 3


class TestGroupSections:
    def group(self, members, kept, size=None, suspect=False):
        return {"size": size or len(members), "folders": {"a": len(members)}, "status": {}, "kept": kept,
                "members": members, "members_truncated": False, "chaining_suspect": suspect, "similarity": None}

    def duplicates(self, largest, suspects=()):
        return {"columns": {"group_id": {"kind": "duplicate", "groups": len(largest), "largest": largest,
                                         "chaining": {"suspects": len(suspects), "groups": list(suspects)}}}}

    def rows(self, paths):
        manifest = make_manifest([{"path": p} for p in paths])
        return _rows(manifest, DATA, None)

    def test_kept_member_first_and_marked(self):
        sections = group_sections(self.duplicates([self.group(["a/1", "a/2"], kept=["a/2"])]), self.rows(["a/1", "a/2"]))
        items = sections[0]["groups"][0]["items"]
        assert [(i["rel_path"], i["kept"]) for i in items] == [("a/2", True), ("a/1", False)]
        assert items[0]["abs_path"] == os.path.abspath(f"{DATA}/a/2")

    def test_suspects_first_without_repeats(self):
        big = self.group(["a/1", "a/2", "a/3"], kept=[], suspect=True)
        small = self.group(["a/4", "a/5"], kept=["a/4"])
        groups = group_sections(self.duplicates([big, small], suspects=[big]), self.rows([f"a/{i}" for i in range(1, 6)]))[0]["groups"]
        assert [g["size"] for g in groups] == [3, 2]
        assert groups[0]["chaining_suspect"] is True

    def test_caps(self):
        groups = [self.group([f"a/{g}{i}" for i in range(20)], kept=[], size=30) for g in "abcdefg"]
        paths = [f"a/{g}{i}" for g in "abcdefg" for i in range(20)]
        section = group_sections(self.duplicates(groups), self.rows(paths), max_groups=2, max_members=5)[0]
        assert len(section["groups"]) == 2
        assert len(section["groups"][0]["items"]) == 5 and section["groups"][0]["hidden"] == 25

    def test_no_duplicates(self):
        assert group_sections(None, self.rows(["a"])) == []
        empty = {"columns": {"hash": {"kind": "exact", "groups": 0, "largest": [], "chaining": {"suspects": 0, "groups": []}}}}
        assert group_sections(empty, self.rows(["a"])) == []


class TestBaselineSection:
    def test_ok_rows_only_and_deterministic(self):
        manifest = make_manifest([{"path": f"{i:02d}.png", "action": "drop" if i % 2 else "keep",
                                   "reason": "exact_duplicate" if i % 2 else ""} for i in range(40)])
        rows = _rows(manifest, DATA, None)
        first, second = baseline_section(rows, 5, seed=1), baseline_section(rows, 5, seed=1)
        assert first == second and first["total"] == 20
        assert all(i["status"] == "ok" for i in first["items"])
        assert [i["rel_path"] for i in first["items"]] == sorted(i["rel_path"] for i in first["items"])
        assert baseline_section(rows, 5, seed=2)["items"] != first["items"]

    def test_fewer_ok_rows_than_the_cap(self):
        rows = _rows(make_manifest([{"path": "a.png"}]), DATA, None)
        assert len(baseline_section(rows, 60)["items"]) == 1


class TestRenderHtml:
    def page(self, rel_path="a.png", thumbs=None):
        item = {"rel_path": rel_path, "abs_path": "/x/a.png", "status": "review", "caption": "blur_score = 5"}
        tags = [{"tag": "blurry", "severity": "warning", "description": "d", "total": 1, "metric": "blur_score", "items": [item]}]
        baseline = {"total": 0, "seed": 0, "items": []}
        return render_html(tags, [], baseline, thumbs or {}, {"generated_at": "t", "fingerprint": "f", "file_count": 1})

    def test_escapes_paths(self):
        page = self.page(rel_path='<script>alert("x")</script>.png')
        assert "<script>" not in page and "&lt;script&gt;" in page

    def test_self_contained(self):
        page = self.page(thumbs={"/x/a.png": "data:image/jpeg;base64,AAAA"})
        assert not re.search(r'(src|href)="https?://', page)
        assert "<script" not in page and "data:image/jpeg;base64,AAAA" in page

    def test_missing_thumbnail_placeholder(self):
        assert "could not be decoded" in self.page()


class TestBuildReviewSheet:
    def test_real_dataset(self, dataset):
        manifest = dataset._manifest.copy()
        page, thumbnails = build_review_sheet(manifest, build_report(dataset), dataset.data_dir)
        assert thumbnails == page.count("<figure")
        assert "blank_or_solid" in page and 'class="tile kept"' in page
        assert dataset.data_dir not in page

    def test_does_not_modify_manifest(self, dataset):
        before = dataset._manifest.copy()
        build_review_sheet(dataset._manifest, build_report(dataset), dataset.data_dir)
        pd.testing.assert_frame_equal(dataset._manifest, before)


class TestWriteReviewSheet:
    def test_writes_utf8(self, tmp_path):
        path = tmp_path / "out" / "s.html"
        write_review_sheet("<p>é</p>", str(path))
        assert path.read_text(encoding="utf-8") == "<p>é</p>"


class TestGenerateReportReviewSheet:
    def outputs(self, path):
        return {o["path"]: o for o in json.load(open(path))["outputs"]}

    def test_off_by_default(self, dataset):
        path = generate_report(dataset)
        assert REVIEW_SHEET_FILE not in os.listdir(os.path.dirname(path))

    def test_on(self, dataset):
        path = generate_report(dataset, ReportConfig(review_sheet=True, max_per_section=2))
        assert os.path.exists(os.path.join(os.path.dirname(path), REVIEW_SHEET_FILE))
        entry = self.outputs(path)[f"report/{REVIEW_SHEET_FILE}"]
        assert entry["rows"] > 0 and entry["description"]

    def test_turning_it_off_removes_the_old_sheet(self, dataset):
        generate_report(dataset, ReportConfig(review_sheet=True))
        path = generate_report(dataset)
        assert REVIEW_SHEET_FILE not in os.listdir(os.path.dirname(path))

    def test_failure_with_force(self, dataset, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(report_module, "build_review_sheet", broken)
        path = generate_report(dataset, ReportConfig(review_sheet=True, force=True))
        report = json.load(open(path))
        assert report["valid"] is False
        assert "review_sheet_failed" in {i["code"] for i in report["issues"]}
        assert f"report/{REVIEW_SHEET_FILE}" not in self.outputs(path)
        assert REVIEW_SHEET_FILE not in os.listdir(os.path.dirname(path))
        assert os.path.exists(os.path.join(os.path.dirname(path), REPORT_FILE))

    def test_failure_without_force(self, dataset, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(report_module, "build_review_sheet", broken)
        with pytest.raises(RuntimeError, match="boom"):
            generate_report(dataset, ReportConfig(review_sheet=True))
