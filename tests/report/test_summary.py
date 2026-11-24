import json
import os
import re

import numpy as np
import pytest
from PIL import Image

import src.report as report_module
from src.cli.lib import group_by_hash
from src.integrity import check_decode
from src.photometric import check_photometric
from src.report import ReportConfig, build_report, generate_report, to_jsonable
from src.report.summary import (
    SUMMARY_HTML,
    SUMMARY_MD,
    build_blocks,
    fmt_bytes,
    fmt_folder,
    fmt_int,
    fmt_num,
    fmt_params,
    fmt_pct,
    fmt_share,
    histogram_svg,
    remove_summary,
    render_html,
    render_markdown,
    write_summary,
)
from src.utils.dataset import ImageDataset


@pytest.fixture
def dataset(tmp_path):
    """Two exact copies of a noise image in different folders, one more noise image, one gray blank frame."""
    rng = np.random.default_rng(0)
    noise = [Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for _ in range(2)]
    files = {"Happy/n0.png": noise[0], "Sad/n0.png": noise[0], "Happy/n1.png": noise[1],
             "Happy/gray.png": Image.new("RGB", (64, 64), (128, 128, 128))}
    root = tmp_path / "Data"
    for rel, img in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(root / rel)
    ds = ImageDataset(str(root), str(tmp_path / "results"))
    check_decode(ds, min_width=4, min_height=4)
    check_photometric(ds)
    group_by_hash(ds, seed=0)
    return ds


@pytest.fixture
def report(dataset):
    return to_jsonable(build_report(dataset))


def text_of(blocks) -> str:
    return render_markdown(blocks)


class TestFormatting:
    def test_int_pct_share(self):
        assert fmt_int(59100) == "59,100" and fmt_int(None) == "–"
        assert fmt_pct(52.04) == "52.0%"
        assert fmt_pct(0.0, count=1) == "<0.1%" and fmt_pct(0.0, count=0) == "0.0%"
        assert fmt_share({"count": 30755, "of": 59100, "pct": 52.04}) == "30,755 (52.0%)"
        assert fmt_share(None) == "–"

    def test_num(self):
        assert fmt_num(32608.51) == "32,609"
        assert fmt_num(8192.0) == "8,192"
        assert fmt_num(0.21104) == "0.211"
        assert fmt_num(2.80213) == "2.802"
        assert fmt_num(None) == "–"

    def test_bytes_params_folder(self):
        assert fmt_bytes(733890704) == "733.9 MB" and fmt_bytes(500) == "500 B"
        assert fmt_params({"a": 1, "b": "x"}) == "a=1, b=x" and fmt_params({}) == "defaults"
        assert fmt_folder(".") == "(top level)" and fmt_folder("Happy") == "Happy"


class TestHistogramSvg:
    histogram = {"edges": [0.0, 10.0, 20.0, 30.0], "counts": [5, 0, 2]}

    def test_one_bar_per_bin(self):
        assert histogram_svg(self.histogram).count('<rect class="bar"') == 3

    def test_threshold_line_only_in_range(self):
        inside = {"threshold": 15.0, "parameter": "min_blur_score"}
        outside = {"threshold": 99.0, "parameter": "min_blur_score"}
        assert 'class="threshold"' in histogram_svg(self.histogram, inside)
        assert 'class="threshold"' not in histogram_svg(self.histogram, outside)

    def test_empty(self):
        assert histogram_svg({"edges": [], "counts": []}) == ""


class TestBuildBlocks:
    def test_sections_and_numbers(self, report):
        text = text_of(build_blocks(report))
        for heading in ("## Summary", "## Findings", "## What is in the delivery", "## Duplicates",
                        "## Methods and thresholds", "## Caveats"):
            assert heading in text
        assert "We received 4 files" in text
        assert "blank_or_solid" in text and "exact_duplicate" in text
        assert "min_blur_score (blur_score)" in text

    def test_descriptions_come_from_the_catalog(self, report):
        from src.report.reasons import get_reason
        assert get_reason("blank_or_solid").description in text_of(build_blocks(report))

    def test_no_banner_when_valid(self, report):
        assert not any(b[0] == "banner" for b in build_blocks(report))

    def test_banner_when_invalid(self, report):
        report["valid"] = False
        report["issues"] = [{"code": "unknown_action", "severity": "error", "message": "bad", "count": 1, "examples": []},
                            {"code": "orphaned_duplicate", "severity": "warning", "message": "look", "count": 1, "examples": []}]
        blocks = build_blocks(report)
        banner = next(b for b in blocks if b[0] == "banner")
        assert banner[2] == ["unknown_action: bad"]
        assert "orphaned_duplicate: look" in text_of(blocks)  # warnings go to the caveats

    def test_missing_sections(self, report):
        for key in ("flags", "profile", "duplicates", "status_counts"):
            report[key] = None
        report["valid"] = False
        text = text_of(build_blocks(report))
        assert "could not be built" in text

    def test_does_not_change_the_report(self, report):
        before = json.dumps(report, sort_keys=True)
        build_blocks(report)
        assert json.dumps(report, sort_keys=True) == before


class TestRenderHtml:
    def test_filled_template_self_contained(self, report):
        page = render_html(build_blocks(report))
        assert page.startswith("<!doctype html>") and "$body" not in page and "$title" not in page
        assert "<script" not in page and not re.search(r'(src|href)="https?://', page)
        assert "<svg" in page

    def test_escapes_text(self):
        page = render_html([("p", "<b>x</b> & y"), ("table", ["<h>"], [["<c>"]], set())])
        assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in page and "&lt;c&gt;" in page


class TestRenderMarkdown:
    def test_table_alignment_and_pipes(self):
        text = render_markdown([("table", ["Name", "Count"], [["a|b", "1"]], {1})])
        assert "| Name | Count |\n| --- | ---: |\n| a\\|b | 1 |" in text

    def test_charts_are_referred_to_html(self, report):
        assert "Histograms are in the HTML version" in render_markdown(build_blocks(report))


class TestWriteSummary:
    def test_writes_both_files(self, report, tmp_path):
        out = tmp_path / "out"
        assert write_summary(report, str(out)) == [SUMMARY_HTML, SUMMARY_MD]
        assert (out / SUMMARY_HTML).read_text(encoding="utf-8").startswith("<!doctype html>")
        assert (out / SUMMARY_MD).read_text(encoding="utf-8").startswith("# Dataset validation report")

    def test_remove_summary(self, report, tmp_path):
        out = tmp_path / "out"
        write_summary(report, str(out))
        (out / "keep.json").write_text("{}")
        remove_summary(str(out))
        assert os.listdir(out) == ["keep.json"]

    def test_files_section_needs_outputs(self, report):
        assert "## Files" not in text_of(build_blocks(report))
        report["outputs"] = [{"path": "report/ok_manifest.csv", "rows": 3, "description": "d"}]
        assert "- [ok_manifest.csv](ok_manifest.csv): d (3 rows)" in text_of(build_blocks(report))


class TestGenerateReportSummary:
    def test_summary_is_a_rendering_of_the_json(self, dataset):
        path = generate_report(dataset)
        out_dir = os.path.dirname(path)
        blocks = build_blocks(json.load(open(path)))
        assert open(os.path.join(out_dir, SUMMARY_HTML), encoding="utf-8").read() == render_html(blocks)
        assert open(os.path.join(out_dir, SUMMARY_MD), encoding="utf-8").read() == render_markdown(blocks)

    def test_lists_itself_and_links_work(self, dataset):
        path = generate_report(dataset)
        out_dir = os.path.dirname(path)
        outputs = [o["path"] for o in json.load(open(path))["outputs"]]
        assert "report/report.html" in outputs and "report/report.md" in outputs
        page = open(os.path.join(out_dir, SUMMARY_HTML), encoding="utf-8").read()
        for href in re.findall(r'<a href="([^"]+)"', page):
            assert os.path.exists(os.path.join(out_dir, href)), href

    def test_no_local_paths(self, dataset, tmp_path):
        out_dir = os.path.dirname(generate_report(dataset))
        for name in (SUMMARY_HTML, SUMMARY_MD):
            assert str(tmp_path) not in open(os.path.join(out_dir, name), encoding="utf-8").read()

    def test_turned_off_removes_old_files(self, dataset):
        generate_report(dataset)
        path = generate_report(dataset, ReportConfig(summary=False))
        assert not set(os.listdir(os.path.dirname(path))) & {SUMMARY_HTML, SUMMARY_MD}
        assert "report/report.html" not in [o["path"] for o in json.load(open(path))["outputs"]]

    def test_failure_with_force(self, dataset, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(report_module, "write_summary", broken)
        path = generate_report(dataset, ReportConfig(force=True))
        report = json.load(open(path))
        assert report["valid"] is False and "summary_failed" in {i["code"] for i in report["issues"]}
        assert "report/report.html" not in [o["path"] for o in report["outputs"]]

    def test_failure_without_force(self, dataset, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(report_module, "write_summary", broken)
        with pytest.raises(RuntimeError, match="boom"):
            generate_report(dataset)

    def test_invalid_report_shows_banner(self, dataset):
        dataset._manifest.loc[0, "action"] = "delete"
        path = generate_report(dataset, ReportConfig(force=True))
        page = open(os.path.join(os.path.dirname(path), SUMMARY_HTML), encoding="utf-8").read()
        assert 'class="banner"' in page and "unknown_action" in page
