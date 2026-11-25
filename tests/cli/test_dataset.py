import argparse

import pytest

from src.cli.dataset import DatasetArgs


def parse(argv):
    parser = argparse.ArgumentParser(parents=[DatasetArgs.get_parser()])
    return parser.parse_args(["--datadir", "data", "--resultdir", "results", *argv])


class TestGetParser:
    def test_group_keep_strategy_defaults_to_random_unseeded(self):
        args = parse([])
        assert args.group_keep_strategy == "random"
        assert args.group_keep_seed is None

    def test_group_keep_seed_is_int(self):
        assert parse(["--group_keep_seed", "7"]).group_keep_seed == 7

    def test_scalar_choices_are_enforced(self):
        with pytest.raises(SystemExit):
            parse(["--group_keep_strategy", "sharpest"])


import json
import os

import numpy as np
from PIL import Image

from src.cli.dataset import dataset_cli


@pytest.fixture
def delivery(tmp_path):
    """Data/a: two exact copies of a noise image and a gray blank frame; Data/b: another noise image."""
    rng = np.random.default_rng(0)
    noise = [Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for _ in range(2)]
    files = {"a/n0.png": noise[0], "a/n0_copy.png": noise[0], "a/gray.png": Image.new("RGB", (64, 64), (128, 128, 128)),
             "b/n1.png": noise[1]}
    for rel, img in files.items():
        (tmp_path / "Data" / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(tmp_path / "Data" / rel)
    return str(tmp_path / "Data"), str(tmp_path / "results")


def run(delivery, *flags):
    data, results = delivery
    parser = argparse.ArgumentParser(parents=[DatasetArgs.get_parser()])
    dataset_cli(parser.parse_args(["--datadir", data, "--resultdir", results, *flags]))
    return os.path.join(results, "report")


def break_manifest(delivery):
    """Gives the first manifest row an unknown action, which validate_manifest reports as an error."""
    import pandas as pd
    path = os.path.join(delivery[1], "manifest.csv")
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    manifest.loc[0, "action"] = "delete"
    manifest.to_csv(path, index=False)


class TestReportFlags:
    def test_defaults(self):
        args = parse([])
        assert (args.report, args.review_sheet, args.force) == (False, False, False)
        assert args.max_per_section == 60

    def test_parsed(self):
        args = parse(["--report", "--review_sheet", "--force", "--max_per_section", "10"])
        assert (args.report, args.review_sheet, args.force, args.max_per_section) == (True, True, True, 10)


class TestDatasetCliReport:
    def test_report_after_checks(self, delivery):
        out = run(delivery, "--check_decode", "--check_photometric", "--group_hash", "--group_keep_seed", "0", "--report")
        assert sorted(os.listdir(out)) == sorted([
            "validation_report.json", "report.html", "report.md",
            "ok_manifest.csv", "review_manifest.csv", "drop_manifest.csv", "duplicate_groups.csv",
        ])
        report = json.load(open(os.path.join(out, "validation_report.json")))
        assert report["valid"] is True
        assert set(report["checks"]) == {"check_decode", "check_photometric", "group_by_hash"}
        assert report["status_counts"]["drop"]["count"] == 1  # the copy
        assert report["status_counts"]["review"]["count"] == 1  # the gray frame

    def test_report_alone_on_an_existing_manifest(self, delivery):
        run(delivery, "--check_decode", "--group_hash", "--group_keep_seed", "0")
        manifest = os.path.join(delivery[1], "manifest.csv")
        before = open(manifest, "rb").read()
        out = run(delivery, "--report")
        assert open(manifest, "rb").read() == before
        assert json.load(open(os.path.join(out, "validation_report.json")))["status_counts"]["drop"]["count"] == 1

    def test_review_sheet(self, delivery):
        out = run(delivery, "--check_decode", "--report", "--review_sheet", "--max_per_section", "2")
        assert "review_sheet.html" in os.listdir(out)

    def test_errors_exit_without_force(self, delivery, capsys):
        run(delivery, "--check_decode")
        break_manifest(delivery)
        with pytest.raises(SystemExit) as exc:
            run(delivery, "--report")
        assert exc.value.code == 1
        assert "--force" in capsys.readouterr().out
        assert not os.path.exists(os.path.join(delivery[1], "report", "validation_report.json"))

    def test_force_writes_invalid_report(self, delivery):
        run(delivery, "--check_decode")
        break_manifest(delivery)
        out = run(delivery, "--report", "--force")
        report = json.load(open(os.path.join(out, "validation_report.json")))
        assert report["valid"] is False

    @pytest.mark.parametrize("flags", [["--review_sheet"], ["--force"], ["--max_per_section", "5"]])
    def test_report_options_need_report(self, delivery, flags):
        with pytest.raises(SystemExit, match="add --report"):
            run(delivery, *flags)


class TestDatasetCliEmbeddings:
    @pytest.fixture
    def calls(self, monkeypatch):
        """Records which embedding function each flag calls, without loading a model."""
        import src.cli.dataset as cli
        calls = []
        monkeypatch.setattr(cli, "make_and_save_embeddings", lambda ds, model_name, batch_size: calls.append(("make", model_name)))
        monkeypatch.setattr(cli, "cached_embeddings", lambda ds, model_name, batch_size: calls.append(("cached", model_name)))
        return calls

    def test_make_general_embeddings_always_recomputes(self, delivery, calls):
        run(delivery, "--make_general_embeddings", "--embedding_model_name", "alexnet")
        assert calls == [("make", "alexnet")]

    def test_make_expert_embeddings_still_uses_the_cache(self, delivery, calls):
        run(delivery, "--make_expert_embeddings")
        assert calls == [("cached", "facenet-vggface2")]


class TestFromNamespace:
    def test_tuple_fields_stay_tuples(self):
        data_args = DatasetArgs.from_namespace(parse(["--expected_modes", "RGB", "L"]))
        assert data_args.expected_modes == ("RGB", "L")
        assert data_args == DatasetArgs(datadir="data", resultdir="results", expected_modes=("RGB", "L"))
