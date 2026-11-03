import argparse

import pytest

from src.cli.classification import ClassificationArgs, classification_cli


def parse(argv):
    parser = argparse.ArgumentParser(parents=[ClassificationArgs.get_parser()])
    return parser.parse_args(argv)


class TestGetParser:
    def test_bool_fields_default_to_false(self):
        args = parse(["--datadir", "data", "--resultdir", "results"])
        assert args.keep_first_in_group is False
        assert args.verify_exact_duplicates is False
        assert args.verify_strict_duplicates is False

    def test_bool_fields_are_store_true(self):
        args = parse(["--datadir", "data", "--resultdir", "results", "--verify_exact_duplicates", "--keep_first_in_group"])
        assert args.verify_exact_duplicates is True
        assert args.keep_first_in_group is True

    def test_field_without_default_is_required(self):
        with pytest.raises(SystemExit):
            parse([])

class TestFromNamespace:
    def test_ignores_extra_attributes(self):
        args = parse(["--datadir", "data", "--resultdir", "results", "--verify_strict_duplicates"])
        args.command = "cls"
        cls_args = ClassificationArgs.from_namespace(args)
        assert cls_args == ClassificationArgs(datadir="data", resultdir="results", verify_strict_duplicates=True)

