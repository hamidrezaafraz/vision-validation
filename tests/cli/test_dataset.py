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
