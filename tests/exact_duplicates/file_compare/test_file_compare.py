import pandas as pd

from src.exact_duplicates.file_compare import find_strict_duplicates


def make_df(rows, index=None):
    """rows: list of (group_id, path)."""
    return pd.DataFrame(rows, columns=["group_id", "path"], index=index)


class TestFindStrictDuplicates:
    def test_identical_group_needs_no_split(self, copy_example_images):
        a, b = copy_example_images
        df = make_df([("g1", a), ("g1", b)])
        assert find_strict_duplicates(df) == []

    def test_singleton_groups_are_ignored(self, make_file):
        df = make_df([("g1", make_file("a.bin", b"a")), ("g2", make_file("b.bin", b"b"))])
        assert find_strict_duplicates(df) == []

    def test_mismatched_file_is_split_out(self, make_file):
        # e.g. a hash collision put a different file in the group
        df = make_df([
            ("g1", make_file("a.bin", b"same")),
            ("g1", make_file("b.bin", b"same")),
            ("g1", make_file("c.bin", b"other")),
        ])
        assert find_strict_duplicates(df) == [[2]]

    def test_multiple_mismatched_classes(self, make_file):
        df = make_df([
            ("g1", make_file("a.bin", b"x")),
            ("g1", make_file("b.bin", b"y")),
            ("g1", make_file("c.bin", b"z")),
            ("g1", make_file("d.bin", b"y")),
        ])
        assert find_strict_duplicates(df) == [[1, 3], [2]]

    def test_returns_index_labels_not_positions(self, make_file):
        df = make_df(
            [("g1", make_file("a.bin", b"x")), ("g1", make_file("b.bin", b"y"))],
            index=[100, 200],
        )
        assert find_strict_duplicates(df) == [[200]]

    def test_only_broken_groups_are_returned(self, make_file):
        df = make_df([
            ("ok", make_file("a.bin", b"x")),
            ("ok", make_file("b.bin", b"x")),
            ("bad", make_file("c.bin", b"p")),
            ("bad", make_file("d.bin", b"q")),
        ])
        assert find_strict_duplicates(df) == [[3]]
