import json

import pandas as pd
import pytest

from src.report.sections.flags import (
    UNKNOWN,
    canonical_tag,
    cooccurrence_matrix,
    cooccurrence_pairs,
    flag_statistics,
    row_tags,
    severity_counts,
    share,
    status_counts,
    tag_combinations,
    tag_counts,
)


def make_manifest(rows):
    """rows: list of (action, reason)."""
    return pd.DataFrame({
        "path": [f"Data/{i:04d}.png" for i in range(len(rows))],
        "group_id": [f"g{i}" for i in range(len(rows))],
        "action": [a for a, _ in rows],
        "reason": [r for _, r in rows],
    })


# 10 rows, 6 flagged. Per-tag counts add up to 9, more than the 6 flagged images.
EXAMPLE = [
    ("", ""),
    ("keep", ""),
    ("keep", ""),
    ("", ""),
    ("review", "too_small;blurry"),
    ("review", "too_small;blurry"),
    ("review", "too_small"),
    ("drop", "corrupt"),
    ("drop", "blurry;exact_duplicate"),
    ("drop", "exact_duplicate"),
]


@pytest.fixture
def example():
    return make_manifest(EXAMPLE)


class TestShare:
    def test_count_of_and_pct(self):
        assert share(1, 3) == {"count": 1, "of": 3, "pct": 33.33}

    def test_zero_denominator(self):
        assert share(0, 0) == {"count": 0, "of": 0, "pct": 0.0}

    def test_returns_python_ints(self):
        import numpy as np
        result = share(np.int64(2), np.int64(4))
        assert type(result["count"]) is int and type(result["of"]) is int


class TestCanonicalTag:
    def test_alias_resolves_to_catalog_tag(self):
        assert canonical_tag("exact hash duplicates") == "exact_duplicate"

    def test_prefix_tag_keeps_its_settings(self):
        assert canonical_tag("near_duplicate_phash_4") == "near_duplicate_phash_4"

    def test_prefix_alias_resolves_to_prefix(self):
        assert canonical_tag("near duplicates") == "near_duplicate_"

    def test_unknown_tag_is_kept_as_written(self):
        assert canonical_tag("made_up") == "made_up"


class TestRowTags:
    def test_one_entry_per_row_and_tag(self, example):
        tags = row_tags(example)
        assert tags.name == "tag"
        assert tags.loc[4].tolist() == ["blurry", "too_small"]
        assert set(tags.index) == {4, 5, 6, 7, 8, 9}

    def test_repeated_and_aliased_tags_count_once(self):
        manifest = make_manifest([("drop", "exact_duplicate;exact hash duplicates;exact_duplicate")])
        assert row_tags(manifest).tolist() == ["exact_duplicate"]

    def test_nan_and_blank_reasons_have_no_tags(self):
        manifest = make_manifest([("", float("nan")), ("", " ; ")])
        assert row_tags(manifest).empty


class TestStatusCounts:
    def test_counts_add_up_to_total(self, example):
        counts = status_counts(example)
        assert {s: c["count"] for s, c in counts.items()} == {"ok": 4, "review": 3, "drop": 3}
        assert all(c["of"] == 10 for c in counts.values())
        assert counts["ok"]["pct"] == 40.0

    def test_missing_status_is_zero(self):
        assert status_counts(make_manifest([("", "")]))["drop"] == {"count": 0, "of": 1, "pct": 0.0}

    def test_unknown_action_raises(self):
        with pytest.raises(ValueError, match="unknown action"):
            status_counts(make_manifest([("delete", "corrupt")]))


class TestTagCounts:
    def test_counts_order_and_metadata(self, example):
        rows = tag_counts(example)
        assert [(r["tag"], r["count"]) for r in rows] == [
            ("blurry", 3), ("too_small", 3), ("exact_duplicate", 2), ("corrupt", 1),
        ]
        blurry = rows[0]
        assert blurry["of"] == 10 and blurry["pct"] == 30.0
        assert (blurry["category"], blurry["severity"]) == ("photometric", "warning")

    def test_per_tag_counts_exceed_flagged_images(self, example):
        assert sum(r["count"] for r in tag_counts(example)) == 9
        assert row_tags(example).index.nunique() == 6

    def test_by_status(self, example):
        by_tag = {r["tag"]: r["by_status"] for r in tag_counts(example)}
        assert by_tag["blurry"] == {"ok": 0, "review": 2, "drop": 1}
        assert by_tag["corrupt"] == {"ok": 0, "review": 0, "drop": 1}

    def test_unknown_tag_is_counted(self):
        rows = tag_counts(make_manifest([("drop", "made_up")]))
        assert rows == [{"tag": "made_up", "category": UNKNOWN, "severity": UNKNOWN,
                         "count": 1, "of": 1, "pct": 100.0, "by_status": {"ok": 0, "review": 0, "drop": 1}}]

    def test_no_tags(self):
        assert tag_counts(make_manifest([("", "")])) == []


class TestSeverityCounts:
    def test_any_overlaps_and_worst_partitions(self, example):
        counts = severity_counts(example)
        any_ = {s: c["count"] for s, c in counts["any"].items()}
        worst = {s: c["count"] for s, c in counts["worst"].items()}
        # Row 8 (blurry + exact_duplicate) is both a warning and info.
        assert any_ == {"error": 1, "warning": 4, "info": 2, UNKNOWN: 0}
        assert worst == {"error": 1, "warning": 4, "info": 1, UNKNOWN: 0}
        assert sum(worst.values()) == 6

    def test_unknown_only_decides_without_known_tags(self):
        counts = severity_counts(make_manifest([("drop", "made_up"), ("drop", "made_up;corrupt")]))
        assert counts["worst"][UNKNOWN]["count"] == 1
        assert counts["worst"]["error"]["count"] == 1

    def test_no_tags(self):
        counts = severity_counts(make_manifest([("", "")]))
        assert all(c["count"] == 0 for c in counts["worst"].values())


class TestCooccurrenceMatrix:
    def test_values(self, example):
        matrix = cooccurrence_matrix(example)
        assert list(matrix.index) == list(matrix.columns) == ["blurry", "corrupt", "exact_duplicate", "too_small"]
        assert matrix.at["blurry", "blurry"] == 3
        assert matrix.at["blurry", "too_small"] == matrix.at["too_small", "blurry"] == 2
        assert matrix.at["blurry", "exact_duplicate"] == 1
        assert matrix.at["corrupt", "too_small"] == 0

    def test_empty(self):
        assert cooccurrence_matrix(make_manifest([("", "")])).empty


class TestCooccurrencePairs:
    def test_pairs_and_conditional_shares(self, example):
        pairs = cooccurrence_pairs(example)
        assert [p["tags"] for p in pairs] == [["blurry", "too_small"], ["blurry", "exact_duplicate"]]
        top = pairs[0]
        assert (top["count"], top["of"]) == (2, 10)
        assert top["given"]["too_small"] == {"count": 2, "of": 3, "pct": 66.67}
        assert pairs[1]["given"]["exact_duplicate"] == {"count": 1, "of": 2, "pct": 50.0}

    def test_no_pairs_for_single_tag_rows(self):
        assert cooccurrence_pairs(make_manifest([("drop", "corrupt"), ("review", "blurry")])) == []


class TestTagCombinations:
    def test_exact_tag_sets(self, example):
        combos = tag_combinations(example)
        assert combos["distinct"] == 5
        assert [(c["tags"], c["count"]) for c in combos["top"]] == [
            (["blurry", "too_small"], 2),
            (["blurry", "exact_duplicate"], 1),
            (["corrupt"], 1),
            (["exact_duplicate"], 1),
            (["too_small"], 1),
        ]

    def test_top_limits_the_list_not_distinct(self, example):
        combos = tag_combinations(example, top=2)
        assert len(combos["top"]) == 2 and combos["distinct"] == 5

    def test_order_of_tags_in_reason_does_not_matter(self):
        combos = tag_combinations(make_manifest([("review", "blurry;too_small"), ("review", "too_small;blurry")]))
        assert combos["top"] == [{"tags": ["blurry", "too_small"], "count": 2, "of": 2, "pct": 100.0}]


class TestFlagStatistics:
    def test_section(self, example):
        section = flag_statistics(example)
        assert section["total_images"] == 10
        assert section["images_flagged"] == {"count": 6, "of": 10, "pct": 60.0}
        assert section["note"]
        assert {"status", "tags", "severity", "cooccurrence", "combinations"} <= section.keys()

    def test_is_strict_json(self, example):
        json.dumps(flag_statistics(example), allow_nan=False)

    def test_does_not_modify_manifest(self, example):
        before = example.copy()
        flag_statistics(example)
        pd.testing.assert_frame_equal(example, before)

    def test_empty_manifest(self):
        section = flag_statistics(make_manifest([]))
        assert section["total_images"] == 0
        assert section["images_flagged"] == {"count": 0, "of": 0, "pct": 0.0}
        assert section["tags"] == [] and section["cooccurrence"] == []
        json.dumps(section, allow_nan=False)

    def test_non_default_index(self, example):
        example.index = example.index + 100
        assert flag_statistics(example)["images_flagged"]["count"] == 6
