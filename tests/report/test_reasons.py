import ast
from pathlib import Path

import pytest

from src.report.reasons import (
    CATALOG,
    DUPLICATE,
    SEVERITIES,
    get_reason,
    is_duplicate_tag,
    is_known_tag,
    parse_tags,
)

SRC = Path(__file__).resolve().parents[2] / "src"

# Calls that write a reason tag, and where the tag is: (positional index, keyword name).
TAG_WRITERS = {
    "apply_policy": (3, "tag"),
    "append_reason": (1, "tag"),
    "keep_one_from_group": (0, "reason"),
    "keep_first_in_group": (0, "reason"),
    "keep_first_in_split_group": (0, "reason"),
}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _tag_argument(call: ast.Call, position: int, keyword: str):
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return call.args[position] if len(call.args) > position else None


def written_tags():
    """
    Every tag literal passed to a TAG_WRITERS call anywhere in src/, as (tag, is_prefix, location).
    f-strings give their leading literal as a prefix (f"near_duplicate_{h}_{t}" -> 'near_duplicate_').
    Variables (e.g. a `reason` parameter passed through) are skipped: their literal is found at the call site.
    Direct assignments like manifest.loc[..., 'reason'] = '...' are found too.
    """
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            value = None
            if isinstance(node, ast.Call) and _call_name(node) in TAG_WRITERS:
                value = _tag_argument(node, *TAG_WRITERS[_call_name(node)])
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Subscript) and "'reason'" in ast.unparse(t.slice).replace('"', "'") for t in node.targets
            ):
                value = node.value
            where = f"{path.relative_to(SRC.parent)}:{node.lineno}" if hasattr(node, "lineno") else str(path)
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value:
                found.append((value.value, False, where))
            elif isinstance(value, ast.JoinedStr) and value.values and isinstance(value.values[0], ast.Constant):
                found.append((value.values[0].value, True, where))
    return found


class TestCatalog:
    def test_tags_and_aliases_are_unique(self):
        names = [r.tag for r in CATALOG] + [a for r in CATALOG for a in r.aliases]
        assert len(names) == len(set(names))

    def test_entries_are_complete(self):
        for entry in CATALOG:
            assert entry.severity in SEVERITIES, entry.tag
            assert entry.description.endswith("."), entry.tag
            assert entry.check and entry.category, entry.tag
            assert ";" not in entry.tag, entry.tag

    def test_every_written_tag_is_in_the_catalog(self):
        tags = written_tags()
        found = {t for t, _, _ in tags}
        # Sanity check that the scan finds tags from each kind of writer.
        assert {"corrupt", "blurry", "exact_duplicate", "near_duplicate_"} <= found
        missing = [
            (tag, where) for tag, is_prefix, where in tags
            if not (any(r.prefix and tag.startswith(r.tag) for r in CATALOG) if is_prefix else is_known_tag(tag))
        ]
        assert missing == []


class TestGetReason:
    @pytest.mark.parametrize("tag", ["corrupt", "blurry", "exact_duplicate", "label_conflict"])
    def test_exact_tag(self, tag):
        assert get_reason(tag).tag == tag

    @pytest.mark.parametrize("alias, tag", [
        ("exact hash duplicates", "exact_duplicate"),
        ("exact byte-level duplicates", "exact_duplicate"),
        ("near-perceptual-hash duplicates", "near_duplicate_"),
        ("Conflicting labels in group", "label_conflict"),
    ])
    def test_alias(self, alias, tag):
        assert get_reason(alias).tag == tag

    @pytest.mark.parametrize("tag", ["near_duplicate_phash_4", "near_duplicate_alexnet_0_95"])
    def test_prefix(self, tag):
        assert get_reason(tag).tag == "near_duplicate_"

    @pytest.mark.parametrize("tag", ["near_duplicate_", "Blurry", "duplicate", "", "blurry "])
    def test_unknown(self, tag):
        assert get_reason(tag) is None


class TestParseTags:
    @pytest.mark.parametrize("reason, tags", [
        ("", []),
        (None, []),
        (float("nan"), []),
        ("blurry", ["blurry"]),
        ("too_small;blurry", ["too_small", "blurry"]),
        ("blurry;; too_small ", ["blurry", "too_small"]),
    ])
    def test_parse(self, reason, tags):
        assert parse_tags(reason) == tags


class TestIsDuplicateTag:
    @pytest.mark.parametrize("tag", ["exact_duplicate", "near_duplicate_phash_4", "exact hash duplicates"])
    def test_duplicate_tags(self, tag):
        assert is_duplicate_tag(tag)
        assert get_reason(tag).category == DUPLICATE

    @pytest.mark.parametrize("tag", ["corrupt", "blurry", "label_conflict", "", "duplicate"])
    def test_other_tags(self, tag):
        assert not is_duplicate_tag(tag)
