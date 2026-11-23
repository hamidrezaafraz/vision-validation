"""
One final status per manifest row, for reporting.

`action` in the manifest has four possible values, but two of them mean the same thing to a reader:
'' (written by load_manifest/restart, and left alone by check_decode/check_photometric, which only
write 'drop' or 'review') and 'keep' (written by keep_one_from_group for every eligible row, singletons
included). Both mean "no check flagged this image". Mapping them here, instead of rewriting the
manifest, keeps the report read-only and makes the clean count independent of which steps happened to run.
"""
import pandas as pd

OK, REVIEW, DROP = "ok", "review", "drop"
STATUSES = (OK, REVIEW, DROP)

ACTION_TO_STATUS = {
    "": OK,
    "keep": OK,
    "review": REVIEW,
    "drop": DROP,
}

MAX_EXAMPLE_PATHS = 5


def resolve_status(manifest: pd.DataFrame) -> pd.Series:
    """
    Maps every row's `action` to 'ok', 'review' or 'drop' and returns it as a Series named 'status'
    with the manifest's index. Does not change `manifest`.

    Raises ValueError for any other action (including NaN), listing the bad values and a few example
    paths, because an unknown action means the manifest is broken and must not be reported on.
    """
    status = manifest["action"].map(ACTION_TO_STATUS)
    unknown = status.isna()
    if unknown.any():
        bad_values = sorted(map(repr, manifest.loc[unknown, "action"].unique()))
        examples = manifest.loc[unknown, "path"].head(MAX_EXAMPLE_PATHS).tolist() if "path" in manifest else []
        raise ValueError(
            f"{int(unknown.sum())} manifest rows have an unknown action {', '.join(bad_values)}; "
            f"expected one of {sorted(ACTION_TO_STATUS)}. Example paths: {examples}"
        )
    return status.rename("status")


def is_duplicate_representative(manifest: pd.DataFrame, column: str = "group_id") -> pd.Series:
    """
    True for 'ok' rows whose group in `column` has more than one member, i.e. the image kept on behalf
    of its duplicates. Group sizes count every row, dropped ones included, because the duplicate copies
    are exactly the dropped rows.

    This cannot be read from action == 'keep': keep_one_from_group marks singletons 'keep' too.
    A 'review' row in a duplicate group is not a representative. Does not change `manifest`.
    """
    group_size = manifest.groupby(column)[column].transform("size")
    return ((resolve_status(manifest) == OK) & (group_size > 1)).rename("is_duplicate_representative")
