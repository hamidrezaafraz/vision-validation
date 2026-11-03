import filecmp
import pandas as pd


def find_strict_duplicates(df: pd.DataFrame) -> list[list[int]]:
    """
    Byte-compares the files in each group and returns the subgroups that are not
    identical to the group's first file, so they can be given new group ids.

    df must have 'group_id' and 'path' columns. Only groups with more than one row are checked.
    Returns a list of subgroups, each a list of df index labels.
    example:

        new_groups = find_strict_duplicates(ds._manifest)
        ds.group(new_groups)

    """
    dup =df.groupby('group_id').filter(lambda x: len(x) > 1)

    new_groups = []

    for gid, g in dup.groupby('group_id'):
        first = g['path'].iloc[0]
        if not all(filecmp.cmp(first, p, shallow=False) for p in g['path'].iloc[1:]):
            rows = df.index[df['group_id'] == gid]
            classes = []  # [(representative_path, [indices])]
            for idx in rows:
                path = df.at[idx, 'path']
                for rep, members in classes:
                    if filecmp.cmp(rep, path, shallow=False):
                        members.append(idx)
                        break
                else:
                    classes.append((path, [idx]))

            # First class keeps the original id, the others get new ones
            new_groups += [members for _, members in classes[1:]]
    return new_groups