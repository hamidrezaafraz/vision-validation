from src.utils.dataset import ImageClassificationDataset, ImageDataset
from src.exact_duplicates.hashing import get_paths_by_hash
from src.exact_duplicates.file_compare import find_strict_duplicates

def verify_exact_duplicates(dataset: ImageClassificationDataset):
    """only flags files that have identical hashes."""
    paths_by_hash = get_paths_by_hash(zip(list(dataset.df.index), list(dataset.df["path"].values)))
    exact_groups = [p for p in paths_by_hash.values() if len(p) > 1]
    if exact_groups:
        lengths = [len(g) for g in exact_groups]
        total = sum(lengths)
        dataset.group(exact_groups)
        print(f"Found {len(exact_groups)} groups of exact duplicates, with each group containing an average of {total/len(exact_groups):.2f} files in total. Maximum group size: {max(lengths)}")
    else:
        print("No exact duplicates found.")

def verify_strict_duplicates(dataset: ImageClassificationDataset):
    """Byte-compares files in each group and splits groups that are not identical.
    this function is used to double check the grouped files after verify_exact_duplicates, 
    because even if two files have the same hash, they may not be identical byte for byte only due to Pigeonhole Principle.
    """
    exact_groups = find_strict_duplicates(dataset.df)
    if exact_groups:
        print(exact_groups)
        dataset.group(exact_groups)
    else:
        print("No exact duplicates found.")


__all__ = (
    'verify_exact_duplicates', 
    'verify_strict_duplicates',
    )
