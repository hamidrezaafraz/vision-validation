from src.near_duplicates.perceptual_hashes import get_perceptual_hashes, hamming_pairs
from src.utils.dataset import ImageClassificationDataset


def verify_near_duplicates(dataset: ImageClassificationDataset, hash_name='phash', threshold=0):
    """
    Verify near duplicates in the dataset using perceptual hashes.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - hash_name: str, name of the hash function (default: 'phash')
    - threshold: int, Hamming distance threshold for considering images as near duplicates

    Returns:
    - None
    """
    # Compute perceptual hashes for all images in the dataset
    H = get_perceptual_hashes(dataset.df['path'], hash_name)

    # Find pairs of images with Hamming distance less than the threshold
    pairs = hamming_pairs(H, threshold)
    

    if len(pairs) > 0:
        print(f"Found {len(pairs)} pairs of near duplicates with Hamming distance < {threshold}.")
        idx2idx = dataset.get_idx2idx()
        for i, d in enumerate(pairs):
            pairs[i] = [idx2idx[idx] for idx in d]
        dataset.group(pairs)
    else:
        print("No near duplicates found.")
        
__all__ = (
    'verify_near_duplicates',
    )