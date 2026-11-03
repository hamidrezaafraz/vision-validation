import torch
from src.near_duplicates.embeddings import get_embeddings, get_model, similar_pairs, get_transform
from src.near_duplicates.perceptual_hashes import get_perceptual_hashes, hamming_pairs
from src.utils.dataset import ImageClassificationDataset
from torch.utils.data import Subset

def get_device():
    dev = "cpu"
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    return dev

def verify_near_embeddings(dataset: ImageClassificationDataset, model_name="alexnet", threshold=0.95, batch_size=128):
    """
    Verify near embeddings in the dataset.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - threshold: float, similarity threshold for considering embeddings as near
    - batch_size: int, size of the batches to process

    Returns:
    - None
    """
    
    device = get_device()

    model = get_model(model_name=model_name, device=device)
    transform = get_transform(model_name=model_name)
    dataset.dataset.transform = transform
    sub_ds = Subset(dataset, dataset.df.index)
    
    emb = get_embeddings(model, sub_ds, batch_size=batch_size, device=device)
    
    pairs = similar_pairs(emb, thr=threshold, device=device)

    if len(pairs) > 0:
        print(f"Found {len(pairs)} pairs of near embeddings.")
        idx2idx = dataset.get_idx2idx()
        for i, d in enumerate(pairs):
            pairs[i] = [idx2idx[idx] for idx in d]
        dataset.group(pairs)
    else:
        print("No near embeddings found.")


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
    'verify_near_embeddings'
)