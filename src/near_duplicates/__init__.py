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

def verify_near_embeddings(dataset: ImageClassificationDataset, model_name="alexnet", threshold=0.95, batch_size=128, ignore_label_conflict=False):
    """
    Verify near embeddings in the dataset (rows not dropped). Only merges the
    'split_group' column; group_id, action and reason are left untouched.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - threshold: float, similarity threshold for considering embeddings as near
    - ignore_label_conflict: bool, whether to ignore label conflicts when grouping
    - batch_size: int, size of the batches to process

    Returns:
    - None
    """
    
    device = get_device()

    model = get_model(model_name=model_name, device=device)
    transform = get_transform(model_name=model_name)
    dataset.dataset.transform = transform
    # get all the embeddings for the dataset using the specified model
    
    # sub_ds = Subset(dataset, dataset.df.index)
    
    emb = dataset.load_embeddings(model_name=model_name)
    if emb is None:
        emb = get_embeddings(model, dataset, batch_size=batch_size, device=device)
        dataset.save_embeddings(emb, model_name=model_name)
    
    emb = emb[dataset.df.index]  # Ensure embeddings are in the same order as the dataset
    emb = torch.tensor(emb)
    pairs = similar_pairs(emb, thr=threshold, device=device)

    if len(pairs) > 0:
        print(f"Found {len(pairs)} pairs of near embeddings.")
        idx2idx = dataset.get_idx2idx()
        for i, d in enumerate(pairs):
            pairs[i] = [idx2idx[idx] for idx in d]
        # embeddings only link images for splitting; they never decide keep/drop
        if ignore_label_conflict:
            dataset.group(pairs)
        else:
            dataset.group(pairs, column='split_group')
        dataset.save_manifest()
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
    H = dataset.load_perceptual_hashes(hash_name=hash_name)
    if H is None:
        H = get_perceptual_hashes(dataset._manifest['path'], hash_name)
        dataset.save_perceptual_hashes(H, hash_name=hash_name)
    
    H = H[dataset.df.index]  # Ensure hashes are in the same order as the dataset

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