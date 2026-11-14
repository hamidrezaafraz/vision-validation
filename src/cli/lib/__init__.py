
from src.utils.dataset import ImageClassificationDataset, ImageDataset
from src.exact_duplicates.hashing import get_hash_hexdigest
import numpy as np
from src.near_duplicates.perceptual_hashes import get_perceptual_hashes, hamming_pairs
from src.near_duplicates.embeddings import get_embeddings, similar_pairs, get_model, get_transform, get_device
import torch

def make_hashes(dataset: ImageDataset | ImageClassificationDataset, hash_name="blake2b"):
    """Hashes all images in the dataset and saves the hashes."""
    dataset._manifest['hash'] = dataset._manifest['path'].apply(
        lambda x: get_hash_hexdigest(x, hash_name=hash_name)
    )
    dataset.save_manifest()

def group_by_hash(dataset: ImageDataset | ImageClassificationDataset):
    """Groups images that share the same file hash using dataset.group. Hashes first if needed."""
    if not dataset.has_hash():
        print("Hashing images before grouping...")
        make_hashes(dataset)
    groups = [
        idx.tolist()
        for idx in dataset.df.groupby('hash').groups.values()
        if len(idx) > 1
    ]
    print(f"Found {len(groups)} groups of exact hash duplicates.")
    if groups:
        print(f"Grouping {sum(len(g) for g in groups)} images into {len(groups)} groups based on exact hash duplicates.")
        dataset.group(groups)
        dataset.keep_first_in_group(reason="exact hash duplicates")
        print("Remaining images after grouping: ", len(dataset.df), "\n")
        
def make_perceptual_hashes(dataset: ImageDataset | ImageClassificationDataset, hash_name="phash"):
    """Hashes all images in the dataset using perceptual hashing and saves the hashes
    in the manifest's '{hash_name}_perceptual_hash' column as hex strings (one per image)."""
    H = get_perceptual_hashes(dataset._manifest['path'], hash_name)
    dataset.save_perceptual_hashes(H, hash_name=hash_name)


def group_by_perceptual_hash(dataset: ImageDataset | ImageClassificationDataset, hash_name='phash', threshold=0):
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
        dataset.keep_first_in_group(reason="near-perceptual-hash duplicates")
    else:
        print("No near duplicates found.")

def verify_near_embeddings(dataset: ImageDataset | ImageClassificationDataset, model_name="alexnet", threshold=0.95, batch_size=128, embed_full_dataset=False):
    """
    Verify near embeddings in the dataset (rows not dropped). Only merges the
    'split_group' column; group_id, action and reason are left untouched.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - threshold: float, similarity threshold for considering embeddings as near
    - ignore_label_conflict: bool, whether to ignore label conflicts when grouping
    - batch_size: int, size of the batches to process
    - embed_full_dataset: bool, whether to embed the full dataset or just the current subset

    Returns:
    - None
    """
    
    device = get_device()

    model = get_model(model_name=model_name, device=device)
    transform = get_transform(model_name=model_name)
    dataset.transform = transform
    # get all the embeddings for the dataset using the specified model
    
    
    emb = dataset.load_embeddings(model_name=model_name)
    if emb is None:
        if embed_full_dataset:
            print("Embedding the full dataset...")
            emb = get_embeddings(model, dataset, batch_size=batch_size, device=device)
        else:
            print("Embedding the current subset of the dataset...")
            sub_ds = dataset.get_subset()
            emb = get_embeddings(model, sub_ds, batch_size=batch_size, device=device)
        dataset.save_embeddings(emb, model_name=model_name)
    
    emb = emb[dataset.df.index]  # Ensure embeddings are in the same order as the dataset
    emb = torch.tensor(emb)
    pairs = similar_pairs(emb, thr=threshold, device=device)

    if len(pairs) > 0:
        print(f"Found {len(pairs)} pairs of near embeddings.")
        idx2idx = dataset.get_idx2idx()
        for i, d in enumerate(pairs):
            pairs[i] = [idx2idx[idx] for idx in d]
        dataset.group(pairs, column='split_group')
        dataset.save_manifest()
    else:
        print("No near embeddings found.")