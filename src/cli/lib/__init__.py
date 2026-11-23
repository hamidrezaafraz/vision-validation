
from src.utils.dataset import ImageClassificationDataset, ImageDataset, embedding_group_column
from src.exact_duplicates.hashing import get_hash_hexdigest
import numpy as np
from src.near_duplicates.perceptual_hashes import get_perceptual_hashes, hamming_pairs
from src.near_duplicates.embeddings import get_embeddings, similar_pairs, get_model, get_transform, get_device
import torch

def merge_pairs(pairs) -> list[list[int]]:
    """Merges overlapping pairs into connected groups: [[1,3],[3,5],[4,6]] -> [[1,3,5],[4,6]]."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        root = find(pair[0])
        for x in pair[1:]:
            parent[find(x)] = root

    groups = {}
    for x in sorted(parent):
        groups.setdefault(find(x), []).append(int(x))
    return list(groups.values())

def make_hashes(dataset: ImageDataset | ImageClassificationDataset, hash_name="blake2b"):
    """Hashes all images in the dataset and saves the hashes."""
    dataset._manifest['hash'] = dataset._manifest['path'].apply(
        lambda x: get_hash_hexdigest(x, hash_name=hash_name)
    )
    dataset.save_manifest()

def group_by_hash(dataset: ImageDataset | ImageClassificationDataset, group_keep_strategy="random", seed=None):
    """Groups images that share the same file hash using dataset.group, then keeps one image per
    group (picked by group_keep_strategy) and tags the rest 'exact_duplicate'. Hashes first if needed."""
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
        dataset.keep_one_from_group(reason="exact_duplicate", group_keep_strategy=group_keep_strategy, seed=seed)
        print("Remaining images after grouping: ", len(dataset.df), "\n")
        
def make_perceptual_hashes(dataset: ImageDataset | ImageClassificationDataset, hash_name="phash"):
    """Hashes all images in the dataset using perceptual hashing and saves the hashes
    in the manifest's '{hash_name}_perceptual_hash' column as hex strings (one per image)."""
    df = dataset.df
    H = get_perceptual_hashes(df['path'], hash_name)
    dataset.save_perceptual_hashes(H, df.index, hash_name=hash_name)


def group_by_perceptual_hash(dataset: ImageDataset | ImageClassificationDataset, hash_name='phash', threshold=0, group_keep_strategy="random", seed=None):
    """
    Verify near duplicates in the dataset using perceptual hashes.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - hash_name: str, name of the hash function (default: 'phash')
    - threshold: int, Hamming distance threshold for considering images as near duplicates
    - group_keep_strategy: str, how the one kept image per group is picked (see GROUP_KEEP_STRATEGIES)
    - seed: int or None, makes the 'random' strategy reproducible

    Returns:
    - None
    """
    # Compute perceptual hashes for all images in the dataset
    H, indices = dataset.load_perceptual_hashes(hash_name=hash_name)
    df = dataset.df
    if H is None or indices is None or len(H) != len(df) or not np.array_equal(indices, df.index.to_numpy()):
        H = get_perceptual_hashes(df['path'], hash_name)
        dataset.save_perceptual_hashes(H, df.index, hash_name=hash_name)
    
    # Find pairs of images with Hamming distance less than the threshold
    pairs = hamming_pairs(H, threshold)
    
    if len(pairs) > 0:
        groups = merge_pairs(pairs)
        print(f"Found {len(pairs)} pairs of near duplicates with Hamming distance <= {threshold}, merged into {len(groups)} groups.")
        idx2idx = dataset.get_idx2idx()
        groups = [[idx2idx[idx] for idx in g] for g in groups]
        dataset.group(groups)
        dataset.keep_one_from_group(reason=f"near_duplicate_{hash_name}_{threshold}", group_keep_strategy=group_keep_strategy, seed=seed)
    else:
        print("No near duplicates found.")

def make_embeddings(dataset: ImageDataset | ImageClassificationDataset, model_name="alexnet", batch_size=128):
    """
    make general embeddings for the dataset.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - model_name: str, name of the model to use for embedding
    - batch_size: int, size of the batches to process

    Returns:
    - None
    """
    
    device = get_device()
    model = get_model(model_name=model_name, device=device)
    transform = get_transform(model_name=model_name)
    dataset.transform = transform        
    print("Embedding the full dataset...")
    emb = get_embeddings(model, dataset, batch_size=batch_size, device=device)
    return emb

def group_by_embeddings(dataset: ImageDataset | ImageClassificationDataset, model_name="alexnet", threshold=0.95):
    """
    Verify near embeddings in the dataset (rows not dropped). Merges the per-model
    'group_<model>_<threshold>' column (e.g. group_alexnet_0_95) and 'split_group';
    group_id, action and reason are left untouched.

    Parameters:
    - dataset: ImageClassificationDataset instance
    - model_name: str, name of the model to use for embedding
    - threshold: float, similarity threshold for considering embeddings as near

    Returns:
    - None
    """
        
    emb = dataset.load_embeddings(model_name=model_name)
    if emb is None:
        print("Embedding the full dataset...")
        emb = make_embeddings(dataset, model_name=model_name, batch_size=64)
        dataset.save_embeddings(emb, model_name=model_name)
        
    device = get_device()

    emb = torch.tensor(emb)
    pairs = similar_pairs(emb, thr=threshold, device=device)

    if len(pairs) > 0:
        groups = merge_pairs(pairs)
        print(f"Found {len(pairs)} close embeddings with similarity <= {threshold}, merged into {len(groups)} groups.")
        idx2idx = dataset.get_idx2idx()
        groups = [[idx2idx[idx] for idx in g] for g in groups]
        # per-model column records which model linked the images; split_group takes the union
        dataset.group(groups, column=embedding_group_column(model_name, threshold))
        dataset.group(groups, column='split_group')
        dataset.save_manifest()
    else:
        print("No near embeddings found.")    
