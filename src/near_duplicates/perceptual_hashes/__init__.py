
import imagehash
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

def get_hash_func(hash_name: str) -> callable:
    """
    Get the hash function based on the hash name.

    Parameters:
    - hash_name: str, name of the hash function ('phash', 'average_hash', 'dhash', 'whash')

    Returns:
    - hash_func: function to compute the hash
    """
    if hash_name == 'phash':
        return imagehash.phash
    elif hash_name == 'average_hash':
        return imagehash.average_hash
    elif hash_name == 'dhash':
        return imagehash.dhash
    elif hash_name == 'whash':
        return imagehash.whash
    else:
        raise ValueError(f"Unknown hash name: {hash_name}")

def get_perceptual_hashes(paths: list[str], hash_name: str = 'phash', rgb=False) -> np.ndarray:
    """
    Compute perceptual hashes for all images in the dataset.

    Parameters:
    - paths: list of image file paths
    - hash_name: str, name of the hash function (default: 'phash')

    Returns:
    - H: numpy array of shape (num_images, hash_size) containing the hashes
    """
    hash_func = get_hash_func(hash_name)
    if rgb:
        H = np.array([hash_func(Image.open(p)).hash.flatten() for p in tqdm(paths, desc="Computing perceptual hashes")])
    else:
        H = np.array([hash_func(Image.open(p).convert("L")).hash.flatten() for p in tqdm(paths, desc="Computing perceptual hashes")])
    return H

def hamming_pairs(ndarray: np.ndarray, threshold: int, chunk_size: int = 2048):
    """
    ndarray:     (N, n_bits) binary (bool or 0/1) hashes, on any device
    threshold:  pairs with Hamming distance <= threshold are returned (inclusive, like --hamming_threshold
                and the near_duplicate_<hash>_<threshold> tag: 0 means identical hashes only)
    chunk_size: number of rows compared against the rest per iteration
    returns:    (M, 2) numpy array of index pairs (i, j) with i < j
    """
    tensor = torch.tensor(ndarray, dtype=torch.bool)
    dev = "cpu"
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    tensor = tensor.to(dev)
    X = tensor.float()          # exact for 0/1 values, and matmul works on cpu/cuda/mps
    ones = X.sum(1)             # number of set bits per hash
    out = []

    for s in tqdm(range(0, X.shape[0], chunk_size), desc="Finding pairs"):
        a = X[s:s + chunk_size]                 # (c, B)
        b = X[s:]                               # only compare against rows at or after the chunk (upper triangle)
        # for binary vectors: hamming(a, b) = |a| + |b| - 2 * (a . b)
        d = ones[s:s + chunk_size, None] + ones[None, s:] - 2 * (a @ b.T)
        i, j = torch.where(d <= threshold)
        i, j = i + s, j + s
        keep = i < j                            # drops self-pairs and mirrored duplicates
        out.append(torch.stack([i[keep], j[keep]], dim=1).cpu())

    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return torch.cat(out).numpy()