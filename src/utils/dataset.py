from collections import defaultdict
import uuid

import numpy as np
import os
from src.utils.logger import Logger
import json
import torch
from torch.utils.data import Subset, Dataset as TorchDataset
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
import pandas as pd


class ImageClassificationDataset(TorchDataset):
    def __init__(self, root_dir, result_dir, transform=None):
        self.root_dir = root_dir
        self.result_dir = result_dir
        self._embeddings_dir = os.path.join(self.result_dir, "embeddings")
        if not os.path.exists(self._embeddings_dir):
            os.makedirs(self._embeddings_dir, exist_ok=True)
        if not os.path.exists(result_dir):
            os.makedirs(self.result_dir, exist_ok=True)
            self._manifest_path = os.path.join(self.result_dir, "manifest.csv")
            self._audit_path = os.path.join(self.result_dir, "audits.json")
        else:
            self._manifest_path = os.path.join(self.result_dir, "manifest.csv")
            self._audit_path = os.path.join(self.result_dir, "audits.json")
        
        # self._manifest_path = "/Users/threewestrick/Documents/workspace/emotion_classification/emotion_back/emotion/results/manifest.csv"
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.load_manifest()
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
    
    def group(self, exact_groups: list[list[int]], column: str = 'group_id'):
        """
        Merges each list of row indices into one new group in `column`.

        'group_id' is the duplicate group (drives keep/drop). Merges there are mirrored
        into 'split_group', so split_group is always at least as coarse as group_id.
        'split_group' is only used for group-aware splitting and never drives keep/drop.
        """
        columns = ['group_id', 'split_group'] if column == 'group_id' else [column]
        for group in exact_groups:
            for col in columns:
                new_group_id = str(uuid.uuid4())
                target_group_ids = self._manifest.loc[group, col].unique()
                mask = self._manifest[col].isin(target_group_ids)
                self._manifest.loc[mask, col] = new_group_id
        self.update_label_conflict()

    def update_label_conflict(self):
        """True for rows whose split_group contains more than one distinct label."""
        n_labels = self._manifest.groupby('split_group')['label'].transform('nunique')
        self._manifest['label_conflict'] = n_labels > 1
            
    def save_embeddings(self, embeddings, model_name):
        embeddings_path = os.path.join(self._embeddings_dir, f"{model_name}_embeddings.npy")
        np.save(embeddings_path, embeddings)
    
    def load_embeddings(self, model_name):
        embeddings_path = os.path.join(self._embeddings_dir, f"{model_name}_embeddings.npy")
        if os.path.exists(embeddings_path):
            return np.load(embeddings_path)
        return None
    def save_perceptual_hashes(self, hashes, hash_name):
        hashes_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes.npy")
        np.save(hashes_path, hashes)
    
    def load_perceptual_hashes(self, hash_name):
        hashes_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes.npy")
        if os.path.exists(hashes_path):
            return np.load(hashes_path)
        return None
        
    def keep_first_in_group(self, reason="duplicate", column='group_id', ignore_label_conflict=False):
        """
        Looks for images sharing a UUID group_id. Sets the first one to 'keep' 
        and the rest to 'drop' with the provided reason.
        """
        eligable = (self._manifest['action'] != 'drop')
        if not ignore_label_conflict:
            eligable = eligable & ~self._manifest['label_conflict']
        is_duplicate = self._manifest.duplicated(subset=[column], keep='first')
        
        
        self._manifest.loc[~is_duplicate & eligable, 'action'] = 'keep'
        self._manifest.loc[is_duplicate & eligable, 'action'] = 'drop'
        self._manifest.loc[is_duplicate & eligable, 'reason'] = reason
        
        if not ignore_label_conflict:
            conflicting = self._manifest['label_conflict'] & (self._manifest['action'] != 'drop')
            self._manifest.loc[conflicting, 'action'] = 'drop'
            self._manifest.loc[conflicting, 'reason'] = 'Conflicting labels in group'
        
        self.save_manifest()
        
    def keep_first_in_split_group(self, reason="split group duplicate"):
        """
        Within each split_group, keeps the first non-dropped row and drops the rest,
        but only for groups without a label conflict. Rows in conflicting groups are
        left untouched, so different-label items (e.g. same person, other expression) survive.
        """
        eligible = (self._manifest['action'] != 'drop') & ~self._manifest['label_conflict']
        is_duplicate = (
            self._manifest[eligible]
            .duplicated(subset=['split_group'], keep='first')
            .reindex(self._manifest.index, fill_value=False)
            .astype(bool)
        )
        first = eligible & ~is_duplicate

        self._manifest.loc[first & (self._manifest['action'] == ''), 'action'] = 'keep'
        self._manifest.loc[is_duplicate, 'action'] = 'drop'
        no_reason = self._manifest['reason'] == ''
        self._manifest.loc[is_duplicate & no_reason, 'reason'] = reason
        self.save_manifest()

    def update_group_ids(self, indices, assigned_group_ids):
        for index, group_id in zip(indices, assigned_group_ids):
            self._manifest.loc[index, 'group_id'] = group_id
        self.save_manifest()
        
    def get_idx2idx(self):
        return {i: idx for i, idx in enumerate(self.df.index)}
    
    @property
    def df(self):
        return self._manifest[self._manifest['action'] != 'drop'].copy()
    
    def save_manifest(self):
        self._manifest.to_csv(self._manifest_path, index=False)
    
    def restart(self):
        """Deletes the manifest and starts over."""
        self.delete_manifest()
        self.load_manifest()
        
    def delete_manifest(self):
        if os.path.exists(self._manifest_path):
            os.remove(self._manifest_path)
            
    def load_manifest(self, path=None):
        if path is None:
            path = self._manifest_path
        if not os.path.exists(path):
            self._manifest = pd.DataFrame(columns=["path", "label", "group_id", "split_group", "label_conflict", "action", "reason"])
            dataset_len = len(self.dataset.samples)
            self._manifest["path"] = [s[0] for s in self.dataset.samples]
            self._manifest["label"] = [s[1] for s in self.dataset.samples]
            self._manifest["group_id"] = [str(uuid.uuid4()) for _ in range(dataset_len)]
            self._manifest["split_group"] = self._manifest["group_id"]
            self._manifest["label_conflict"] = False
            self._manifest["action"] = [""] * dataset_len
            self._manifest["reason"] = [""] * dataset_len
            self._manifest.to_csv(path, index=False)
        else:
            self._manifest = pd.read_csv(
                path, 
                dtype={'group_id':str,'split_group':str,'label_conflict':bool,'action': str, 'reason': str},
                keep_default_na=False  # Prevents empty strings from being converted to NaN
            )
            if 'split_group' not in self._manifest.columns:  # manifest from before split_group existed
                self._manifest.insert(3, 'split_group', self._manifest['group_id'])
                self.save_manifest()
            if 'label_conflict' not in self._manifest.columns:  # manifest from before label_conflict existed
                self._manifest.insert(4, 'label_conflict', False)
                self.update_label_conflict()
                self.save_manifest()

def similar_pairs(X, thr, chunk=2048):
    logger = Logger(__name__)
    """X: (N,D) L2-normalised. Returns (i,j), i<j, with cosine >= thr."""
    dev = "cpu"
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    logger.info(f"Using device: {dev}")
    X = X.to(dev)
    out = []
    if len(X) == 0:
        return np.empty((0, 2), dtype=int)
    for s in tqdm(range(0, len(X), chunk), desc="Finding similar pairs"):
        i, j = torch.where(X[s:s + chunk] @ X.T >= thr)
        i = i + s
        m = i < j
        out.append(torch.stack([i[m], j[m]], 1).cpu())
    return torch.cat(out).numpy()

def get_train_data(get_dataset):
    train_full, _ = get_dataset()
    if isinstance(train_full, Subset):
        base_dataset = train_full.dataset
        abs_indices = train_full.indices
        paths = [base_dataset.samples[i][0] for i in abs_indices]
        labels = np.array([base_dataset.targets[i] for i in abs_indices])
    else:
        # It's a standard ImageFolder
        paths = [p for p, _ in train_full.samples]
        labels = np.array(train_full.targets)
        abs_indices = np.arange(len(paths))
        
    # RETURN the absolute indices as well!
    return paths, labels, np.array(abs_indices)

        
        
def verify_perceptual_hashes(get_dataset, task_name, save_indices=True, perceptual_hash_threshold=3):
    """(pHash) compares what the image looks like, so it also flags re-encoded, resized or slightly edited copies. These are exactly the duplicates MD5 can’t see, so it finding more is the point of that step."""
    logger = Logger(__name__)
    import imagehash
    from PIL import Image
    
    paths, labels, abs_indices = get_train_data(get_dataset)
    logger.info(f"Number of images in dataset: {len(paths)}")

    def phash_vec(p):
        h = imagehash.phash(Image.open(p).convert("L"))   # 64 bits
        return torch.tensor(h.hash.flatten().astype(np.float32) * 2 - 1) / 8  # ±1/8 -> unit norm
    H = torch.stack([phash_vec(p) for p in tqdm(paths, desc="Computing perceptual hashes")])
    thr = 1 - 2 * 6 / 64 - 1e-6
    logger.info(f"Perceptual hash similarity threshold: {thr}")
    pairs_hash = similar_pairs(H, thr=thr)   # k = 6
    logger.critical(f"Number of perceptual hash pairs: {len(pairs_hash)}")
    
    cos = (H[pairs_hash[:, 0]] * H[pairs_hash[:, 1]]).sum(1)

    ham = torch.round((1 - cos) * 32).int().numpy()
    
    T = perceptual_hash_threshold
    pairs_sel = pairs_hash[ham <= T]        # use the threshold T you settled on

    neighbors = defaultdict(list)
    for i, j in pairs_sel:                  # i < j guaranteed
        neighbors[i].append(j)

    removed = set()
    for i in sorted(neighbors):
        if i in removed:                    # already a duplicate of an earlier image, so it keeps nothing alive
            continue
        removed.update(neighbors[i])        # i is kept; discard everything similar to it

    keep_idx_local = np.array(sorted(set(range(len(paths))) - removed))
    
    if len(removed) > 0:
        os.makedirs(f"{task_name}/results", exist_ok=True)
        np.save(f"{task_name}/results/perceptual_hash_pairs.npy", pairs_hash)
        np.save(f"{task_name}/results/perceptual_hash_ham.npy", ham)
        
        msg = f"{len(removed)} duplicates found"
        logger.critical(msg)
        
        if save_indices:
            # MAP LOCAL TO ABSOLUTE BEFORE SAVING
            keep_idx_absolute = abs_indices[keep_idx_local]
            
            np.save(f"{task_name}/results/data_indices.npy", keep_idx_absolute)
            logger.info(f"Saved {len(keep_idx_absolute)} unique indices out of {len(paths)} total images.")
    else:
        logger.info("No perceptual duplicates found.")
        

def visualize_perceptual_hashes(get_dataset, task_name):
    paths, _, _ = get_train_data(get_dataset)
    def show_pairs(pairs, ham, d, n=6):
        sel = pairs[ham == d][:n]
        if len(sel) == 0:
            print(f"no pairs at distance {d}"); return
        fig, axes = plt.subplots(2, len(sel), figsize=(2.2 * len(sel), 4.6), squeeze=False)
        for c, (i, j) in enumerate(sel):
            for r, k in enumerate((i, j)):
                axes[r, c].imshow(Image.open(paths[k]), cmap="gray")
                axes[r, c].axis("off")
        fig.suptitle(f"Hamming distance = {d}")
        plt.savefig(f"{task_name}/results/perceptual_hash_pairs_d{d}.png")
        plt.close()
        
    if not os.path.exists(f"{task_name}/results/perceptual_hash_pairs.npy") or not os.path.exists(f"{task_name}/results/perceptual_hash_ham.npy"):
        verify_perceptual_hashes(get_dataset, task_name, save_indices=False, perceptual_hash_threshold=3)
    pairs_hash = np.load(f"{task_name}/results/perceptual_hash_pairs.npy")
    ham = np.load(f"{task_name}/results/perceptual_hash_ham.npy")
    for d in (0, 2, 4, 6):
        show_pairs(pairs_hash, ham, d)
        

def verify_embeddings(get_dataset, task_name, save_indices=True, fresh_embedding=False):
    logger = Logger(__name__)
    from torchvision import models, datasets
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # 1. Fetch absolute indices first so we ALWAYS know the dataset size
    _, _, abs_indices = get_train_data(get_dataset)
    num_images = len(abs_indices)

    # 2. Proper device selection order
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    pairs_path = f"{task_name}/results/embedding_pairs.npy"

    if not os.path.exists(pairs_path) or fresh_embedding:
        w = models.AlexNet_Weights.DEFAULT
        net2 = models.alexnet(weights=w)
        net2.classifier = torch.nn.Identity()
        net2.eval().to(dev)
        
        train_ds, _ = get_dataset(w.transforms(), w.transforms())
        logger.info(f"Number of images in dataset: {len(train_ds)}")
        
        feats = []
        with torch.no_grad():
            for x, _ in tqdm(DataLoader(train_ds, batch_size=128, shuffle=False), desc="Computing features"):
                feats.append(F.normalize(net2(x.to(dev)), dim=1).cpu())
                
        emb = torch.cat(feats)
        pairs = similar_pairs(emb, thr=0.95)
        
        if len(pairs) > 0:
            os.makedirs(f"{task_name}/results", exist_ok=True)
            np.save(pairs_path, pairs)
            logger.critical(f"Number of embedding pairs: {len(pairs)}")
            logger.info(f"Embedding pairs saved to {pairs_path}")
        else:
            logger.info("No embedding pairs found.")
    else:
        logger.info("Found existing embedding_pairs.npy, loading pairs.")
        pairs = np.load(pairs_path)
        logger.critical(f"Number of embedding pairs: {len(pairs)}")
    
    if save_indices:
        # Safeguard in case pairs is completely empty
        if len(pairs) > 0:
            removed = set(pairs[:, 1])
        else:
            removed = set()
            
        # Use num_images instead of train_ds to avoid NameError
        keep_idx_local = np.array(sorted(set(range(num_images)) - removed))
        
        # MAP LOCAL TO ABSOLUTE BEFORE SAVING
        keep_idx_absolute = abs_indices[keep_idx_local]
        
        np.save(f"{task_name}/results/data_indices.npy", keep_idx_absolute)
        logger.info(f"Saved {len(keep_idx_absolute)} unique indices out of {num_images} total images.")

def visualize_embedding_pairs(get_dataset, task_name, n=6):
    if not os.path.exists(f"{task_name}/results/embedding_pairs.npy"):
        verify_embeddings(get_dataset, task_name, save_indices=False, fresh_embedding=True)
    paths, _, _ = get_train_data(get_dataset)
    embedding_pairs = np.load(f"{task_name}/results/embedding_pairs.npy")
    

    
    for idx, pairs in enumerate(embedding_pairs[:n]):
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
        for c, k in enumerate(pairs):
            axes[c].imshow(Image.open(paths[k]), cmap="gray")
            axes[c].axis("off")
        fig.suptitle(f"Embedding pair")
        plt.savefig(f"{task_name}/results/embedding_pair_{idx}.png")
        plt.close()