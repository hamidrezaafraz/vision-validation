from collections import defaultdict
import uuid

import numpy as np
import os
from src.utils.images import find_image_paths
from src.utils.logger import Logger
from src.utils.audit import MANIFEST_CREATED, RESTART, record_event
import json
import torch
from torch.utils.data import Subset, Dataset as TorchDataset
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logger = Logger(__name__)


# How keep_one_from_group picks the survivor of each duplicate group.
GROUP_KEEP_STRATEGIES = ("random",)


def embedding_group_column(model_name: str, threshold: float) -> str:
    """Name of the per-model group column, e.g. ('alexnet', 0.95) -> 'group_alexnet_0_95'."""
    return f"group_{model_name}_{str(threshold).replace('.', '_')}"


class GroupColumnsMixin:
    """Per-model group columns and the split_group rebuild shared by both dataset classes."""

    def _ensure_group_column(self, column: str):
        """Adds `column` with a unique UUID per row when it doesn't exist yet."""
        if column not in self._manifest.columns:
            self._manifest[column] = [str(uuid.uuid4()) for _ in range(len(self._manifest))]

    def group_columns(self) -> list[str]:
        """Per-model group columns currently in the manifest."""
        return [c for c in self._manifest.columns if c.startswith('group_') and c != 'group_id']

    def rebuild_split_group(self, columns=()):
        """
        Rebuilds 'split_group' from scratch as the union of 'group_id' and the given
        per-model group columns: two rows end up in the same split_group when they share
        a value in any of those columns (directly or through a chain). Columns not listed
        no longer contribute, and merges made directly on split_group are discarded.
        """
        columns = ['group_id'] + [c for c in columns if c != 'group_id']
        missing = [c for c in columns if c not in self._manifest.columns]
        if missing:
            raise KeyError(f"Unknown group columns: {missing}")

        parent = list(range(len(self._manifest)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for col in columns:
            first_of = {}
            for pos, value in enumerate(self._manifest[col]):
                if value in first_of:
                    parent[find(pos)] = find(first_of[value])
                else:
                    first_of[value] = pos

        ids = {}
        self._manifest['split_group'] = [ids.setdefault(find(i), str(uuid.uuid4())) for i in range(len(parent))]
        self._after_split_group_change()
        self.save_manifest()

    def _after_split_group_change(self):
        pass

    def _embedding_paths(self, model_name: str) -> tuple[str, str]:
        base = os.path.join(self._embeddings_dir, model_name)
        return f"{base}_embeddings.npy", f"{base}_embeddings_indices.npy"

    def save_embeddings(self, embeddings, model_name: str, indices):
        """
        Saves the embedding matrix to <model>_embeddings.npy and, row for row, the manifest index of each
        embedding to <model>_embeddings_indices.npy. Without the indices, row i would only mean "the i-th
        row not dropped at the time", which changes whenever a rerun keeps different images.
        """
        embeddings, indices = np.asarray(embeddings), np.asarray(indices)
        if len(embeddings) != len(indices):
            raise ValueError(f"{len(embeddings)} embeddings but {len(indices)} indices")
        matrix_path, indices_path = self._embedding_paths(model_name)
        os.makedirs(self._embeddings_dir, exist_ok=True)
        np.save(matrix_path, embeddings)
        np.save(indices_path, indices)

    def load_embeddings(self, model_name: str, indices=None):
        """
        The cached embeddings of the rows `indices` (default: the rows not dropped, self.df.index), in that
        order, or None when they have to be computed: no cache, a cache saved without row indices (older
        caches; their rows can't be matched to the manifest), or a cache that lacks any of the rows.
        A cache holding more rows than needed is fine; only the requested ones are returned.
        """
        matrix_path, indices_path = self._embedding_paths(model_name)
        if not os.path.exists(matrix_path):
            return None
        if not os.path.exists(indices_path):
            logger.warning(f"{matrix_path} has no row indices; recomputing the {model_name} embeddings")
            return None
        matrix, cached = np.load(matrix_path, mmap_mode="r"), pd.Index(np.load(indices_path))
        wanted = pd.Index(self.df.index if indices is None else indices)
        if len(matrix) != len(cached) or not cached.is_unique or not wanted.isin(cached).all():
            logger.info(f"Cached {model_name} embeddings don't cover the current rows; recomputing")
            return None
        return np.asarray(matrix[cached.get_indexer(wanted)])

    def _select_one_per_group(self, eligible: pd.Series, group_keep_strategy: str = "random", seed=None) -> pd.Index:
        """
        Index labels of the one eligible row per group_id that should survive.

        - 'random': picks uniformly at random among the group's eligible rows; `seed` makes it reproducible.
        """
        # TODO: add select by sorting blurness and width and height
        if group_keep_strategy not in GROUP_KEEP_STRATEGIES:
            raise ValueError(f"group_keep_strategy must be one of {GROUP_KEEP_STRATEGIES}, got {group_keep_strategy!r}")
        candidates = self._manifest[eligible]
        if group_keep_strategy == "random":
            candidates = candidates.sample(frac=1, random_state=np.random.default_rng(seed))
        return candidates.drop_duplicates(subset=['group_id'], keep='first').index


class ImageDataset(GroupColumnsMixin, TorchDataset):
    def __init__(self, data_dir, result_dir, transform=None):
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.transform = transform
        
        
        self.paths = find_image_paths(data_dir)
        print(f"Found {len(self.paths)} files\n")
        
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
        
        self.load_manifest()
        
    def has_hash(self):
        return 'hash' in self._manifest.columns and not self._manifest['hash'].isnull().all()
    
    def has_perceptual_hash(self, hash_name):
        return f'{hash_name}_perceptual_hash' in self._manifest.columns and not self._manifest[f'{hash_name}_perceptual_hash'].isnull().all()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_subset(self):
        """Returns a Subset of the dataset containing only the rows that are not dropped."""
        return Subset(self, self.df.index)
    
    def save_perceptual_pairs(self, pairs, hash_name, threshold):
        if not os.path.exists(os.path.join(self.result_dir, "perceptual_pairs")):
            os.makedirs(os.path.join(self.result_dir, "perceptual_pairs"), exist_ok=True)
        pairs_path = os.path.join(self.result_dir, "perceptual_pairs", f"{hash_name}_thr_{threshold}.npy")
        np.save(pairs_path, pairs)
        
    def load_perceptual_pairs(self, hash_name, threshold):
        pairs_path = os.path.join(self.result_dir, "perceptual_pairs", f"{hash_name}_thr_{threshold}.npy")
        if os.path.exists(pairs_path):
            return np.load(pairs_path)
        return None
        
    def group(self, exact_groups: list[list[int]], column: str = 'group_id', workers: int = 4):
            """
            Merges each list of row indices into one new group in `column`.
    
            'group_id' is the duplicate group (drives keep/drop). Merges there are mirrored
            into 'split_group', so split_group is always at least as coarse as group_id.
            'split_group' is only used for group-aware splitting and never drives keep/drop.
    
            The current ids of every group are looked up in parallel (`workers` threads), then
            overlapping groups are merged with a union-find and every merged set gets one new
            UUID. This gives the same result as merging the groups one after another.
            """
            print(f"\nMerging {len(exact_groups)} groups into column '{column}' with {workers} workers...\n")
            columns = ['group_id', 'split_group'] if column == 'group_id' else [column]
            self._ensure_group_column(column)
            
            if not exact_groups:
                return
            
            chunks = [exact_groups[i::workers] for i in range(workers)]
    
            for col in columns:
                series = self._manifest[col]
    
                def lookup(chunk):
                    return [series.loc[g].unique().tolist() for g in chunk]
    
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    per_chunk = list(pool.map(lookup, chunks))
    
                parent = {}
                def find(x):
                    parent.setdefault(x, x)
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
    
                for ids in (ids for chunk in per_chunk for ids in chunk):
                    root = find(ids[0])
                    for other in ids[1:]:
                        parent[find(other)] = root
    
                new_ids = {}
                mapping = {old: new_ids.setdefault(find(old), str(uuid.uuid4())) for old in list(parent)}
                self._manifest[col] = series.map(lambda v: mapping.get(v, v))     
                
                 
    
    def save_perceptual_hashes(self, indices, hashes, hash_name):
        hashes_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes.npy")
        hash_indices_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes_indices.npy")
        np.save(hashes_path, hashes)
        np.save(hash_indices_path, indices)

    def load_perceptual_hashes(self, hash_name):
        hashes_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes.npy")
        hash_indices_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes_indices.npy")
        if os.path.exists(hashes_path) and os.path.exists(hash_indices_path):
            return np.load(hashes_path), np.load(hash_indices_path)
        return None, None
        
    def keep_one_from_group(self, reason: str, group_keep_strategy="random", seed=None):
        """
        Within each group_id, marks one eligible row 'keep' and the other eligible rows 'drop',
        appending `reason` to their existing reason tags (earlier tags are never lost).

        Eligible rows are those not already 'drop' or 'review'. `group_keep_strategy` picks
        which row survives (see GROUP_KEEP_STRATEGIES). Pass `seed` for a reproducible choice.
        """
        eligible = (self._manifest['action'] != 'drop') & (self._manifest['action'] != 'review')
        chosen = self._select_one_per_group(eligible, group_keep_strategy, seed)
        drop = eligible & ~self._manifest.index.isin(chosen)

        self._manifest.loc[eligible & ~drop, 'action'] = 'keep'
        self._manifest.loc[drop, 'action'] = 'drop'
        self._manifest.loc[drop, 'reason'] = append_reason(self._manifest.loc[drop, "reason"], reason)
        self.save_manifest()

    def update_group_ids(self, indices, assigned_group_ids):
        for index, group_id in zip(indices, assigned_group_ids):
            self._manifest.loc[index, 'group_id'] = group_id
        self.save_manifest()

    def get_idx2idx(self):
        return {i: idx for i, idx in enumerate(self._manifest[self._manifest['action'] != 'drop'].index)}
    
    @property
    def df(self):
        return self._manifest[self._manifest['action'] != 'drop'].copy()
    
    def save_manifest(self):
        self._manifest.to_csv(self._manifest_path, index=False)
    
    def restart(self):
        """Gives every row a fresh group_id/split_group and per-model group column (group_<model>_<thr>),
        and clears action and reason. Without resetting the per-model columns, group() would merge each
        run's links into the previous runs'. Other columns (hash, metrics, ...) are left untouched."""
        self._manifest["group_id"] = [str(uuid.uuid4()) for _ in range(len(self._manifest))]
        self._manifest["split_group"] = self._manifest["group_id"]
        for column in self.group_columns():
            self._manifest[column] = [str(uuid.uuid4()) for _ in range(len(self._manifest))]
        self._manifest["action"] = ""
        self._manifest["reason"] = ""
        self.save_manifest()
        record_event(self, RESTART)

    def delete_manifest(self):
        if os.path.exists(self._manifest_path):
            os.remove(self._manifest_path)

    def load_manifest(self, path=None):
        if path is None:
            # is_decodable	exif_orientation
            path = self._manifest_path
        if not os.path.exists(path):
            self._manifest = pd.DataFrame(columns=["path", "group_id", "split_group", "action", "reason", "is_decodable", "exif_orientation"])
            dataset_len = len(self.paths)
            self._manifest["path"] = self.paths
            self._manifest["group_id"] = [str(uuid.uuid4()) for _ in range(dataset_len)]
            self._manifest["split_group"] = self._manifest["group_id"]
            self._manifest["action"] = [""] * dataset_len
            self._manifest["reason"] = [""] * dataset_len
            self._manifest["is_decodable"] = [True] * dataset_len
            self._manifest["exif_orientation"] = [1] * dataset_len
            self._manifest.to_csv(path, index=False)
            record_event(self, MANIFEST_CREATED)
        else:
            self._manifest = pd.read_csv(
                path, 
                dtype={'group_id':str,'split_group':str,'action': str, 'reason': str, 'is_decodable': bool, 'exif_orientation': int},
                keep_default_na=False  # Prevents empty strings from being converted to NaN
            )
            if 'split_group' not in self._manifest.columns:  # manifest from before split_group existed
                self._manifest.insert(3, 'split_group', self._manifest['group_id'])
                self.save_manifest()

class ImageClassificationDataset(GroupColumnsMixin, TorchDataset):
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
        self._ensure_group_column(column)
        for group in exact_groups:
            for col in columns:
                new_group_id = str(uuid.uuid4())
                target_group_ids = self._manifest.loc[group, col].unique()
                mask = self._manifest[col].isin(target_group_ids)
                self._manifest.loc[mask, col] = new_group_id
        self.update_label_conflict()

    def _after_split_group_change(self):
        self.update_label_conflict()

    def update_label_conflict(self):
        """True for rows whose split_group contains more than one distinct label."""
        n_labels = self._manifest.groupby('split_group')['label'].transform('nunique')
        self._manifest['label_conflict'] = n_labels > 1
            
    def save_perceptual_hashes(self, hashes, hash_name):
        hashes_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes.npy")
        np.save(hashes_path, hashes)
    
    def load_perceptual_hashes(self, hash_name):
        hashes_path = os.path.join(self._embeddings_dir, f"{hash_name}_hashes.npy")
        if os.path.exists(hashes_path):
            return np.load(hashes_path)
        return None
        
    def keep_one_from_group(self, reason: str, group_keep_strategy="random", ignore_label_conflict=False, seed=None):
        """
        Within each group_id, marks one eligible row 'keep' and the other eligible rows 'drop'.
        Eligible rows are not dropped and, unless ignore_label_conflict, have no label conflict.
        `group_keep_strategy` picks the survivor (see GROUP_KEEP_STRATEGIES); pass `seed` for
        a reproducible choice. Reasons are appended, so earlier tags are never lost.
        """
        eligible = (self._manifest['action'] != 'drop')
        if not ignore_label_conflict:
            eligible = eligible & ~self._manifest['label_conflict']
        chosen = self._select_one_per_group(eligible, group_keep_strategy, seed)
        drop = eligible & ~self._manifest.index.isin(chosen)

        self._manifest.loc[eligible & ~drop, 'action'] = 'keep'
        self._manifest.loc[drop, 'action'] = 'drop'
        self._manifest.loc[drop, 'reason'] = append_reason(self._manifest.loc[drop, 'reason'], reason)

        if not ignore_label_conflict:
            conflicting = self._manifest['label_conflict'] & (self._manifest['action'] != 'drop')
            self._manifest.loc[conflicting, 'action'] = 'drop'
            self._manifest.loc[conflicting, 'reason'] = append_reason(self._manifest.loc[conflicting, 'reason'], 'label_conflict')

        self.save_manifest()

    def keep_first_in_group(self, reason: str, column='group_id', ignore_label_conflict=False):
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
        
    def keep_first_in_split_group(self, reason: str):
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
            record_event(self, MANIFEST_CREATED)
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

def append_reason(series: pd.Series, tag: str) -> pd.Series:
    """Appends a semicolon-delimited failure tag without duplicating existing tags."""
    return series.fillna("").apply(lambda r: tag if not r else (r if tag in r.split(";") else f"{r};{tag}"))
 

def apply_policy(df: pd.DataFrame, mask: pd.Series, policy: str, tag: str) -> None:
    """Applies 'drop' or 'review' without downgrading an existing 'drop' to 'review'."""
    if not mask.any():
        return
    if policy == "drop":
        df.loc[mask, "action"] = "drop"
    elif policy == "review":
        df.loc[mask & (df["action"] != "drop"), "action"] = "review"
    df.loc[mask, "reason"] = append_reason(df.loc[mask, "reason"], tag)


NULLABLE_DTYPES = {
    "bool": "boolean",
    "int64": "Int64",
    "float64": "Float64",
    "object": "string",
}

def write_back(dataset: ImageDataset | ImageClassificationDataset, df: pd.DataFrame, new_columns: list[str]) -> None:
    """Copies the checked rows back into the full manifest and saves it.

    New columns are created first: assigning through .loc aligns on column names,
    so columns missing from the manifest would otherwise be silently discarded.
    """
    manifest = dataset._manifest
    for col in new_columns:
        if col not in manifest.columns:
            target_dtype = NULLABLE_DTYPES.get(str(df[col].dtype), "object")
            manifest[col] = pd.Series(pd.NA, index=manifest.index, dtype=target_dtype)
    columns = ["action", "reason", *new_columns]
    manifest.loc[df.index, columns] = df[columns]
    dataset.save_manifest()