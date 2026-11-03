import hashlib
from collections import defaultdict
from typing import Callable

def get_hash_function(hash_name: str) -> Callable[..., "hashlib._Hash"]:
    """Returns a hash function based on the given hash name."""
    if hash_name == "blake2b":
        return hashlib.blake2b
    elif hash_name == "sha256":
        return hashlib.sha256
    elif hash_name == "md5":
        return hashlib.md5
    elif hash_name == "sha1":
        return hashlib.sha1
    elif hash_name == "sha384":
        return hashlib.sha384
    else:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}")

def get_hash_hexdigest(filepath, chunk_size=65536, hash_name="blake2b") -> str:
    """Generates a hash for a file, reading it in chunks to save RAM.
        return the hexdigest of the hash.
    """
    hash_algorithm = get_hash_function(hash_name)
    file_hash = hash_algorithm()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def get_paths_by_hash(paths: list[tuple[int, str]], chunk_size=65536, hash_name="blake2b") -> defaultdict[str, list[int]]:
    """
    Returns a dictionary mapping hashes to lists of file paths indices.
    example:
        
        paths_by_hash = get_paths_by_hash(zip(list(df.index), list(df["path"].values)))
        
        paths_by_hash = get_paths_by_hash(list(enumerate(list(df["path"].values))))
        
    """
    hashed_paths = defaultdict(list)
    
    for idx, path in paths:
        file_hash = get_hash_hexdigest(path, chunk_size=chunk_size, hash_name=hash_name)
        hashed_paths[file_hash].append(idx)
    return hashed_paths