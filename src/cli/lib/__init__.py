
from src.utils.dataset import ImageClassificationDataset, ImageDataset
from src.exact_duplicates.hashing import get_hash_hexdigest
from src.near_duplicates.perceptual_hashes import get_perceptual_hashes

def make_hashes(dataset: ImageDataset | ImageClassificationDataset, hash_name="blake2b"):
    """Hashes all images in the dataset and saves the hashes."""
    dataset._manifest['hash'] = dataset._manifest['path'].apply(
        lambda x: get_hash_hexdigest(x, hash_name=hash_name)
    )
    dataset.save_manifest()
    
def make_perceptual_hashes(dataset: ImageDataset | ImageClassificationDataset, hash_name="phash"):
    """Hashes all images in the dataset using perceptual hashing and saves the hashes."""
    dataset._manifest['perceptual_hash'] = get_perceptual_hashes(
        dataset._manifest['path'].tolist(), 
        hash_name=hash_name
    )
    dataset.save_manifest()
