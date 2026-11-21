import argparse
from dataclasses import MISSING, dataclass, field, fields
from typing import ClassVar

from src.photometric import check_photometric
from src.utils.dataset import ImageDataset
from src.utils.images import find_image_paths
from src.near_duplicates import verify_near_duplicates, verify_near_embeddings
from src.cli.lib import make_embeddings, make_hashes, group_by_hash, make_perceptual_hashes, group_by_perceptual_hash, group_by_embeddings
from src.integrity import check_decode

def arg(help: str, group: str, default=MISSING, choices=None):
    """Declares a dataclass field as a CLI flag. Fields without a default become required flags."""
    return field(default=default, metadata={"help": help, "group": group, "choices": choices})


@dataclass
class DatasetArgs:
    """
    Options shared by every task (cls, seg, ...) for validating its dataset.
    Each field becomes a --<field name> flag in the argument group named in its metadata.
    bool fields become store_true flags.
    example:

        parser = argparse.ArgumentParser(parents=[DatasetArgs.get_parser()])
        cls_args = DatasetArgs.from_namespace(parser.parse_args())

    """
    GROUPS: ClassVar[dict[str, str]] = {
        "Data": "Dataset-related options",
        "Integrity": "Image file integrity and photometric quality checks",
        "Validation": "Dataset validation steps",
        "Exact hash duplicates": "Exact hash duplicate detection",
        "Exact byte-level duplicates": "Exact byte-level duplicate detection",
        "Near duplicates": "Near duplicate detection using perceptual hashes",
        "Semantic embeddings": "Semantic embedding detection using pre-trained models",
        "Domain expert": "Embedding detection using domain expert models (e.g., face recognition)",
    }

    datadir: str = arg("Root directory of the dataset", "Data")
    resultdir: str = arg("Root directory of the results", "Data")
    num_workers: int = arg("Worker threads for parallel image I/O checks", "Data", default=8)
    check_decode: bool = arg("Decode full pixel buffers, check EXIF orientation, validate color modes and bit-depths, and check width, height and aspect ratio", "Integrity", default=False)
    expected_modes: tuple = arg(
        "Allowed Pillow image modes (e.g., RGB L RGBA)",
        "Integrity",
        default=("RGB",),
        choices=["RGB", "L", "RGBA", "CMYK", "P", "LA", "1", "I;16", "I", "F"],
    )
    expected_dtypes: tuple = arg(
        "Allowed NumPy pixel dtypes (e.g., uint8 uint16)",
        "Integrity",
        default=("uint8",),
        choices=["uint8", "uint16", "int32", "float32", "bool"],
    )
    format_mismatch_policy: str = arg(
        "Action when color mode or bit-depth mismatches ('review' or 'drop')",
        "Integrity",
        default="review",
        choices=["review", "drop"],
    )
    min_width: int = arg("Minimum allowed image width in pixels", "Integrity", default=32)
    min_height: int = arg("Minimum allowed image height in pixels", "Integrity", default=32)
    max_width: int = arg("Maximum allowed image width in pixels", "Integrity", default=8192)
    max_height: int = arg("Maximum allowed image height in pixels", "Integrity", default=8192)
    max_aspect_ratio: float = arg("Maximum allowed symmetric aspect ratio max(w/h, h/w)", "Integrity", default=5.0)
    dimension_violation_policy: str = arg(
        "Action when resolution or aspect ratio exceeds bounds ('review' or 'drop')",
        "Integrity",
        default="review",
        choices=["review", "drop"],
    )
    
    check_photometric: bool = arg("Check for blank frames, extreme exposure, and blur", "Integrity", default=False)
    min_pixel_std: float = arg("Minimum normalized pixel std in [0, 1] before flagging blank/solid frames", "Integrity", default=0.02)
    max_clip_ratio: float = arg("Maximum fraction of near-black or near-white pixels before flagging exposure", "Integrity", default=0.90)
    min_blur_score: float = arg("Minimum Laplacian variance on <=512px luminance before flagging blur", "Integrity", default=40.0)
    photometric_violation_policy: str = arg(
        "Action when photometric quality fails ('review' or 'drop')",
        "Integrity",
        default="review",
        choices=["review", "drop"],
    )
    
    
    restart: bool = arg("Restart the validation process from scratch, ignoring any previous results", "Validation", default=False)
    
    make_hashes : bool = arg("Hash every image and store the hashes", "Exact hash duplicates", default=False)
    group_hash: bool = arg("Group images with identical file hashes (hashes them first if needed)", "Exact hash duplicates", default=False)
    
    make_perceptual_hashes: bool = arg("Compute perceptual hashes for every image and store the hashes", "Near duplicates", default=False)
    perceptual_hash: str = arg("Perceptual hash function to use ('phash', 'average_hash', 'dhash', 'whash')", "Near duplicates", default='phash', choices=['phash', 'average_hash', 'dhash', 'whash'])
    group_perceptual_hash: bool = arg("Group images whose perceptual hashes are within --hamming_threshold bits (hashes them first if needed)", "Near duplicates", default=False)
    hamming_threshold: int = arg("Max Hamming distance (inclusive) for two perceptual hashes to count as near duplicates", "Near duplicates", default=1)
    
    make_general_embeddings: bool = arg("Compute general embeddings for every image and store the embeddings", "Semantic embeddings", default=False)
    group_embeddings: bool = arg("Group images whose embeddings have cosine similarity >= --embedding_threshold (embeds them first if needed)", "Semantic embeddings", default=False)
    embedding_model_name: str = arg("Model used to compute the embeddings", "Semantic embeddings", default='alexnet')
    embedding_threshold: float = arg("Min cosine similarity for two embeddings to count as near duplicates", "Semantic embeddings", default=0.95)
    embedding_batch_size: int = arg("Batch size used when computing embeddings", "Semantic embeddings", default=128)
    
    make_expert_embeddings: bool = arg("Compute domain expert embeddings for every image and store the embeddings", "Domain expert", default=False)
    group_expert_embeddings: bool = arg("Group images whose domain expert embeddings have cosine similarity >= --domain_expert_threshold (embeds them first if needed)", "Domain expert", default=False)
    domain_expert_model_names: tuple = arg("Names of the domain expert models to apply one by one, e.g. --domain_expert_model_names facenet-vggface2 facenet-casia-webface", "Domain expert", default=('facenet-vggface2',), choices=['facenet-vggface2', 'facenet-casia-webface'])
    domain_expert_threshold: float = arg("Min cosine similarity for two domain expert embeddings to count as near duplicates", "Domain expert", default=0.95)
    domain_expert_batch_size: int = arg("Batch size used when computing domain expert embeddings", "Domain expert", default=128)
    
        
    # keep_first_in_group: bool = arg("Keep the first image of each group and mark the rest as drop", "Validation", default=False)
    # restart: bool = arg("Restart the validation process from scratch, ignoring any previous results", "Validation", default=False)
    
    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """Builds a parent parser (add_help=False) with one argument group per entry in GROUPS."""
        parser = argparse.ArgumentParser(add_help=False)
        groups = {name: parser.add_argument_group(name, description) for name, description in cls.GROUPS.items()}
        for f in fields(cls):
            kwargs = {"help": f.metadata["help"]}
            if f.type is bool:
                kwargs["action"] = "store_true"
            elif f.type in (list, tuple):
                kwargs.update(type=str, nargs="+", default=list(f.default), choices=f.metadata["choices"])
            elif f.default is MISSING:
                kwargs.update(type=f.type, required=True)
            else:
                kwargs.update(type=f.type, default=f.default)
            groups[f.metadata["group"]].add_argument(f"--{f.name}", **kwargs)
        return parser

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "DatasetArgs":
        """Picks this class's fields out of a parsed namespace, ignoring extras like `command`."""
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})



def dataset_cli(args: argparse.Namespace):
    """CLI for dataset tasks."""
    data_args = DatasetArgs.from_namespace(args)
    
    print(f"Validating dataset in \"{data_args.datadir}\"")
    print(f"Storing results in \"{data_args.resultdir}\"\n")
       
    dataset = ImageDataset(data_args.datadir, data_args.resultdir)

    
    if data_args.restart:
        dataset.restart()
    if data_args.check_decode:
        check_decode(
            dataset,
            num_workers=data_args.num_workers,
            expected_modes=data_args.expected_modes,
            expected_dtypes=data_args.expected_dtypes,
            on_mismatch=data_args.format_mismatch_policy,
            min_width=data_args.min_width,
            min_height=data_args.min_height,
            max_width=data_args.max_width,
            max_height=data_args.max_height,
            max_aspect_ratio=data_args.max_aspect_ratio,
            on_dimension_violation=data_args.dimension_violation_policy,
        )
    if data_args.check_photometric:
        check_photometric(
            dataset,
            min_std=data_args.min_pixel_std,
            max_clip_ratio=data_args.max_clip_ratio,
            min_blur_score=data_args.min_blur_score,
            on_violation=data_args.photometric_violation_policy,
            num_workers=data_args.num_workers,
        )
    if data_args.make_hashes:
        make_hashes(dataset)
    if data_args.group_hash:
        group_by_hash(dataset)
    if data_args.make_perceptual_hashes:
        make_perceptual_hashes(dataset, hash_name=data_args.perceptual_hash)
    if data_args.group_perceptual_hash:
        group_by_perceptual_hash(dataset, hash_name=data_args.perceptual_hash, threshold=data_args.hamming_threshold)
        
    if data_args.make_general_embeddings:
        if not dataset.has_hash():
            print("Hashing images before computing embeddings...")
            make_hashes(dataset)
        make_embeddings(dataset, model_name=data_args.embedding_model_name, batch_size=data_args.embedding_batch_size)
    
    if data_args.group_embeddings:
        if not dataset.has_hash():
            print("Hashing images before grouping by embeddings...")
            make_hashes(dataset)
        group_by_embeddings(dataset, model_name=data_args.embedding_model_name, threshold=data_args.embedding_threshold)
        
    if data_args.make_expert_embeddings:
        if not dataset.has_hash():
            print("Hashing images before computing domain expert embeddings...")
            make_hashes(dataset)
        for model_name in data_args.domain_expert_model_names:
            make_embeddings(dataset, model_name=model_name, batch_size=data_args.domain_expert_batch_size)

    if data_args.group_expert_embeddings:
        if not dataset.has_hash():
            print("Hashing images before grouping by domain expert embeddings...")
            make_hashes(dataset)
        for model_name in data_args.domain_expert_model_names:
            group_by_embeddings(dataset, model_name=model_name, threshold=data_args.domain_expert_threshold)
    print(f"Remaining images after validation: {len(dataset.df)}")
   