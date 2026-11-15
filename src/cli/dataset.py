import argparse
from dataclasses import MISSING, dataclass, field, fields
from typing import ClassVar

from src.utils.dataset import ImageDataset
from src.utils.images import find_image_paths
from src.near_duplicates import verify_near_duplicates, verify_near_embeddings
from src.cli.lib import make_embeddings, make_hashes, group_by_hash, make_perceptual_hashes, group_by_perceptual_hash, group_by_embeddings

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
        "Validation": "Dataset validation steps",
        "Exact hash duplicates": "Exact hash duplicate detection",
        "Exact byte-level duplicates": "Exact byte-level duplicate detection",
        "Near duplicates": "Near duplicate detection using perceptual hashes",
        "Semantic embeddings": "Semantic embedding detection using pre-trained models",
        "Domain expert": "Embedding detection using domain expert models (e.g., face recognition)",
    }

    datadir: str = arg("Root directory of the dataset", "Data")
    resultdir: str = arg("Root directory of the results", "Data")
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

    print(f"Validating images...")
    image_paths = find_image_paths(data_args.datadir)
    print(f"Found {len(image_paths)} valid images\n")

    dataset = ImageDataset(image_paths, data_args.resultdir)

    
    if data_args.restart:
        dataset.restart()
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
   