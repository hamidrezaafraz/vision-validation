import argparse
from dataclasses import MISSING, dataclass, field, fields
from typing import ClassVar

from src.utils.dataset import ImageDataset
from src.exact_duplicates import verify_exact_duplicates, verify_strict_duplicates
from src.near_duplicates import verify_near_duplicates, verify_near_embeddings

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
    cls_args = DatasetArgs.from_namespace(args)
    
    print(f"Validating dataset in \"{cls_args.datadir}\"")
    print(f"Storing results in \"{cls_args.resultdir}\"")
    
    # dataset = ImageClassificationDataset(cls_args.datadir, cls_args.resultdir)
    
    # if cls_args.restart:
    #     dataset.restart()
    # if cls_args.verify_exact_duplicates:
    #     verify_exact_duplicates(dataset)
    #     if cls_args.keep_first_in_group: dataset.keep_first_in_group("exact hash duplicates", ignore_label_conflict=cls_args.ignore_label_conflict)
    # if cls_args.verify_strict_duplicates:
    #     verify_strict_duplicates(dataset)
    #     if cls_args.keep_first_in_group: dataset.keep_first_in_group("exact byte-level duplicates", ignore_label_conflict=cls_args.ignore_label_conflict)
    # if cls_args.verify_near_duplicates:
    #     verify_near_duplicates(dataset, hash_name=cls_args.perceptual_hash, threshold=cls_args.hamming_threshold)
    #     if cls_args.keep_first_in_group: dataset.keep_first_in_group("near duplicates", ignore_label_conflict=cls_args.ignore_label_conflict)
    # if cls_args.verify_near_embeddings:
    #     verify_near_embeddings(dataset, model_name=cls_args.embedding_model_name, threshold=cls_args.embedding_threshold, batch_size=cls_args.embedding_batch_size, ignore_label_conflict=cls_args.ignore_label_conflict)
    #     if cls_args.keep_first_in_group: dataset.keep_first_in_group("semantic embeddings", column='split_group', ignore_label_conflict=cls_args.ignore_label_conflict)
    # if cls_args.domain_expert_embedding:
    #     for model_name in cls_args.domain_expert_model_names:
    #         print(f"Applying domain expert model \"{model_name}\"")
    #         verify_near_embeddings(dataset, model_name=model_name, threshold=cls_args.domain_expert_threshold, batch_size=cls_args.domain_expert_batch_size, ignore_label_conflict=cls_args.ignore_label_conflict)
    #         if cls_args.keep_first_in_group: dataset.keep_first_in_group(f"domain expert embeddings ({model_name})", column='split_group', ignore_label_conflict=cls_args.ignore_label_conflict)

