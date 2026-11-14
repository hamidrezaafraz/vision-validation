import argparse
# from src.utils.dataset import ImageClassificationDataset
from src.cli import ClassificationArgs, classification_cli, DatasetArgs, dataset_cli
# from src.cli import 


def main():
    parser = argparse.ArgumentParser(description="DL Model CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available tasks")
    cls_val_parser = ClassificationArgs.get_parser()
    data_val_parser = DatasetArgs.get_parser()

    data_parser = subparsers.add_parser("data", parents=[data_val_parser], help="Just Dataset")
    cls_parser = subparsers.add_parser("cls", parents=[cls_val_parser], help="Classification tasks")

    # train_group = cls_parser.add_argument_group("Training")
    # train_group.add_argument("--lr", type=float, default=1e-3)
    # train_group.add_argument("--epochs", type=int, default=10)

    subparsers.add_parser("seg", parents=[cls_val_parser], help="Segmentation tasks")

    args = parser.parse_args()

    match args.command:
        case "cls":
            classification_cli(args)
            # dataset = ImageClassificationDataset(args.datadir, args.resultdir)
            # print(dataset.df.head())
        case "data":
            dataset_cli(args)
        case "seg":
            raise NotImplementedError("Segmentation datasets are not supported yet.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
