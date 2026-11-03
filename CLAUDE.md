# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Facial emotion image classification (PyTorch/torchvision). Current work is **dataset cleaning**: finding exact, perceptual and embedding-level duplicates before training. `todo.md` (gitignored) tracks the roadmap. The plan there: fix the test/val split once, drop train items that duplicate test/val items, then compare training on increasingly aggressive cleaning (none → exact → exact+pHash → exact+pHash+embedding similarity).

## Environment & commands

- Python 3.10, managed with `uv` (`pyproject.toml` + `uv.lock`, venv in `.venv`).
- Install deps: `uv sync`
- Run anything with the venv: `uv run python main.py --help`
- Run from the repo root. Imports are absolute from the root (`from src.utils...`, `from emotion...`), and many output paths are relative (`emotion/results/...`, `f"{task_name}/results"`).
- Tests use pytest (dev dependency; config is in `pyproject.toml`, which puts the repo root on `sys.path`):
  - all: `uv run pytest`
  - one file: `uv run pytest tests/exact_duplicates/hashing/test_hashing.py`
  - one test: `uv run pytest "tests/utils/test_dataset.py::TestGrouping::test_keep_first_in_group"`
- No linter or formatter is configured.

## Tests layout

- `tests/` mirrors `src/`: `src/<pkg>/<module>` is tested in `tests/<pkg>/<module>/test_<module>.py` (or `tests/<pkg>/test_<module>.py` for single-file modules like `src/utils/dataset.py`). Each test folder needs an `__init__.py` because test packages share names with `src` packages.
- Group the tests for each function in a `Test<FunctionName>` class.
- Shared fixtures are in `tests/conftest.py`: `make_file` (bytes → tmp file), `make_image_folder` (builds a tmp `ImageFolder` tree from solid-color PNGs, where equal colors give byte-identical files) and `example_duplicate_images` / `copy_example_images` (the identical PNGs in `src/example_data/`). Tests must write only to `tmp_path`. Pass `result_dir=str(tmp_path)` to `ImageClassificationDataset`. It is now a required argument; it used to be the optional `manifest_dir`, which defaulted to `./Resultssss`.

## Data layout

- Images live at `emotion/Data/<Class>/*.jpg` (classes: Angry, Fear, Happy, Sad, Suprise — spelling is from the dataset, so don't "fix" it). They're loaded with `torchvision.datasets.ImageFolder`, so labels are alphabetical class indices.
- Outputs go to `emotion/results/` (gitignored, as are `*/Data` and all `*.ipynb`).
- Absolute paths under `/Users/threewestrick/...` are hardcoded in `emotion/dataset.py` and `emotion/cli.py`.

## Architecture

- `src/` holds generic, task-agnostic code. `emotion/` is the task-specific layer that plugs into it. A task supplies a `get_dataset(train_tf, val_tf)` callable that returns `(train_full, val_full)`, i.e. two views of the same files that differ only in transform. It also supplies a `task_name` string that is used as the output directory prefix. Functions in `src/utils/dataset.py` take `(get_dataset, task_name, ...)`.
- **Manifest table** (`ImageClassificationDataset(root_dir, result_dir, transform=None)` in `src/utils/dataset.py`, which writes `manifest.csv` and `audits.json` into `result_dir`): one CSV row per image with columns `path, label, group_id, action, reason`. Files are never deleted. Cleaning is recorded as `action = keep|drop` plus a `reason`. Each image starts with a unique UUID `group_id`. `group()` merges the indices of a duplicate set into one new UUID group. `keep_first_in_group()` keeps the first row of each group and drops the rest, without overwriting an existing `reason`. The `.df` property returns the rows that are not dropped. The manifest must be read with `dtype=str` and `keep_default_na=False` so that empty strings survive.
- **Exact-duplicate pipeline** (`src/exact_duplicates/`):
  1. `hashing.get_paths_by_hash` buckets `(index, path)` pairs by file hash (default blake2b, read in chunks).
  2. Buckets go into `group()`.
  3. `file_compare.find_strict_duplicates` byte-compares the members of each multi-member group with `filecmp` and returns the subgroups that need new ids, which guards against hash collisions.
  4. `keep_first_in_group(reason=...)` marks the result.
  This is currently driven from notebooks (`exact_duplicates.ipynb`), not the CLI.
- **Older near-duplicate path** (still in `src/utils/dataset.py`): `verify_perceptual_hashes` builds ±1 pHash vectors so that cosine similarity maps to Hamming distance. `verify_embeddings` uses AlexNet features with cosine ≥ 0.95. Both use `similar_pairs`, a chunked matmul on mps/cuda/cpu. Their results go to `.npy` files (`*_pairs.npy`, `data_indices.npy`), not to the manifest. `get_train_data` maps `Subset` indices back to absolute `ImageFolder` indices. Keep that local→absolute mapping whenever you save indices. `verify_exact_duplicates` is currently an empty stub.
- `emotion/dataset.py` is mid-migration. `EmotionDataset` subclasses the manifest class, but `get_dataset`/`get_loader` still build raw `ImageFolder` + `Subset` objects and write their own manifest variant.
- CLI: `main.py` (argparse) has `cls` and `seg` subcommands. Both inherit a shared parent parser built by `ClassificationArgs.get_parser()` in `src/cli/classification.py`. `ClassificationArgs` is a dataclass and is the single source of truth for the flags: each field declared with `arg(help, group, default=...)` becomes `--<field name>` in the argument group named in `GROUPS`. bool fields become `store_true` flags, and fields without a default become required. To add a flag, add a field; don't call `add_argument` directly. `classification_cli(args)` turns the namespace into a `ClassificationArgs` via `from_namespace` but doesn't act on the flags yet. `seg` raises `NotImplementedError`. CLI argument definitions go in `src/cli/`, and the logic they call stays in `src/exact_duplicates/` and `src/utils/`. The older `emotion verify_dataset` subcommand in `emotion/cli.py` (`EmitionCLI`) is no longer wired into `main.py`.
- `a.py` (tracked) is a scratch demo of `questionary` prompts. It isn't imported anywhere.
- `src/utils/logger.py` provides a colored `Logger(__name__)` wrapper. Use it instead of `print`/`logging` directly.
