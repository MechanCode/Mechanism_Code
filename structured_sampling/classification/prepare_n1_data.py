#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path

from connect_privacy.data import prepare_classification_cache, resolve_split_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HAR", "USC"), required=True)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--validation-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_split_paths(
        args.dataset, args.train_path, args.validation_path, args.test_path
    )
    metadata = prepare_classification_cache(
        *paths,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
