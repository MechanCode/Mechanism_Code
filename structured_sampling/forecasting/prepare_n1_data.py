#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path

from connect_privacy.data import prepare_standardized_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-path",
        type=Path,
        required=True,
        help="Directory containing the source CSV file",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="CSV path relative to --root-path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory in which to write the standardized cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_standardized_cache(
        root_path=args.root_path,
        data_path=args.data_path,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
