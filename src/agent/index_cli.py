"""Standalone CLI to build/rebuild the search index without starting the chat agent."""

import argparse
import logging
import warnings

from agent.core import ensure_index
from agent.utils import DEFAULT_DATA_DIR, DEFAULT_INDEX_DIR


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Build QMS search index")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Path to the QMS document directory",
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Path to store the search index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the index even if a prepared index already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ensure_index(data_dir=args.data_dir, index_dir=args.index_dir, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
