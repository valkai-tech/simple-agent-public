"""Validate that the prepared search index can serve retrieval requests."""

from __future__ import annotations

import argparse

from agent.search import SearchIndex
from agent.utils import DEFAULT_INDEX_DIR


def main() -> None:
    """Load the index and run a retrieval-only setup check."""
    parser = argparse.ArgumentParser(description="Check the prepared QMS search index")
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Path to the prepared search index",
    )
    args = parser.parse_args()

    index = SearchIndex(args.index_dir)
    documents = index.list_all_documents()
    unique_document_ids = {document["doc_id"] for document in documents}
    results = index.search(
        "MX1 top-level Bill of Materials",
        mode="hybrid",
        doc_type="BOM",
        latest_only=True,
        n_results=5,
    )

    if not documents or not results:
        raise SystemExit("Index check failed: the corpus or retrieval index is empty")

    top_matches = ", ".join(
        f"{result['doc_id']} Rev {result['revision']}" for result in results
    )
    print(
        f"Index ready: {len(documents)} revisions across "
        f"{len(unique_document_ids)} document IDs."
    )
    print(f"Retrieval check: {top_matches}")


if __name__ == "__main__":
    main()
