"""LangChain tool wrappers for the QMS search agent.

Thin formatting layer on top of SearchIndex. Tools are created via
create_tools() which closes over the SearchIndex instance, avoiding globals.
"""

from __future__ import annotations

import json
import logging
import sys
import threading

from langchain_core.tools import tool

from agent.search import SearchIndex
from agent.tracing import _truncate
from agent.utils import MAX_READ_CHARS, VALID_SEARCH_MODES, SearchResult

logger = logging.getLogger(__name__)

# Live tool call logging (prints to stderr so it appears immediately)
_tool_step = 0
_live_logging = True
DIM = "\033[2m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def reset_tool_steps() -> None:
    """Reset step counter. Call before each user turn."""
    global _tool_step
    _tool_step = 0


def set_live_logging(enabled: bool) -> None:
    """Enable/disable live tool call logging."""
    global _live_logging
    _live_logging = enabled


LIVE_OUTPUT_MAX = 300
GREEN = "\033[32m"

_step_local = threading.local()


def _log_tool_call(name: str, **kwargs) -> None:
    if not _live_logging:
        return
    global _tool_step
    _tool_step += 1
    step = _tool_step
    _step_local.step = step
    args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in kwargs.items() if v)
    print(
        f"\n{DIM}[step {step}]{RESET} {CYAN}{BOLD}{name}{RESET}({args_str})",
        file=sys.stderr,
        flush=True,
    )


def _log_tool_output(text: str) -> None:
    if not _live_logging:
        return
    step = getattr(_step_local, "step", "?")
    truncated = _truncate(text, LIVE_OUTPUT_MAX)
    print(
        f"  {DIM}[step {step}]{RESET} {GREEN}→{RESET} {DIM}{truncated}{RESET}",
        file=sys.stderr,
        flush=True,
    )


def _validate_mode(mode: str) -> str:
    if mode in VALID_SEARCH_MODES:
        return mode
    logger.warning("Invalid search mode '%s', falling back to 'hybrid'", mode)
    return "hybrid"


def _format_result_for_json(r: SearchResult) -> dict:
    """Build a clean JSON-serializable result dict."""
    entry: dict = {
        "doc_id": r["doc_id"],
        "revision": r["revision"],
        "title": r["title"],
        "type": r.get("doc_type_label", r.get("doc_type", "")),
    }
    if r.get("is_latest"):
        entry["is_latest"] = True
    if r.get("is_obsolete"):
        entry["is_obsolete"] = True
    entry["matching_snippets"] = r.get("matching_snippets", [])
    return entry


def create_tools(index: SearchIndex) -> list:
    """Create LangChain tools bound to the given SearchIndex.

    Returns a list of tool callables that close over the index instance.
    """

    @tool
    def search(
        query: str,
        mode: str = "hybrid",
        doc_type: str = "",
        latest_only: bool = False,
        n_results: int = 10,
    ) -> str:
        """Search the QMS document corpus.

        Finds the most relevant documents by searching within their content, then
        returns one result per document with matching passage snippets.

        Args:
            query: Search query. Can be natural language, document IDs, keywords, etc.
            mode: Search mode - "hybrid" (default, combines keyword+semantic),
                  "keyword" (exact term matching, good for doc IDs and specific terms),
                  "semantic" (meaning-based, good for conceptual queries).
            doc_type: Filter by document type prefix (e.g. "VVPR", "BOM", "RSK",
                      "MEMO", "PLN", "ECR"). Leave empty for all types.
            latest_only: If true, only return the latest revision of each document.
            n_results: Number of results to return (default 10).

        Returns:
            JSON array of search results. Each result has doc_id, revision, title,
            type, matching_snippets, and status flags. Use doc_id and revision to
            call read_document() for the full text.
        """
        _log_tool_call(
            "search",
            query=query,
            mode=mode,
            doc_type=doc_type,
            latest_only=latest_only if latest_only else "",
            n_results=n_results if n_results != 10 else "",
        )
        validated_mode = _validate_mode(mode)

        results = index.search(
            query=query,
            mode=validated_mode,
            doc_type=doc_type.upper() or None,
            latest_only=latest_only,
            n_results=n_results,
        )

        if not results:
            result = json.dumps({"count": 0, "results": []})
            _log_tool_output(result)
            return result

        output: dict = {
            "count": len(results),
            "results": [_format_result_for_json(r) for r in results],
        }
        if len(results) >= n_results:
            output["has_more"] = True

        result = json.dumps(output, indent=2)
        _log_tool_output(result)
        return result

    @tool
    def read_document(doc_id: str, revision: str = "") -> str:
        """Read the full content of a specific document.

        Use this AFTER searching, when you need to read a document's complete text
        to extract specific details, answer questions about its content, or
        compare it with other documents.

        Args:
            doc_id: The document ID (e.g. "BOM-055", "VVPR-P01-081", "RSK-P01-010").
            revision: Specific revision letter (e.g. "A", "B", "G").
                      If empty, returns the latest revision.

        Returns:
            Full document text with metadata header.
        """
        _log_tool_call("read_document", doc_id=doc_id, revision=revision or "")
        doc, fuzzy_warning = index.read_document(doc_id, revision=revision or None)

        if doc is None:
            result = (
                f"Document '{doc_id}' (revision: {revision or 'latest'}) not found."
            )
            _log_tool_output(result)
            return result

        body = doc.get("body", "")
        if not body:
            result = (
                f"Document {doc['doc_id']} Rev {doc['revision']} ({doc.get('filename', '')}) "
                f"exists but has no extractable text content."
            )
            _log_tool_output(result)
            return result

        header = (
            f"=== {doc['doc_id']} Rev {doc['revision']} ===\n"
            f"Title: {doc.get('title', '')}\n"
            f"Filename: {doc.get('filename', '')}\n"
            f"{'=' * 50}\n\n"
        )
        _log_tool_output(
            f"{doc['doc_id']} Rev {doc['revision']}: {doc.get('title', '')} ({len(body)} chars)"
        )

        # Fuzzy matches return a stub so the agent can verify before reading in full
        if fuzzy_warning:
            preview = body[:500] + "..." if len(body) > 500 else body
            return (
                f"NOTE: {fuzzy_warning}\n"
                f'Showing preview only. Call read_document(doc_id="{doc["doc_id"]}") '
                f"to read the full document.\n\n"
                f"{header}{preview}"
            )

        if len(body) > MAX_READ_CHARS:
            body = (
                body[:MAX_READ_CHARS]
                + f"\n\n... [truncated, {len(body)} total characters]"
            )

        return header + body

    @tool
    def list_documents(doc_type: str = "") -> str:
        """List all documents in the corpus, optionally filtered by type.

        Use this for enumeration queries ("how many X do we have?") or
        to get a complete inventory of documents.

        Args:
            doc_type: Filter by type prefix (e.g. "VVPR", "BOM", "ECR", "RSK",
                      "PLN", "MEMO"). Leave empty to list all documents.

        Returns:
            JSON with count, unique_ids count, and documents grouped by ID.
        """
        _log_tool_call("list_documents", doc_type=doc_type or "")
        docs = index.list_all_documents(doc_type=doc_type.upper() if doc_type else None)

        if not docs:
            result = json.dumps({"count": 0, "documents": []})
            _log_tool_output(result)
            return result

        by_id: dict[str, list[dict]] = {}
        for d in docs:
            by_id.setdefault(d["doc_id"], []).append(d)

        documents = []
        for did in sorted(by_id):
            revs = sorted(by_id[did], key=lambda x: x["revision"])
            rev_list = []
            for r in revs:
                entry: dict = {"revision": r["revision"]}
                if r.get("is_latest"):
                    entry["is_latest"] = True
                if r.get("is_obsolete"):
                    entry["is_obsolete"] = True
                rev_list.append(entry)
            documents.append(
                {
                    "doc_id": did,
                    "title": revs[0].get("title", ""),
                    "type": revs[0].get("doc_type_label", revs[0].get("doc_type", "")),
                    "revisions": rev_list,
                }
            )

        output: dict = {
            "count": len(docs),
            "unique_ids": len(by_id),
            "documents": documents,
        }
        if doc_type:
            output["filtered_by"] = doc_type.upper()

        result = json.dumps(output, indent=2)
        _log_tool_output(result)
        return result

    return [search, read_document, list_documents]
