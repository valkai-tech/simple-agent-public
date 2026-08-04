"""Search index: wraps ChromaDB + BM25 for hybrid search over QMS documents.

Pure search logic -- no LangChain/framework dependencies. The LangChain
tool wrappers live in tools.py.
"""

from __future__ import annotations

import json
from typing import Literal

import chromadb
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

from agent.utils import (
    RRF_K,
    RRF_OVERFETCH_FACTOR,
    SEARCH_OVERFETCH_FACTOR,
    SNIPPET_MAX_CHARS,
    BodyStoreEntry,
    SearchResult,
    make_unique_id,
    strip_section_header,
    tokenize,
)


class SearchIndex:
    """Wraps ChromaDB + BM25 for hybrid search over QMS documents.

    Searching operates on the *sections* collection and deduplicates
    results by parent document, keeping the best-scoring section per doc.
    Document-level metadata (for list/read) comes from the body store JSON.
    """

    def __init__(self, persist_dir: str = ".chroma_index"):
        self.persist_dir = persist_dir
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._section_collection = self._client.get_collection("sections")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open ChromaDB index at {persist_dir}. "
                f"Run `uv run index` first. Error: {e}"
            ) from e

        self._body_store: dict[str, BodyStoreEntry] = self._load_body_store()
        self._bm25, self._bm25_ids = self._build_bm25()

    def _load_body_store(self) -> dict[str, BodyStoreEntry]:
        path = f"{self.persist_dir}/body_store.json"
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Body store not found at {path}. Run `uv run index` first."
            )

    def _build_bm25(self) -> tuple[BM25Okapi, list[str]]:
        """Build a BM25 index over the sections collection."""
        all_data = self._section_collection.get(include=["documents"])
        ids = all_data["ids"]
        docs = all_data["documents"]
        tokenized = [tokenize(d) for d in docs]
        return BM25Okapi(tokenized), ids

    def search(
        self,
        query: str,
        mode: Literal["keyword", "semantic", "hybrid"] = "hybrid",
        doc_type: str | None = None,
        latest_only: bool = False,
        n_results: int = 10,
    ) -> list[SearchResult]:
        """Search the corpus and return the best-matching section per document."""
        where_filter = self._build_filter(doc_type=doc_type, latest_only=latest_only)
        fetch_n = max(n_results * SEARCH_OVERFETCH_FACTOR, 50)

        if mode == "keyword":
            raw = self._keyword_search(query, fetch_n, where_filter)
        elif mode == "semantic":
            raw = self._semantic_search(query, fetch_n, where_filter)
        else:
            raw = self._hybrid_search(query, fetch_n, where_filter)

        return self._group_and_rank_snippets(raw, n_results, query)

    FUZZY_MATCH_THRESHOLD = 80

    def read_document(
        self,
        doc_id: str,
        revision: str | None = None,
    ) -> tuple[BodyStoreEntry | None, str | None]:
        """Read full document content by doc_id and optional revision.

        Returns (document, fuzzy_warning). fuzzy_warning is None for exact matches,
        or a message like "No exact match for 'BOM-55'. Showing closest match: BOM-055."
        """
        # 1. Exact match on full key
        if revision:
            key = make_unique_id(doc_id, revision)
            if key in self._body_store:
                return self._body_store[key], None

        # 2. Exact match on doc_id
        matches = [(k, v) for k, v in self._body_store.items() if v["doc_id"] == doc_id]

        # 3. Case-insensitive match
        if not matches:
            doc_id_upper = doc_id.upper()
            matches = [
                (k, v)
                for k, v in self._body_store.items()
                if v["doc_id"].upper() == doc_id_upper
            ]

        # 4. Fuzzy match as last resort
        if not matches:
            fuzzy_doc = self._fuzzy_find_doc_id(doc_id)
            if fuzzy_doc:
                warning = f"No exact match for '{doc_id}'. Showing closest match: {fuzzy_doc}."
                return self.read_document(fuzzy_doc, revision)[0], warning
            return None, None

        if revision:
            rev_upper = revision.upper()
            for _, v in matches:
                if v["revision"].upper() == rev_upper:
                    return v, None

        matches.sort(key=lambda x: x[1]["revision"])
        return matches[-1][1], None

    def _fuzzy_find_doc_id(self, doc_id: str) -> str | None:
        """Find the closest doc_id using fuzzy string matching."""
        all_doc_ids = list({v["doc_id"] for v in self._body_store.values()})
        result = process.extractOne(
            doc_id,
            all_doc_ids,
            scorer=fuzz.ratio,
            score_cutoff=self.FUZZY_MATCH_THRESHOLD,
        )
        if result:
            return result[0]
        return None

    def list_all_documents(self, doc_type: str | None = None) -> list[dict]:
        """List all documents from the body store (metadata only, no body text).

        This is an O(n) loop over the body store. For a large corpus, we would build an index on important fields like doc_id and doc_type.
        """
        results = []
        for entry in self._body_store.values():
            if doc_type and entry.get("doc_type", "").upper() != doc_type.upper():
                continue
            results.append({k: v for k, v in entry.items() if k != "body"})
        return results

    @staticmethod
    def _build_filter(
        doc_type: str | None = None,
        latest_only: bool = False,
    ) -> dict | None:
        conditions = []
        if doc_type:
            conditions.append({"doc_type": doc_type.upper()})
        if latest_only:
            conditions.append({"is_latest": True})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _semantic_search(
        self,
        query: str,
        n_results: int,
        where_filter: dict | None,
    ) -> list[dict]:
        kwargs: dict = {
            "query_texts": [query],
            "n_results": min(n_results, self._section_collection.count()),
            "include": ["metadatas", "documents", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter
        results = self._section_collection.query(**kwargs)
        return self._format_chroma_results(results)

    def _keyword_search(
        self,
        query: str,
        n_results: int,
        where_filter: dict | None,
    ) -> list[dict]:
        # Score all sections against the query using BM25
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # If filtering (e.g. doc_type), get the set of valid section IDs from ChromaDB
        if where_filter:
            filtered = self._section_collection.get(
                where=where_filter, include=["metadatas"]
            )
            valid_ids = set(filtered["ids"])
        else:
            valid_ids = None

        # Pair each section ID with its BM25 score, excluding filtered-out sections
        scored = []
        for idx, score in enumerate(scores):
            iid = self._bm25_ids[idx]
            if valid_ids is not None and iid not in valid_ids:
                continue
            scored.append((iid, score))

        # Take the top N with positive scores
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [s[0] for s in scored[:n_results] if s[1] > 0]

        if not top_ids:
            return []

        # Fetch full metadata + text from ChromaDB for the top results
        data = self._section_collection.get(
            ids=top_ids, include=["metadatas", "documents"]
        )

        # ChromaDB doesn't guarantee return order matches request order,
        # so re-order results to match our BM25 ranking
        id_to_idx = {iid: i for i, iid in enumerate(data["ids"])}
        results = []
        for iid in top_ids:
            if iid in id_to_idx:
                i = id_to_idx[iid]
                entry = dict(data["metadatas"][i])
                entry["snippet"] = (data["documents"][i] or "")[:SNIPPET_MAX_CHARS]
                entry["_id"] = iid
                results.append(entry)
        return results

    def _hybrid_search(
        self,
        query: str,
        n_results: int,
        where_filter: dict | None,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of keyword and semantic results."""
        fetch_n = n_results * RRF_OVERFETCH_FACTOR

        keyword_results = self._keyword_search(query, fetch_n, where_filter)
        semantic_results = self._semantic_search(query, fetch_n, where_filter)

        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, r in enumerate(keyword_results):
            rid = r["_id"]
            rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (RRF_K + rank + 1)
            result_map[rid] = r

        for rank, r in enumerate(semantic_results):
            rid = r["_id"]
            rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (RRF_K + rank + 1)
            if rid not in result_map:
                result_map[rid] = r

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [result_map[rid] for rid in sorted_ids[:n_results]]

    @staticmethod
    def _format_chroma_results(raw: dict) -> list[dict]:
        """Format ChromaDB query results into a flat list of dicts."""
        results = []
        for i, meta in enumerate(raw["metadatas"][0]):
            entry = dict(meta)
            doc_text = raw["documents"][0][i] if raw["documents"] else ""
            entry["snippet"] = (doc_text or "")[:SNIPPET_MAX_CHARS]
            entry["_id"] = raw["ids"][0][i]
            if raw.get("distances"):
                entry["_distance"] = raw["distances"][0][i]
            results.append(entry)
        return results

    def _group_and_rank_snippets(
        self, sections: list[dict], n_results: int, query: str
    ) -> list[SearchResult]:
        """Collect the top matching snippets per (doc_id, revision) pair."""
        grouped: dict[tuple[str, str], list[str]] = {}
        group_order: list[tuple[str, str]] = []
        first_section: dict[tuple[str, str], dict] = {}

        for sec in sections:
            key = (sec["doc_id"], sec["revision"])
            if key not in grouped:
                grouped[key] = []
                group_order.append(key)
                first_section[key] = sec
            grouped[key].append(sec.get("snippet", ""))

        query_terms = set(tokenize(query))

        results: list[SearchResult] = []
        for did, rev in group_order:
            if len(results) >= n_results:
                break

            snippets = self._select_snippets(grouped[(did, rev)], query_terms)
            sec = first_section[(did, rev)]
            doc_meta = self._body_store.get(make_unique_id(did, rev), {})

            results.append(
                {
                    "doc_id": did,
                    "doc_type": sec["doc_type"],
                    "doc_type_label": doc_meta.get("doc_type_label", sec["doc_type"]),
                    "title": sec["title"],
                    "revision": rev,
                    "is_latest": sec.get("is_latest", doc_meta.get("is_latest", False)),
                    "is_obsolete": sec.get("is_obsolete", False),
                    "is_signed": doc_meta.get("is_signed", False),
                    "filename": doc_meta.get("filename", ""),
                    "matching_snippets": snippets,
                }
            )

        return results

    @staticmethod
    def _select_snippets(raw_snippets: list[str], query_terms: set[str]) -> list[str]:
        """Pick the most relevant unique snippets up to the char budget.

        Strips the indexing header, then re-ranks snippets by how many query
        terms appear in the body text (not the header). This ensures the
        most informative sections surface first.
        """
        cleaned_snippets: list[str] = []
        seen: set[str] = set()
        for snippet in raw_snippets:
            cleaned = strip_section_header(snippet)
            if not cleaned.strip() or cleaned in seen:
                continue
            seen.add(cleaned)
            cleaned_snippets.append(cleaned)

        def _relevance_score(text: str) -> tuple[int, int]:
            """(query term overlap count, text length) -- higher is better."""
            text_tokens = set(tokenize(text))
            return (len(query_terms & text_tokens), len(text))

        cleaned_snippets.sort(key=_relevance_score, reverse=True)

        budget = SNIPPET_MAX_CHARS
        selected: list[str] = []
        for snippet in cleaned_snippets:
            trimmed = snippet[:budget]
            if not trimmed.strip():
                continue
            selected.append(trimmed)
            budget -= len(trimmed)
            if budget <= 0:
                break
        return selected
