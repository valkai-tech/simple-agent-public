"""Document ingestion pipeline.

Parses documents (docx, xlsx), extracts metadata from filenames,
extracts body text, and builds a ChromaDB index for semantic search.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from agent.extraction import extract_text
from agent.extraction import extract_docx_text as extract_docx_text
from agent.extraction import extract_xlsx_text as extract_xlsx_text
from agent.utils import (
    CHROMA_MAX_CHARS,
    DOC_TYPE_LABELS,
    INDEX_BATCH_SIZE,
    MIN_SECTION_CHARS,
    PARSE_WORKER_THRESHOLD,
    BodyStoreEntry,
    make_unique_id,
)

logger = logging.getLogger(__name__)

PARSE_PROGRESS_INTERVAL = 10
DOWNLOAD_COPY_SUFFIX_RE = re.compile(r"\(\d+\)(?=\.[^.]+$|$)")


@dataclass
class ParsedDocument:
    """Represents a parsed document with metadata and body text."""

    doc_id: str
    doc_type: str
    doc_type_label: str
    title: str
    revision: str
    is_obsolete: bool
    is_signed: bool
    file_format: str
    filename: str
    filepath: str
    body: str
    sections: list[str] = field(default_factory=list)


class MetadataExtractor(Protocol):
    """Interface for extracting document metadata from filenames.

    Implement this protocol to support different filename conventions.
    The default implementation is QMSFilenameExtractor, which uses a regex.
    Future implementations could use lightweight LLM-based extraction for more complex patterns.
    """

    def parse(self, filename: str) -> dict | None:
        """Extract metadata from a filename.

        Returns a dict with keys: doc_id, doc_type, doc_type_label, title,
        revision, is_obsolete, is_signed, file_format, filename.
        Returns None if the filename doesn't match the expected pattern.
        """
        ...


class QMSFilenameExtractor:
    """Extracts metadata from QMS filenames following the pattern:
    {TYPE}-{ID} - {TITLE}_{REVISION}[-STATUS][.ext]

    Handles edge cases: 3P-P01-32 (short number), IFU-MX1 (alpha suffix),
    VVPR-P01-152- (dash glued), MEMO-P01-660 (space), and status suffix spacing.
    """

    FILENAME_RE = re.compile(
        r"^(?P<doc_id>[A-Z0-9]{2,4}-(?:[A-Z0-9]+-)*[A-Z0-9]+)"
        r"[\s-]+"
        r"(?P<title>.+?)"
        r"_(?P<revision>[A-Z])"
        r"(?P<status>[\s_-]*(?:signed|Signed|Obsolete|obsolete))?"
        r"(?:\(\d+\))?"
        r"(?:\.(?P<ext>docx|xlsx))?$"
    )

    def __init__(self, type_labels: dict[str, str] | None = None):
        self._type_labels = type_labels or DOC_TYPE_LABELS

    def parse(self, filename: str) -> dict | None:
        match = self.FILENAME_RE.match(filename)
        if not match:
            logger.warning("Could not parse filename: %s", filename)
            return None

        doc_id = match.group("doc_id")
        title = match.group("title").strip()
        revision = match.group("revision")
        status = match.group("status") or ""
        ext = match.group("ext") or "unknown"

        doc_type = doc_id.split("-")[0]
        doc_type_label = self._type_labels.get(doc_type)
        if doc_type_label is None:
            logger.warning(
                "Unknown document type prefix: %s (from %s)", doc_type, filename
            )
            doc_type_label = f"Unknown ({doc_type})"

        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "doc_type_label": doc_type_label,
            "title": title,
            "revision": revision,
            "is_obsolete": "obsolete" in status.lower(),
            "is_signed": "signed" in status.lower(),
            "file_format": ext,
            "filename": filename,
        }


def parse_filename(filename: str) -> dict | None:
    """Parse a QMS filename using the default extractor. Convenience wrapper."""
    return QMSFilenameExtractor().parse(filename)


def _extract_all_texts(
    tasks: list[tuple[str, str]],
    total_entries: int,
    max_workers: int | None = None,
) -> list[tuple[str, list[str]]]:
    """Extract text for every task, in parallel processes when it pays off.

    Document parsing is CPU-bound inside lxml/openpyxl, so worker processes
    scale with cores while threads do not.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

    def _serial() -> Iterable[tuple[str, list[str]]]:
        return (extract_text(task) for task in tasks)

    def _with_progress(
        results: Iterable[tuple[str, list[str]]],
    ) -> list[tuple[str, list[str]]]:
        collected = []
        for processed, result in enumerate(results, start=1):
            collected.append(result)
            if processed % PARSE_PROGRESS_INTERVAL == 0 or processed == total_entries:
                logger.info("Parsing documents: %d/%d", processed, total_entries)
        return collected

    if max_workers <= 1 or len(tasks) < PARSE_WORKER_THRESHOLD:
        return _with_progress(_serial())

    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # A spawned worker importing the caller's __main__ module: starting
        # more workers here would recurse, so let the pool in the parent break
        # and fall back to serial parsing there.
        raise RuntimeError(
            "parse_all_documents() ran while importing __main__ in a worker "
            'process; call it under `if __name__ == "__main__":`'
        )

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return _with_progress(pool.map(extract_text, tasks, chunksize=4))
    except (OSError, RuntimeError, BrokenProcessPool) as e:
        # Worker processes are unavailable, for example when the caller starts
        # parsing at import time on a spawn platform.
        logger.warning("Parallel parsing unavailable (%s); parsing serially", e)
        return _with_progress(_serial())


def parse_all_documents(
    data_dir: str,
    extractor: MetadataExtractor | None = None,
    max_workers: int | None = None,
) -> list[ParsedDocument]:
    """Parse all documents in the given directory.

    Args:
        data_dir: Path to directory containing document files.
        extractor: Metadata extractor to use. Defaults to QMSFilenameExtractor.
        max_workers: Worker processes used for text extraction. Defaults to the
            core count (capped at 8); 1 forces serial parsing.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if extractor is None:
        extractor = QMSFilenameExtractor()

    entries = sorted(entry for entry in data_path.iterdir() if entry.is_file())

    metas: list[dict | None] = []
    tasks: list[tuple[str, str]] = []
    for entry in entries:
        meta = extractor.parse(entry.name)
        if meta is not None:
            meta["filepath"] = str(entry)
        metas.append(meta)
        # Unparsed filenames keep a task so progress still counts every entry;
        # an unsupported format extracts to empty text without reading the file.
        tasks.append((str(entry), meta["file_format"] if meta else ""))

    extracted = _extract_all_texts(
        tasks, total_entries=len(entries), max_workers=max_workers
    )

    docs = [
        ParsedDocument(
            doc_id=meta["doc_id"],
            doc_type=meta["doc_type"],
            doc_type_label=meta["doc_type_label"],
            title=meta["title"],
            revision=meta["revision"],
            is_obsolete=meta["is_obsolete"],
            is_signed=meta["is_signed"],
            file_format=meta["file_format"],
            filename=meta["filename"],
            filepath=meta["filepath"],
            body=body,
            sections=sections,
        )
        for meta, (body, sections) in zip(metas, extracted)
        if meta is not None
    ]

    logger.info("Parsed %d documents from %s", len(docs), data_dir)
    return docs


def _document_revision_signature(document: ParsedDocument) -> tuple[object, ...]:
    return (
        document.doc_type,
        document.doc_type_label,
        document.title,
        document.is_obsolete,
        document.is_signed,
        document.file_format,
        document.body,
        tuple(document.sections),
    )


def deduplicate_document_revisions(
    docs: list[ParsedDocument],
) -> list[ParsedDocument]:
    """Collapse identical file copies and reject conflicting document revisions."""
    grouped: dict[tuple[str, str], list[ParsedDocument]] = {}
    for document in docs:
        grouped.setdefault((document.doc_id, document.revision), []).append(document)

    unique_documents: list[ParsedDocument] = []
    for (doc_id, revision), copies in grouped.items():
        reference_signature = _document_revision_signature(copies[0])
        if any(
            _document_revision_signature(copy) != reference_signature
            for copy in copies[1:]
        ):
            filenames = ", ".join(sorted(copy.filename for copy in copies))
            raise ValueError(
                f"Conflicting files for {doc_id} Rev {revision}: {filenames}"
            )

        canonical_copy = min(
            copies,
            key=lambda copy: (
                DOWNLOAD_COPY_SUFFIX_RE.search(copy.filename) is not None,
                copy.filename,
            ),
        )
        unique_documents.append(canonical_copy)

    return unique_documents


def determine_latest_revisions(docs: list[ParsedDocument]) -> dict[str, str]:
    """For each doc_id, map to its latest revision letter: the highest non-obsolete revision.

    A doc_id with at least one non-obsolete revision maps to the max of those
    non-obsolete revisions. A doc_id whose every revision is obsolete is OMITTED
    from the returned map entirely, rather than falling back to the max obsolete
    revision.

    Why: a fully-obsolete document has no "current" revision. Flagging one of its
    obsolete revisions as latest would surface a retired document as if it were
    live in latest_only searches. Callers set is_latest via
    ``latest_map.get(doc_id) == revision``, so an absent doc_id yields
    is_latest=False for all of its revisions -- exactly the desired result.
    Revision-specific reads are unaffected because they do not depend on is_latest.
    """
    id_revisions: dict[str, list[tuple[str, bool]]] = {}
    for doc in docs:
        id_revisions.setdefault(doc.doc_id, []).append((doc.revision, doc.is_obsolete))

    latest: dict[str, str] = {}
    for doc_id, revisions in id_revisions.items():
        non_obsolete = [(r, obs) for r, obs in revisions if not obs]
        if non_obsolete:
            latest[doc_id] = max(non_obsolete, key=lambda x: x[0])[0]

    return latest


def _dedup_id(base_id: str, seen: set[str]) -> str:
    """Append a suffix if base_id already exists in seen."""
    if base_id not in seen:
        seen.add(base_id)
        return base_id
    n = 2
    while f"{base_id}_{n}" in seen:
        n += 1
    deduped = f"{base_id}_{n}"
    seen.add(deduped)
    return deduped


def build_index(
    docs: list[ParsedDocument],
    persist_dir: str = ".chroma_index",
) -> tuple[chromadb.ClientAPI, dict[str, str]]:
    """Build a ChromaDB sections collection and return (client, latest_map).

    Only creates one collection ("sections") for semantic/keyword search.
    Document-level metadata lives in the body store JSON.
    """
    docs = deduplicate_document_revisions(docs)
    latest_map = determine_latest_revisions(docs)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection("sections")
    except Exception:
        pass

    section_collection = client.create_collection(
        name="sections",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ONNXMiniLM_L6_V2(),
    )

    section_ids: list[str] = []
    section_documents: list[str] = []
    section_metadatas: list[dict] = []
    seen_ids: set[str] = set[str]()

    for doc in docs:
        is_latest = latest_map.get(doc.doc_id) == doc.revision
        unique_id = _dedup_id(make_unique_id(doc.doc_id, doc.revision), seen_ids)

        for i, section in enumerate(doc.sections):
            # Skip tiny sections -- assumed to be noise.
            if len(section) < MIN_SECTION_CHARS:
                continue

            section_embed = (
                f"[{doc.doc_id} Rev {doc.revision}: {doc.title} | {doc.doc_type_label}]\n"
                f"{section}"
            )

            section_ids.append(f"{unique_id}_sec{i}")
            section_documents.append(section_embed[:CHROMA_MAX_CHARS])
            section_metadatas.append(
                {
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    "revision": doc.revision,
                    "is_latest": is_latest,
                    "is_obsolete": doc.is_obsolete,
                    "parent_unique_id": unique_id,
                    "section_index": i,
                }
            )

    for i in range(0, len(section_ids), INDEX_BATCH_SIZE):
        end = min(i + INDEX_BATCH_SIZE, len(section_ids))
        section_collection.add(
            ids=section_ids[i:end],
            documents=section_documents[i:end],
            metadatas=section_metadatas[i:end],
        )
        logger.info("Indexed sections batch %d-%d / %d", i + 1, end, len(section_ids))

    logger.info(
        "Indexed %d documents (%d sections) into ChromaDB at %s",
        len(docs),
        len(section_ids),
        persist_dir,
    )
    return client, latest_map


def _body_store_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "body_store.json")


def save_body_store(
    docs: list[ParsedDocument],
    latest_map: dict[str, str],
    persist_dir: str = ".chroma_index",
) -> None:
    """Save full document bodies + metadata to a JSON file."""
    docs = deduplicate_document_revisions(docs)
    os.makedirs(persist_dir, exist_ok=True)
    store: dict[str, BodyStoreEntry] = {}
    for doc in docs:
        uid = make_unique_id(doc.doc_id, doc.revision)
        store[uid] = {
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "doc_type_label": doc.doc_type_label,
            "title": doc.title,
            "revision": doc.revision,
            "is_latest": latest_map.get(doc.doc_id) == doc.revision,
            "is_obsolete": doc.is_obsolete,
            "is_signed": doc.is_signed,
            "file_format": doc.file_format,
            "filename": doc.filename,
            "has_body": bool(doc.body),
            "body": doc.body,
        }
    path = _body_store_path(persist_dir)
    with open(path, "w") as f:
        json.dump(store, f)
    logger.info("Saved body store (%d docs) to %s", len(store), path)


def ingest(
    data_dir: str,
    persist_dir: str = ".chroma_index",
    extractor: MetadataExtractor | None = None,
) -> chromadb.ClientAPI:
    """Full ingestion pipeline: parse, index, and save body store.

    Args:
        data_dir: Path to document directory.
        persist_dir: Path to store the ChromaDB index and body store.
        extractor: Metadata extractor for filename parsing. Defaults to QMS format.
    """
    logger.info("Starting ingestion from %s ...", data_dir)
    docs = parse_all_documents(data_dir, extractor=extractor)
    client, latest_map = build_index(docs, persist_dir=persist_dir)
    save_body_store(docs, latest_map=latest_map, persist_dir=persist_dir)
    logger.info("Ingestion complete.")
    return client
