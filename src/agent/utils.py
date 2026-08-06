"""Shared utilities, constants, and type definitions."""

from __future__ import annotations

import re
from typing import TypedDict

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = "example_qms_data"
DEFAULT_INDEX_DIR = ".chroma_index"
DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

SNIPPET_MAX_CHARS = 500
MIN_SECTION_CHARS = 50
MAX_READ_CHARS = 30_000
CHROMA_MAX_CHARS = 8_000
INDEX_BATCH_SIZE = 100
CHUNK_MAX_CHARS = 2_000
MAX_RECURSION_LIMIT = 100

RRF_K = 60
SEARCH_OVERFETCH_FACTOR = 5
RRF_OVERFETCH_FACTOR = 3

VALID_SEARCH_MODES = ("keyword", "semantic", "hybrid")

# Document type categories for human-readable labeling
DOC_TYPE_LABELS: dict[str, str] = {
    "3P": "Third-Party Report",
    "BOM": "Bill of Materials",
    "DHF": "Design History File",
    "DMR": "Device Master Record",
    "DR": "Design Review",
    "ECR": "Engineering Change Request",
    "ESF": "Engineering Specification",
    "IFU": "Instructions for Use",
    "MEMO": "Memo",
    "PLN": "Plan",
    "QSR": "Quality System Record",
    "RSK": "Risk Management",
    "TRA": "Training",
    "VVAM": "Verification & Validation Acceptance Matrix",
    "VVPR": "Verification & Validation Protocol/Report",
}


# ── Typed dicts ──────────────────────────────────────────────────────────────


class BodyStoreEntry(TypedDict):
    doc_id: str
    doc_type: str
    doc_type_label: str
    title: str
    revision: str
    is_latest: bool
    is_obsolete: bool
    is_signed: bool
    file_format: str
    filename: str
    has_body: bool
    body: str


class SearchResult(TypedDict):
    doc_id: str
    doc_type: str
    doc_type_label: str
    title: str
    revision: str
    is_latest: bool
    is_obsolete: bool
    is_signed: bool
    filename: str
    matching_snippets: list[str]


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_unique_id(doc_id: str, revision: str) -> str:
    """Build the canonical key for a (doc_id, revision) pair."""
    return f"{doc_id}_Rev{revision}"


_SECTION_HEADER_RE = re.compile(r"^\[.*?\]\n")


def strip_section_header(text: str) -> str:
    """Remove the [DOC_ID Rev X: Title | Type] header prepended at index time."""
    return _SECTION_HEADER_RE.sub("", text, count=1)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())
