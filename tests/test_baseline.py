"""Deterministic contracts for the supplied search baseline."""

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from agent.ingestion import (
    ParsedDocument,
    deduplicate_document_revisions,
    determine_latest_revisions,
    parse_all_documents,
    parse_filename,
)
from agent.prompts import build_prompt, build_tools_section
from agent.search import SearchIndex
from agent.tools import create_tools
from agent.utils import BodyStoreEntry, make_unique_id


def _document(revision: str, *, obsolete: bool = False) -> ParsedDocument:
    return ParsedDocument(
        doc_id="RSK-P01-010",
        doc_type="RSK",
        doc_type_label="Risk Management",
        title="MX1 Risk Assessment",
        revision=revision,
        is_obsolete=obsolete,
        is_signed=False,
        file_format="docx",
        filename=f"RSK-P01-010 - MX1 Risk Assessment_{revision}.docx",
        filepath="/tmp/example.docx",
        body="",
        sections=[],
    )


class FilenameOnlyExtractor:
    """Return valid metadata without reading file contents."""

    def parse(self, filename: str) -> dict[str, object]:
        return {
            "doc_id": filename,
            "doc_type": "TEST",
            "doc_type_label": "Test Document",
            "title": filename,
            "revision": "A",
            "is_obsolete": False,
            "is_signed": False,
            "file_format": "unknown",
            "filename": filename,
        }


class FixedReadIndex:
    """Return one controlled document lookup result to the tool wrapper."""

    def __init__(
        self,
        document: BodyStoreEntry | None,
        warning: str | None,
    ) -> None:
        self._document = document
        self._warning = warning

    def read_document(
        self,
        _doc_id: str,
        revision: str | None = None,
    ) -> tuple[BodyStoreEntry | None, str | None]:
        return self._document, self._warning


def test_medai_filename_parses_project_and_status() -> None:
    parsed = parse_filename(
        "VVPR-P01-152- Pediatric Filtration Verification Protocol and "
        "Report_B-signed.docx"
    )

    assert parsed is not None
    assert parsed["doc_id"] == "VVPR-P01-152"
    assert parsed["revision"] == "B"
    assert parsed["is_signed"] is True


def test_duplicate_download_suffix_preserves_document_identity() -> None:
    parsed = parse_filename(
        "VVPR-SWV-027 - Galden Verification Firmware Verification and "
        "Validation Protocol and Report_B(1).docx"
    )

    assert parsed is not None
    assert parsed["doc_id"] == "VVPR-SWV-027"
    assert parsed["revision"] == "B"


def test_duplicate_document_revisions_keep_the_canonical_copy() -> None:
    canonical = replace(
        _document("B"),
        body="same extracted text",
        sections=["same extracted text"],
    )
    downloaded_copy = replace(
        canonical,
        filename="RSK-P01-010 - MX1 Risk Assessment_B(1).docx",
        filepath="/tmp/downloaded-copy.docx",
    )

    documents = deduplicate_document_revisions([downloaded_copy, canonical])

    assert documents == [canonical]


def test_conflicting_duplicate_document_revisions_are_rejected() -> None:
    canonical = replace(
        _document("B"),
        body="canonical text",
        sections=["canonical text"],
    )
    conflicting_copy = replace(
        canonical,
        filename="RSK-P01-010 - MX1 Risk Assessment_B(1).docx",
        filepath="/tmp/conflicting-copy.docx",
        body="different text",
        sections=["different text"],
    )

    with pytest.raises(ValueError, match="Conflicting files for RSK-P01-010 Rev B"):
        deduplicate_document_revisions([canonical, conflicting_copy])


def test_parsing_reports_every_ten_files_and_final_count(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for number in range(12):
        (tmp_path / f"document-{number:02d}").touch()

    with caplog.at_level(logging.INFO, logger="agent.ingestion"):
        documents = parse_all_documents(
            str(tmp_path),
            extractor=FilenameOnlyExtractor(),
        )

    progress_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Parsing documents:")
    ]
    assert len(documents) == 12
    assert progress_messages == [
        "Parsing documents: 10/12",
        "Parsing documents: 12/12",
    ]


def test_latest_revision_excludes_obsolete_revision() -> None:
    documents = [
        _document("A"),
        _document("B"),
        _document("C", obsolete=True),
    ]

    assert determine_latest_revisions(documents) == {"RSK-P01-010": "B"}


@pytest.mark.parametrize(
    ("document", "terminal_message"),
    [
        (None, "not found"),
        (
            BodyStoreEntry(
                doc_id="RSK-P01-010",
                doc_type="RSK",
                doc_type_label="Risk Management",
                title="MX1 Risk Assessment",
                revision="A",
                is_latest=True,
                is_obsolete=False,
                is_signed=False,
                file_format="docx",
                filename="RSK-P01-010 - MX1 Risk Assessment_A.docx",
                has_body=False,
                body="",
            ),
            "exists but has no extractable text content",
        ),
    ],
)
def test_read_document_keeps_fuzzy_warning_on_terminal_results(
    document: BodyStoreEntry | None,
    terminal_message: str,
) -> None:
    warning = "No exact match for 'RSK-P01-01'. Showing closest match: RSK-P01-010."
    index = FixedReadIndex(document, warning)
    read_document = create_tools(index)[1]

    result = read_document.invoke({"doc_id": "RSK-P01-01", "revision": "C"})

    assert result.startswith(f"NOTE: {warning}")
    assert terminal_message in result


def test_unique_id_includes_revision() -> None:
    assert make_unique_id("BOM-055", "G") == "BOM-055_RevG"


def test_tool_section_describes_only_available_tools() -> None:
    section = build_tools_section(["search", "read_document"])

    assert "search" in section
    assert "read_document" in section
    assert "list_documents" not in section


def test_prompt_uses_current_project_examples() -> None:
    prompt = build_prompt(tool_names=["search", "read_document", "list_documents"])

    assert "MedAI" in prompt
    assert "VVPR-P01-081" in prompt
    assert "RSK-P01-010" in prompt
    assert "<cite>" in prompt


def test_additional_prompt_instructions_take_final_precedence() -> None:
    prompt = build_prompt(
        tool_names=["search", "read_document", "list_documents"],
        additional_instructions=["Keep the final answer under 1,000 characters."],
    )

    assert prompt.endswith("Keep the final answer under 1,000 characters.")


def test_snippet_selection_prioritizes_query_overlap() -> None:
    snippets = [
        "[DOC Rev A: Title | Type]\nGeneral device description.",
        "[DOC Rev A: Title | Type]\nElectrical leakage acceptance criteria.",
    ]

    selected = SearchIndex._select_snippets(
        snippets,
        {"electrical", "leakage", "acceptance"},
    )

    assert selected[0] == "Electrical leakage acceptance criteria."
