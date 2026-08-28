"""Evaluation cases.

Each case is a question plus a small set of expectations the scorers check
against the run. The three required risk types are covered explicitly:

- ``known``        one question with a known answer/source;
- ``completeness`` one question needing multiple documents / full enumeration;
- ``insufficient`` one question whose correct behavior is to report that the
                   evidence is missing or insufficient.

Remaining representative questions ride along as ``general`` cases: they still
get the citation-grounding and non-empty-answer checks, but no bespoke oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    """A single evaluation question and its expectations."""

    id: str
    question: str
    kind: str  # "known" | "completeness" | "insufficient" | "general"

    # --- expectations (all optional; scorers apply what is set) ---
    # doc_ids that SHOULD appear as citations in a correct answer.
    expect_cited_docs: tuple[str, ...] = ()
    # a tool the agent SHOULD call (e.g. list_documents for enumeration).
    expect_tool: str | None = None
    # doc_type the enumeration should cover. When set, the completeness_recall
    # scorer diffs the answer's mentioned ids of this type against the corpus
    # ground truth from list_documents(type).
    expect_doc_type: str | None = None
    # gold-fact oracle: substrings a correct answer MUST contain (case-
    # insensitive), and substrings it must NOT contain. Derived from corpus
    # ground truth, e.g. the latest revision letter of a known document.
    gold_contains: tuple[str, ...] = ()
    gold_absent: tuple[str, ...] = ()
    # true when the correct answer is "the corpus does not contain / cannot
    # support this" -- the answer should signal absence and NOT fabricate cites.
    expect_absent: bool = False
    # phrases any of which signal a correct "absent / insufficient" answer.
    absent_markers: tuple[str, ...] = field(
        default=(
            "not contain",
            "does not",
            "no 510",
            "no such",
            "not found",
            "not present",
            "not in the corpus",
            "could not find",
            "unable to find",
            "insufficient",
            "not enough",
            "no evidence",
        )
    )


CASES: list[Case] = [
    # 1) Known answer / source: the top-level MX1 bill of materials is BOM-055.
    Case(
        id="bom_latest",
        question="Find the latest MX1 Bill of Materials and summarize what it covers.",
        kind="known",
        expect_cited_docs=("BOM-055",),
        # Corpus ground truth: BOM-055 revisions are E, F, G; G is latest.
        # A correct "latest BOM" answer must land on Rev G, not F.
        gold_contains=("BOM-055", "Rev G"),
    ),
    # 2) Completeness / multi-document: every ECR requires the exhaustive
    #    inventory tool, not ranked search.
    Case(
        id="ecr_inventory",
        question="List every engineering change request in the QMS and its status.",
        kind="completeness",
        expect_tool="list_documents",
        expect_doc_type="ECR",
    ),
    # 3) Insufficient evidence: a 510(k) summary is a regulatory submission
    #    document; the QMS design corpus is not expected to contain one. The
    #    correct behavior is to say so, not to invent a citation.
    Case(
        id="510k_absent",
        question="Does the corpus contain a 510(k) summary for the device?",
        kind="insufficient",
        expect_absent=True,
    ),
    # 4) Known answer with a numeric acceptance criterion: the MX1 residual-
    #    radiation verification (VVPR-P01-226 Rev B) sets the limit at
    #    <150 uGy/hr per IEC 60601-2-54:2022 Table 203.106; measured <3.6 uGy/hr
    #    (Pass). A correct answer must land on the exact limit, not just name a
    #    protocol.
    Case(
        id="residual_radiation_dose",
        question=(
            "What is the acceptance criterion for residual radiation in the MX1 "
            "verification, and did the device pass?"
        ),
        kind="known",
        expect_cited_docs=("VVPR-P01-226",),
        gold_contains=("VVPR-P01-226", "150", "uGy/hr"),
    ),
    # 5) Data-quality contradiction: BOM-079's latest revision (Rev P) is
    #    software v3.4.0, while its earlier, non-latest Rev M is software
    #    v4.0.0 -- the highest revision letter maps to an OLDER software
    #    version. A correct answer surfaces both versions and flags the
    #    inconsistency rather than trusting "latest letter = newest".
    Case(
        id="bom079_version_anomaly",
        question=(
            "What software version does the latest revision of BOM-079 "
            "correspond to, and is there anything inconsistent about its "
            "revision history?"
        ),
        kind="general",
        expect_cited_docs=("BOM-079",),
        gold_contains=("v3.4.0", "v4.0.0"),
    ),
    # 6) Obsolete-leak (default exclusion): the only DFMEA is RSK-P01-012 Rev C,
    #    and it is OBSOLETE (is_latest=True AND is_obsolete=True -- a known
    #    latest-selection bug). The correct answer is that there is no current /
    #    active DFMEA, flagging RSK-P01-012 as obsolete rather than presenting it
    #    as current. Custom absent_markers recognize obsolescence phrasing so the
    #    insufficient oracle scores this case's correct behavior.
    Case(
        id="current_dfmea_obsolete",
        question=(
            "What is the current, effective DFMEA for the MX1? Is it an active, "
            "released document?"
        ),
        kind="insufficient",
        expect_absent=True,
        gold_contains=("RSK-P01-012", "obsolete"),
        absent_markers=(
            "obsolete",
            "superseded",
            "not active",
            "no active",
            "no current",
            "retired",
            "not released",
            "no effective",
        ),
    ),
    # 7) Obsolete retrieve-when-named (regression guard for the obsolete fix):
    #    obsolete must be excluded by DEFAULT but still retrievable when a
    #    revision is explicitly named. RSK-P01-010 Rev A is obsolete + non-latest
    #    and must be returned/cited when asked for by name.
    Case(
        id="obsolete_rev_explicit",
        question=(
            "Retrieve the superseded Revision A of the MX1 Risk Assessment "
            "(RSK-P01-010), even though it is obsolete."
        ),
        kind="known",
        expect_cited_docs=("RSK-P01-010",),
        gold_contains=("RSK-P01-010", "Rev A"),
    ),
    # 8) Second completeness / multi-document case (the multi-doc dimension
    #    previously rested only on ecr_inventory). Third-Party test reports need
    #    the exhaustive inventory tool, not ranked search. Corpus ground truth:
    #    doc_type "3P" is exactly {3P-P01-32, 3P-P01-33} (verified count=2, both
    #    latest & non-obsolete). Sibling ids -26/-28 are referenced elsewhere but
    #    are NOT in the corpus, so ranked search can silently under-enumerate.
    Case(
        id="third_party_reports_inventory",
        question=(
            "List every third-party test report in the QMS and what each covers."
        ),
        kind="completeness",
        expect_tool="list_documents",
        expect_doc_type="3P",
    ),
    # 9) Single-item completeness case (low value but a clean recall oracle).
    #    Corpus ground truth: doc_type "DR" is exactly {DR-P01-005} (verified
    #    count=1, Rev G, latest & non-obsolete). Guards against an answer that
    #    invents additional design-review ids or misses the one that exists.
    Case(
        id="design_review_inventory",
        question="List every design review document in the QMS.",
        kind="completeness",
        expect_tool="list_documents",
        expect_doc_type="DR",
    ),
    # 10) Empty-body document. Corpus ground truth: IFU-MX1 exists but its body
    #     is empty (verified has_body=False, body==""). Correct behavior: report
    #     the IFU exists but its content is not extractable, WITHOUT fabricating
    #     content from neighboring docs that quote the IFU. The default
    #     absent_markers don't cover empty-doc phrasing, so a custom tuple adds
    #     the "no extractable / no readable text / empty" vocabulary. gold_contains
    #     forces the answer to actually name the doc it's reporting on.
    Case(
        id="ifu_no_extractable_text",
        question=(
            "Summarize the contents of the MX1 Instructions For Use (IFU-MX1)."
        ),
        kind="insufficient",
        expect_absent=True,
        gold_contains=("IFU-MX1",),
        absent_markers=(
            "no extractable",
            "no readable",
            "no text",
            "empty",
            "not available",
            "no content",
            "cannot",
            "unable",
            "not extractable",
            "could not extract",
        ),
    ),
    # 11) P00/P01 fuzzy crossing (retrieval bug guard). Corpus ground truth:
    #     there are ZERO P00 documents (verified: no doc_id contains "P00"), but
    #     the same-numbered VVPR-P01-081 DOES exist (Usability Summative Protocol,
    #     latest & active). The fuzzy id resolver crosses P00->P01, so asking for
    #     the non-existent VVPR-P00-081 risks silently returning VVPR-P01-081.
    #     Correct behavior: report VVPR-P00-081 is not in the corpus; do NOT
    #     substitute the same-numbered P01 doc. gold_absent forbids presenting the
    #     verified fuzzy-match target VVPR-P01-081 as the answer. Default
    #     absent_markers ("not in the corpus", "not found", ...) cover this.
    Case(
        id="missing_p00_document",
        question="Find document VVPR-P00-081 and summarize what it covers.",
        kind="insufficient",
        expect_absent=True,
        gold_contains=("VVPR-P00-081",),
        gold_absent=("VVPR-P01-081",),
    ),
    # 12) Exact count. Corpus ground truth: doc_type "ECR" has exactly 3 members
    #     (verified {ECR-577, ECR-587, ECR-593}). Counting correctly requires the
    #     enumeration tool, not ranked search, so expect_tool guards that.
    #     NOTE (weak oracle): the count is asserted via gold_contains=("3",); this
    #     relies on gold_facts matching being word-boundary-safe (a bare "3" would
    #     otherwise match inside larger numbers/ids). It also fails if the agent
    #     spells the count as "three" instead of "3". Judgment-adjacent oracle.
    Case(
        id="ecr_count",
        question="How many engineering change requests are in the QMS?",
        kind="general",
        expect_tool="list_documents",
        gold_contains=("3",),
    ),
    # --- Remaining representative questions (general grounding checks) ---
    # electrical_safety_protocols: enriched from a bare non-empty/tool check with
    # a concrete gold-fact. Corpus ground truth: the electrical-safety verification
    # evidence (e.g. 3P-P01-33 Intertek dielectric & leakage report, ESF-P01-003,
    # and multiple VVPRs) is governed by IEC 60601, verified present across those
    # docs. A correct answer about electrical-safety protocols should name the
    # standard. Kept kind=general (still gets grounding + >=1-tool check).
    Case(
        id="electrical_safety_protocols",
        question="What verification protocols exist for electrical safety?",
        kind="general",
        gold_contains=("IEC 60601",),
    ),
    # leakage_trace: enriched with the verified verification-evidence anchor for
    # the leakage chain. Corpus ground truth: 3P-P01-33 is the Intertek IEC
    # 60601-1 MX1 Dielectric and Leakage Summary Report (latest & active) -- the
    # verification evidence end of the risk->verification chain. The risk-record
    # end is NOT anchored here because the primary risk assessment RSK-P01-010 is
    # obsolete (the known obsolete-leak bug), so a correct answer may legitimately
    # flag it rather than cite it. Kept kind=general.
    Case(
        id="leakage_trace",
        question=(
            "Trace electrical-leakage risk from the risk records to "
            "verification evidence."
        ),
        kind="general",
        expect_cited_docs=("3P-P01-33",),
    ),
    Case(
        id="revision_diff",
        question="What changed between two revisions of an MX1 document?",
        kind="general",
    ),
    Case(
        id="dhf_coverage",
        question=(
            "Does the Design History File provide enough evidence to support a "
            "complete design-control coverage claim?"
        ),
        kind="insufficient",
        expect_absent=True,
    ),
]


CORE_CASE_IDS = ("bom_latest", "ecr_inventory", "510k_absent")
"""The three cases that satisfy the required risk-type coverage."""
