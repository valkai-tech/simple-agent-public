# Search-Agent Evaluation and Improvement: Write-up

This documents the evaluation built for the supplied QMS search agent, the
shortcomings it measured, and the two improvements made to the system. It maps
to the seven required outcomes.

## How to run

```bash
uv run eval                         # all 16 cases, deterministic scorers
uv run eval --core                  # the 3 required risk-type cases
uv run eval --json out.json --trace out.trace.json   # scorecard + raw tool outputs
EVAL_LLM_JUDGE=1 uv run eval        # add the optional LLM-judge dimension
```

The runner loads the index once and reuses one agent across cases. `--json`
writes a lean scorecard; `--trace` writes a separate verbose file with the raw
tool outputs, which lets any scorer be re-run offline against saved answers
without calling the model again.

## 1. Evaluation runner

`src/agent/eval/` (`cases.py`, `scorers.py`, `runner.py`), registered as
`uv run eval`. It invokes the real agent end to end, extracts the observable
trace (tool calls, tool results, citations, final answer), scores each case on
several dimensions, and prints a per-case report plus an aggregate scorecard.

## 2. Cases

16 cases covering the three required risk types and several failure modes.

- **Known answer / source:** `bom_latest` (latest BOM is BOM-055 Rev G),
  `residual_radiation_dose` (limit 150 uGy/hr in VVPR-P01-226),
  `obsolete_rev_explicit` (named obsolete RSK-P01-010 Rev A must be returned).
- **Completeness / multi-document:** `ecr_inventory` (ECR set {577,587,593}),
  `third_party_reports_inventory` (3P set {32,33}), `design_review_inventory`
  (DR set {005}).
- **Report insufficient / missing evidence:** `510k_absent` (no 510(k) summary),
  `current_dfmea_obsolete` (only DFMEA is obsolete, so no active one),
  `ifu_no_extractable_text` (IFU exists but has no readable text),
  `missing_p00_document` (a P00 predecessor id not in the corpus), `dhf_coverage`
  (is the DHF evidence sufficient).
- **Other:** `bom079_version_anomaly` (rev letter maps to an older software
  version), `electrical_safety_protocols`, `leakage_trace`, `revision_diff`,
  `ecr_count`.

Every oracle was verified against the corpus before use.

## 3. Scoring dimensions

Six dimensions; five deterministic, one opt-in.

- **citation_grounding** (the required source/citation dimension): every cited
  document must have appeared in a tool result during the run.
- **expectation:** per-kind oracle. Known cites the expected source;
  completeness used the exhaustive tool; insufficient reports absence without a
  fabricated citation. A genuine clarifying question with no tool call is
  accepted rather than failed.
- **gold_facts:** deterministic presence/absence of known corpus facts, matched
  on token boundaries and unit-normalized (so `150` does not match `1500`, and
  `uGy/hr` matches whether the answer used ASCII `u` or a micro glyph).
- **completeness_recall:** set recall of the answer's enumerated ids against the
  `list_documents` ground truth for the type.
- **claim_support:** numeric fidelity of citations. Every number in a claim must
  appear in the cited document's retrieved text (this is what catches a cite
  that says 100 when the source says 150). Prose/semantic support is not judged
  here.
- **llm_judge** (opt-in, `EVAL_LLM_JUDGE=1`): a model grades faithfulness and
  whether the answer addresses the question, including reporting absence. This
  is the dimension for semantic support that token overlap cannot judge.

## 4. Baseline and final runs

The baseline run of the supplied system (`eval_baseline_v2.json`, 11 cases)
passed most cases; the failures were dominated by scorer artifacts rather than
agent errors, which drove the scorer hardening below.

Final run on the improved system, 16 cases, five deterministic dimensions:

```
citation_grounding   16/16
expectation          15/16
gold_facts            8/9
claim_support        14/15
completeness_recall   3/3
-> 13/16 passed all applicable dimensions
```

The three cases that do not pass all dimensions are understood and are not agent
defects on the changed behavior:

- `dhf_coverage` (expectation): on this run the answer asserted completeness, so
  the absence oracle correctly failed it. See shortcoming S3.
- `missing_p00_document` (gold_facts): the agent correctly reports the P00 id
  does not exist, then transparently offers the labeled P01 successor. The
  oracle forbids any mention of the successor, so it is too strict, not the
  agent.
- `leakage_trace` (claim_support 0.86): a line naming a standard number was
  attributed to a document whose snippet lacks that exact number. This is the
  known coarse attribution limit, and 0.86 is the correct soft signal.

## 5. Shortcomings

Detail in `docs/shortcomings.md`. Summary:

- **S1 (retrieval, HIGH):** obsolete documents surface as current. When every
  revision of a document is obsolete, the latest-revision logic still marked one
  as latest, so 11 documents (including the DFMEA RSK-P01-012 Rev C) were flagged
  `is_latest` while obsolete, and default search applies no obsolete filter.
- **S2 (reclassified):** an apparent "answered without retrieving" failure on
  `revision_diff` was in fact a correct clarifying question. This was a scorer
  false-fail, now corrected, not an agent shortcoming.
- **S3 (synthesis, MEDIUM):** on a sufficiency question the agent can assert
  completeness instead of hedging. Open.
- **S4 (tool behavior, MEDIUM-LOW):** the fuzzy document lookup crossed project
  and number boundaries, so a missing P00 predecessor id resolved to the
  same-numbered P01 document.

## 6. Improvements

Two changes to the supplied system, each tied to a measured shortcoming.

- **S4, P00 fuzzy guard** (`search.py`, `_fuzzy_find_doc_id`): candidates are
  pre-filtered to the same project segment and trailing number before fuzzy
  scoring, so fuzzy can only absorb type/title typos, never cross to a different
  document.
- **S1, obsolete-as-latest fix** (`ingestion.py`, `determine_latest_revisions`):
  a document whose every revision is obsolete is omitted from the latest map, so
  no revision of it is flagged latest. The main index has been rebuilt so this is
  live by default.

## 7. Before and after, regression risk, remaining uncertainty

Both fixes are retrieval-layer changes, so the cleanest evidence is a
deterministic retrieval-level before/after rather than a model rerun.

- **P00 guard.** Before: `read_document("VVPR-P00-081")` returned
  `VVPR-P01-081`; `MEMO-P00-159` returned `MEMO-P01-859`; adjacent ids
  (`VVPR-P01-080`, `-100`) returned neighbors. After: all return not found;
  valid ids and genuine type/title typos still resolve. End to end,
  `missing_p00_document` now reports the document is absent.
- **Obsolete fix.** Before: 11 documents had `is_latest` and `is_obsolete` both
  true, and `search(latest_only=True, doc_type="RSK")` returned obsolete
  RSK-P01-010/011/012 as "latest". After: that count is 0 and those obsolete
  documents no longer appear as latest, while `BOM-055` latest is still Rev G.

**Regression risk.**
- Obsolete fix: guarded by `test_latest_revision_excludes_obsolete_revision`
  (documents with a non-obsolete revision are unchanged) and by
  `obsolete_rev_explicit` (a named obsolete revision is still retrievable). All
  17 contract tests pass.
- P00 guard: a typo that changes the number or project is now unrecoverable by
  fuzzy, by design. Valid ids and same-family typos still resolve. Contract
  tests pass.

**Remaining uncertainty.**
- The live agent already partially masks the obsolete leak by reading the
  `is_obsolete` flag on results, so the answer-level gain on `current_dfmea_obsolete`
  is smaller than the retrieval-level gain. The retrieval-level before/after is
  the stronger evidence.
- `claim_support` is a numeric-fidelity check with coarse document-to-text
  attribution; it can miss a wrong number that happens to appear elsewhere in a
  large body, and prose support is delegated to the LLM judge.
- S3 (sufficiency hedging) is unaddressed, and the `missing_p00_document`
  `gold_absent` oracle is too strict; both are noted for follow-up.

## What the scores do and do not measure

- Grounding proves a citation was retrieved, not that it supports the claim or
  that every claim is cited.
- `gold_facts` confirms a fact string is present as a token, not that it is used
  correctly in context.
- `claim_support` confirms cited numbers are in the source, not semantic
  entailment.
- `completeness_recall` measures recall against the inventory, not precision.
- The LLM judge is non-deterministic and can be fooled by confident wrong
  answers.

The suite favors a few defensible checks over many weak ones, and each score's
limit is stated so it is clear what a pass does and does not guarantee.

## Speed

Retrieval is close to free (about a second across all tool calls). Almost all
wall-clock time is sequential model round-trips, so the levers for latency are
fewer turns and smaller per-turn payloads (for example trimming `MAX_READ_CHARS`
and discouraging near-duplicate searches), not the index. This is a candidate
direction for a future latency-focused improvement, measurable once per-case
wall-clock and per-tool timing are added to the trace.

## Reproduce

```bash
uv sync --frozen
uv run index            # builds .chroma_index with both fixes live
uv run eval --json eval_after.json --trace eval_after.trace.json
```
