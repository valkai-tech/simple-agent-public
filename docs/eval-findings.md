# Eval findings & backlog

Working notes for the search-agent evaluation. Parked here to revisit; current
focus is accuracy scoring. Baselines: `eval_baseline_full.json` (7 cases, offline
re-scored on 4 dims) and `eval_gold.json` (7 cases, live, 4 dims).

Scoring dimensions today: `citation_grounding`, `expectation`, `gold_facts`,
`completeness_recall` (see `src/agent/eval/scorers.py`).

## Measured shortcomings (from the gold run)

1. **`revision_diff` answers with zero tool calls.** The agent produced a
   non-empty answer with no `search`/`read_document`/`list_documents` call at all
   — an ungrounded answer. Caught by the `expectation` scorer (general kind
   requires >=1 tool call). Failure layer: tool-selection / synthesis.
2. **`dhf_coverage` false-passes the absence oracle.** Scored "reports
   insufficiency: True" while the answer actually *asserts coverage is
   complete*. The `insufficient` scorer's substring markers matched incidental
   "not yet / pending / deferred" phrasing. Scorer weakness, not agent success.
   Fix: tighten the absence oracle (stance detection, not substring), or require
   the answer to NOT conclude sufficiency.

## Scorer hardening backlog

- **Extend `absent_markers`** to recognize empty-doc phrasing ("no extractable",
  "no readable", "no text content", "empty") — needed before the
  `ifu_no_extractable_text` case scores correctly.
- **Claim-level support** (beyond grounding): does the cited doc's *retrieved
  snippet* actually contain the claim's numbers/terms? Keyword-overlap or LLM
  judge. Grounding only proves the doc was retrieved, not that it supports the
  claim.
- **Citation coverage**: fraction of factual sentences carrying any cite (catches
  under-citation, the inverse of grounding).
- **Latest-revision correctness**: cited revision == corpus `is_latest`.

## Proposed eval cases (backlog)

Verified oracles read from `.chroma_index`. Two already added to `cases.py`.

| id | kind | oracle (verified) | status |
|---|---|---|---|
| `residual_radiation_dose` | known | cite VVPR-P01-226; answer has `150` + `uGy/hr` (limit <150, Pass) | ADDED |
| `bom079_version_anomaly` | general | cite BOM-079; answer has `v3.4.0` (Rev P latest) + `v4.0.0` (Rev M) | ADDED |
| `ifu_no_extractable_text` | insufficient | IFU-MX1 exists but body empty; report unreadable, don't fabricate from neighbors | backlog |
| `current_dfmea_obsolete` | insufficient | only DFMEA = RSK-P01-012 Rev C, obsolete; no active DFMEA | backlog |
| `obsolete_rev_explicit` | known | return/cite RSK-P01-010 Rev A (obsolete, non-latest) when named | backlog |
| `bom055_revF_ecr` | known | ECR-577 released BOM-055 Rev F (forces non-latest) | backlog |
| `third_party_reports_inventory` | completeness | exactly {3P-P01-32, 3P-P01-33}; siblings -26/-28 referenced but absent | backlog |
| `design_review_inventory` | completeness | exactly {DR-P01-005} (single-item; low value) | backlog |

Case kinds map to the three required risk types: known (answer/source),
completeness (multi-doc), insufficient (report missing/insufficient). The
obsolete pair `current_dfmea_obsolete` + `obsolete_rev_explicit` is the
before/after for the obsolete-leak retrieval fix.

Paste-ready `Case(...)` definitions for the backlog items are in the
eval-coverage subagent output; oracles above are sufficient to reconstruct them.

## Retrieval bugs (improvement candidates, from the revision audit)

Not eval cases — actual system fixes tied to metrics. Detail in memory
(`search-agent-shortcomings`). Smallest first:

1. **Obsolete-as-latest leak.** `determine_latest_revisions`
   (`ingestion.py:354-359`) falls back to `max()` incl. obsolete when all revs
   of an id are obsolete → 11 docs flagged `is_latest=True AND is_obsolete=True`
   (RSK-P01-010 E, RSK-P01-012 C, MEMO-P01-776/777, ...). Default
   `search(latest_only=False)` applies no status filter → 410 obsolete sections
   retrievable. Fix: don't mark is_latest when all obsolete; exclude obsolete by
   default in `_build_filter`/`search` with a named-doc override. Metric:
   obsolete-leak rate.
2. **Fuzzy match crosses P00→P01 / adjacent ids.** `_fuzzy_find_doc_id`
   (`search.py:137-148`, cutoff 80) resolves `VVPR-P00-081`→`VVPR-P01-081`;
   19 docs reference P00 predecessors. Guard: require project + numeric tail to
   match, only fuzz the title. Metric: correct-doc/correct-revision rate.

Already-solved (don't rebuild): metadata IS lifted into section metadata and
search filters on doc_type/latest_only; base_id grouping exists; extraction IS
table-aware (reads tables, doesn't parse them into fields). Filename-vs-internal
revision disagreement overstated (no real tip disagreement found).

## Follow-ups (open, prioritized)

1. **`claim_support` earns little on this agent.** On the final run it caught
   nothing real (14/15; the one fail was a coarse-attribution artifact). It only
   checks numeric claims against a large haystack, so a wrong number can appear
   elsewhere and slip through; prose support is punted to `llm_judge`. Options:
   narrow it to a few high-value numeric facts (e.g. the dose criterion), or lean
   on `llm_judge` for accuracy and demote claim_support to a spot check.
2. **Run `llm_judge` in a scored run.** It is off by default and absent from the
   final scorecard, so the semantic/faithfulness signal is unmeasured. Turn it on
   (`EVAL_LLM_JUDGE=1`) over the saved `--trace` and record its numbers.
3. **S3 sufficiency hedging (agent).** `dhf_coverage` can assert completeness
   instead of flagging gaps. Fix is a prompt instruction to distinguish "evidence
   exists" from "sufficient" + a stance-based insufficient oracle (not substring).
4. **Relax `missing_p00_document` `gold_absent`.** It forbids any mention of the
   P01 successor, but the agent transparently labels it as the successor. Allow a
   labeled mention; only fail a silent substitution.
5. **Finish the obsolete fix.** Only the `ingestion.py` is_latest part shipped.
   Add the default `is_obsolete` filter to `_build_filter`/`search` with a
   named-doc override so obsolete is excluded by default, not just de-flagged.
6. **Latency instrumentation + improvement.** Retrieval is ~free; wall-clock is
   sequential model round-trips. Add per-case wall-clock + per-tool seconds to the
   trace, then measure a latency change (trim `MAX_READ_CHARS` 30k→~12k;
   discourage near-duplicate searches). Currently no timing is captured.
7. **Eval robustness: a case can hang the run.** `bom079_version_anomaly` spawns
   a nested sub-agent that once hung the runner for ~72 min. Add a per-case
   wall-clock timeout (and/or lower the nested recursion cap) so one case can't
   stall the suite.
