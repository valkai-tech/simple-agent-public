# Measured shortcomings

Evidence-backed weaknesses in the supplied search agent, with likely cause and
priority. Measured against the eval baseline (`eval_gold.json`, 7 cases live on
`claude-sonnet-4-6`, 4 scoring dimensions) and read-only probes of the corpus /
index. Two primary shortcomings plus two secondary.

## S1 (primary, HIGH): Obsolete documents surface as "current"

**What.** For a document whose every stored revision is obsolete, the system
still marks one revision as the latest, and default search applies no obsolete
filter. So a retired document is returned as if it were the current one.

**Evidence.** 11 doc_ids carry `is_latest=True AND is_obsolete=True`, including
`RSK-P01-012 Rev C` (the only DFMEA), `RSK-P01-010 Rev E` (Risk Assessment),
and `MEMO-P01-776/777` (the obsolete predecessors of the live SRS/SDS). Verified
directly from the index. `search(latest_only=True)` returns these; default
`search(latest_only=False)` exposes all 410 obsolete sections (of 2820). The
eval case `current_dfmea_obsolete` targets this: the correct answer is that
there is no active DFMEA.

**Likely cause.** `determine_latest_revisions` (`ingestion.py:354-359`) filters
obsolete revisions, but when none remain it falls back to `max()` over all
revisions, including obsolete ones. Separately, `_build_filter`
(`search.py:162-176`) never adds an `is_obsolete` condition, so obsolete is not
excluded by default.

**Priority: HIGH.** In a regulated QMS, presenting a retired risk document as
current is a substantive correctness failure, not a cosmetic one. The fix is
small (metadata-only for the latest flag; a filter change for default search)
and has a clean regression guard (`obsolete_rev_explicit`: obsolete must still
be retrievable when named). This is the selected improvement.

## S2 (RECLASSIFIED: scorer false-fail, not an agent shortcoming)

**What we first saw.** `revision_diff` ("What changed between two revisions of an
MX1 document?") ran with `tool_calls = 0` and `expectation` scored 0.00, which
looked like an ungrounded answer.

**What it actually was.** Reading the answer, the agent did NOT fabricate: it
asked a clarifying question, because the query names no document and no two
revisions. The real answer was "I'd be happy to help... Could you provide: 1.
Document ID 2. Which two revisions...". Asking for clarification on an
under-specified query is correct behavior, not a failure.

**Cause of the false score.** Our `expectation` general-kind oracle required
>=1 tool call, so it failed a legitimate clarifying question. This was a SCORER
bug, not an agent bug.

**Resolution.** Fixed in `score_expectation` (`scorers.py`): a 0-tool answer now
passes when it is a genuine clarification (cue phrase + question/enumerated-request
signal + no fabricated citation). `revision_diff` now scores PASS ("clarifying
question (0 tools, accepted)"). Contract tests still 17/17.

**Takeaway.** This is itself a finding about evaluation design: a naive
"answered without a tool call = failure" rule mislabels correct abstention /
clarification. Left here as a documented reclassification. The remaining
evidence-backed *agent* shortcomings are S1, S3, S4.

## S3 (secondary, MEDIUM): Over-confident sufficiency claims

**What.** On an evidence-sufficiency question the agent asserts completeness
rather than flagging gaps.

**Evidence.** `dhf_coverage` ("Does the DHF provide enough evidence to support a
complete design-control coverage claim?") answered: "The DHF provides strong
evidence for a complete design-control coverage claim." It emitted 39 grounded
citations but concluded sufficiency where the safer, arguably correct behavior
is to enumerate gaps and hedge.

**Likely cause.** The prompt's Response Quality section rewards thoroughness and
does not instruct the model to hedge or to distinguish "evidence exists" from
"evidence is sufficient." Layer: synthesis.

**Measurement caveat.** Our `insufficient` scorer FALSE-PASSED this case: its
substring absence-markers matched incidental "not yet / pending" phrasing and
reported "insufficiency: True" even though the conclusion was the opposite. So
this shortcoming also exposes a scorer weakness (absence detection should test
the answer's stance, not substrings). Both are logged in
[eval-findings](eval-findings.md).

**Priority: MEDIUM.** Judgment-heavy and harder to score deterministically;
valuable but lower-confidence than S1.

## S4 (secondary, MEDIUM-LOW): Fuzzy doc lookup crosses project/ID boundaries

**What.** `read_document` resolves a non-existent ID to a different project's or
a numerically adjacent document, with only a soft warning.

**Evidence.** `read_document("VVPR-P00-081")` returns `VVPR-P01-081` (rapidfuzz
91.7). No P00 documents exist, but 19 documents reference P00 predecessor
pointers in their bodies (e.g. `DR-P01-005` cites `MEMO-P00-159`), so an agent
chasing such a pointer silently lands on the same-numbered P01 doc. Adjacent
drift also occurs (`VVPR-P01-080` -> `VVPR-P01-081`).

**Likely cause.** `_fuzzy_find_doc_id` (`search.py:137-148`) uses `fuzz.ratio`
with `score_cutoff=80` and no constraint that the project segment (P01/P00) or
numeric tail match the query.

**Priority: MEDIUM-LOW.** Only reachable via non-existent / typo'd IDs (valid
IDs short-circuit before fuzzy), but P00 references make that path real. Fix:
require project + numeric-tail match, fuzz only the title.

---

## Priority summary

| id | shortcoming | layer | priority |
|---|---|---|---|
| S1 | Obsolete surfaced as current | retrieval | HIGH (selected fix) |
| S2 | Answers without retrieving | tool selection | MEDIUM |
| S3 | Over-confident sufficiency claims | synthesis | MEDIUM |
| S4 | Fuzzy lookup crosses project/ID | tool behavior | MEDIUM-LOW |

S1 is the improvement target (deliverable 6): smallest change, highest-confidence
measured failure, built-in regression guard. Before/after in the eval on
`current_dfmea_obsolete` (should improve) and `obsolete_rev_explicit` (must not
regress).
