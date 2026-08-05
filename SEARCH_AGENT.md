# Supplied Search Agent

The interview repository contains a working search assistant over a fixed document corpus. This document describes its public contract so the timed exercise can focus on evaluation.

## Corpus

The corpus contains 189 files representing 188 distinct document revisions across 165 document identifiers for MedAI's MX1 portable X-ray system. The files cover bills of materials, risk records, verification protocols, engineering changes, design reviews, plans, and third-party reports.

Current project documents generally include `P01` in their identifiers. Some predecessor references use `P00`. Several document identifiers have multiple revisions, and some files are obsolete or contain no extractable text.

## Runtime flow

1. `agent.ingestion` parses filenames and DOCX content.
2. ChromaDB stores document sections for semantic retrieval.
3. An in-memory BM25 index provides keyword retrieval.
4. `SearchIndex` combines keyword and semantic rankings.
5. A LangChain deep agent selects tools and writes the cited answer.

The implementation is intentionally a realistic baseline, not a claim that every answer is correct.

## Agent tools

### `search`

Searches document sections and returns the best matching passages per document.

Important inputs include:

- `query`;
- `mode`: `keyword`, `semantic`, or `hybrid`;
- optional `doc_type`;
- optional `latest_only`; and
- `n_results`.

### `read_document`

Returns a document's extracted text for a document identifier and optional revision. Without a revision, it returns the latest available revision.

### `list_documents`

Returns document metadata, optionally filtered by type. This is the only exhaustive inventory tool. Ranked search results are not exhaustive.

## Result messages

`agent.invoke(...)` returns a dictionary whose `messages` entry contains the full run:

- the user message;
- assistant messages containing `tool_calls`;
- `ToolMessage` objects containing tool results; and
- the final assistant answer.

The following loop prints the observable trace:

```python
for message in result["messages"]:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print(tool_calls)

    if type(message).__name__ == "ToolMessage":
        print(message.name, message.content)
```

Treat tool calls and outputs as product behavior. They can help distinguish a retrieval failure from a final-answer failure.

## Citation contract

The system prompt asks the agent to cite factual claims using:

```text
<cite>BOM-055 Rev G</cite>
```

For an obsolete source, the prompt permits:

```text
<cite>RSK-P01-010 Rev A OBSOLETE</cite>
```

The prompt does not itself guarantee that a citation exists, was retrieved, supports the nearby claim, or covers every factual claim. Those are separate properties an evaluation may choose to measure.
