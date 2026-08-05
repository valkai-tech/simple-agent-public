# Valkai Coding Onsite: Evaluate and Improve a Search Agent

You are joining a working search system. The repository contains an end-to-end assistant that searches a fixed Quality Management System (QMS) corpus for MedAI's MX1 portable X-ray system and returns cited answers.

Your task is to measure how well it works, find an important shortcoming, and improve it.

## Schedule

The 90-minute session includes setup:

- **5 minutes:** Introduction and setup kickoff.
- **35 minutes:** Inspect the agent, build the baseline evaluation, and choose a failure to improve.
- **5 minutes:** Checkpoint.
- **30 minutes:** Complete the baseline, improve the system, and rerun the evaluation.
- **15 minutes:** Walk through your approach, evidence, change, and next steps.

Normal corpus extraction, indexing, smoke testing, and the first agent run are part of the exercise. Ask the interviewer for help with credential, dependency, or environment failures. Those failures are not interview signal.

## Start here

Prerequisites:

- Python 3.13 or newer;
- the `uv` package manager;
- `Example_QMS_-_MedAI.zip`; and
- an Anthropic API key supplied by the interviewer.

From the repository root:

```bash
uv sync --frozen
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

mkdir -p example_qms_data
unzip -j /path/to/Example_QMS_-_MedAI.zip \
  'Example QMS - MedAI/*.docx' \
  -d example_qms_data

uv run index
uv run smoke
uv run chat
```

Ask the chat agent:

> Find the latest Bill of Materials for the MX1 system and summarize what it covers.

Notice which tools it chooses, what evidence they return, and how the final citations relate to that evidence. Exit the chat after the answer. This first trace is your starting point for the evaluation.

The first index build parses 189 documents and downloads a local embedding model. Later runs load the saved index.

## Representative user questions

These illustrate how people use this search system. You may use them directly, refine them, or create alternatives. Query originality is not evaluated.

- Find the latest MX1 Bill of Materials and summarize what it covers.
- List every engineering change request in the QMS and its status.
- What verification protocols exist for electrical safety?
- Trace electrical-leakage risk from the risk records to verification evidence.
- What changed between two revisions of an MX1 document?
- Does the corpus contain a 510(k) summary for the device?
- Does the Design History File provide enough evidence to support a complete design-control coverage claim?

## Required outcome

By the end of the session, produce:

1. An executable evaluation runner.
2. At least three cases that test meaningfully different risks:
   - one question with a known answer or source;
   - one question that requires multiple documents or tests completeness; and
   - one question where the correct behavior is to report missing or insufficient evidence.
3. Automated scoring for at least two quality dimensions. One dimension must evaluate sources or citations.
4. A baseline run with per-case answers, scores, tool evidence, and useful failure details.
5. At least two evidence-backed shortcomings, their likely causes, and a priority.
6. One improvement to the supplied system that addresses a measured failure.
7. A before-and-after result for the affected case, plus any regression risk or remaining uncertainty.

The improvement may change retrieval, tool behavior, agent instructions, answer synthesis, citation handling, or another relevant layer. The change does not need to succeed. A well-supported hypothesis, bounded experiment, and correct interpretation are more useful than an unmeasured change that happens to look better.

We should be able to run one command and see the per-case evidence. Prefer a few defensible checks over a large set of weak assertions.

## Checkpoint

At the checkpoint, be ready to show one complete case and answer:

- What does the current score measure?
- What could it incorrectly pass?
- Is the most interesting failure in retrieval, tool selection, synthesis, or citation use?
- What is the smallest change that could improve it?
- How will you measure the change and notice a regression?

If the first case is not running, reduce scope to one script, three literal cases, and two checks.

## Supplied system

The agent has three tools:

- `search`: ranked keyword, semantic, or hybrid retrieval;
- `read_document`: full extracted text for a selected document and revision; and
- `list_documents`: exhaustive document metadata for inventory questions.

Answers are instructed to cite sources as `<cite>DOC_ID Rev LETTER</cite>`. Correct citation syntax does not prove that the document exists, was retrieved, supports the nearby claim, or covers every factual claim.

See [SEARCH_AGENT.md](SEARCH_AGENT.md) for the architecture and programmatic interface. The retained FastAPI server and React frontend are not part of the exercise.

## Running the agent in Python

```python
from dotenv import load_dotenv

from agent.core import make_agent

load_dotenv()
agent = make_agent()
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Your question"}]}
)

messages = result["messages"]
answer = messages[-1].content
```

The returned messages include assistant tool calls and tool-result messages. Treat those messages as product behavior. They can distinguish a retrieval failure from a final-answer failure.

## Working norms

- Internet access and AI coding tools are allowed and expected.
- Ask questions when requirements or domain facts are ambiguous.
- Establish a baseline before changing the supplied system.
- Keep the evaluation runnable by the next engineer.
- Explain the limits of every score you rely on.
