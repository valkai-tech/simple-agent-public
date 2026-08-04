# Valkai Coding Onsite: Evaluate a Search Agent

You are joining a working system, not building search from scratch. The repository contains an end-to-end assistant that searches a fixed Quality Management System (QMS) corpus for MedAI's MX1 portable X-ray system and returns cited answers.

Your task is to determine how well it works.

## The exercise

Build a small evaluation system that runs the search assistant against representative questions, measures answer quality, and makes its shortcomings concrete.

By the end of the session, we should be able to run one command and see:

- which cases passed or failed;
- enough evidence to understand each result;
- the most important shortcomings you found; and
- what you would improve next.

Use any structure or libraries you think fit. We care about the quality of the evaluation and your reasoning, not a particular framework.

## Required scope

Your submission should include:

1. An executable eval runner.
2. At least three cases that test meaningfully different risks. Include:
   - one question with a known answer or source;
   - one question that requires multiple documents or tests completeness; and
   - one question where the correct behavior is to report missing or insufficient evidence.
3. Automated scoring for at least two dimensions. One dimension must evaluate sources or citations.
4. A baseline run with per-case evidence, not only one aggregate score.
5. A short summary of at least two shortcomings, their likely causes, and which one you would address first.

Good evals distinguish a plausible-looking answer from a supported answer. They also make their own blind spots visible.

## Out of scope

Do not spend the core exercise rebuilding ingestion, retrieval, the CLI or web chat interface, or the system prompt. Treat the supplied agent as the product under evaluation. The existing FastAPI server and React frontend are retained from the current repository but are not part of the task.

If the required evaluation is complete, you may use remaining time to improve one shortcoming and show the before-and-after result. This is a stretch goal, not a requirement.

## Timeline

- **5 minutes:** Introduction and questions.
- **40 minutes:** Build the first end-to-end cases and scoring path.
- **5 minutes:** Checkpoint. Show one complete result and decide what to finish.
- **25 minutes:** Complete the run and diagnose the baseline.
- **15 minutes:** Walk through your approach, findings, tradeoffs, and next step.

## What is provided

The agent has three tools:

- `search`: ranked keyword, semantic, or hybrid retrieval;
- `read_document`: full text for a selected document and revision; and
- `list_documents`: exhaustive document metadata for inventory questions.

Answers are instructed to cite sources as `<cite>DOC_ID Rev LETTER</cite>`. A correctly formatted citation is not necessarily a correct or well-supported citation.

See [SEARCH_AGENT.md](SEARCH_AGENT.md) for the architecture and programmatic interface.

## Setup before the interview

Setup is not part of the 90-minute exercise. Do not start the timer until the final chat check succeeds.

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
uv run chat --quiet
```

Ask the chat agent `Find the Bill of Materials for the MX1 system`, then exit. Once it returns a cited answer, setup is complete.

The first index build parses 189 documents and downloads a local embedding model. It may take several minutes. Later runs load the saved index.

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

The returned messages include assistant tool calls and tool-result messages. Your evals may inspect them when that produces a better diagnosis than final-answer checks alone.

## Working norms

- Internet access and AI coding tools are allowed and expected.
- Ask questions when requirements or domain facts are ambiguous.
- Keep the evaluation runnable by the next engineer.
- Prefer a few defensible cases over a large set of weak assertions.
