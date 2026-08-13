"""Agent factory: creates a deep agent wired up with search tools."""

from __future__ import annotations

import logging
import os

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from agent.ingestion import ingest
from agent.index_lock import index_lock
from agent.prompts import build_prompt
from agent.search import SearchIndex
from agent.tools import create_tools
from agent.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL,
    MAX_RECURSION_LIMIT,
)

logger = logging.getLogger(__name__)


def ensure_index(
    data_dir: str = DEFAULT_DATA_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    force: bool = False,
) -> SearchIndex:
    """Build or load the search index.

    If the index already exists on disk and force=False, loads it.
    Otherwise, runs the full ingestion pipeline.
    """
    with index_lock(index_dir):
        body_store_path = os.path.join(index_dir, "body_store.json")
        if os.path.exists(body_store_path) and not force:
            logger.info("Loading existing index from %s", index_dir)
        else:
            logger.info("Building index from %s ...", data_dir)
            ingest(data_dir, persist_dir=index_dir)
            logger.info("Index built.")

        return SearchIndex(persist_dir=index_dir)


def make_agent(
    model_str: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    force_reindex: bool = False,
):
    """Create a deep agent with QMS search tools.

    Args:
        model_str: Provider and model in "provider:model" format.
        system_prompt: Optional system prompt override.
        data_dir: Path to the QMS document directory.
        index_dir: Path to store/load the ChromaDB index.
        force_reindex: If True, rebuild index even if it already exists.

    Returns:
        A compiled LangGraph agent with search capabilities.
    """
    index = ensure_index(data_dir=data_dir, index_dir=index_dir, force=force_reindex)

    model = init_chat_model(model_str)
    tools = create_tools(index)
    tool_names = [t.name for t in tools]

    prompt = system_prompt or build_prompt(tool_names=tool_names)

    agent = create_deep_agent(model=model, tools=tools, system_prompt=prompt)

    # The deep agent framework uses internal graph steps for middleware
    # (summarization, prompt caching, etc.) beyond just tool calls.
    # 50 allows roughly 15 tool calls while preventing runaway loops.
    # Default without this is 1000 (from deepagents).
    # This is to prevent runaway loops.
    return agent.with_config({"recursion_limit": MAX_RECURSION_LIMIT})
