"""Interactive CLI chat REPL with QMS search capabilities."""

import argparse
import logging
import warnings

from dotenv import load_dotenv

from agent.core import make_agent
from agent.prompts import build_prompt
from agent.tools import reset_tool_steps, set_live_logging
from agent.tracing import print_system_prompt, print_trace
from agent.utils import DEFAULT_DATA_DIR, DEFAULT_INDEX_DIR, DEFAULT_MODEL


def main():
    load_dotenv()
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="QMS Document Search Agent")
    parser.add_argument(
        "--model",
        default=None,
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-sonnet-4-6",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the QMS document directory",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Path to store/load the search index",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild of the search index",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show system prompt and full tool outputs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide tool call trace (only show final response)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("Initializing QMS search agent...")
    agent = make_agent(
        model_str=args.model or DEFAULT_MODEL,
        system_prompt=args.system,
        data_dir=args.data_dir or DEFAULT_DATA_DIR,
        index_dir=args.index_dir or DEFAULT_INDEX_DIR,
        force_reindex=args.reindex,
    )

    # Configure live tool call logging
    set_live_logging(not args.quiet)

    if args.verbose:
        prompt = args.system or build_prompt(
            tool_names=["search", "read_document", "list_documents"]
        )
        print_system_prompt(prompt)

    print("Ready. Type 'quit' to exit.\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        reset_tool_steps()
        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})

        all_messages = result["messages"]

        # In verbose mode, also show full tool outputs after the run
        if args.verbose:
            print_trace(all_messages, verbose=True)

        ai_msg = all_messages[-1]
        content = (
            ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
        )
        print(f"\nAssistant: {content}\n")
        messages = all_messages


if __name__ == "__main__":
    main()
