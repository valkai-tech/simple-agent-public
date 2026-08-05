"""Verbose tracing for agent execution.

Walks through the message list after invoke() and prints a structured
trace of what happened: tool calls, tool results, and agent reasoning.
"""

from __future__ import annotations

import json
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

TOOL_OUTPUT_COMPACT = 200
TOOL_OUTPUT_VERBOSE = 1000


def print_system_prompt(prompt: str, file=sys.stderr) -> None:
    """Print the system prompt (verbose mode only)."""
    _write(file, f"\n{DIM}{'─' * 60}{RESET}")
    _write(file, f"{DIM}SYSTEM PROMPT ({len(prompt)} chars):{RESET}")
    for line in prompt.split("\n")[:20]:
        _write(file, f"{DIM}│ {line}{RESET}")
    if prompt.count("\n") > 20:
        _write(file, f"{DIM}│ ... [{prompt.count(chr(10))} total lines]{RESET}")
    _write(file, f"{DIM}{'─' * 60}{RESET}\n")


def print_trace(messages: list, verbose: bool = False, file=sys.stderr) -> None:
    """Print a structured trace of the agent's execution.

    In normal mode: shows tool calls with args + compact outputs + reasoning.
    In verbose mode: adds full tool outputs.
    """
    # Build a map of tool_call_id -> ToolMessage for interleaving
    tool_outputs: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = msg

    step = 0
    max_output = TOOL_OUTPUT_VERBOSE if verbose else TOOL_OUTPUT_COMPACT

    for msg in messages:
        if isinstance(msg, (HumanMessage, SystemMessage, ToolMessage)):
            continue

        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Show thinking text before tool calls
            text = msg.content
            if text and isinstance(text, str) and text.strip() and tool_calls:
                _write(file, f"  {DIM}(thinking): {_truncate(text, 200)}{RESET}")

            # Show each tool call immediately followed by its output
            for tc in tool_calls:
                step += 1
                name = tc["name"]
                args = tc["args"]
                args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
                _write(
                    file,
                    f"\n{DIM}[step {step}]{RESET} {CYAN}{BOLD}{name}{RESET}({args_str})",
                )

                # Find and print the matching tool output
                tool_msg = tool_outputs.get(tc.get("id", ""))
                if tool_msg:
                    content = (
                        tool_msg.content
                        if isinstance(tool_msg.content, str)
                        else str(tool_msg.content)
                    )
                    truncated = _truncate(content, max_output)
                    _write(file, f"  {GREEN}→{RESET} {DIM}{truncated}{RESET}")

    _write(file, "")


def _truncate(text: str, max_chars: int) -> str:
    # If it's JSON, pretty-print then truncate line-by-line
    if text.strip().startswith("{"):
        try:
            parsed = json.loads(text)
            pretty = json.dumps(parsed, indent=2)
            lines = pretty.split("\n")
            result_lines = []
            char_count = 0
            for line in lines:
                if char_count + len(line) > max_chars:
                    result_lines.append(f"... [{len(text)} total chars]")
                    break
                result_lines.append(line)
                char_count += len(line)
            return "\n       ".join(result_lines)
        except (json.JSONDecodeError, ValueError):
            pass
    if len(text) <= max_chars:
        return text.replace("\n", " ")
    return text[:max_chars].replace("\n", " ") + f" ... [{len(text)} chars]"


def _write(file, text: str) -> None:
    print(text, file=file)
