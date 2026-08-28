"""Evaluation harness for the QMS search agent.

Runs a small set of representative questions through the supplied agent,
captures the observable trace (tool calls, tool results, final answer), and
scores each case on quality dimensions -- including whether the answer's
citations are actually grounded in retrieved evidence.

Entry point: ``uv run eval`` (see agent.eval.runner:main).
"""
