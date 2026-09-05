"""Evaluation runner.

Runs cases through the supplied agent and prints, per case: the final answer,
the tool evidence (calls + the retrieved-evidence doc set), the scores for each
dimension, and failure details. Ends with an aggregate scorecard.

    uv run eval                 # all cases
    uv run eval --core          # only the 3 required-risk-type cases
    uv run eval --case bom_latest --case 510k_absent
    uv run eval --json out.json # also dump structured results

The runner reuses one agent (one index load) across all cases.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv

from agent.core import make_agent
from agent.eval.cases import CASES, CORE_CASE_IDS, Case
from agent.eval.scorers import Score, Trace, extract_trace, score_case
from agent.search import SearchIndex
from agent.tools import reset_tool_steps, set_live_logging
from agent.utils import DEFAULT_DATA_DIR, DEFAULT_INDEX_DIR, DEFAULT_MODEL

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass
class CaseResult:
    case_id: str
    kind: str
    question: str
    answer: str
    tool_calls: list[dict]
    grounded_doc_ids: list[str]
    citations: list[str]
    scores: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s["passed"] for s in self.scores)


def _run_one(
    agent, case: Case, truth_ids: set[str] | None
) -> tuple[Trace, list[Score]]:
    reset_tool_steps()
    result = agent.invoke({"messages": [{"role": "user", "content": case.question}]})
    trace = extract_trace(result["messages"])
    return trace, score_case(case, trace, truth_ids)


def _mark(passed: bool) -> str:
    return f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"


def _print_case(result: CaseResult) -> None:
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}[{result.case_id}] {CYAN}({result.kind}){RESET} {result.question}")
    print(f"{'=' * 78}")

    # Tool evidence
    if result.tool_calls:
        print(f"\n{DIM}Tool calls:{RESET}")
        for i, tc in enumerate(result.tool_calls, 1):
            args = ", ".join(
                f"{k}={json.dumps(v)}" for k, v in tc["args"].items() if v != ""
            )
            print(f"  {DIM}{i}.{RESET} {tc['name']}({args})")
    else:
        print(f"\n{YELLOW}No tool calls.{RESET}")
    print(
        f"{DIM}Retrieved evidence ({len(result.grounded_doc_ids)} docs):{RESET} "
        f"{', '.join(sorted(result.grounded_doc_ids)[:20])}"
        f"{' ...' if len(result.grounded_doc_ids) > 20 else ''}"
    )
    print(
        f"{DIM}Citations in answer:{RESET} "
        f"{', '.join(result.citations) if result.citations else '(none)'}"
    )

    # Answer
    print(f"\n{DIM}Answer:{RESET}")
    answer = result.answer.strip() or "(empty)"
    print("  " + answer.replace("\n", "\n  "))

    # Scores
    print(f"\n{DIM}Scores:{RESET}")
    for s in result.scores:
        print(
            f"  {_mark(s['passed'])} {s['dimension']:<20} "
            f"{s['value']:.2f}  {DIM}{s['detail']}{RESET}"
        )


def run(
    cases: list[Case], model: str, data_dir: str, index_dir: str
) -> tuple[list[CaseResult], list[dict]]:
    """Run cases. Returns (lean CaseResults, verbose trace records).

    The verbose records add the raw tool outputs (heavy) and are meant for the
    separate --trace file, not the main --json scorecard.
    """
    print(f"Initializing agent (model={model}) ...")
    agent = make_agent(model_str=model, data_dir=data_dir, index_dir=index_dir)
    set_live_logging(False)  # keep eval output clean; we print our own trace

    # Ground-truth inventories for the completeness_recall oracle, one read of
    # the same index the agent uses. Cached per doc_type.
    oracle = SearchIndex(persist_dir=index_dir)
    truth_by_type: dict[str, set[str]] = {}
    for case in cases:
        dt = case.expect_doc_type
        if dt and dt not in truth_by_type:
            truth_by_type[dt] = {
                d["doc_id"] for d in oracle.list_all_documents(doc_type=dt)
            }

    results: list[CaseResult] = []
    verbose: list[dict] = []
    for case in cases:
        print(f"\n{DIM}Running {case.id} ...{RESET}", flush=True)
        truth_ids = truth_by_type.get(case.expect_doc_type) if case.expect_doc_type else None
        trace, scores = _run_one(agent, case, truth_ids)
        scores_json = [
            {
                "dimension": s.dimension,
                "value": s.value,
                "passed": s.passed,
                "detail": s.detail,
            }
            for s in scores
        ]
        results.append(
            CaseResult(
                case_id=case.id,
                kind=case.kind,
                question=case.question,
                answer=trace.answer,
                tool_calls=[{"name": tc.name, "args": tc.args} for tc in trace.tool_calls],
                grounded_doc_ids=sorted(trace.grounded_doc_ids),
                citations=[c.raw for c in trace.citations],
                scores=scores_json,
            )
        )
        # Verbose record: everything above PLUS the raw tool outputs, so a
        # claim-support scorer can re-check the answer against retrieved text.
        verbose.append(
            {
                "case_id": case.id,
                "kind": case.kind,
                "question": case.question,
                "answer": trace.answer,
                "tool_calls": [{"name": tc.name, "args": tc.args} for tc in trace.tool_calls],
                "tool_results": [
                    {"name": tr.name, "tool_call_id": tr.tool_call_id, "content": tr.content}
                    for tr in trace.tool_results
                ],
                "citations": [c.raw for c in trace.citations],
                "scores": scores_json,
            }
        )
    return results, verbose


def _print_summary(results: list[CaseResult]) -> None:
    print(f"\n\n{BOLD}{'=' * 100}{RESET}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'=' * 100}")

    # Render whatever dimensions actually appeared, in first-seen order.
    dims: list[str] = []
    for r in results:
        for s in r.scores:
            if s["dimension"] not in dims:
                dims.append(s["dimension"])

    header = f"{'case':<26}{'kind':<14}" + "".join(f"{d:<22}" for d in dims)
    print(DIM + header + RESET)
    for r in results:
        by_dim = {s["dimension"]: s for s in r.scores}
        row = f"{r.case_id:<26}{r.kind:<14}"
        for d in dims:
            s = by_dim.get(d)
            if s:
                cell = f"{_mark(s['passed'])} {s['value']:.2f}"
                pad = 22 - len(f"{'PASS' if s['passed'] else 'FAIL'} {s['value']:.2f}")
            else:
                cell, pad = f"{DIM}-{RESET}", 22 - 1
            row += cell + " " * pad
        print(row)

    total = len(results)
    passed = sum(r.passed for r in results)
    # Per-dimension pass rates make the weak/strong dimensions visible.
    print()
    for d in dims:
        scored = [s for r in results for s in r.scores if s["dimension"] == d]
        p = sum(s["passed"] for s in scored)
        print(f"{DIM}  {d:<22} {p}/{len(scored)} passed{RESET}")
    print(f"\n{BOLD}{passed}/{total} cases passed ALL applicable dimensions.{RESET}")


def main() -> None:
    load_dotenv()
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="QMS search-agent evaluation runner")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--core", action="store_true", help="Run only the 3 required-risk-type cases"
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this case id (repeatable)",
    )
    parser.add_argument("--json", default=None, help="Write structured results to this path")
    parser.add_argument(
        "--trace",
        default=None,
        help="Write a separate verbose trace (incl. raw tool outputs) to this path",
    )
    args = parser.parse_args()

    selected = CASES
    if args.core:
        selected = [c for c in CASES if c.id in CORE_CASE_IDS]
    if args.case:
        selected = [c for c in CASES if c.id in set(args.case)]
    if not selected:
        raise SystemExit("No cases selected.")

    results, verbose = run(selected, args.model, args.data_dir, args.index_dir)

    for r in results:
        _print_case(r)
    _print_summary(results)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"\n{DIM}Wrote structured results to {args.json}{RESET}")

    if args.trace:
        with open(args.trace, "w") as fh:
            json.dump(verbose, fh, indent=2)
        print(f"{DIM}Wrote verbose trace (with raw tool outputs) to {args.trace}{RESET}")


if __name__ == "__main__":
    main()
