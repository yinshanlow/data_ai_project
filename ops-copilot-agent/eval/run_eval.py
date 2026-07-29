"""Evaluation harness for the Ops Copilot orchestrator.

Scores against eval/eval_questions.json:
  1. Routing accuracy — did the orchestrator call the expected tool(s)?
  2. Keyword grounding — does the final answer contain the expected facts?
  3. Hallucination flags — for the invalid-customer case, did it fabricate
     a risk band instead of saying "not found"?
  4. Correct refusal rate — for out-of-scope questions, did it call no tools?
  5. Compound-question handling — a separate breakout, since mock mode is
     known (by design) to only route to one tool per question.

Run with:
    python -m eval.run_eval            # mock mode (default, no API key needed)
    python -m eval.run_eval --mode live
"""
import argparse
import json
from pathlib import Path

from orchestrator.agent import OpsCopilotAgent

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_MD_PATH = Path(__file__).parent / "eval_results.md"


def load_questions() -> list[dict]:
    return json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))


def run(mode: str | None) -> dict:
    agent = OpsCopilotAgent()
    questions = load_questions()
    rows = []

    for q in questions:
        result = agent.ask(q["question"], mode=mode)
        tools_called = [tc.tool for tc in result.tool_calls]
        expected_tools = set(q["expected_tools"])
        routing_correct = set(tools_called) == expected_tools

        answer_lower = result.answer.lower()
        must_contain = q.get("must_contain", [])
        keyword_hit = all(kw.lower() in answer_lower for kw in must_contain) if must_contain else None

        must_not_contain = q.get("must_not_contain", [])
        hallucination_flag = any(kw.lower() in answer_lower for kw in must_not_contain) if must_not_contain else False

        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_tools": sorted(expected_tools),
            "tools_called": tools_called,
            "routing_correct": routing_correct,
            "keyword_hit": keyword_hit,
            "hallucination_flag": hallucination_flag,
            "handled": result.handled,
            "mode_used": result.mode_used,
            "answer": result.answer,
        })

    n = len(rows)
    routing_accuracy = sum(1 for r in rows if r["routing_correct"]) / n if n else 0.0
    keyword_rows = [r for r in rows if r["keyword_hit"] is not None]
    keyword_accuracy = sum(1 for r in keyword_rows if r["keyword_hit"]) / len(keyword_rows) if keyword_rows else 0.0
    hallucination_flags = sum(1 for r in rows if r["hallucination_flag"])

    out_of_scope_rows = [r for r in rows if r["category"] == "out_of_scope"]
    correct_refusal_rate = (
        sum(1 for r in out_of_scope_rows if not r["tools_called"]) / len(out_of_scope_rows)
        if out_of_scope_rows else 0.0
    )

    compound_rows = [r for r in rows if r["category"] == "compound"]
    compound_full_success_rate = (
        sum(1 for r in compound_rows if r["routing_correct"]) / len(compound_rows) if compound_rows else 0.0
    )

    summary = {
        "mode": rows[0]["mode_used"] if rows else mode,
        "total_questions": n,
        "routing_accuracy": routing_accuracy,
        "keyword_grounding_accuracy": keyword_accuracy,
        "hallucination_flags": hallucination_flags,
        "correct_refusal_rate": correct_refusal_rate,
        "compound_full_success_rate": compound_full_success_rate,
        "rows": rows,
    }
    return summary


def write_report(summary: dict) -> None:
    lines = [
        "# Evaluation Results",
        "",
        f"Mode used: **{summary['mode']}**",
        "",
        f"- Questions evaluated: {summary['total_questions']}",
        f"- Routing accuracy (expected tool(s) called): **{summary['routing_accuracy']:.0%}**",
        f"- Keyword grounding accuracy (answer contains expected fact): **{summary['keyword_grounding_accuracy']:.0%}**",
        f"- Hallucination flags raised: **{summary['hallucination_flags']}**",
        f"- Correct refusal rate on out-of-scope questions: **{summary['correct_refusal_rate']:.0%}**",
        f"- Compound-question full-success rate (both expected tools called): **{summary['compound_full_success_rate']:.0%}**",
        "",
        "## Per-question results",
        "",
        "| ID | Category | Question | Expected tools | Tools called | Routing correct | Keyword hit | Hallucination |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['question']} | {', '.join(r['expected_tools']) or '(none)'} | "
            f"{', '.join(r['tools_called']) or '(none)'} | {r['routing_correct']} | {r['keyword_hit']} | "
            f"{r['hallucination_flag']} |"
        )
    RESULTS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live"], default=None)
    args = parser.parse_args()

    summary = run(mode=args.mode)

    print(f"Mode: {summary['mode']}")
    print(f"Routing accuracy: {summary['routing_accuracy']:.0%}")
    print(f"Keyword grounding accuracy: {summary['keyword_grounding_accuracy']:.0%}")
    print(f"Hallucination flags: {summary['hallucination_flags']}")
    print(f"Correct refusal rate (out-of-scope): {summary['correct_refusal_rate']:.0%}")
    print(f"Compound-question full-success rate: {summary['compound_full_success_rate']:.0%}")

    write_report(summary)
    print(f"\nFull report written to {RESULTS_MD_PATH}")
