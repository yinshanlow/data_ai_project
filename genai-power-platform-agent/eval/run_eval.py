"""Evaluation harness for the RAG pipeline.

Scores two things against eval/eval_questions.json:
  1. Retrieval accuracy  — for in-scope questions, does the expected source
     document show up among the top-k retrieved chunks?
  2. Grounding / hallucination risk — for in-scope questions, did the pipeline
     actually answer (rather than hedge) and does the answer contain the
     expected keywords? For deliberately out-of-scope questions, did the
     pipeline correctly refuse instead of fabricating an answer?

Run with:
    python -m eval.run_eval            # mock mode (default, no API key needed)
    python -m eval.run_eval --mode live
"""
import argparse
import json
from pathlib import Path

from rag.pipeline import RAGPipeline

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_MD_PATH = Path(__file__).parent / "eval_results.md"


def load_questions() -> list[dict]:
    return json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))


def retrieval_hit(result, expected_source: str | None) -> bool | None:
    if expected_source is None:
        return None
    return any(c.source_file == expected_source for c in result.chunks)


def keyword_hit(result, expected_keywords: list[str]) -> bool | None:
    if not expected_keywords:
        return None
    answer_lower = result.answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def run(mode: str | None) -> dict:
    pipeline = RAGPipeline()
    questions = load_questions()

    rows = []
    for q in questions:
        result = pipeline.ask(q["question"], mode=mode)
        in_scope = q["in_scope"]

        r_hit = retrieval_hit(result, q.get("expected_source")) if in_scope else None
        k_hit = keyword_hit(result, q.get("expected_keywords", [])) if in_scope else None

        if in_scope:
            # Hallucination risk: pipeline answered (grounded=True) but the
            # expected keywords never showed up anywhere in the answer.
            hallucination_flag = bool(result.grounded and k_hit is False)
        else:
            # For out-of-scope questions, the ONLY correct behavior is the
            # not-found fallback. Answering at all here is the hallucination risk.
            hallucination_flag = bool(result.grounded)

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "in_scope": in_scope,
            "expected_source": q.get("expected_source"),
            "retrieval_hit": r_hit,
            "keyword_hit": k_hit,
            "grounded": result.grounded,
            "best_distance": round(result.best_distance, 3) if result.best_distance is not None else None,
            "hallucination_flag": hallucination_flag,
            "mode_used": result.mode_used,
            "answer": result.answer,
        })

    in_scope_rows = [r for r in rows if r["in_scope"]]
    out_scope_rows = [r for r in rows if not r["in_scope"]]

    retrieval_accuracy = (
        sum(1 for r in in_scope_rows if r["retrieval_hit"]) / len(in_scope_rows)
        if in_scope_rows else 0.0
    )
    keyword_accuracy = (
        sum(1 for r in in_scope_rows if r["keyword_hit"]) / len(in_scope_rows)
        if in_scope_rows else 0.0
    )
    correct_refusals = (
        sum(1 for r in out_scope_rows if not r["grounded"]) / len(out_scope_rows)
        if out_scope_rows else 0.0
    )
    hallucination_flags = sum(1 for r in rows if r["hallucination_flag"])

    summary = {
        "mode": rows[0]["mode_used"] if rows else mode,
        "total_questions": len(rows),
        "in_scope_questions": len(in_scope_rows),
        "out_of_scope_questions": len(out_scope_rows),
        "retrieval_accuracy": retrieval_accuracy,
        "keyword_grounding_accuracy": keyword_accuracy,
        "correct_refusal_rate": correct_refusals,
        "hallucination_flags": hallucination_flags,
        "rows": rows,
    }
    return summary


def write_report(summary: dict) -> None:
    lines = [
        "# Evaluation Results",
        "",
        f"Mode used: **{summary['mode']}**",
        "",
        f"- Questions evaluated: {summary['total_questions']} "
        f"({summary['in_scope_questions']} in-scope, {summary['out_of_scope_questions']} deliberately out-of-scope)",
        f"- Retrieval accuracy (expected doc in top-k): **{summary['retrieval_accuracy']:.0%}**",
        f"- Keyword grounding accuracy (answer contains expected fact): **{summary['keyword_grounding_accuracy']:.0%}**",
        f"- Correct refusal rate on out-of-scope questions: **{summary['correct_refusal_rate']:.0%}**",
        f"- Hallucination flags raised: **{summary['hallucination_flags']}**",
        "",
        "## Per-question results",
        "",
        "| ID | Question | In scope | Retrieval hit | Keyword hit | Grounded | Best distance | Hallucination flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['id']} | {r['question']} | {r['in_scope']} | {r['retrieval_hit']} | "
            f"{r['keyword_hit']} | {r['grounded']} | {r['best_distance']} | {r['hallucination_flag']} |"
        )
    RESULTS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live"], default=None)
    args = parser.parse_args()

    summary = run(mode=args.mode)

    print(f"Mode: {summary['mode']}")
    print(f"Retrieval accuracy: {summary['retrieval_accuracy']:.0%}")
    print(f"Keyword grounding accuracy: {summary['keyword_grounding_accuracy']:.0%}")
    print(f"Correct refusal rate (out-of-scope): {summary['correct_refusal_rate']:.0%}")
    print(f"Hallucination flags: {summary['hallucination_flags']}")

    write_report(summary)
    print(f"\nFull report written to {RESULTS_MD_PATH}")
