"""Policy tool: calls genai-power-platform-agent's existing FastAPI service.

Preferred path: an HTTP call to the already-built rag/api.py service rather than
importing that project's Python modules directly. This is the same pattern
Project 2's Power Automate stub was designed around — the FastAPI layer
exists specifically so more than one caller can reuse it. This project is
that second caller.

Fallback path: Streamlit Community Cloud only runs one process per app, so
there's no way to also run Project 2's FastAPI service alongside this one when
hosted. Rather than ship a hosted demo where every HR/IT question silently
fails, this falls back to importing Project 2's RAG pipeline directly in the
same process when the HTTP service isn't reachable — both projects live in the
same repo, so Project 2's code and policy documents are physically present
right alongside this one wherever this repo is cloned. This is a deliberate,
documented adaptation to a single-process hosting constraint, not a reversal
of the HTTP-reuse design: local multi-service development still prefers HTTP.

For local dev:
    cd ../genai-power-platform-agent && uvicorn rag.api:app --port 8000
"""
import os
import sys
from pathlib import Path

import requests

POLICY_SERVICE_URL = os.environ.get("POLICY_SERVICE_URL", "http://localhost:8000").rstrip("/")

_embedded_pipeline = None  # lazily constructed only if the HTTP service is unreachable


def policy_service_healthy() -> bool:
    try:
        resp = requests.get(f"{POLICY_SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _get_embedded_pipeline():
    global _embedded_pipeline
    if _embedded_pipeline is None:
        genai_agent_dir = Path(__file__).resolve().parent.parent.parent / "genai-power-platform-agent"
        if not genai_agent_dir.exists():
            raise FileNotFoundError(
                f"genai-power-platform-agent not found at {genai_agent_dir} — can't fall back to "
                "an embedded policy pipeline without it."
            )
        sys.path.insert(0, str(genai_agent_dir))
        from rag.pipeline import RAGPipeline  # noqa: PLC0415 - deliberately lazy, sibling-project import

        _embedded_pipeline = RAGPipeline()
    return _embedded_pipeline


def _ask_via_http(question: str) -> dict:
    resp = requests.post(f"{POLICY_SERVICE_URL}/ask", json={"question": question}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _ask_via_embedded_pipeline(question: str) -> dict:
    pipeline = _get_embedded_pipeline()
    result = pipeline.ask(question)
    return {
        "question": result.question,
        "answer": result.answer,
        "mode_used": result.mode_used,
        "grounded": result.grounded,
        "citations": result.citations,
    }


def ask_hr_it_policy(question: str) -> dict:
    if policy_service_healthy():
        try:
            return _ask_via_http(question)
        except requests.exceptions.RequestException:
            pass  # fall through to the embedded pipeline below

    try:
        return _ask_via_embedded_pipeline(question)
    except Exception as exc:  # noqa: BLE001 - surface as tool data, not a crash
        return {
            "error": (
                f"HR/IT policy lookup unavailable: no reachable service at {POLICY_SERVICE_URL}, "
                f"and the embedded fallback failed too ({exc}). For local dev, start the service with: "
                f"cd ../genai-power-platform-agent && uvicorn rag.api:app --port 8000"
            )
        }
