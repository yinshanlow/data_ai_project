"""Policy tool: calls genai-power-platform-agent's existing FastAPI service.

Deliberately an HTTP call to the already-built rag/api.py service rather than
importing that project's Python modules directly. This is the same pattern
Project 2's Power Automate stub was designed around — the FastAPI layer
exists specifically so more than one caller can reuse it. This project is
that second caller.

Requires the service to be running:
    cd ../genai-power-platform-agent && uvicorn rag.api:app --port 8000
"""
import os

import requests

POLICY_SERVICE_URL = os.environ.get("POLICY_SERVICE_URL", "http://localhost:8000").rstrip("/")


def ask_hr_it_policy(question: str) -> dict:
    try:
        resp = requests.post(f"{POLICY_SERVICE_URL}/ask", json={"question": question}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        return {
            "error": (
                f"HR/IT policy service unavailable at {POLICY_SERVICE_URL}. Start it with: "
                f"cd ../genai-power-platform-agent && uvicorn rag.api:app --port 8000 "
                f"(underlying error: {exc})"
            )
        }


def policy_service_healthy() -> bool:
    try:
        resp = requests.get(f"{POLICY_SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
