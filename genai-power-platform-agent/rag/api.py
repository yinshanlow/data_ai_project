"""Minimal HTTP wrapper around the RAG pipeline.

This is the piece a Power Automate flow (or Copilot Studio / Teams bot) would
call in a real deployment — see power_automate/ for the documented flow stub
that targets this exact endpoint.

Run with:
    uvicorn rag.api:app --host 0.0.0.0 --port 8000
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from rag.pipeline import RAGPipeline

app = FastAPI(
    title="Meridian Holdings HR/IT Policy Assistant API",
    description="RAG-backed Q&A over synthetic HR/IT policy documents.",
    version="1.0.0",
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str
    mode: Literal["mock", "live"] | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    mode_used: str
    grounded: bool
    citations: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    result = get_pipeline().ask(payload.question, mode=payload.mode)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        mode_used=result.mode_used,
        grounded=result.grounded,
        citations=result.citations,
    )
