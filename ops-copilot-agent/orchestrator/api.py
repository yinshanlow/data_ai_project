"""HTTP wrapper around the orchestrator — same pattern as Project 2's rag/api.py,
so a Power Automate/Teams flow can call this copilot the same way it would
call any single-purpose service.

Run with:
    uvicorn orchestrator.api:app --host 0.0.0.0 --port 8100
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator.agent import OpsCopilotAgent

app = FastAPI(
    title="RetailPulse Ops Copilot API",
    description="Orchestrator routing between HR/IT policy and customer analytics tools.",
    version="1.0.0",
)

_agent: OpsCopilotAgent | None = None


def get_agent() -> OpsCopilotAgent:
    global _agent
    if _agent is None:
        _agent = OpsCopilotAgent()
    return _agent


class AskRequest(BaseModel):
    question: str
    mode: Literal["mock", "live"] | None = None


class ToolCallOut(BaseModel):
    tool: str
    input: dict
    output: dict


class AskResponse(BaseModel):
    question: str
    answer: str
    mode_used: str
    handled: bool
    tool_calls: list[ToolCallOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    result = get_agent().ask(payload.question, mode=payload.mode)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        mode_used=result.mode_used,
        handled=result.handled,
        tool_calls=[ToolCallOut(tool=tc.tool, input=tc.input, output=tc.output) for tc in result.tool_calls],
    )
