"""Orchestrator: routes a question to the HR/IT policy tool and/or the
customer-analytics tools, via real Claude tool-use (live mode) or a
rule-based keyword router (mock mode, no API key needed).
"""
import json
import os
import re
from dataclasses import dataclass, field

from anthropic import Anthropic

from orchestrator.tool_schemas import SYSTEM_PROMPT, TOOLS
from tools.analytics import get_churn_risk_drivers, get_country_kpis, get_customer_churn_risk, valid_countries
from tools.policy import ask_hr_it_policy

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
LIVE_MODE_AVAILABLE = bool(ANTHROPIC_API_KEY)

MAX_TOOL_ITERATIONS = 4

NOT_HANDLED_MESSAGE = (
    "That's outside what this copilot can help with — it only covers HR/IT policy "
    "questions and RetailPulse customer/business analytics. Please check with the "
    "relevant team directly."
)

TOOL_FUNCTIONS = {
    "ask_hr_it_policy": lambda input_: ask_hr_it_policy(input_["question"]),
    "get_customer_churn_risk": lambda input_: get_customer_churn_risk(input_["customer_id"]),
    "get_country_kpis": lambda input_: get_country_kpis(input_["country"]),
    "get_churn_risk_drivers": lambda input_: get_churn_risk_drivers(),
}


@dataclass
class ToolCall:
    tool: str
    input: dict
    output: dict


@dataclass
class AgentResult:
    question: str
    answer: str
    mode_used: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    handled: bool = True


def execute_tool(name: str, tool_input: dict) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return func(tool_input)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure back to the model as data, not a crash
        return {"error": f"Tool '{name}' failed: {exc}"}


def run_live(question: str) -> AgentResult:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": question}]
    tool_calls: list[ToolCall] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text").strip()
            return AgentResult(question=question, answer=final_text, mode_used="live", tool_calls=tool_calls)

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = execute_tool(block.name, block.input)
            tool_calls.append(ToolCall(tool=block.name, input=block.input, output=output))
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
            )
        messages.append({"role": "user", "content": tool_results})

    return AgentResult(
        question=question,
        answer="I wasn't able to resolve this within the allotted number of tool calls.",
        mode_used="live",
        tool_calls=tool_calls,
        handled=False,
    )


_HR_KEYWORDS = [
    "leave", "expense", "reimburs", "laptop", "it equipment", "remote work", "hybrid",
    "in the office", "wfh", "work from home", "onboard", "probation", "password",
    "mfa", "phishing", "business travel", "per diem", "flight", "performance review",
    "rating", "pip", "maternity", "paternity", "parental leave", "harassment",
    "grievance", "workplace conduct",
]
_CUSTOMER_ID_RE = re.compile(r"\bCUST\d{6}\b", re.IGNORECASE)
_DRIVER_KEYWORDS = ["driver", "why are customers churning", "feature importance", "model accuracy", "how accurate"]
# A bare country name isn't enough signal on its own — "weather in Singapore" isn't
# a KPI question. Require an accompanying analytics-context word too.
_ANALYTICS_CONTEXT_KEYWORDS = ["customer", "revenue", "kpi", "churn", "risk", "market", "sales"]


def run_mock(question: str) -> AgentResult:
    """Rule-based keyword router — a real fallback, not a demo of true tool-use.

    Deliberately simpler than live mode: it picks ONE route per question and
    does not attempt compound/multi-tool questions the way live mode can.
    See the README's honest failure case for what this costs on ambiguous
    or compound questions.
    """
    q = question.lower()
    tool_calls: list[ToolCall] = []

    customer_match = _CUSTOMER_ID_RE.search(question)
    if customer_match:
        customer_id = customer_match.group().upper()
        output = get_customer_churn_risk(customer_id)
        tool_calls.append(ToolCall(tool="get_customer_churn_risk", input={"customer_id": customer_id}, output=output))
        if "error" in output:
            answer = output["error"]
        else:
            answer = (
                f"Customer {output['customer_id']} ({output['country']}, {output['segment']} segment): "
                f"churn probability {output['churn_probability']:.0%}, risk band {output['churn_risk_band']}. "
                f"Recency {output['recency_days']} days, frequency {output['frequency']}, "
                f"monetary SGD {output['monetary_sgd']}, tenure {output['tenure_days']} days."
            )
        return AgentResult(question=question, answer=answer, mode_used="mock", tool_calls=tool_calls)

    for country in valid_countries():
        if country.lower() in q and any(kw in q for kw in _ANALYTICS_CONTEXT_KEYWORDS):
            output = get_country_kpis(country)
            tool_calls.append(ToolCall(tool="get_country_kpis", input={"country": country}, output=output))
            answer = (
                f"{output['country']}: {output['customer_count']} customers, "
                f"average churn probability {output['avg_churn_probability']:.0%}, "
                f"{output['high_risk_count']} high-risk, total revenue SGD {output['total_monetary_sgd']:,.2f}."
            )
            return AgentResult(question=question, answer=answer, mode_used="mock", tool_calls=tool_calls)

    if any(kw in q for kw in _DRIVER_KEYWORDS):
        output = get_churn_risk_drivers()
        tool_calls.append(ToolCall(tool="get_churn_risk_drivers", input={}, output=output))
        drivers = ", ".join(f"{d['feature']} ({d['importance']:.0%})" for d in output["top_drivers"])
        answer = (
            f"Top churn drivers: {drivers}. Model accuracy {output['model_accuracy']:.1%}, "
            f"ROC-AUC {output['model_roc_auc']:.2f}."
        )
        return AgentResult(question=question, answer=answer, mode_used="mock", tool_calls=tool_calls)

    if any(kw in q for kw in _HR_KEYWORDS):
        output = ask_hr_it_policy(question)
        tool_calls.append(ToolCall(tool="ask_hr_it_policy", input={"question": question}, output=output))
        answer = output.get("error") or output.get("answer", "")
        return AgentResult(question=question, answer=answer, mode_used="mock", tool_calls=tool_calls)

    return AgentResult(
        question=question, answer=NOT_HANDLED_MESSAGE, mode_used="mock", tool_calls=tool_calls, handled=False
    )


class OpsCopilotAgent:
    def ask(self, question: str, mode: str | None = None) -> AgentResult:
        resolved_mode = mode or ("live" if LIVE_MODE_AVAILABLE else "mock")
        if resolved_mode == "live":
            return run_live(question)
        return run_mock(question)
