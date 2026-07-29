"""Streamlit chat UI for the Ops Copilot — shows which tool(s) handled each turn.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

try:
    # Same Streamlit Community Cloud sqlite3 fix as genai-power-platform-agent —
    # needed here too because tools/policy.py's embedded fallback imports chromadb
    # when Project 2's HTTP service isn't reachable (always true when hosted).
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.agent import LIVE_MODE_AVAILABLE, OpsCopilotAgent
from tools.policy import POLICY_SERVICE_URL, policy_service_healthy

st.set_page_config(page_title="RetailPulse — Ops Copilot", page_icon="🧭")


@st.cache_resource(show_spinner=False)
def load_agent() -> OpsCopilotAgent:
    return OpsCopilotAgent()


st.title("🧭 RetailPulse — Ops Copilot")
st.caption(
    "Ask an HR/IT policy question, or a customer/business analytics question "
    "(e.g. \"what's the churn risk for CUST100005?\", \"KPIs for Singapore\"). "
    "One copilot, two backend systems, routed automatically."
)

with st.sidebar:
    st.subheader("Mode")
    if LIVE_MODE_AVAILABLE:
        st.success("ANTHROPIC_API_KEY detected — live Claude tool-use routing available.")
        default_mode = "live"
    else:
        st.info("No ANTHROPIC_API_KEY found — running in **mock mode** (rule-based router, zero cost).")
        default_mode = "mock"

    mode = st.radio(
        "Orchestration mode",
        options=["mock", "live"] if LIVE_MODE_AVAILABLE else ["mock"],
        index=0,
        help="Live mode uses real Claude tool-use to decide which tool(s) to call.",
    )

    st.subheader("Policy service")
    if policy_service_healthy():
        st.success(f"Connected: {POLICY_SERVICE_URL}")
    else:
        st.info(
            f"No HTTP service at {POLICY_SERVICE_URL} — using the embedded RAG pipeline "
            "fallback instead (same underlying code, just in-process). For local multi-service "
            "dev: `cd ../genai-power-platform-agent && uvicorn rag.api:app --port 8000`"
        )

    st.subheader("Tools available")
    st.markdown(
        "- `ask_hr_it_policy`\n"
        "- `get_customer_churn_risk`\n"
        "- `get_country_kpis`\n"
        "- `get_churn_risk_drivers`"
    )

agent = load_agent()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("tool_calls"):
            with st.expander(f"Tool calls ({len(turn['tool_calls'])})"):
                for tc in turn["tool_calls"]:
                    st.markdown(f"**{tc['tool']}**")
                    st.json({"input": tc["input"], "output": tc["output"]})

question = st.chat_input("Ask about leave, expenses, a customer ID, or a country's KPIs...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing..."):
            result = agent.ask(question, mode=mode)
        st.markdown(result.answer)
        tool_calls = [{"tool": tc.tool, "input": tc.input, "output": tc.output} for tc in result.tool_calls]
        if tool_calls:
            with st.expander(f"Tool calls ({len(tool_calls)})"):
                for tc in tool_calls:
                    st.markdown(f"**{tc['tool']}**")
                    st.json({"input": tc["input"], "output": tc["output"]})
        else:
            st.caption("No tools called — this question wasn't routed anywhere.")

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "tool_calls": tool_calls}
    )
