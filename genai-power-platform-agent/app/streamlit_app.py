"""Streamlit chat UI for the HR/IT policy RAG assistant.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import LIVE_MODE_AVAILABLE, POLICIES_DIR
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="Meridian Holdings — HR/IT Assistant", page_icon="💬")


@st.cache_resource(show_spinner="Loading policy knowledge base...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


st.title("💬 Meridian Holdings — HR/IT Policy Assistant")
st.caption(
    "Ask a question about leave, expenses, IT equipment, remote work, or any other HR/IT "
    "policy. All documents in this demo are synthetic — see the README for details."
)

with st.sidebar:
    st.subheader("Mode")
    if LIVE_MODE_AVAILABLE:
        default_mode = "live"
        st.success("ANTHROPIC_API_KEY detected — live LLM generation available.")
    else:
        default_mode = "mock"
        st.info(
            "No ANTHROPIC_API_KEY found — running in **mock mode** "
            "(extractive answers, no LLM call, zero cost)."
        )

    mode = st.radio(
        "Generation mode",
        options=["mock", "live"] if LIVE_MODE_AVAILABLE else ["mock"],
        index=0 if default_mode == "mock" else 0,
        help="Live mode calls the Anthropic API and requires ANTHROPIC_API_KEY in .env.",
    )

    st.subheader("Knowledge base")
    policy_files = sorted(p.stem.replace("_", " ").title() for p in POLICIES_DIR.glob("*.md"))
    for name in policy_files:
        st.markdown(f"- {name}")

pipeline = load_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("citations"):
            with st.expander("Sources"):
                for c in turn["citations"]:
                    st.markdown(f"- {c}")

question = st.chat_input("Ask about leave, expenses, IT equipment, remote work...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            result = pipeline.ask(question, mode=mode)
        st.markdown(result.answer)
        if result.citations:
            with st.expander("Sources"):
                for c in result.citations:
                    st.markdown(f"- {c}")
        else:
            st.caption("No sources cited — this was a not-found fallback response.")

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "citations": result.citations}
    )
