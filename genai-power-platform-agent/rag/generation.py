"""Generation layer.

Two modes:
- mock:  purely extractive, zero API calls, zero cost. Stitches together the
         retrieved chunk text itself rather than paraphrasing it. This is the
         default so the whole pipeline is testable with no API key at all.
- live:  calls the Anthropic API with a system prompt that restricts the model
         to the retrieved context and instructs it to say so when the context
         doesn't cover the question, instead of guessing.
"""
from anthropic import Anthropic

from rag.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LIVE_MODE_AVAILABLE
from rag.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are an internal HR/IT policy assistant for a company called Meridian Holdings.

Answer the user's question using ONLY the policy excerpts provided below. Follow these rules strictly:
1. Do not use any outside knowledge or make assumptions beyond what the excerpts say.
2. If the excerpts do not contain enough information to answer confidently, say so plainly \
instead of guessing — do not invent a policy detail that isn't in the excerpts.
3. Cite which document and section each part of your answer comes from, e.g. "(Leave Policy > Carry-forward and forfeiture)".
4. Keep answers concise and practical — this is for an employee who wants a direct answer, not an essay.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.citation}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def generate_mock_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Extractive fallback: no LLM call, just the retrieved text with citations.

    This is intentionally NOT a paraphrase — it quotes the source chunks directly
    so there is no risk of the "generation" step introducing an unsupported claim.
    """
    if not chunks:
        return ""

    lines = ["Here's what the policy documents say (mock mode — extractive, no LLM used):\n"]
    for c in chunks[:3]:
        snippet = c.text.strip()
        lines.append(f"**{c.citation}**\n{snippet}\n")
    return "\n".join(lines)


def generate_live_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not LIVE_MODE_AVAILABLE:
        raise RuntimeError(
            "Live mode requested but ANTHROPIC_API_KEY is not set. "
            "Set it in .env or fall back to mock mode."
        )

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    context = _format_context(chunks)
    user_message = (
        f"Policy excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"
    )

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def generate_answer(question: str, chunks: list[RetrievedChunk], mode: str) -> str:
    if mode == "live":
        return generate_live_answer(question, chunks)
    return generate_mock_answer(question, chunks)
