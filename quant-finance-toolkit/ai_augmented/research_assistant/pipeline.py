"""RAG-style research assistant over the synthetic financial news corpus.

Same architectural pattern as the genai-power-platform-agent HR/IT assistant
(embed -> retrieve -> generate, with a similarity-threshold not-found
fallback and a mock/live generation split) reimplemented here rather than
imported, so this repo doesn't take a hard runtime dependency on a sibling
project. With a corpus this small (a dozen short notes), an in-memory
cosine-similarity search does the same job as a vector database — Chroma's
advantage shows up at a scale this demo doesn't need.

This is explicitly a research-idea generator, not a source of truth: see
ai_augmented/README.md for its documented failure modes (hallucination risk
in live mode, no verification against actual filings, small/synthetic corpus
coverage) and why every answer should be treated as a starting point for
human research, not a citable fact.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ai_augmented.research_assistant.corpus import CORPUS, NewsItem

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Cosine similarity threshold below which we refuse to answer rather than force
# a response from a weakly-related note. Calibrated against real queries — see
# ai_augmented/README.md for the calibration numbers.
MIN_SIMILARITY = 0.30

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
LIVE_MODE_AVAILABLE = bool(ANTHROPIC_API_KEY)

NOT_FOUND_MESSAGE = (
    "Nothing in the research note corpus is closely related to this question. "
    "Rather than guess, treat this as a gap to research through primary sources "
    "(filings, transcripts, real news) instead of this tool."
)

SYSTEM_PROMPT = """You are a buy-side research assistant surfacing CANDIDATE signal ideas and \
sentiment summaries from a small set of internal research notes. You are explicitly NOT a \
source of truth. Follow these rules:
1. Use ONLY the note excerpts provided — do not add outside knowledge or invented figures.
2. Never state a note's content as verified fact. Frame everything as "the notes suggest" or \
"one analyst view is" — these are opinions and observations, not confirmed data.
3. If the notes don't clearly address the question, say so plainly instead of stretching a \
weakly-related note to fit.
4. Cite which note(s) each point comes from.
5. End every answer with a one-line reminder that this is a starting point for research, not \
investment advice or a verified fact."""


@dataclass
class RetrievedNote:
    item: NewsItem
    similarity: float


@dataclass
class ResearchAnswer:
    question: str
    answer: str
    mode_used: str
    grounded: bool
    citations: list[str] = field(default_factory=list)


class ResearchAssistant:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._corpus = CORPUS
        texts = [f"{item.headline}. {item.text}" for item in self._corpus]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self._embeddings = np.asarray(embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedNote]:
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ query_emb
        top_idx = np.argsort(-sims)[:k]
        return [RetrievedNote(item=self._corpus[i], similarity=float(sims[i])) for i in top_idx]

    def _generate_mock(self, notes: list[RetrievedNote]) -> str:
        lines = ["Candidate signal ideas from the research notes (mock mode — extractive, no LLM used):\n"]
        for n in notes[:3]:
            lines.append(f"**[{n.item.id}] {n.item.headline}** ({n.item.ticker}, {n.item.date}, similarity={n.similarity:.2f})")
            lines.append(n.item.text)
            lines.append("")
        lines.append("Reminder: these are notes, not verified facts — treat as a starting point for research, not investment advice.")
        return "\n".join(lines)

    def _generate_live(self, question: str, notes: list[RetrievedNote]) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        context = "\n\n---\n\n".join(f"[{n.item.id}] {n.item.headline} ({n.item.ticker}, {n.item.date})\n{n.item.text}" for n in notes)
        message = f"Research notes:\n\n{context}\n\n---\n\nQuestion: {question}"
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=500, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def ask(self, question: str, mode: str | None = None, k: int = 3) -> ResearchAnswer:
        resolved_mode = mode or ("live" if LIVE_MODE_AVAILABLE else "mock")
        notes = self.retrieve(question, k=k)
        best_sim = notes[0].similarity if notes else -1.0

        if not notes or best_sim < MIN_SIMILARITY:
            return ResearchAnswer(question=question, answer=NOT_FOUND_MESSAGE, mode_used=resolved_mode, grounded=False)

        answer = self._generate_live(question, notes) if resolved_mode == "live" else self._generate_mock(notes)
        citations = [f"[{n.item.id}] {n.item.headline}" for n in notes]
        return ResearchAnswer(question=question, answer=answer, mode_used=resolved_mode, grounded=True, citations=citations)
