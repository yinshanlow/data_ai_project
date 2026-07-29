"""End-to-end RAG pipeline: retrieve -> (guard) -> generate -> cite."""
import re
from dataclasses import dataclass, field

from rag.config import LIVE_MODE_AVAILABLE, MAX_RELEVANT_DISTANCE, NOT_FOUND_MESSAGE, TOP_K
from rag.generation import generate_answer
from rag.ingest import get_vector_store
from rag.retrieval import RetrievedChunk, retrieve

# Boilerplate that appears in nearly every policy document ("Meridian Holdings",
# "Document owner", etc.) and therefore carries no real topical signal — it must
# be excluded or it defeats the whole point of the lexical grounding check below.
_BOILERPLATE_WORDS = {
    "meridian", "holdings", "policy", "policies", "employee", "employees",
    "document", "owner", "effective", "applies", "team", "company", "confirmed",
    "what", "how", "when", "where", "does", "for", "the", "and", "with", "this",
    "that", "have", "from", "into", "under", "about", "there", "their", "your",
}
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")


def _normalize(word: str) -> str:
    """Naive singularization (laptops -> laptop) so plural/singular mismatches
    between question and chunk text don't defeat the overlap check."""
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


_NORMALIZED_BOILERPLATE = {_normalize(w) for w in _BOILERPLATE_WORDS}


def _significant_words(text: str) -> set[str]:
    words = {_normalize(w.lower()) for w in _WORD_RE.findall(text)}
    return words - _NORMALIZED_BOILERPLATE


def _is_lexically_grounded(question: str, top_chunk: RetrievedChunk) -> bool:
    """Second gate on top of the distance threshold.

    Embedding similarity alone lets some off-topic questions slip through when
    a chunk's boilerplate ("Meridian Holdings", generic policy phrasing) happens
    to sit close in vector space to a formally-worded question. This check
    requires at least one real, non-boilerplate content word from the question
    to actually appear in the top retrieved chunk (including its title/section) —
    catching cases like "stock buyback policy for shareholders", which is close
    enough in distance to pass but shares no real topic with any policy chunk.
    """
    question_words = _significant_words(question)
    if not question_words:
        return True  # nothing meaningful to check against; don't block on it
    chunk_words = _significant_words(
        f"{top_chunk.doc_title} {top_chunk.section} {top_chunk.text}"
    )
    return bool(question_words & chunk_words)


@dataclass
class AnswerResult:
    question: str
    answer: str
    mode_used: str
    citations: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = True  # False when we hit the not-found fallback
    best_distance: float | None = None


class RAGPipeline:
    def __init__(self):
        self.vector_store = get_vector_store()

    def ask(self, question: str, mode: str | None = None, k: int = TOP_K) -> AnswerResult:
        """Answer a question.

        mode: "mock", "live", or None (auto — uses live if ANTHROPIC_API_KEY is set,
        otherwise falls back to mock automatically).
        """
        resolved_mode = mode or ("live" if LIVE_MODE_AVAILABLE else "mock")

        chunks = retrieve(self.vector_store, question, k=k)
        best_distance = min((c.distance for c in chunks), default=None)

        distance_ok = chunks and best_distance is not None and best_distance <= MAX_RELEVANT_DISTANCE
        lexically_ok = distance_ok and _is_lexically_grounded(question, chunks[0])

        if not distance_ok or not lexically_ok:
            return AnswerResult(
                question=question,
                answer=NOT_FOUND_MESSAGE,
                mode_used=resolved_mode,
                citations=[],
                chunks=chunks,
                grounded=False,
                best_distance=best_distance,
            )

        answer = generate_answer(question, chunks, mode=resolved_mode)
        return AnswerResult(
            question=question,
            answer=answer,
            mode_used=resolved_mode,
            citations=[c.citation for c in chunks],
            chunks=chunks,
            grounded=True,
            best_distance=best_distance,
        )
