"""Retrieval layer: turn a question into a ranked list of scored chunks."""
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import TOP_K


@dataclass
class RetrievedChunk:
    document: Document
    distance: float  # Chroma cosine distance — lower is more similar

    @property
    def source_file(self) -> str:
        return self.document.metadata.get("source_file", "unknown")

    @property
    def doc_title(self) -> str:
        return self.document.metadata.get("doc_title", "Unknown Policy")

    @property
    def section(self) -> str:
        return self.document.metadata.get("section", "Overview")

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def citation(self) -> str:
        return f"{self.doc_title} > {self.section}"


def retrieve(vector_store: Chroma, question: str, k: int = TOP_K) -> list[RetrievedChunk]:
    results = vector_store.similarity_search_with_score(question, k=k)
    return [RetrievedChunk(document=doc, distance=score) for doc, score in results]
