"""Chunk the synthetic policy corpus, embed it locally, and persist to Chroma.

Run directly to (re)build the vector store:
    python -m rag.ingest
"""
import re
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    POLICIES_DIR,
)

_SECTION_HEADER_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _doc_title(text: str) -> str:
    match = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled Policy"


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a policy markdown file into (section_title, section_text) pairs.

    Content before the first '## ' header (the title + metadata line) is
    kept as its own "Overview" section so it's still retrievable.
    """
    headers = list(_SECTION_HEADER_RE.finditer(text))
    if not headers:
        return [("Overview", text.strip())]

    sections = []
    preamble = text[: headers[0].start()].strip()
    if preamble:
        sections.append(("Overview", preamble))

    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section_title = h.group(1).strip()
        section_body = text[start:end].strip()
        sections.append((section_title, section_body))
    return sections


def load_and_chunk_documents() -> list[Document]:
    """Load every policy markdown file, split by section, then chunk each section.

    Each chunk carries metadata: source_file, doc_title, section — this is what
    lets the pipeline cite "Leave Policy > Carry-forward and forfeiture" instead
    of just a filename.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    md_files = sorted(POLICIES_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No policy documents found in {POLICIES_DIR}")

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        doc_title = _doc_title(text)
        for section_title, section_body in _split_into_sections(text):
            if not section_body:
                continue
            for piece in splitter.split_text(section_body):
                all_chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source_file": path.name,
                            "doc_title": doc_title,
                            "section": section_title,
                        },
                    )
                )
    return all_chunks


def build_vector_store(rebuild: bool = True) -> Chroma:
    """Embed all chunks with a local sentence-transformers model and persist to Chroma."""
    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    chunks = load_and_chunk_documents()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def get_vector_store() -> Chroma:
    """Load the persisted vector store, building it first if it doesn't exist yet."""
    if not CHROMA_DIR.exists():
        return build_vector_store(rebuild=False)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    print(f"Loaded {len(chunks)} chunks from {len(list(POLICIES_DIR.glob('*.md')))} policy documents.")
    store = build_vector_store(rebuild=True)
    print(f"Built and persisted Chroma vector store at {CHROMA_DIR}")
