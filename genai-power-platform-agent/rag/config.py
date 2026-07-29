"""Shared configuration for the RAG pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Chroma's telemetry client has a known incompatibility with recent posthog
# versions (harmless "capture() takes 1 positional argument" warnings on every
# call) — disable it outright rather than let it spam stdout.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT_DIR = Path(__file__).resolve().parent.parent
POLICIES_DIR = ROOT_DIR / "data" / "policies"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"
COLLECTION_NAME = "hr_it_policies"

# Local, free embedding model — no API key required.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

# Retrieval
TOP_K = 4
# Chroma reports cosine *distance* (lower = more similar) via similarity_search_with_score.
# Above this distance, the best match is considered too weak to answer from —
# this is the knob that drives the "not found in knowledge base" fallback.
# Calibrated empirically against real questions (see eval/eval_results.md):
# genuinely-covered questions scored 0.56-0.97 on this corpus/embedding model,
# genuinely out-of-scope questions scored 1.13+.
MAX_RELEVANT_DISTANCE = 1.05

# Generation
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
LIVE_MODE_AVAILABLE = bool(ANTHROPIC_API_KEY)

NOT_FOUND_MESSAGE = (
    "I couldn't find anything in the HR/IT policy knowledge base that answers this "
    "question confidently. Please check with the People Team or IT Service Desk directly "
    "rather than relying on a guess."
)
