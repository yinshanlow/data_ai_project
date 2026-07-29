"""Convenience entry point: python scripts/build_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.ingest import build_vector_store

if __name__ == "__main__":
    build_vector_store(rebuild=True)
    print("Vector store rebuilt.")
