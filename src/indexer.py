"""Build and persist FAISS index + metadata for offline use."""

from __future__ import annotations

import json
import os
import pickle
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    METADATA_PATH,
    PROBLEMS_DIR,
)
from src.utils import build_embedding_text


def load_problems(problems_dir: str | None = None) -> list[dict]:
    """Load all problem JSON files from disk."""
    directory = problems_dir or PROBLEMS_DIR
    problems: list[dict] = []

    if not os.path.isdir(directory):
        return problems

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path, encoding="utf-8") as f:
                problems.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {fname}: {exc}", file=sys.stderr)

    return problems


def build_index(
    problems_dir: str | None = None,
    model_name: str = EMBEDDING_MODEL,
    show_progress: bool = True,
) -> tuple[faiss.IndexFlatL2, list[dict], np.ndarray]:
    """Embed problems and build an in-memory FAISS index."""
    problems = load_problems(problems_dir)
    if not problems:
        raise FileNotFoundError(
            f"No problem JSON files found in {problems_dir or PROBLEMS_DIR}. "
            "Run `python -m src.scraper` first."
        )

    texts = [build_embedding_text(p) for p in problems]
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, problems, vectors


def save_index(
    index: faiss.IndexFlatL2,
    problems: list[dict],
    vectors: np.ndarray,
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
) -> None:
    """Persist FAISS index and metadata mapping to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    metadata = {
        "problems": problems,
        "vectors": vectors,
        "model_name": EMBEDDING_MODEL,
        "count": len(problems),
    }
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved FAISS index ({len(problems)} problems) -> {index_path}")
    print(f"Saved metadata -> {metadata_path}")


def load_index_from_disk(
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
) -> tuple[faiss.IndexFlatL2, list[dict], np.ndarray, str]:
    """Load pre-built FAISS index and metadata from disk."""
    if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Index not found. Run `python -m src.indexer` to build "
            f"{index_path} and {metadata_path}."
        )

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return (
        index,
        metadata["problems"],
        metadata["vectors"],
        metadata.get("model_name", EMBEDDING_MODEL),
    )


def main() -> None:
    """CLI entry point: build and save the FAISS index."""
    print(f"Building index from {PROBLEMS_DIR} ...")
    index, problems, vectors = build_index(show_progress=True)
    save_index(index, problems, vectors)
    print(f"Done. Indexed {len(problems)} problems into {INDEX_DIR}")


if __name__ == "__main__":
    main()
