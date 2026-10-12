"""Build and persist a FAISS index for Codeforces problems."""

from __future__ import annotations

import argparse
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
    """Load all valid problem JSON files."""
    directory = problems_dir or PROBLEMS_DIR
    if not os.path.isdir(directory):
        return []

    problems = []

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
) -> tuple[faiss.IndexFlatIP, list[dict], np.ndarray, list[dict]]:
    """Load problems, create embeddings, and build a cosine FAISS index."""

    problems = load_problems(problems_dir)

    if not problems:
        raise FileNotFoundError(
            f"No problem JSON files found in {problems_dir or PROBLEMS_DIR}. Run `python -m src.scraper` first."
        )

    print(f"Loaded {len(problems)} problems.")

    model = SentenceTransformer(model_name)
    max_seq_len = model.max_seq_length or 256

    print(f"Embedding model: {model_name}")
    print(f"Maximum sequence length: {max_seq_len}")

    texts = []
    chunk_metadata = []

    for problem_index, problem in enumerate(problems):
        chunks = build_embedding_text(problem,model.tokenizer,max_seq_len)

        for chunk in chunks:
            texts.append(chunk["text"])
            chunk_metadata.append({
                "problem_index": problem_index,
                "problem_id": problem.get("id"),
                "chunk_id": chunk["chunk_id"],
                "total_chunks": chunk["total_chunks"],
            })

    print(f"Created {len(texts)} embedding chunks from {len(problems)} problems.")
    print("Generating embeddings...")

    vectors = model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got {vectors.shape}.")

    if len(vectors) != len(chunk_metadata):
        raise ValueError("Number of embeddings does not match chunk metadata.")

    # Normalize so inner product = cosine similarity.
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print(f"Built cosine-similarity FAISS index with {index.ntotal} vectors of dimension {dimension}.")
    return index, problems, vectors, chunk_metadata


def save_index(
    index: faiss.IndexFlatIP,
    problems: list[dict],
    vectors: np.ndarray,
    chunk_metadata: list[dict],
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """Save FAISS index and metadata."""

    for path in (index_path, metadata_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    faiss.write_index(index, index_path)

    metadata = {
        "problems": problems,
        "vectors": vectors,
        "chunk_metadata": chunk_metadata,
        "model_name": model_name,
        "problem_count": len(problems),
        "vector_count": len(chunk_metadata),
        "similarity": "cosine",
        "faiss_metric": "inner_product",
        "normalized": True,
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved FAISS index ({len(problems)} problems, {len(chunk_metadata)} vectors) -> {index_path}")
    print(f"Saved metadata -> {metadata_path}")


def load_index_from_disk(index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_PATH,
) -> tuple[
    faiss.IndexFlatIP,
    list[dict],
    np.ndarray,
    list[dict],
    str,
]:
    """Load and validate a stored FAISS cosine-similarity index."""

    if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Index not found. Run `python -m src.indexer` to build {index_path} and {metadata_path}.")

    index = faiss.read_index(index_path)

    if not isinstance(index, faiss.IndexFlatIP):
        raise TypeError("Stored index is not an IndexFlatIP cosine-similarity index. Rebuild the index.")

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if metadata.get("similarity") not in (None, "cosine"):
        raise ValueError(f"Expected cosine similarity metadata, got {metadata.get('similarity')!r}. Rebuild the index.")

    if metadata.get("normalized") is False:
        raise ValueError(f"Stored vectors are not normalized. Rebuild the index.")

    problems = metadata["problems"]
    vectors = metadata["vectors"]
    chunk_metadata = metadata["chunk_metadata"]
    model_name = metadata.get("model_name", EMBEDDING_MODEL)

    if len(chunk_metadata) != index.ntotal:
        raise ValueError( f"FAISS index contains {index.ntotal} vectors, but metadata contains {len(chunk_metadata)} chunks. Rebuild the index.")

    if len(vectors) != index.ntotal:
        raise ValueError(f"Stored vector count does not match FAISS index count. Rebuild the index.")

    return index, problems, vectors, chunk_metadata, model_name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the CP Chatbot FAISS index.")
    parser.add_argument("--problems-dir",default=PROBLEMS_DIR,help="Directory containing problem JSON files.")
    parser.add_argument("--model-name",default=EMBEDDING_MODEL,help="Sentence-transformer model used for embeddings.")
    parser.add_argument("--no-progress",action="store_true",help="Disable embedding progress output.",)
    args = parser.parse_args(argv)

    print(f"Building cosine-similarity index from {args.problems_dir} ...")

    index, problems, vectors, chunk_metadata = build_index(
        problems_dir=args.problems_dir,
        model_name=args.model_name,
        show_progress=not args.no_progress,
    )

    save_index(index,problems,vectors,chunk_metadata,model_name=args.model_name)

    print(f"Done. Indexed {len(problems)} problems into {INDEX_DIR}.")
    print(f"FAISS vectors: {index.ntotal}")

if __name__ == "__main__":
    main()