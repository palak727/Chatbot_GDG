"""Build and persist FAISS index + metadata for offline use."""

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
            print(
                f"Warning: skipping {fname}: {exc}",
                file=sys.stderr,
            )

    return problems


def build_index(
    problems_dir: str | None = None,
    model_name: str = EMBEDDING_MODEL,
    show_progress: bool = True,
) -> tuple[faiss.IndexFlatIP, list[dict], np.ndarray]:
    """Embed problems and build an in-memory FAISS cosine-similarity index."""

    problems = load_problems(problems_dir)

    if not problems:
        raise FileNotFoundError(
            f"No problem JSON files found in "
            f"{problems_dir or PROBLEMS_DIR}. "
            "Run `python -m src.scraper` first."
        )

    print(f"Loaded {len(problems)} problems.")

    # Load the sentence-transformer model.
    model = SentenceTransformer(model_name)

    max_seq_len = (
        model.max_seq_length
        if model.max_seq_length is not None
        else 256
    )

    print(f"Embedding model: {model_name}")
    print(f"Maximum sequence length: {max_seq_len}")

    # Build the text used for embeddings.
    #
    # IMPORTANT:
    # build_embedding_text() should put high-value information such as
    # title and tags before the long problem statement so that important
    # topic information is less likely to be truncated.
    texts = [
        build_embedding_text(
            problem,
            model.tokenizer,
            max_seq_len,
        )
        for problem in problems
    ]

    print("Generating embeddings...")

    vectors = model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {vectors.shape}."
        )

    # Normalize every vector to unit length.
    #
    # After L2 normalization:
    #
    #       dot(x, y) == cosine_similarity(x, y)
    #
    # Therefore IndexFlatIP below performs cosine similarity search.
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    # Inner Product on normalized vectors is equivalent to
    # cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    print(
        f"Built cosine-similarity FAISS index "
        f"with {index.ntotal} vectors of dimension {dimension}."
    )

    return index, problems, vectors


def save_index(
    index: faiss.IndexFlatIP,
    problems: list[dict],
    vectors: np.ndarray,
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """Persist FAISS index and metadata mapping to disk."""

    index_directory = os.path.dirname(index_path)

    if index_directory:
        os.makedirs(index_directory, exist_ok=True)

    metadata_directory = os.path.dirname(metadata_path)

    if metadata_directory:
        os.makedirs(metadata_directory, exist_ok=True)

    # Save FAISS index.
    faiss.write_index(index, index_path)

    # Save metadata and normalized vectors.
    metadata = {
        "problems": problems,
        "vectors": vectors,
        "model_name": model_name,
        "count": len(problems),
        "similarity": "cosine",
        "faiss_metric": "inner_product",
        "normalized": True,
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(
        f"Saved FAISS index ({len(problems)} problems) "
        f"-> {index_path}"
    )
    print(f"Saved metadata -> {metadata_path}")


def load_index_from_disk(
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
) -> tuple[faiss.IndexFlatIP, list[dict], np.ndarray, str]:
    """Load a pre-built cosine-similarity FAISS index from disk."""

    if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            "Index not found. Run `python -m src.indexer` "
            f"to build {index_path} and {metadata_path}."
        )

    index = faiss.read_index(index_path)

    # Ensure the loaded index is an IndexFlatIP index.
    if isinstance(index, faiss.IndexFlatIP):
        flat_index = index
    else:
        raise TypeError(
            "The stored FAISS index is not an IndexFlatIP "
            "cosine-similarity index. Rebuild the index with "
            "`python -m src.indexer`."
        )

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Validate metadata when available.
    similarity = metadata.get("similarity")

    if similarity is not None and similarity != "cosine":
        raise ValueError(
            f"Expected cosine similarity metadata, got {similarity!r}. "
            "Rebuild the FAISS index."
        )

    if metadata.get("normalized") is False:
        raise ValueError(
            "Stored vectors are not normalized. "
            "Rebuild the FAISS index."
        )

    problems = metadata["problems"]
    vectors = metadata["vectors"]
    model_name = metadata.get(
        "model_name",
        EMBEDDING_MODEL,
    )

    return (
        flat_index,
        problems,
        vectors,
        model_name,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: build and save the FAISS index."""

    parser = argparse.ArgumentParser(
        description="Build the CP Chatbot FAISS index."
    )

    parser.add_argument(
        "--problems-dir",
        default=PROBLEMS_DIR,
        help="Directory containing problem JSON files.",
    )

    parser.add_argument(
        "--model-name",
        default=EMBEDDING_MODEL,
        help="Sentence-transformers model used for embeddings.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable embedding progress output.",
    )

    args = parser.parse_args(argv)

    print(
        f"Building cosine-similarity index "
        f"from {args.problems_dir} ..."
    )

    index, problems, vectors = build_index(
        problems_dir=args.problems_dir,
        model_name=args.model_name,
        show_progress=not args.no_progress,
    )

    save_index(
        index,
        problems,
        vectors,
        model_name=args.model_name,
    )

    print(
        f"Done. Indexed {len(problems)} problems "
        f"into {INDEX_DIR}"
    )


if __name__ == "__main__":
    main()