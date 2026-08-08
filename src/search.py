"""Search and retrieval logic for the CP chatbot with BM25 + FAISS Hybrid Search."""

from __future__ import annotations

import re
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH
from src.indexer import load_index_from_disk
from src.utils import extract_rating


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


def match_by_id(problems: list[dict], query: str) -> dict | None:
    """Exact match by problem ID (e.g. 1000A)."""
    match = re.search(r"(\d+[A-Z]\d*)", query.strip(), re.IGNORECASE)
    if not match:
        return None
    pid = match.group(1).upper()
    return next((p for p in problems if p.get("id", "").upper() == pid), None)


def match_by_title(problems: list[dict], query: str) -> dict | None:
    """Substring match on problem title."""
    q = query.lower().strip()
    if len(q) < 3:
        return None
    return next((p for p in problems if q in p.get("title", "").lower()), None)


def filter_problems(
    problems: list[dict],
    tags: list[str] | None = None,
    rating_min: int = 800,
    rating_max: int = 2400,
) -> list[dict]:
    """Filter problems by tags and difficulty rating range."""
    filtered = []
    for p in problems:
        rating = p.get("rating") or extract_rating(p.get("tags", []))
        if rating is not None and not (rating_min <= rating <= rating_max):
            continue
        if tags:
            problem_tags = {t.lower() for t in p.get("tags", [])}
            if not any(t.lower() in problem_tags for t in tags):
                continue
        filtered.append(p)
    return filtered


def semantic_search(
    query: str,
    index: faiss.IndexFlatL2,
    problems: list[dict],
    model: SentenceTransformer,
    k: int = 5,
    candidate_indices: list[int] | None = None,
) -> list[tuple[dict, float]]:
    """Perform semantic search using FAISS."""
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = np.asarray(query_vec, dtype=np.float32)

    if candidate_indices is not None:
        if not candidate_indices:
            return []
        sub_vectors = np.array(
            [index.reconstruct(i) for i in candidate_indices], dtype=np.float32
        )
        sub_index = faiss.IndexFlatL2(sub_vectors.shape[1])
        sub_index.add(sub_vectors)
        distances, local_idx = sub_index.search(query_vec, min(k, len(candidate_indices)))
        results = []
        for dist, li in zip(distances[0], local_idx[0]):
            if li >= 0:
                global_idx = candidate_indices[li]
                results.append((problems[global_idx], float(dist)))
        return results

    distances, indices = index.search(query_vec, min(k, len(problems)))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            results.append((problems[idx], float(dist)))
    return results


def find_similar_problems(
    problem: dict[str, Any],
    index: faiss.IndexFlatL2,
    problems: list[dict],
    vectors: np.ndarray,
    model: SentenceTransformer,
    k: int = 3,
    rating_tolerance: int = 200,
) -> list[dict]:
    """Find similar problems with comparable difficulty ratings."""
    try:
        idx = next(i for i, p in enumerate(problems) if p["id"] == problem["id"])
    except StopIteration:
        return []

    source_rating = problem.get("rating") or extract_rating(problem.get("tags", []))
    query_vec = vectors[idx : idx + 1].astype(np.float32)

    search_k = min(len(problems), k + 15)
    distances, indices = index.search(query_vec, search_k)

    similar: list[dict] = []
    for dist, i in zip(distances[0], indices[0]):
        if i < 0 or i == idx:
            continue
        candidate = problems[i]
        cand_rating = candidate.get("rating") or extract_rating(candidate.get("tags", []))

        if source_rating is not None and cand_rating is not None:
            if abs(source_rating - cand_rating) > rating_tolerance:
                continue

        similar.append(candidate)
        if len(similar) >= k:
            break

    return similar


class SearchEngine:
    """Loads persisted indices and exposes hybrid search methods."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatL2 | None = None
        self.problems: list[dict] = []
        self.vectors: np.ndarray | None = None
        self.model: SentenceTransformer | None = None
        self.bm25: BM25Okapi | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.index, self.problems, self.vectors, model_name = load_index_from_disk(
            FAISS_INDEX_PATH, METADATA_PATH
        )
        self.model = SentenceTransformer(model_name)

        # Build BM25 index on tokenized statements and titles
        corpus = [
            tokenize(f"{p.get('title', '')} {p.get('statement', '')}")
            for p in self.problems
        ]
        self.bm25 = BM25Okapi(corpus)
        self._loaded = True

    def hybrid_search(
        self,
        query: str,
        candidate_indices: list[int],
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Combine BM25 and FAISS results using Reciprocal Rank Fusion (RRF)."""
        if not candidate_indices or self.bm25 is None or self.index is None or self.model is None:
            return []

        # 1. FAISS Dense Retrieval
        dense_results = semantic_search(
            query, self.index, self.problems, self.model, k=len(candidate_indices), candidate_indices=candidate_indices
        )

        # 2. BM25 Sparse Retrieval
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Filter BM25 scores for candidates only
        candidate_bm25 = [(idx, bm25_scores[idx]) for idx in candidate_indices]
        candidate_bm25.sort(key=lambda x: x[1], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {idx: 0.0 for idx in candidate_indices}

        for rank, (problem, _) in enumerate(dense_results):
            p_idx = self.problems.index(problem)
            rrf_scores[p_idx] += 1.0 / (rrf_k + rank + 1)

        for rank, (p_idx, _) in enumerate(candidate_bm25):
            rrf_scores[p_idx] += 1.0 / (rrf_k + rank + 1)

        # Sort candidate indices by fused score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        return [self.problems[i] for i in sorted_indices[:k]]

    def search(
        self,
        query: str,
        mode: str = "semantic",
        tags: list[str] | None = None,
        rating_min: int = 800,
        rating_max: int = 2400,
        k: int = 5,
    ) -> list[dict]:
        self.load()
        assert self.index is not None and self.model is not None

        pool = filter_problems(self.problems, tags, rating_min, rating_max)

        if mode == "exact_id":
            result = match_by_id(pool, query) or match_by_id(self.problems, query)
            return [result] if result else []

        # Exact ID and Title match priority
        id_match = match_by_id(pool, query)
        if id_match:
            return [id_match]

        title_match = match_by_title(pool, query)
        if title_match:
            return [title_match]

        candidate_indices = [self.problems.index(p) for p in pool if p in self.problems]
        return self.hybrid_search(query, candidate_indices, k=k)

    def get_similar(self, problem: dict, k: int = 3) -> list[dict]:
        self.load()
        assert self.index is not None and self.vectors is not None and self.model is not None
        return find_similar_problems(
            problem, self.index, self.problems, self.vectors, self.model, k=k
        )

    @property
    def all_tags(self) -> list[str]:
        self.load()
        tag_set: set[str] = set()
        for p in self.problems:
            for t in p.get("tags", []):
                if not t.startswith("*"):
                    tag_set.add(t)
        return sorted(tag_set)