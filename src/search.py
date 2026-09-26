"""Search and retrieval logic for the CP chatbot with BM25 + FAISS Hybrid Search."""

from __future__ import annotations

import heapq
import re
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import FAISS_INDEX_PATH, METADATA_PATH
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


def is_problem_id_query(query: str) -> bool:
    """Return whether the complete query is a Codeforces problem ID."""
    return bool(re.fullmatch(r"\d+[A-Z]\d*", query.strip(), re.IGNORECASE))


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
    rating_max: int = 3500,
) -> list[dict]:
    """Filter problems by all selected tags and an inclusive rating range."""
    filtered = []
    for p in problems:
        rating = p.get("rating") or extract_rating(p.get("tags", []))
        if rating is None:
            continue
        if not (rating_min <= rating <= rating_max):
            continue
        if tags:
            problem_tags = {t.lower() for t in p.get("tags", [])}
            if not all(t.lower() in problem_tags for t in tags):
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
        target_count = min(k, len(candidate_indices))
        candidate_ids = np.asarray(candidate_indices, dtype=np.int64)
        selector = faiss.IDSelectorBatch(
            len(candidate_ids), faiss.swig_ptr(candidate_ids)
        )
        search_params = faiss.SearchParameters()
        search_params.sel = selector
        distances, indices = index.search(query_vec, target_count, params=search_params)
        return [
            (problems[idx], float(dist))
            for dist, idx in zip(distances[0], indices[0])
            if idx >= 0
        ]

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
    problem_id_to_index: dict[str, int] | None = None,
) -> list[dict]:
    """Find similar problems with comparable difficulty ratings."""
    if problem_id_to_index is None:
        try:
            idx = next(i for i, p in enumerate(problems) if p["id"] == problem["id"])
        except StopIteration:
            return []
    else:
        problem_id = problem.get("id")
        if not isinstance(problem_id, str):
            return []
        idx = problem_id_to_index.get(problem_id)
        if idx is None:
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
        self.problem_id_to_index: dict[str, int] = {}
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
        self.problem_id_to_index = {
            problem["id"]: index for index, problem in enumerate(self.problems)
        }
        self.model = SentenceTransformer(model_name)

        # Repeat short topic fields so BM25 does not dilute them in long statements.
        corpus = [
            tokenize(
                f"{p.get('title', '')} {p.get('title', '')} {p.get('title', '')} "
                f"{' '.join(p.get('tags', []))} {' '.join(p.get('tags', []))} "
                f"{p.get('statement', '')} {p.get('input', '')} {p.get('output', '')}"
            )
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

        retrieval_k = min(len(candidate_indices), max(k * 20, 50))

        # 1. FAISS Dense Retrieval. Fuse only the useful head of each list;
        dense_results = semantic_search(
            query,
            self.index,
            self.problems,
            self.model,
            k=retrieval_k,
            candidate_indices=candidate_indices,
        )

        # 2. BM25 Sparse Retrieval
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Filter BM25 scores for candidates only
        candidate_bm25 = heapq.nlargest(
            retrieval_k,
            ((idx, bm25_scores[idx]) for idx in candidate_indices),
            key=lambda item: item[1],
        )

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}

        for rank, (problem, _) in enumerate(dense_results):
            p_idx = self.problem_id_to_index[problem["id"]]
            rrf_scores[p_idx] = rrf_scores.get(p_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (p_idx, _) in enumerate(candidate_bm25):
            rrf_scores[p_idx] = rrf_scores.get(p_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort candidate indices by fused score
        sorted_indices = sorted(
            rrf_scores.keys(), key=lambda idx: (-rrf_scores[idx], idx)
        )
        return [self.problems[i] for i in sorted_indices[:k]]

    def search(
        self,
        query: str,
        mode: str = "semantic",
        tags: list[str] | None = None,
        rating_min: int = 800,
        rating_max: int = 3500,
        k: int = 5,
    ) -> list[dict]:
        self.load()
        assert self.index is not None and self.model is not None

        pool = filter_problems(self.problems, tags, rating_min, rating_max)

        # An empty query is an intentional browse request, not a semantic query.
        if not query.strip():
            return pool[:k]

        if is_problem_id_query(query):
            result = match_by_id(pool, query)
            return [result] if result else []

        # Exact ID and Title match priority
        id_match = match_by_id(pool, query)
        if id_match:
            return [id_match]

        title_match = match_by_title(pool, query)
        if title_match:
            return [title_match]

        candidate_indices = [self.problem_id_to_index[p["id"]] for p in pool]
        return self.hybrid_search(query, candidate_indices, k=k)

    def find_problem_by_id(self, query: str) -> dict | None:
        """Find an ID in the complete index, independent of active filters."""
        self.load()
        return match_by_id(self.problems, query)

    def filter_count(
        self,
        tags: list[str] | None = None,
        rating_min: int = 800,
        rating_max: int = 3500,
    ) -> int:
        """Return the number of problems matching the active filters."""
        self.load()
        return len(filter_problems(self.problems, tags, rating_min, rating_max))

    def get_similar(self, problem: dict, k: int = 3) -> list[dict]:
        self.load()
        assert self.index is not None and self.vectors is not None and self.model is not None
        return find_similar_problems(
            problem,
            self.index,
            self.problems,
            self.vectors,
            self.model,
            k=k,
            problem_id_to_index=self.problem_id_to_index,
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