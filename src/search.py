"""Hybrid search for the CP chatbot: FAISS semantic search + BM25."""

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
    return re.findall(r"\w+", text.lower())

def match_by_id(problems: list[dict], query: str) -> dict | None:
    match = re.search(r"(\d+[A-Z]\d*)", query.strip(), re.I)
    if not match:
        return None
    pid = match.group(1).upper()
    return next((p for p in problems if str(p.get("id", "")).upper() == pid), None)

def is_problem_id_query(query: str) -> bool:
    return bool(re.fullmatch(r"\d+[A-Z]\d*", query.strip(), re.I))

def match_by_title(problems: list[dict], query: str) -> dict | None:
    q = query.lower().strip()
    if len(q) < 3:
        return None
    return next((p for p in problems if q in str(p.get("title", "")).lower()), None)

def _get_rating(problem: dict) -> int | None:
    rating = problem.get("rating")
    if rating is not None:
        try:
            return int(rating)
        except (TypeError, ValueError):
            pass

    tags = problem.get("tags", [])
    if isinstance(tags, list):
        return extract_rating([t for t in tags if isinstance(t, str)])
    return None

def _get_tags(problem: dict) -> set[str]:
    tags = problem.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {t.lower() for t in tags if isinstance(t, str) and not t.startswith("*")}

def filter_problems(
    problems: list[dict],
    tags: list[str] | None = None,
    rating_min: int = 800,
    rating_max: int = 3500,
) -> list[dict]:
    result = []
    wanted = {t.lower() for t in tags or []}
    for problem in problems:
        rating = _get_rating(problem)
        if rating is None or not rating_min <= rating <= rating_max:
            continue
        if wanted and not wanted.issubset(_get_tags(problem)):
            continue
        result.append(problem)
    return result

def _default_chunk_metadata(problems: list[dict], ntotal: int) -> list[dict]:
    return [
        {
            "problem_index": i,
            "problem_id": problems[i].get("id"),
            "chunk_id": 0,
            "total_chunks": 1,
        }
        for i in range(min(ntotal, len(problems)))
    ]

def semantic_search(
    query: str,
    index: faiss.IndexFlatIP,
    problems: list[dict],
    model: SentenceTransformer,
    k: int = 5,
    candidate_indices: list[int] | None = None,
    chunk_metadata: list[dict] | None = None,
) -> list[tuple[dict, float]]:
    if not problems or index.ntotal == 0:
        return []

    query_vec = np.asarray(model.encode([query], convert_to_numpy=True), dtype=np.float32)
    faiss.normalize_L2(query_vec)

    candidates = set(range(len(problems))) if candidate_indices is None else set(candidate_indices)
    if not candidates:
        return []

    metadata = chunk_metadata or _default_chunk_metadata(problems, index.ntotal)
    search_k = min(index.ntotal, max(k * 20, 50))
    distances, indices = index.search(query_vec, search_k)

    best: dict[int, float] = {}
    for score, vector_idx in zip(distances[0], indices[0]):
        if vector_idx < 0 or vector_idx >= len(metadata):
            continue
        problem_idx = metadata[vector_idx].get("problem_index")
        if not isinstance(problem_idx, int) or problem_idx not in candidates:
            continue
        if problem_idx >= len(problems):
            continue
        score = float(score)
        best[problem_idx] = max(score, best.get(problem_idx, -np.inf))

    ranked = sorted(best.items(), key=lambda x: (-x[1], x[0]))[:k]
    return [(problems[i], score) for i, score in ranked]


def find_similar_problems(
    problem: dict[str, Any],
    index: faiss.IndexFlatIP,
    problems: list[dict],
    vectors: np.ndarray,
    model: SentenceTransformer,
    k: int = 3,
    rating_tolerance: int = 200,
    problem_id_to_index: dict[str, int] | None = None,
    chunk_metadata: list[dict] | None = None,
) -> list[dict]:
    problem_id = problem.get("id")
    if not isinstance(problem_id, str):
        return []

    if problem_id_to_index is None:
        problem_idx = next((i for i, p in enumerate(problems) if p.get("id") == problem_id), None)
    else:
        problem_idx = problem_id_to_index.get(problem_id)
    if problem_idx is None:
        return []

    source_rating = _get_rating(problem)
    metadata = chunk_metadata or _default_chunk_metadata(problems, index.ntotal)
    source_vectors = [
        i for i, m in enumerate(metadata)
        if m.get("problem_index") == problem_idx
    ]

    if source_vectors:
        query_vec = vectors[source_vectors].astype(np.float32).mean(axis=0, keepdims=True)
        faiss.normalize_L2(query_vec)
    elif problem_idx < len(vectors):
        query_vec = vectors[problem_idx:problem_idx + 1].astype(np.float32)
        faiss.normalize_L2(query_vec)
    else:
        return []

    search_k = min(index.ntotal, max(k * 50, 100))
    distances, indices = index.search(query_vec, search_k)

    best: dict[int, float] = {}
    for score, vector_idx in zip(distances[0], indices[0]):
        if vector_idx < 0 or vector_idx >= len(metadata):
            continue
        candidate_idx = metadata[vector_idx].get("problem_index")
        if not isinstance(candidate_idx, int) or candidate_idx == problem_idx:
            continue
        if candidate_idx >= len(problems):
            continue
        best[candidate_idx] = max(float(score), best.get(candidate_idx, -np.inf))

    result = []
    for candidate_idx, _ in sorted(best.items(), key=lambda x: -x[1]):
        candidate = problems[candidate_idx]
        candidate_rating = _get_rating(candidate)
        if (
            source_rating is not None
            and candidate_rating is not None
            and abs(source_rating - candidate_rating) > rating_tolerance
        ):
            continue
        result.append(candidate)
        if len(result) == k:
            break
    return result


class SearchEngine:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.problems: list[dict] = []
        self.problem_id_to_index: dict[str, int] = {}
        self.vectors: np.ndarray | None = None
        self.chunk_metadata: list[dict] = []
        self.model: SentenceTransformer | None = None
        self.bm25: BM25Okapi | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        (
            self.index,
            self.problems,
            self.vectors,
            self.chunk_metadata,
            model_name,
        ) = load_index_from_disk(FAISS_INDEX_PATH, METADATA_PATH)

        self.problem_id_to_index = {
            str(p["id"]): i for i, p in enumerate(self.problems) if "id" in p
        }
        self.model = SentenceTransformer(model_name)

        corpus = []
        for p in self.problems:
            title = str(p.get("title", ""))
            tags = p.get("tags", [])
            tags_text = " ".join(str(t) for t in tags) if isinstance(tags, list) else ""
            text = (
                f"{title} {title} {title} "
                f"{tags_text} {tags_text} "
                f"{p.get('statement', '')} {p.get('input', '')} {p.get('output', '')}"
            )
            corpus.append(tokenize(text))
        self.bm25 = BM25Okapi(corpus)
        self._loaded = True

    def hybrid_search(
        self,
        query: str,
        candidate_indices: list[int],
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        if not candidate_indices or self.bm25 is None or self.index is None or self.model is None:
            return []

        retrieval_k = min(len(candidate_indices), max(k * 20, 50))
        dense = semantic_search(
            query, self.index, self.problems, self.model,
            retrieval_k, candidate_indices, self.chunk_metadata,
        )

        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse = heapq.nlargest(
            retrieval_k,
            ((i, bm25_scores[i]) for i in candidate_indices if 0 <= i < len(bm25_scores)),
            key=lambda x: x[1],
        )

        rrf: dict[int, float] = {}
        for rank, (problem, _) in enumerate(dense):
            idx = self.problem_id_to_index.get(str(problem.get("id")))
            if idx is not None:
                rrf[idx] = rrf.get(idx, 0.0) + 1 / (rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(sparse):
            rrf[idx] = rrf.get(idx, 0.0) + 1 / (rrf_k + rank + 1)

        ranked = sorted(rrf, key=lambda i: (-rrf[i], i))
        return [self.problems[i] for i in ranked[:k]]

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
        pool = filter_problems(self.problems, tags, rating_min, rating_max)

        if not query.strip():
            return pool[:k]
        if is_problem_id_query(query):
            result = match_by_id(pool, query)
            return [result] if result else []

        result = match_by_id(pool, query)
        if result:
            return [result]

        result = match_by_title(pool, query)
        if result:
            return [result]

        candidate_indices = [
            self.problem_id_to_index[p["id"]]
            for p in pool
            if p.get("id") in self.problem_id_to_index
        ]
        return self.hybrid_search(query, candidate_indices, k)

    def find_problem_by_id(self, query: str) -> dict | None:
        self.load()
        return match_by_id(self.problems, query)

    def filter_count(
        self,
        tags: list[str] | None = None,
        rating_min: int = 800,
        rating_max: int = 3500,
    ) -> int:
        self.load()
        return len(filter_problems(self.problems, tags, rating_min, rating_max))

    def get_similar(self, problem: dict, k: int = 3) -> list[dict]:
        self.load()
        if self.index is None or self.vectors is None or self.model is None:
            return []
        return find_similar_problems(
            problem, self.index, self.problems, self.vectors, self.model,
            k=k, problem_id_to_index=self.problem_id_to_index,
            chunk_metadata=self.chunk_metadata,
        )

    @property
    def all_tags(self) -> list[str]:
        self.load()
        return sorted({
            tag
            for p in self.problems
            for tag in p.get("tags", [])
            if isinstance(tag, str) and not tag.startswith("*")
        })
