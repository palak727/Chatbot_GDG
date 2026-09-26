import unittest
from typing import cast

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.search import filter_problems, is_problem_id_query, match_by_id, semantic_search


PROBLEMS = [
    {"id": "1A", "title": "Binary Search", "rating": 1200, "tags": ["binary search"]},
    {"id": "4C", "title": "Graph Paths", "rating": 1600, "tags": ["graphs", "shortest paths"]},
    {"id": "1000A", "title": "No Rating", "rating": None, "tags": ["strings"]},
]


class SearchFilterTests(unittest.TestCase):
    def test_exact_id_detection_and_matching(self):
        self.assertTrue(is_problem_id_query("1000a"))
        match = match_by_id(PROBLEMS, "4C")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["id"], "4C")
        self.assertIsNone(match_by_id(PROBLEMS, "999Z"))

    def test_multiple_tags_use_and_semantics(self):
        self.assertEqual(
            [p["id"] for p in filter_problems(PROBLEMS, ["graphs", "shortest paths"])],
            ["4C"],
        )
        self.assertEqual(filter_problems(PROBLEMS, ["graphs", "binary search"]), [])

    def test_rating_bounds_are_inclusive(self):
        self.assertEqual(
            [p["id"] for p in filter_problems(PROBLEMS, rating_min=1200, rating_max=1200)],
            ["1A"],
        )
        self.assertEqual(
            [p["id"] for p in filter_problems(PROBLEMS, rating_min=1200, rating_max=1600)],
            ["1A", "4C"],
        )

    def test_missing_rating_is_excluded_from_rating_browse(self):
        self.assertEqual(
            [p["id"] for p in filter_problems(PROBLEMS, rating_min=800, rating_max=2400)],
            ["1A", "4C"],
        )

    def test_filtered_semantic_search_matches_restricted_faiss_index(self):
        class Model:
            calls = 0

            def encode(self, texts, convert_to_numpy=True):
                self.calls += 1
                return np.array([[0.0]], dtype=np.float32)

        vectors = np.array([[0], [1], [-1], [2], [3], [-4], [4], [5]], dtype=np.float32)
        index = faiss.IndexFlatL2(1)
        index.add(vectors)
        problems = [{"id": f"{i}A"} for i in range(len(vectors))]
        candidate_indices = [1, 2, 5, 6]
        model = cast(SentenceTransformer, Model())

        actual = semantic_search(
            "query", index, problems, model, k=3, candidate_indices=candidate_indices
        )

        restricted_index = faiss.IndexFlatL2(1)
        restricted_index.add(vectors[candidate_indices])
        distances, local_indices = restricted_index.search(
            np.array([[0.0]], dtype=np.float32), 3
        )
        expected = [
            (problems[candidate_indices[i]]["id"], float(distance))
            for distance, i in zip(distances[0], local_indices[0])
        ]

        self.assertEqual(
            [(problem["id"], distance) for problem, distance in actual], expected
        )
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()