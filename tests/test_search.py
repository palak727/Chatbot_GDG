import unittest

from src.search import filter_problems, is_problem_id_query, match_by_id


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


if __name__ == "__main__":
    unittest.main()