import pickle
import unittest
from pathlib import Path

import faiss
import numpy as np

from config import FAISS_INDEX_PATH, METADATA_PATH, PROBLEMS_DIR
from src.indexer import load_problems


class IndexConsistencyTests(unittest.TestCase):
    def test_dataset_and_persisted_index_have_matching_rows(self):
        problems = load_problems(PROBLEMS_DIR)
        with Path(METADATA_PATH).open("rb") as handle:
            metadata = pickle.load(handle)
        index = faiss.read_index(FAISS_INDEX_PATH)

        self.assertEqual(len(problems), len(metadata["problems"]))
        self.assertEqual(len(metadata["problems"]), len(metadata["vectors"]))
        self.assertEqual(len(metadata["vectors"]), index.ntotal)
        self.assertEqual(np.asarray(metadata["vectors"]).shape[0], index.ntotal)
        self.assertEqual(
            {problem["id"] for problem in problems},
            {problem["id"] for problem in metadata["problems"]},
        )


if __name__ == "__main__":
    unittest.main()