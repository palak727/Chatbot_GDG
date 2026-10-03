import pickle
import re
import unittest
from typing import cast
from pathlib import Path

import faiss
import numpy as np
from transformers import PreTrainedTokenizerBase

from config import FAISS_INDEX_PATH, METADATA_PATH, PROBLEMS_DIR
from src.indexer import load_problems
from src.utils import build_embedding_text


class IndexConsistencyTests(unittest.TestCase):
    def test_embedding_text_keeps_topic_tags_before_truncated_statement(self):
        class WhitespaceTokenizer:
            def __init__(self):
                self.tokens: dict[str, int] = {}
                self.token_by_id: dict[int, str] = {}

            def encode(
                self,
                text,
                add_special_tokens=True,
                truncation=False,
                max_length=None,
            ):
                pieces = re.findall(r"\w+|[^\w\s]", text.lower())
                ids = []
                for piece in pieces:
                    if piece not in self.tokens:
                        token_id = len(self.tokens) + 2
                        self.tokens[piece] = token_id
                        self.token_by_id[token_id] = piece
                    ids.append(self.tokens[piece])
                if add_special_tokens:
                    ids = [0, *ids, 1]
                if truncation and max_length is not None:
                    ids = ids[:max_length]
                return ids

            def decode(self, token_ids, skip_special_tokens=True):
                return " ".join(
                    self.token_by_id[token_id]
                    for token_id in token_ids
                    if not skip_special_tokens or token_id not in (0, 1)
                )

        tokenizer = WhitespaceTokenizer()
        problem = {
            "title": "Ordered List Search",
            "tags": ["binary search", "graphs", "*1200"],
            "statement": "First statement sentence. " + "Long detail. " * 100,
            "input": "Input details",
            "output": "Output details",
        }

        text = build_embedding_text(
            problem, cast(PreTrainedTokenizerBase, tokenizer), max_tokens=24
        )
        encoded = tokenizer.encode(text, add_special_tokens=True)

        self.assertLess(text.lower().index("tags"), text.lower().index("statement"))
        self.assertIn("binary search", text.lower())
        self.assertIn("graphs", text.lower())
        self.assertIn("first statement sentence", text.lower())
        self.assertLessEqual(len(encoded), 24)

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