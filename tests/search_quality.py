"""Real-world semantic search evaluation for the current Codeforces index.

Run with ``python tests/search_quality.py``. The script prints every top-five
result so relevance can be judged from the problem title, tags, and statement,
not from a shared query token alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search import SearchEngine  # noqa: E402


TOPICS = [
    ("weighted shortest path", "find shortest path in weighted graph", "find minimum route cost", "shortest path weigth graph"),
    ("Dijkstra", "find shortest path between nodes using dijkstra", "minimum distance from source", "dijsktra shortest path"),
    ("BFS graph", "find shortest path in an unweighted graph", "visit graph layer by layer", "shortes path unweighted grapf"),
    ("DFS cycle", "find cycle in directed graph", "detect cycle using dfs", "cycle in directd graph"),
    ("MST", "connect all cities with minimum cost", "minimum cost to connect all points", "minumum cost connect nodes"),
    ("binary search", "search in sorted array", "find a value in an ordered list", "binry serach sorted aray"),
    ("binary search answer", "find the minimum feasible value with binary search", "optimize an answer over a monotonic condition", "minimun binary serach answer"),
    ("dynamic programming", "solve a problem by reusing answers to smaller states", "minimum cost with overlapping subproblems", "dynamc programing states"),
    ("tree DP", "calculate the best value for every subtree", "dynamic programming on a tree", "tree dynamc progrmming"),
    ("sliding window", "find the longest subarray satisfying a condition", "maintain a moving range over an array", "sliding windw longest subaray"),
    ("two pointers", "find a pair in a sorted array", "move two indices from both ends", "two pointrs sum aray"),
    ("segment tree", "answer range minimum queries with updates", "support range queries and point changes", "segmant tree range query"),
    ("greedy", "choose locally best actions to minimize the answer", "schedule intervals for maximum profit", "greedy algorthm choice"),
    ("strings", "count or compare patterns in strings", "transform one string into another", "string mismtach problem"),
    ("arrays", "find the maximum sum contiguous subarray", "process an array of numbers efficiently", "maximun contigous aray sum"),
    ("kth order statistic", "find kth smallest element", "select the element with a given rank", "kth smalest elment"),
    ("range sums", "answer many subarray sum queries", "calculate sums between array indices", "subaray sum queris"),
    ("connected components", "count groups of connected nodes", "find separate components in a graph", "conected componets graph"),
    ("negative shortest path", "find shortest path with negative edge weights", "handle negative costs between nodes", "negativ edge weigths path"),
    ("graph traversal", "explore all reachable vertices", "visit every node and edge", "travese grap nodes"),
]


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    engine = SearchEngine()
    engine.load()
    print(json.dumps({"indexed_problems": len(engine.problems), "queries": len(TOPICS) * 4}))

    for topic, original, paraphrase, typo in TOPICS:
        variants = (
            ("original", original),
            ("paraphrase", paraphrase),
            ("typo", typo),
            ("vague", topic),
        )
        for variant, query in variants:
            results = engine.search(query, k=5)
            print(f"\n[{topic}] {variant}: {query}")
            for rank, problem in enumerate(results, 1):
                tags = ", ".join(t for t in problem.get("tags", []) if not t.startswith("*"))
                statement = " ".join(problem.get("statement", "").split())[:180]
                print(f"{rank}. {problem['id']} | {problem['title']} | tags={tags} | {statement}")


if __name__ == "__main__":
    main()