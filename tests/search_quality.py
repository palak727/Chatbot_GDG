"""Real-world semantic search evaluation for the current Codeforces index.

Run with:
    python tests/search_quality.py

The queries are designed to represent realistic searches a competitive
programming user might make when looking for problems by topic, technique,
or problem pattern.

The script prints the top-five results so relevance can be judged from
the problem title, tags, and statement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search import SearchEngine  # noqa: E402


TOPICS = [
    # ------------------------------------------------------------------
    # Shortest paths
    # ------------------------------------------------------------------
    (
        "weighted shortest path",
        "find the shortest path in a weighted graph",
        "find the minimum cost route between two nodes",
        "shortest path weigth graph",
    ),
    (
        "Dijkstra",
        "find shortest paths using Dijkstra algorithm",
        "minimum distances from one source in a weighted graph",
        "dijsktra shortest path",
    ),
    (
        "BFS graph",
        "find the shortest path in an unweighted graph",
        "find minimum number of edges between two nodes",
        "shortes path unweighted grapf",
    ),

    # ------------------------------------------------------------------
    # Graph traversal / cycles
    # ------------------------------------------------------------------
    (
        "DFS cycle",
        "detect a cycle in a directed graph using DFS",
        "check whether a directed graph contains a cycle",
        "cycle in directd graph",
    ),
    (
        "graph traversal",
        "traverse a graph and visit reachable vertices",
        "explore all nodes reachable from a starting vertex",
        "travese grap nodes",
    ),
    (
        "connected components",
        "find connected components in an undirected graph",
        "find separate groups of connected vertices",
        "conected componets graph",
    ),

    # ------------------------------------------------------------------
    # Minimum spanning tree
    # ------------------------------------------------------------------
    (
        "MST",
        "build a minimum spanning tree of a weighted graph",
        "connect all vertices with minimum total edge cost",
        "minimum spaning tree graph",
    ),

    # ------------------------------------------------------------------
    # Binary search
    # ------------------------------------------------------------------
    (
        "binary search",
        "search for a target value in a sorted array",
        "find an element efficiently in sorted data",
        "binry serach sorted aray",
    ),
    (
        "binary search answer",
        "solve a problem using binary search on the answer",
        "find the minimum feasible value using a monotonic condition",
        "binary serach on answr",
    ),

    # ------------------------------------------------------------------
    # Dynamic programming
    # ------------------------------------------------------------------
    (
        "dynamic programming",
        "solve a problem using dynamic programming",
        "compute answers by reusing results from smaller states",
        "dynamc programing",
    ),
    (
        "tree DP",
        "solve a tree problem using dynamic programming",
        "compute the best value for each subtree",
        "tree dynamc progrmming",
    ),

    # ------------------------------------------------------------------
    # Sliding window / two pointers
    # ------------------------------------------------------------------
    (
        "sliding window",
        "find the longest subarray satisfying a constraint",
        "maintain a moving window over an array",
        "sliding windw longest subaray",
    ),
    (
        "two pointers",
        "find a pair in a sorted array using two pointers",
        "solve an array problem by moving two indices",
        "two pointrs sorted aray",
    ),

    # ------------------------------------------------------------------
    # Segment tree / range queries
    # ------------------------------------------------------------------
    (
        "segment tree",
        "answer range queries with point updates using a segment tree",
        "maintain an array while processing range queries and updates",
        "segmant tree range query",
    ),

    # ------------------------------------------------------------------
    # Greedy
    # ------------------------------------------------------------------
    (
        "greedy",
        "solve a problem by making the best local choice at each step",
        "choose locally optimal actions to maximize or minimize the result",
        "greedy algorthm",
    ),

    # ------------------------------------------------------------------
    # Strings
    # ------------------------------------------------------------------
    (
        "strings",
        "compare or transform two strings",
        "solve a problem involving string matching or string operations",
        "string mismtach problem",
    ),

    # ------------------------------------------------------------------
    # Arrays / subarrays
    # ------------------------------------------------------------------
    (
        "maximum subarray",
        "find the contiguous subarray with maximum sum",
        "maximize the sum of a contiguous segment of an array",
        "maximun contigous aray sum",
    ),

    # ------------------------------------------------------------------
    # Order statistics
    # ------------------------------------------------------------------
    (
        "kth order statistic",
        "find the kth smallest element in an array",
        "select the element with a given rank after sorting",
        "kth smalest elment",
    ),

    # ------------------------------------------------------------------
    # Range sums
    # ------------------------------------------------------------------
    (
        "range sums",
        "answer many subarray sum queries",
        "calculate the sum between two array indices",
        "subaray sum queris",
    ),

    # ------------------------------------------------------------------
    # Negative edge weights
    # ------------------------------------------------------------------
    (
        "negative edge weights",
        "find shortest paths in a graph with negative edge weights",
        "handle graph edges that can have negative costs",
        "negativ edge weigths path",
    ),
]


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    engine = SearchEngine()
    engine.load()

    total_queries = len(TOPICS) * 4

    print(
        json.dumps(
            {
                "indexed_problems": len(engine.problems),
                "topics": len(TOPICS),
                "queries": total_queries,
            }
        )
    )

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
                tags = ", ".join(
                    tag
                    for tag in problem.get("tags", [])
                    if not tag.startswith("*")
                )

                statement = " ".join(
                    problem.get("statement", "").split()
                )[:180]

                print(
                    f"{rank}. {problem['id']} | "
                    f"{problem['title']} | "
                    f"tags={tags} | "
                    f"{statement}"
                )


if __name__ == "__main__":
    main()