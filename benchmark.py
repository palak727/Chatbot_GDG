"""
CP Chatbot - Performance Benchmarking Script
Measures index sizes, startup time, memory footprint, and search latency.
"""

import os
import time
import statistics
import psutil

from config import FAISS_INDEX_PATH, METADATA_PATH
from src.search import SearchEngine


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark():
    print("--- CP Chatbot Benchmarking Report ---")

    # 1. Disk footprint check
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        faiss_size = os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024)
        meta_size = os.path.getsize(METADATA_PATH) / (1024 * 1024)
        print(f"FAISS Index Size: {faiss_size:.2f} MB")
        print(f"Metadata Size:    {meta_size:.2f} MB")
    else:
        print("Warning: Index files missing. Build the index first before benchmarking.")
        return

    # 2. Startup latency & RAM usage
    mem_before = get_memory_mb()
    t_start = time.perf_counter()

    engine = SearchEngine()
    engine.load()

    load_time_ms = (time.perf_counter() - t_start) * 1000
    mem_after = get_memory_mb()
    ram_used = mem_after - mem_before

    print(f"Total Indexed Problems: {len(engine.problems)}")
    print(f"Index Load Time:        {load_time_ms:.2f} ms")
    print(f"RAM Usage:              {ram_used:.2f} MB")

    # 3. Search latency evaluation
    test_queries = [
        ("dynamic programming on trees", "semantic"),
        ("shortest path dijkstra", "semantic"),
        ("1000A", "exact_id"),
        ("greedy interval scheduling", "semantic"),
        ("binary search on answer", "semantic"),
    ]

    # Warmup query
    engine.search("test query", mode="semantic", k=5)

    latencies = []
    iterations = 20

    print(f"\nRunning search tests ({len(test_queries) * iterations} total queries)...")

    for query, mode in test_queries:
        for _ in range(iterations):
            t0 = time.perf_counter()
            engine.search(query, mode=mode, k=5)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    avg_lat = statistics.mean(latencies)
    p50_lat = statistics.median(latencies)
    p95_lat = statistics.quantiles(latencies, n=20)[18]
    throughput = 1000 / avg_lat if avg_lat > 0 else 0

    print("\nSearch Performance Metrics:")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Median (P50):    {p50_lat:.2f} ms")
    print(f"P95 Latency:     {p95_lat:.2f} ms")
    print(f"Throughput:      {throughput:.1f} queries/sec")

    print("\n--- Summary for Resume ---")
    print(f"Indexed Problems: {len(engine.problems)}")
    print(f"Average Latency:  {avg_lat:.1f} ms (P95: {p95_lat:.1f} ms)")
    print(f"RAM Footprint:    {ram_used:.1f} MB")


if __name__ == "__main__":
    run_benchmark()