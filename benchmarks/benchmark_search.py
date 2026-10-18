"""Reproducible synthetic benchmark for the project's search strategies."""

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hnsw import HNSWIndex
from src.index import Index
from src.search import BruteForceSearch
from src.storage import VectorStore
from src.vector import Vector


@dataclass
class BenchmarkDataset:
    store: VectorStore
    queries: list[Vector]


def generate_dataset(size=1000, dimension=16, query_count=100, seed=42):
    """Create deterministic synthetic vectors and query vectors."""
    if size < 0 or dimension < 1 or query_count < 0:
        raise ValueError("size and query_count must be non-negative; dimension must be positive")
    rng = Random(seed)
    store = VectorStore()
    for _ in range(size):
        store.add(Vector([rng.gauss(0.0, 1.0) for _ in range(dimension)]))
    queries = [Vector([rng.gauss(0.0, 1.0) for _ in range(dimension)]) for _ in range(query_count)]
    return BenchmarkDataset(store, queries)


def recall_at_k(approximate_ids, exact_ids, k):
    """Calculate overlap recall, using available exact results as denominator."""
    denominator = min(k, len(exact_ids))
    if denominator == 0:
        return 0.0
    return len(set(approximate_ids[:k]) & set(exact_ids[:k])) / denominator


def percentile(values, percentage):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def measure_strategy(searcher, queries, k, exact_results):
    latencies = []
    recalls = []
    for query, exact in zip(queries, exact_results):
        started = time.perf_counter()
        result = searcher.search(query, k=k)
        latencies.append(time.perf_counter() - started)
        recalls.append(recall_at_k([item[0] for item in result], [item[0] for item in exact], k))
    total = sum(latencies)
    return {
        "build_time": None,
        "mean_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) * 1000 if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95) * 1000,
        "throughput_qps": len(latencies) / total if total else 0.0,
        "recall_at_k": statistics.mean(recalls) if recalls else 0.0,
    }


def build_timed(factory):
    started = time.perf_counter()
    index = factory()
    return index, time.perf_counter() - started


def run_benchmark(size=1000, dimension=16, query_count=100, k=10, seed=42, lsh_planes=8,
                  hnsw_m=8, hnsw_ef_construction=100, hnsw_ef_search=32):
    if k < 1:
        raise ValueError("k must be positive")
    dataset = generate_dataset(size, dimension, query_count, seed)
    exact = BruteForceSearch(dataset.store)
    exact_results = [exact.search(query, k=k) for query in dataset.queries]
    rows = {"brute force": measure_strategy(exact, dataset.queries, k, exact_results)}
    rows["brute force"]["recall_at_k"] = 1.0 if exact_results else 0.0

    lsh, build_time = build_timed(lambda: Index(dataset.store, num_planes=lsh_planes, seed=seed))
    rows["LSH"] = measure_strategy(BruteForceSearch(dataset.store, lsh), dataset.queries, k, exact_results)
    rows["LSH"]["build_time"] = build_time

    hnsw, build_time = build_timed(lambda: HNSWIndex(
        dataset.store, M=hnsw_m, ef_construction=hnsw_ef_construction,
        ef_search=hnsw_ef_search, seed=seed,
    ))
    rows["HNSW"] = measure_strategy(BruteForceSearch(dataset.store, hnsw), dataset.queries, k, exact_results)
    rows["HNSW"]["build_time"] = build_time
    rows["brute force"]["build_time"] = 0.0
    return rows


def print_report(rows):
    print("Strategy       Build (ms)  Mean (ms)  Median (ms)  Throughput (q/s)  Recall@k")
    print("-" * 80)
    for name, row in rows.items():
        print(f"{name:<14} {row['build_time'] * 1000:>9.3f}  {row['mean_latency_ms']:>9.3f}  "
              f"{row['median_latency_ms']:>11.3f}  {row['throughput_qps']:>16.1f}  {row['recall_at_k']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lsh-planes", type=int, default=8)
    parser.add_argument("--hnsw-m", type=int, default=8)
    parser.add_argument("--hnsw-ef-construction", type=int, default=100)
    parser.add_argument("--hnsw-ef-search", type=int, default=32)
    args = parser.parse_args()
    print(f"Synthetic dataset: size={args.size}, dimension={args.dimension}, queries={args.queries}, k={args.k}")
    print_report(run_benchmark(
        args.size, args.dimension, args.queries, args.k, args.seed, args.lsh_planes,
        args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search,
    ))


if __name__ == "__main__":
    main()
