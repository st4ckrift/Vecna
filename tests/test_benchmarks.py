from benchmarks.benchmark_search import generate_dataset, recall_at_k, run_benchmark


def test_dataset_generation_is_deterministic():
    first = generate_dataset(size=3, dimension=2, query_count=2, seed=5)
    second = generate_dataset(size=3, dimension=2, query_count=2, seed=5)
    assert first.store.vectors[1].values == second.store.vectors[1].values
    assert first.queries[0].values == second.queries[0].values


def test_recall_handles_full_partial_and_short_results():
    assert recall_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0
    assert recall_at_k([1, 9, 3], [1, 2, 3], 3) == 2 / 3
    assert recall_at_k([1], [1, 2], 5) == 1 / 2
    assert recall_at_k([], [], 5) == 0.0


def test_small_benchmark_reports_all_strategies_and_valid_metrics():
    rows = run_benchmark(size=30, dimension=3, query_count=5, k=4, seed=2, lsh_planes=2,
                         hnsw_m=3, hnsw_ef_construction=10, hnsw_ef_search=8)
    assert set(rows) == {"brute force", "LSH", "HNSW"}
    for row in rows.values():
        assert row["mean_latency_ms"] >= 0
        assert row["median_latency_ms"] >= 0
        assert row["throughput_qps"] > 0
        assert 0.0 <= row["recall_at_k"] <= 1.0
