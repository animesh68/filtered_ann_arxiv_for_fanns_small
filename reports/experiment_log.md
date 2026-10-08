# FANNS Research Experiment Log

## Experiment Entry 01: Dataset & Format Verification
- **Date**: 2026-10-08
- **Dataset**: `SPCL/arxiv-for-fanns-small` (1,000 database vectors, 10,000 query vectors, 4,096-dim Float32)
- **Objective**: Verify binary vector structures (`.fvecs`), metadata attributes (`.jsonl`), and variable-length filtered ground truth files (`.ivecs`).
- **Hardware**: Windows 64-bit AMD x86_64, Python 3.14 venv
- **Results**:
  - `database_vectors.fvecs`: 1,000 vectors × 4,096 dimensions. No NaN or Inf. Unit-normalized.
  - `query_vectors.fvecs`: 10,000 vectors × 4,096 dimensions.
  - `ground_truth.ivecs`: Fixed-width (10000 × 100).
  - Variable filtered ground truths: EM (mean 96.3 GT items), Range (mean 98.3 GT items), EMIS (mean 82.1 GT items).
  - Filter attribute bindings verified:
    - **EM**: `database_attributes.number_of_sub_categories == query.label`
    - **Range**: `query.range_start <= database_attributes.update_date <= query.range_end`
    - **EMIS**: `query.label in database_attributes.main_categories`

---

## Experiment Entry 02: Exact Search Ground-Truth Reference
- **Date**: 2026-10-08
- **Method**: `ExactFilteredSearcher` (Euclidean L2 distance over inverted index candidates)
- **Parameters**: `K=10`, `N=500` queries/workload
- **Results**:
  - `unfiltered`: Recall@10 = **1.0000**, Latency = **17.33 ms**, QPS = **57.7**
  - `EM`: Recall@10 = **1.0000**, Latency = **7.40 ms**, QPS = **135.1** (Mean candidates = 362.4)
  - `Range`: Recall@10 = **1.0000**, Latency = **9.50 ms**, QPS = **105.3** (Mean candidates = 434.2)
  - `EMIS`: Recall@10 = **1.0000**, Latency = **3.89 ms**, QPS = **257.3** (Mean candidates = 155.6)
- **Observations**: Exact filtered search matches the provided `.ivecs` ground truth with 100% mathematical precision.

---

## Experiment Entry 03: HNSW Baseline & Naive Post-Filtering
- **Date**: 2026-10-08
- **Methods**: `HNSW_Unfiltered`, `HNSW_PostFilter`
- **Parameters**: `M=16`, `ef_construction=100`, `ef_search=50`, `k=10`, `N=500` queries
- **Results**:
  - `HNSW_Unfiltered`: Recall@10 = **0.9980**, Latency = **2.52 ms**, QPS = **396.5**
  - `EM Post-Filter`: Recall@10 = **0.2608**, Latency = **2.52 ms**, QPS = **396.5**
  - `Range Post-Filter`: Recall@10 = **0.3952**, Latency = **2.64 ms**, QPS = **379.4**
  - `EMIS Post-Filter`: Recall@10 = **0.2030**, Latency = **2.67 ms**, QPS = **374.2**
- **Observations**: Catastrophic recall collapse under post-filtering (down to 20.3% on EMIS and 26.1% on EM) because top-k candidates in semantic space are pruned away by metadata predicates without being replaced.

---

## Experiment Entry 04: Overfetching Factor Sweep
- **Date**: 2026-10-08
- **Method**: `HNSWOverfetchSearcher`
- **Parameters**: Overfetch factors = `[1.0x, 2.0x, 5.0x, 10.0x, 20.0x]`
- **Results (EMIS Workload)**:
  - 1x: Recall = **0.2030**, Latency = 2.67 ms
  - 2x: Recall = **0.2494**, Latency = 2.60 ms
  - 5x: Recall = **0.3705**, Latency = 2.64 ms
  - 10x: Recall = **0.5263**, Latency = 3.93 ms
  - 20x: Recall = **0.6977**, Latency = 5.41 ms
- **Observations**: While overfetching improves recall, it suffers severe diminishing returns on selective workloads (EMIS at 20x fetch only reaches 69.8% recall while latency doubles).

---

## Experiment Entry 05: Comprehensive Controlled Benchmark (11 Strategies)
- **Date**: 2026-10-08
- **Evaluated Methods**: `ExactSearch`, `PreFilter`, `HNSW_PostFilter`, `Overfetch 2x/5x/10x/20x`, `InGraphFilter`, `ACORN_Approx`, `SIEVE_Approx`, `UNIFY_Approx`
- **Number of Evaluations**: 16,500 total query executions (500 queries × 11 methods × 3 workloads)
- **Results Summary**:

| Method | EM Recall@10 | EM Latency (ms) | Range Recall@10 | Range Latency (ms) | EMIS Recall@10 | EMIS Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ExactSearch** | 1.0000 | 7.07 | 1.0000 | 8.30 | 1.0000 | 3.11 |
| **PreFilter** | **1.0000** | **6.92** | **1.0000** | **8.20** | **1.0000** | **3.15** |
| **HNSW_PostFilter** | 0.2608 | 2.52 | 0.3952 | 2.64 | 0.2030 | 2.67 |
| **Overfetch_5x** | 0.9550 | 2.52 | 0.8260 | 2.70 | 0.3705 | 2.64 |
| **Overfetch_20x** | 0.9958 | 5.25 | 0.9710 | 5.11 | 0.6977 | 5.41 |
| **InGraphFilter** | **0.9998** | **4.95** | **0.9984** | **4.64** | **0.9940** | **6.97** |
| **ACORN_Approx** | 0.9998 | 14.93 | 0.9998 | 14.13 | 0.9948 | 29.24 |
| **SIEVE_Approx** | 0.9998 | 5.02 | 0.9984 | 5.02 | 0.9942 | 6.92 |
| **UNIFY_Approx** | 0.9998 | 7.44 | 0.9986 | 7.46 | 0.9940 | 22.21 |

- **Key Finding**: Pre-filtering dominates on low-selectivity (sparse) workloads (3.15 ms with 100% recall), while In-Graph and Filter-Aware methods dominate when candidate pools are dense. Overfetching is ineffective under low selectivity (<10%).
