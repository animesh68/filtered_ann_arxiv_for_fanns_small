# Filtered Approximate Nearest Neighbor Search (FANNS) Benchmark & Research Framework

An academic-grade, reproducible research and benchmarking pipeline for **Filtered Approximate Nearest Neighbor Search (FANNS)** on high-dimensional vector embeddings with metadata predicates.

---

## 📌 Project Overview & Research Goals
This repository investigates the empirical performance trade-offs between retrieval accuracy (Recall@K), query latency, throughput (QPS), and index memory across diverse metadata filtering regimes:
- **Exact Match (EM)** (Categorical point queries)
- **Range (R)** (Continuous interval filters on timestamps)
- **Exact Match In Set (EMIS)** (Set containment / categorical membership)

We systematically evaluate and compare 11 search strategies:
1. **Exact Filtered Search** (Mathematical ground-truth reference)
2. **Inverted-Index Pre-Filtering**
3. **Naive HNSW Post-Filtering**
4. **HNSW Overfetching (2x, 5x, 10x, 20x)**
5. **In-Graph Filtered Traversal**
6. **ACORN-style Filter Traversal** (*Paper-inspired approximation*)
7. **SIEVE-style Entry-Point Routing** (*Paper-inspired approximation*)
8. **UNIFY-style Selectivity-Adaptive Traversal** (*Paper-inspired approximation*)

---

## 📂 Repository Structure

```
filtered-ann-research/
├── data/
│   └── arxiv-for-fanns-small/             # Verified benchmark dataset
├── src/
│   ├── dataset_loader.py                   # Standardized fvecs/ivecs/JSONL loader
│   ├── filters.py                          # Predicates & Inverted Metadata Index
│   ├── exact_search.py                     # Ground-truth exact filtered searcher
│   ├── hnsw_index.py                       # Modular HNSW Graph Index with research hooks
│   ├── postfilter.py                       # Naive post-filtering strategy
│   ├── overfetch.py                        # Overfetching candidate multiplier strategy
│   ├── prefilter.py                        # Inverted-index pre-filtering strategy
│   ├── ingraph.py                          # In-graph traversal strategy
│   ├── filter_aware.py                     # ACORN, SIEVE, and UNIFY approximations
│   ├── metrics.py                          # Recall@K, percentile latencies, QPS, logging
│   └── timing.py                           # Nanosecond precision timing harness
├── experiments/
│   ├── 01_dataset_validation.py            # Dataset integrity & format verification
│   ├── 02_exact_search.py                  # Exact search ground-truth benchmark
│   ├── 03_hnsw_postfilter.py               # HNSW baseline & post-filtering
│   ├── 04_hnsw_overfetch.py                # Overfetching factor sweep
│   ├── 05_prefilter.py                     # Pre-filtering benchmark
│   ├── 06_ingraph.py                       # In-graph filtering benchmark
│   ├── 07_filter_aware_comparison.py       # ACORN, SIEVE, UNIFY benchmark
│   ├── 08_controlled_benchmark_all.py      # Comprehensive 16,500-query benchmark
│   └── 09_generate_plots_and_tables.py     # Publication figure generator
├── results/
│   ├── controlled_benchmark/
│   │   ├── all_queries_raw_results.csv     # Granular per-query logs (16,500 rows)
│   │   ├── benchmark_summary.csv           # Aggregated metrics table
│   │   └── stratified_selectivity_summary.csv
│   ├── exact_search/
│   ├── hnsw_postfilter/
│   ├── hnsw_overfetch/
│   ├── prefilter/
│   └── ingraph/
├── reports/
│   ├── research_progress_report.md         # Comprehensive 28-section research report
│   ├── experiment_log.md                   # Systematic experiment log
│   └── figures/                            # Publication-quality plots (PNG, 300 DPI)
│       ├── recall_vs_selectivity.png
│       ├── recall_vs_latency_tradeoff.png
│       └── qps_comparison.png
├── hpc/
│   ├── hpc_checklist.md                    # Supercomputer checklist & memory sizing
│   └── slurm_job_template.sh               # Slurm batch submission script
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Dataset
```bash
python experiments/01_dataset_validation.py
```

### 3. Run Full Controlled Comparative Benchmark
```bash
python experiments/08_controlled_benchmark_all.py
```

### 4. Generate Visualizations & Tables
```bash
python experiments/09_generate_plots_and_tables.py
```

---

## 📊 Key Experimental Findings

| Strategy | EM Recall@10 | EM Latency | Range Recall@10 | Range Latency | EMIS Recall@10 | EMIS Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ExactSearch** | 1.0000 | 7.07 ms | 1.0000 | 8.30 ms | 1.0000 | 3.11 ms |
| **PreFilter** | **1.0000** | **6.92 ms** | **1.0000** | **8.20 ms** | **1.0000** | **3.15 ms** |
| **HNSW_PostFilter** | 0.2608 | 2.52 ms | 0.3952 | 2.64 ms | 0.2030 | 2.67 ms |
| **HNSW_Overfetch_5x** | 0.9550 | 2.52 ms | 0.8260 | 2.70 ms | 0.3705 | 2.64 ms |
| **HNSW_Overfetch_20x** | 0.9958 | 5.25 ms | 0.9710 | 5.11 ms | 0.6977 | 5.41 ms |
| **InGraphFilter** | **0.9998** | **4.95 ms** | **0.9984** | **4.64 ms** | **0.9940** | **6.97 ms** |
| **ACORN_Approx** | 0.9998 | 14.93 ms | 0.9998 | 14.13 ms | 0.9948 | 29.24 ms |
| **SIEVE_Approx** | 0.9998 | 5.02 ms | 0.9984 | 5.02 ms | 0.9942 | 6.92 ms |
| **UNIFY_Approx** | 0.9998 | 7.44 ms | 0.9986 | 7.46 ms | 0.9940 | 22.21 ms |

---

## 🖥️ Supercomputer / HPC Deployment
To scale experiments to 100K or 1M vectors on a high-performance computing cluster:
1. Review the [HPC Readiness Checklist](file:///c:/Users/anime/Desktop/filtered-ann-research/hpc/hpc_checklist.md).
2. Submit the Slurm batch script:
```bash
sbatch hpc/slurm_job_template.sh
```

---

## 📄 Documentation & Research Progress
- Detailed academic report: [`reports/research_progress_report.md`](file:///c:/Users/anime/Desktop/filtered-ann-research/reports/research_progress_report.md)
- Complete chronological experiment log: [`reports/experiment_log.md`](file:///c:/Users/anime/Desktop/filtered-ann-research/reports/experiment_log.md)
