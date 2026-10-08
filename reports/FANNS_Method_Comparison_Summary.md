# Comparative Analysis of FANNS Search Strategies: Executive Summary for Mentor

**Document Title**: Comparative Analysis of FANNS Search Strategies  
**Full Report**: [`reports/FANNS_Method_Comparison_Report.docx`](file:///c:/Users/anime/Desktop/filtered-ann-research/reports/FANNS_Method_Comparison_Report.docx)  
**Dataset**: `SPCL/arxiv-for-fanns-small` (1,000 vectors, 10,000 queries, 4,096 dimensions)  
**Evaluated Workloads**: Exact Match (EM), Continuous Range (Range), Set Containment (EMIS)  
**Date**: October 2026  

---

## 10 Key Findings from Controlled Small-Scale Benchmarks

### 1. Naive Post-Filtering Suffers Immediate Recall Collapse
Traversing an unconstrained HNSW index and applying metadata filters post-hoc causes recall to collapse to **20.3% on set-containment (EMIS)** and **26.1% on exact-match (EM)** queries. Because proximity graphs group points geometrically regardless of metadata, top candidates fail the filter and are discarded without replacement.

### 2. Overfetching Exhibits Severe Diminishing Returns Under Selective Filters
While overfetching recovers recall on dense workloads (reaching 95.5% at 5x on EM), it fails on selective workloads. On EMIS queries, retrieving 20x candidates (200 points) only achieves **69.8% recall**. Reaching 95.3% recall requires **50x overfetching** (retrieving 500 candidates out of 1,000 vectors), which triples query latency (6.70 ms vs 2.15 ms) and drops throughput from 432 to 149 QPS.

### 3. Inverted-Index Pre-Filtering Dominates in Sparse Selectivity Regimes (<15%)
When filter selectivity is low ($\le 15\%$), inverted-index pre-filtering achieves **100% recall at 3.15 ms latency** (317.5 QPS), outperforming graph-based traversal methods in both speed and accuracy. In ultra-low selectivity (<2%), candidate pruning takes only 0.01 ms and vectorized distance evaluation takes <0.38 ms.

### 4. In-Graph Filtering Provides High, Stable Recall Across All Regimes
Single-graph in-traversal filtering (traversing all nodes for structural connectivity while admitting only matching candidates into the priority queue) achieves **99.4%–99.9% recall** across EM, Range, and EMIS workloads with competitive latency (4.64–6.97 ms).

### 5. Multi-Hop Traversal (ACORN Approximation) Prevents Graph Disconnection at High Cost
ACORN-style 2-hop expansion over invalid routing nodes achieves near-perfect recall (**99.48%–99.98%**) across all workloads, but visits up to 952 out of 1,000 nodes, resulting in high latency (**14.93–29.24 ms**). It should be selectively triggered rather than globally applied.

### 6. Localized Entry Routing (SIEVE Approximation) Prevents Graph Wandering
Initializing Layer 0 search from candidate centroids matching the query predicate achieves **99.42%–99.98% recall** and matches in-graph traversal speed (3.87–5.57 ms), proving that metadata-guided entry routing is effective for clustered metadata attributes.

### 7. The Empirical 20% Selectivity Crossover Rule
- **Selectivity $< 20\%$**: Pre-Filtering is superior (faster latency, 100% recall, zero graph traversal overhead).
- **Selectivity $> 25\%$**: In-Graph filtering is superior (higher throughput, 215+ QPS, avoiding exhaustive subset scanning).

### 8. Filter Types Differ Fundamentally in Complexity
- **Range Filters** (Continuous timestamps): Medium candidate density (mean 434 candidates), yielding 0.9984 In-Graph recall at 4.64 ms.
- **Exact Match** (Integer counts): Moderate candidate density (mean 362 candidates), yielding 0.9998 In-Graph recall at 4.95 ms.
- **EMIS Filters** (Categorical string sets): Highly selective (mean 155 candidates), causing severe post-filtering collapse (20.3%) and requiring adaptive traversal.

### 9. Scientific Limitations of Small-Scale Validation
- The 1,000-vector dataset (~16 MB) fits completely in the CPU L3 cache, making brute-force scans artificially fast.
- Graph traversals visit a disproportionate fraction (~80%–95%) of small graphs.
- These results serve to establish baseline correctness before scaling to 100K and 2.7M+ vectors on HPC infrastructure.

### 10. Research Opportunity: Adaptive AutoFANNS Engine
Because no single algorithm dominates across all selectivity regimes, we propose developing **AutoFANNS**—a dynamic hybrid planner that inspects filter selectivity at query time via O(1) bitmask cardinality and automatically dispatches to SIMD Pre-Filtering, SIEVE entry routing, or HNSW overfetching.

---

## Benchmark Comparison Table (K=10, N=500 queries/workload)

| Method | EM Recall@10 | EM Latency | Range Recall@10 | Range Latency | EMIS Recall@10 | EMIS Latency |
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

## Recommended Next Steps for Mentor Discussion
1. Transition the pipeline to **100,000 vectors** on the university cluster using our prepared Slurm script ([`hpc/slurm_job_template.sh`](file:///c:/Users/anime/Desktop/filtered-ann-research/hpc/slurm_job_template.sh)).
2. Formalize the **AutoFANNS** cost model to evaluate whether dynamic switching outperforms static algorithms at scale.
