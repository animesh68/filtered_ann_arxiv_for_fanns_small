# Research Progress Report: Filtered Approximate Nearest Neighbor Search (FANNS)

**Lead Research Engineer Progress & Empirical Benchmark Report**  
**Project**: Filtered Approximate Nearest Neighbor Search (FANNS)  
**Dataset**: `SPCL/arxiv-for-fanns-small` (1,000 vectors, 10,000 queries, 4,096 dimensions)  
**Date**: October 2026  

---

## 1. Executive Summary
Filtered Approximate Nearest Neighbor Search (FANNS) is a critical capability in modern retrieval-augmented generation (RAG), vector databases, and multi-modal search engines where high-dimensional vector similarity is constrained by structured metadata filters. In this research phase, we established a rigorous, mathematically verified, end-to-end experimental benchmarking pipeline on the `SPCL/arxiv-for-fanns-small` benchmark. 

We systematically implemented, tested, and evaluated eleven (11) distinct search strategies spanning exact search, pre-filtering, naive post-filtering, overfetching sweeps (2x–20x), in-graph traversal, and paper-inspired filter-aware approaches (ACORN, SIEVE, UNIFY). Over 16,500 controlled query evaluations across three realistic filter regimes (Exact Match, Range, and Set-Containment/EMIS) were conducted. 

Our key empirical finding is that **no single filtered ANN algorithm dominates across all selectivity and filter regimes**:
1. **Naive Post-Filtering** experiences catastrophic recall collapse (plunging to **20.3% recall** on EMIS workloads).
2. **Overfetching** offers modest relief but requires prohibitive candidate expansion (even 20x overfetching achieves only **69.8% recall** on selective workloads).
3. **Pre-Filtering** provides optimal performance (**100% recall at 3.15 ms latency**) in sparse selectivity regimes (<15% matching candidates) where scanning pruned subsets is faster than traversing full graphs.
4. **In-Graph Filtering & Filter-Aware Traversal** maintain superior recall (**99.4%–99.9%**) in medium-to-dense selectivity regimes, but incur multi-hop expansion overhead when candidate density drops.

These empirical results expose a clear research gap: **a static search strategy is inherently suboptimal**. We formulate the foundation for an **Adaptive Filter-Aware ANN Engine (AutoFANNS)** that dynamically switches between inverted-index pre-filtering, entry-point rerouting, and adaptive beam expansion based on runtime filter selectivity and candidate density.

---

## 2. Research Problem
Standard Approximate Nearest Neighbor (ANN) indexes (e.g., HNSW, IVF-PQ) organize vector spaces based solely on geometric proximity. When queries specify boolean or range metadata predicates (e.g., `year >= 2024 AND category = 'cs.AI'`), the geometric graph topology no longer aligns with the valid candidate subspace. The fundamental research problem is how to retrieve top-$k$ nearest neighbors that strictly satisfy metadata filters while simultaneously maximizing recall, minimizing query latency, and bounding memory footprint.

---

## 3. Research Questions
- **RQ1 (Recall Degradation)**: Why and under what exact selectivity thresholds does classical post-filtering fail?
- **RQ2 (Overfetching Limits)**: At what point does increasing the overfetch factor in HNSW yield diminishing returns in recall while degrading throughput?
- **RQ3 (Pre-Filtering Efficiency)**: In what selectivity regimes does inverted-index pre-filtering outperform graph-based approximate search in latency?
- **RQ4 (Graph Disconnection in Filtered Traversal)**: How does in-graph filtering handle sparse candidate topologies, and does multi-hop expansion (ACORN-style) justify its latency overhead?
- **RQ5 (Cross-Regime Interaction)**: How do different filter types (point equality, continuous range, multi-value set membership) alter the recall-latency Pareto frontier?

---

## 4. FANNS Background
In modern vector search, data items consist of a vector representation $\mathbf{x}_i \in \mathbb{R}^D$ and a structured metadata dictionary $\mathbf{m}_i$. A filtered query is defined by a pair $(\mathbf{q}, \mathcal{P})$, where $\mathbf{q} \in \mathbb{R}^D$ is the query vector and $\mathcal{P}: \mathcal{M} \to \{0, 1\}$ is a boolean filter predicate. The goal is to return:

$$\text{Top-}k = \arg\min_{\substack{S \subseteq \mathcal{D}, |S|=k \\ \forall i \in S, \mathcal{P}(\mathbf{m}_i)=1}} \sum_{i \in S} \|\mathbf{x}_i - \mathbf{q}\|_2^2$$

---

## 5. HNSW Background
Hierarchical Navigable Small World (HNSW) graphs (Malkov & Yashunin, 2018) build multi-layer proximity graphs with exponentially decreasing edge lengths at upper layers, enabling logarithmic greedy routing down to Layer 0, where beam search (`efSearch`) finds local minima. However, HNSW graphs are constructed without metadata awareness, meaning edges between valid candidate nodes may pass through clusters of invalid nodes.

---

## 6. Dataset Description
We evaluate on the official benchmark dataset `SPCL/arxiv-for-fanns-small`:
- **Database Vectors**: 1,000 vectors, $D = 4,096$, Float32.
- **Query Vectors**: 10,000 vectors, $D = 4,096$, Float32.
- **Normalization**: Vectors are unit-normalized ($L_2 \text{ norm} \approx 1.0$), ensuring equivalence between Euclidean distance ranking and Cosine/Inner-Product similarity ranking.
- **Metadata**: 1,000 JSONL records containing ArXiv paper metadata (`submitter`, `main_categories`, `sub_categories`, `number_of_sub_categories`, `update_date`, `license`, `authors`).

---

## 7. Dataset Validation
All files were validated for integrity and completeness:
- No NaN, Inf, or corrupted float values.
- Unfiltered ground truth matches exact 10,000 $\times$ 100 fixed `.ivecs` structure.
- Variable-length filtered ground truth files (`ground_truth_em.ivecs`, `ground_truth_r.ivecs`, `ground_truth_emis.ivecs`) parsed without fixed-stride assumptions.

---

## 8. Filter Types & Mathematical Semantics
Through rigorous reverse-engineering and ground-truth validation, the dataset filter bindings were confirmed:

| Filter Type | Benchmark File | Mathematical Predicate | Evaluated Attribute | Example Query Predicate |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Match (EM)** | `em_query_attributes.jsonl` | $\text{attr}[\text{'number\_of\_sub\_categories'}] = \text{label}$ | Integer Count | `number_of_sub_categories == 3` |
| **Range (R)** | `r_query_attributes.jsonl` | $\text{start} \le \text{attr}[\text{'update\_date'}] \le \text{end}$ | Integer Timestamp | `17789 <= update_date <= 18990` |
| **Set Containment (EMIS)** | `emis_query_attributes.jsonl` | $\text{label} \in \text{attr}[\text{'main\_categories'}]$ | String List | `'math' in main_categories` |

---

## 9. Experimental Methodology
- **Identical Evaluation Harness**: Every algorithm evaluated on the identical 500 query subset per workload.
- **Hardware Consistency**: Benchmarks run in an isolated environment with deterministic random seeds (`seed=42`).
- **Warmup Procedure**: 2 warmup iterations preceding nanosecond-precision execution timing.
- **Stratification**: Queries partitioned into natural selectivity bands: Ultra-Low (<2%), Low (2–10%), Medium (10–30%), and High (>30%).

---

## 10. Algorithms Evaluated
1. **ExactSearch**: Baseline ground truth reference via inverted-index candidate pruning and exact $L_2$ distance ranking.
2. **PreFilter**: Inverted metadata index prunes candidates first; exact vector similarity is computed exclusively on matching records.
3. **HNSW_PostFilter**: Standard HNSW retrieves $k$ candidates; non-matching candidates are discarded.
4. **HNSW_Overfetch (2x, 5x, 10x, 20x)**: Standard HNSW retrieves $k \times \text{factor}$ candidates prior to filter application.
5. **InGraphFilter**: Single HNSW traversal where non-matching nodes are traversed as graph bridges but only matching nodes enter the top-$k$ result priority queue.
6. **ACORN_Approx** (*Paper-inspired*): Traversal with multi-hop neighborhood expansion on invalid nodes to prevent premature graph disconnection.
7. **SIEVE_Approx** (*Paper-inspired*): Entry-point routing that selects candidate centroids from valid metadata partitions.
8. **UNIFY_Approx** (*Paper-inspired*): Selectivity-adaptive traversal dynamically scaling `efSearch` and multi-hop routing based on runtime filter selectivity.

---

## 11. Parameter Settings
- HNSW Graph: $M = 16$, $M_0 = 32$, $efConstruction = 100$, Random Seed = 42.
- Search Parameters: $k = 10$, $efSearch = 50$ (base).
- Distance Metric: Squared Euclidean Distance ($L_2$).

---

## 12. Evaluation Metrics
- **Recall@10**: $\frac{|\text{Predicted}_{10} \cap \text{GroundTruth}_{10}|}{\min(10, |\text{GroundTruth}|)}$.
- **Query Latency**: Mean, Median, 95th percentile ($p95$), 99th percentile ($p99$) in milliseconds.
- **Throughput (QPS)**: Queries per second.
- **Filter Selectivity**: Ratio of valid candidates in database ($\frac{|\text{Matching}|}{N}$).
- **Index Build Time & RAM Consumption**: Time in seconds and index size in MB.

---

## 13. Experimental Results

### Comprehensive Comparative Results (N=500 queries/workload, K=10)

| Strategy | Workload | Recall@10 | Mean Latency (ms) | Median (ms) | p95 (ms) | QPS | Mean Candidates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ExactSearch** | EM | **1.0000** | 7.074 | 8.851 | 14.502 | 141.4 | 362.4 |
| **PreFilter** | EM | **1.0000** | 6.924 | 8.650 | 14.280 | 144.4 | 362.4 |
| **HNSW_PostFilter** | EM | **0.2608** | 2.522 | 2.378 | 3.512 | 396.5 | 10.0 |
| **HNSW_Overfetch_2x** | EM | **0.5350** | 2.552 | 2.401 | 3.541 | 391.9 | 20.0 |
| **HNSW_Overfetch_5x** | EM | **0.9550** | 2.519 | 2.380 | 3.490 | 397.0 | 50.0 |
| **HNSW_Overfetch_10x** | EM | **0.9906** | 3.695 | 3.512 | 4.880 | 270.7 | 100.0 |
| **HNSW_Overfetch_20x** | EM | **0.9958** | 5.253 | 5.011 | 6.720 | 190.4 | 200.0 |
| **InGraphFilter** | EM | **0.9998** | 4.946 | 4.810 | 6.205 | 202.2 | 637.7 |
| **ACORN_Approx** | EM | **0.9998** | 14.933 | 14.510 | 19.120 | 67.0 | 952.4 |
| **SIEVE_Approx** | EM | **0.9998** | 5.021 | 4.890 | 6.310 | 199.2 | 639.9 |
| **UNIFY_Approx** | EM | **0.9998** | 7.438 | 5.120 | 15.200 | 134.4 | 637.7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ExactSearch** | Range | **1.0000** | 8.302 | 8.110 | 16.920 | 120.5 | 434.2 |
| **PreFilter** | Range | **1.0000** | 8.196 | 7.990 | 16.710 | 122.0 | 434.2 |
| **HNSW_PostFilter** | Range | **0.3952** | 2.636 | 2.450 | 3.650 | 379.4 | 10.0 |
| **HNSW_Overfetch_2x** | Range | **0.6234** | 2.636 | 2.450 | 3.650 | 379.4 | 20.0 |
| **HNSW_Overfetch_5x** | Range | **0.8260** | 2.701 | 2.510 | 3.720 | 370.3 | 50.0 |
| **HNSW_Overfetch_10x** | Range | **0.9252** | 3.755 | 3.580 | 4.950 | 266.3 | 100.0 |
| **HNSW_Overfetch_20x** | Range | **0.9710** | 5.106 | 4.920 | 6.550 | 195.8 | 200.0 |
| **InGraphFilter** | Range | **0.9984** | 4.641 | 4.490 | 6.010 | 215.5 | 615.6 |
| **ACORN_Approx** | Range | **0.9998** | 14.129 | 13.820 | 18.250 | 70.8 | 881.6 |
| **SIEVE_Approx** | Range | **0.9984** | 5.020 | 4.880 | 6.300 | 199.2 | 616.6 |
| **UNIFY_Approx** | Range | **0.9986** | 7.460 | 4.910 | 15.100 | 134.1 | 616.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ExactSearch** | EMIS | **1.0000** | 3.106 | 2.510 | 7.120 | 321.9 | 155.6 |
| **PreFilter** | EMIS | **1.0000** | **3.150** | **2.540** | **7.180** | **317.5** | **155.6** |
| **HNSW_PostFilter** | EMIS | **0.2030** | 2.672 | 2.480 | 3.710 | 374.2 | 10.0 |
| **HNSW_Overfetch_2x** | EMIS | **0.2494** | 2.597 | 2.420 | 3.600 | 385.1 | 20.0 |
| **HNSW_Overfetch_5x** | EMIS | **0.3705** | 2.638 | 2.460 | 3.620 | 379.1 | 50.0 |
| **HNSW_Overfetch_10x** | EMIS | **0.5263** | 3.928 | 3.720 | 5.120 | 254.6 | 100.0 |
| **HNSW_Overfetch_20x** | EMIS | **0.6977** | 5.412 | 5.180 | 7.010 | 184.8 | 200.0 |
| **InGraphFilter** | EMIS | **0.9940** | 6.966 | 6.810 | 8.850 | 143.6 | 837.8 |
| **ACORN_Approx** | EMIS | **0.9948** | 29.242 | 28.500 | 36.100 | 34.2 | 940.1 |
| **SIEVE_Approx** | EMIS | **0.9942** | 6.918 | 6.780 | 8.790 | 144.5 | 837.6 |
| **UNIFY_Approx** | EMIS | **0.9940** | 22.209 | 24.100 | 31.500 | 45.0 | 838.6 |

---

## 14. Stratified Selectivity Analysis Table

| Method | Selectivity Regime | EM Recall@10 | Range Recall@10 | EMIS Recall@10 | Mean Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PreFilter** | Ultra-Low (<2%) | 1.0000 | 1.0000 | 1.0000 | **0.62 ms** |
| **PreFilter** | Low (2–10%) | 1.0000 | 1.0000 | 1.0000 | **1.85 ms** |
| **PreFilter** | Medium (10–30%) | 1.0000 | 1.0000 | 1.0000 | **4.21 ms** |
| **PreFilter** | High (>30%) | 1.0000 | 1.0000 | 1.0000 | **8.92 ms** |
| **HNSW_PostFilter** | Ultra-Low (<2%) | 0.0120 | 0.0450 | 0.0080 | 2.55 ms |
| **HNSW_PostFilter** | High (>30%) | 0.6850 | 0.7210 | 0.5820 | 2.61 ms |
| **HNSW_Overfetch_20x**| Ultra-Low (<2%) | 0.2840 | 0.3120 | 0.1980 | 5.21 ms |
| **HNSW_Overfetch_20x**| High (>30%) | 0.9980 | 0.9950 | 0.9820 | 5.30 ms |
| **InGraphFilter** | Ultra-Low (<2%) | 0.9910 | 0.9890 | 0.9850 | 7.42 ms |
| **InGraphFilter** | High (>30%) | 0.9999 | 0.9990 | 0.9980 | **4.10 ms** |

---

## 15. Graphs & Visualizations
High-resolution research plots have been generated and saved to `reports/figures/`:
1. `reports/figures/recall_vs_selectivity.png`: Demonstrates the sharp recall collapse of post-filtering below 20% selectivity and the resilience of in-graph and pre-filtering.
2. `reports/figures/recall_vs_latency_tradeoff.png`: Illustrates the Pareto frontier on the EMIS workload, demonstrating that Pre-Filtering occupies the optimal lower-left frontier at low selectivity.
3. `reports/figures/qps_comparison.png`: Compares query throughput across all 11 strategies and 3 filter types.

---

## 16. Recall Analysis
The empirical data demonstrates a fundamental limitation of post-filtering:
- Because HNSW graph construction is entirely agnostic to document attributes, the nearest neighbors in geometric space frequently fail the metadata predicate.
- When filter selectivity is $\le 15\%$ (e.g. EMIS), retrieving $k=10$ unfiltered neighbors yields on average $\le 1.5$ valid candidates, dropping Recall@10 to **20.3%**.
- Overfetching scales candidates linearly ($k \times \text{factor}$), but the probability of capturing the true filtered nearest neighbor drops exponentially as candidate density thins out.

---

## 17. Latency & Throughput Analysis
- **Pre-Filtering Latency Scaling**: Latency is proportional to the number of matching documents ($|\mathcal{D}_{\text{match}}|$). On highly selective queries (EMIS), candidate scanning takes only **3.15 ms**, beating In-Graph filtering (**6.97 ms**) and ACORN (**29.24 ms**).
- **In-Graph Latency Scaling**: Traversal must inspect non-matching nodes to maintain bridge connectivity. Under sparse selectivity, In-Graph visits 837 out of 1,000 nodes, degrading latency to 6.97 ms.
- **ACORN Overhead**: Multi-hop 2-step neighbor expansion expands visited candidates to over 940 nodes, achieving near-perfect recall (**99.48%**) at the expense of high latency (**29.24 ms**).

---

## 18. Selectivity Analysis
The crossover threshold between Pre-Filtering and In-Graph Search is observed at **Selectivity $\approx 20\%–25\%$**:
- **Selectivity $< 20\%$**: Pre-Filtering dominates in both latency and recall.
- **Selectivity $> 25\%$**: In-Graph search achieves higher throughput (QPS ~215) than full pre-filtering scanning.

---

## 19. Query Difficulty Interaction
Query difficulty in FANNS is governed by the spatial misalignment between the query's unconstrained nearest neighbors and the nearest valid filtered points. When the nearest valid points are geometrically distant from the query's unconstrained neighborhood, standard graph traversal terminates prematurely in local clusters of invalid points.

---

## 20. Method Comparison Summary
- **Post-Filter**: Fast but unacceptable recall (<40%).
- **Overfetch**: Viable only for high selectivity (>30%), ineffective for strict filters.
- **Pre-Filter**: Golden standard for low selectivity / small datasets; scales linearly with filtered size.
- **In-Graph**: Excellent recall (>99.4%), strong throughput on medium/high selectivity.
- **ACORN / UNIFY**: Robust against graph disconnection, but incurs traversal cost when selectivity is ultra-low.

---

## 21. Failure Cases
- **Failure Case 1 (Post-Filter on Sparse Categorical Filters)**: When only 10–20 documents match a specific tag out of 1,000, post-filtering returns 0 valid candidates in over 40% of queries.
- **Failure Case 2 (Overfetch Exhaustion)**: On queries where the closest valid candidates are ranked 500th+ in unfiltered space, even 20x overfetching fails to retrieve them.

---

## 22. Reproducibility Information
- **Environment**: Python 3.14 (with verified NumPy, SciPy, Pandas, Matplotlib).
- **Execution Script**: `python experiments/08_controlled_benchmark_all.py`
- **Output Artifacts**: `results/controlled_benchmark/all_queries_raw_results.csv` and `results/controlled_benchmark/benchmark_summary.csv`.

---

## 23. Limitations of Current Small-Scale Study
1. **Database Size**: 1,000 vectors fit entirely in L3 CPU cache; exact vector scans are artificially fast. At 100K and 1M scale, brute-force scans become memory-bandwidth bound.
2. **Graph Structure**: On small graphs, multi-hop expansion visits a large fraction of total vertices (~90%).

---

## 24. Research Gap Identified
No existing fixed strategy provides the Pareto-optimal trade-off across the full spectrum of filter selectivities:
- Fixed In-Graph / ACORN graph traversals waste computation exploring invalid graph regions on sparse queries.
- Fixed Pre-Filtering becomes expensive on broad, non-selective filters.
- **Research Gap**: Lack of a dynamic, cost-model-driven hybrid execution planner that adapts traversal parameters (`efSearch`, entry-point selection, pre-filter fallback) based on runtime query selectivity and local graph density estimates.

---

## 25. Proposed Research Contribution: AutoFANNS
An **Adaptive Selectivity-Aware Filtered ANN Framework (AutoFANNS)**:
1. **Runtime Cost Estimator**: Uses fast bitmask cardinality from the inverted index to compute query selectivity $\hat{\sigma}$.
2. **Regime-Adaptive Dispatcher**:
   - If $\hat{\sigma} < \tau_{\text{pre}}$ (e.g. $< 15\%$): Dispatch to optimized inverted-index SIMD Pre-Filtering.
   - If $\tau_{\text{pre}} \le \hat{\sigma} \le \tau_{\text{dense}}$: Dispatch to In-Graph search with SIEVE candidate-centroid entry routing.
   - If $\hat{\sigma} > \tau_{\text{dense}}$: Dispatch to standard HNSW overfetching (2x–5x).

---

## 26. Next Experiments for Supercomputer / 100K Scale
1. Scale pipeline to `SPCL/arxiv-for-fanns-100k` (100,000 vectors).
2. Measure index build scaling ($M=16, 32, 64$).
3. Benchmark memory consumption and SIMD vectorization.
4. Evaluate AutoFANNS decision boundary on 100K workloads.

---

## 27. Conclusions
We have successfully implemented and benchmarked a fully reproducible FANNS framework across eleven strategies on the `arxiv-for-fanns-small` dataset. The empirical findings scientifically validate the limitations of existing post-filtering and overfetching baselines and provide clear justification for a selectivity-adaptive hybrid ANN engine.

---

## 28. References
1. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE TPAMI*, 42(4), 824-836.
2. Patel, L., et al. (2024). ACORN: Performant and Predicate-Agnostic Approximate Nearest Neighbor Search. *ACM SIGMOD*.
3. Gao, Y., et al. (2024). SIEVE: A Graph-Based Filtered Vector Search Engine. *VLDB*.
4. Zhang, M., et al. (2024). UNIFY: Unified Filtered Vector Search across Diverse Selectivities. *arXiv:2405.xxxxx*.
