"""
Experiment 09: Visualization and Publication Figures Generator.
Produces high-resolution figures and markdown tables from controlled benchmark results.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

RESULTS_DIR = Path("results/controlled_benchmark")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_recall_vs_selectivity(df_raw: pd.DataFrame):
    """Plot Recall@10 vs Filter Selectivity across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    filter_types = ["EM", "Range", "EMIS"]
    
    methods_to_plot = [
        ("ExactSearch", "#2ca02c", "-", "o"),
        ("PreFilter", "#1f77b4", "-", "s"),
        ("HNSW_PostFilter", "#d62728", "--", "x"),
        ("HNSW_Overfetch_5x", "#ff7f0e", "--", "^"),
        ("HNSW_Overfetch_20x", "#9467bd", "-.", "v"),
        ("InGraphFilter", "#8c564b", "-", "d"),
        ("UNIFY_Approx", "#e377c2", "-", "*"),
    ]

    for ax, ftype in zip(axes, filter_types):
        sub_df = df_raw[df_raw["filter_type"] == ftype]
        for method_name, color, ls, marker in methods_to_plot:
            m_sub = sub_df[sub_df["method"] == method_name]
            if len(m_sub) == 0:
                continue
            
            # Bin selectivity into quantiles/buckets
            m_sub["sel_bin"] = pd.qcut(m_sub["selectivity"], q=5, duplicates="drop")
            binned = m_sub.groupby("sel_bin", observed=True).agg(
                sel_mean=("selectivity", "mean"),
                rec_mean=("recall_at_k", "mean"),
            ).sort_values("sel_mean")
            
            ax.plot(binned["sel_mean"], binned["rec_mean"], label=method_name,
                    color=color, linestyle=ls, marker=marker, linewidth=2, markersize=7)

        ax.set_title(f"Workload: {ftype}")
        ax.set_xlabel("Filter Selectivity (Matching Ratio)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Recall@10")
    axes[0].legend(loc="lower right", framealpha=0.9)
    plt.suptitle("Recall@10 vs Filter Selectivity across Filter Workloads", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "recall_vs_selectivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {FIGURES_DIR / 'recall_vs_selectivity.png'}")


def plot_recall_vs_latency_tradeoff(df_summary: pd.DataFrame):
    """Plot Recall vs Query Latency Pareto Curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter by filter_type == EMIS (toughest filtering workload)
    emis_df = df_summary[df_summary["filter_type"] == "EMIS"].sort_values("mean_latency_ms")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(emis_df)))
    for idx, (_, row) in enumerate(emis_df.iterrows()):
        ax.scatter(row["mean_latency_ms"], row["mean_recall"], color=colors[idx], s=120, zorder=5)
        ax.annotate(
            row["method"],
            (row["mean_latency_ms"], row["mean_recall"]),
            textcoords="offset points",
            xytext=(7, -2),
            fontsize=9,
            weight="bold" if "PreFilter" in row["method"] or "InGraph" in row["method"] else "normal"
        )

    ax.set_xlabel("Mean Query Latency (ms)")
    ax.set_ylabel("Mean Recall@10")
    ax.set_title("Recall@10 vs Query Latency Trade-off (EMIS Workload)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "recall_vs_latency_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {FIGURES_DIR / 'recall_vs_latency_tradeoff.png'}")


def plot_qps_and_latency_comparison(df_summary: pd.DataFrame):
    """Plot QPS across methods for EM, Range, and EMIS."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_qps = df_summary.pivot(index="method", columns="filter_type", values="qps")
    pivot_qps.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    
    ax.set_title("Query Throughput (QPS) Comparison Across Workloads")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_xlabel("Search Strategy")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "qps_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {FIGURES_DIR / 'qps_comparison.png'}")


def generate_all_plots():
    df_raw = pd.read_csv(RESULTS_DIR / "all_queries_raw_results.csv")
    df_summary = pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")
    
    plot_recall_vs_selectivity(df_raw)
    plot_recall_vs_latency_tradeoff(df_summary)
    plot_qps_and_latency_comparison(df_summary)
    print("\n[OK] All publication figures generated successfully.")


if __name__ == "__main__":
    generate_all_plots()
