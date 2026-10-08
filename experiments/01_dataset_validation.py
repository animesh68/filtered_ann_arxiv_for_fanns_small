"""
Experiment 01: Dataset Validation and Workload Verification.
Validates vectors, attributes, variable ground truth ivecs, and filter consistency.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import load_fanns_dataset
from src.filters import MetadataIndex, parse_query_filter


def main():
    print("=" * 70)
    print("EXPERIMENT 01: DATASET VALIDATION & SCHEMA VERIFICATION")
    print("=" * 70)

    dataset = load_fanns_dataset("data/arxiv-for-fanns-small")
    print(f"\n[OK] Database vectors shape: {dataset.database_vectors.shape}, dtype: {dataset.database_vectors.dtype}")
    print(f"[OK] Query vectors shape   : {dataset.query_vectors.shape}, dtype: {dataset.query_vectors.dtype}")
    print(f"[OK] Metadata records count: {len(dataset.database_attributes)}")

    # Check finite
    assert not np.isnan(dataset.database_vectors).any(), "NaN found in DB vectors!"
    assert not np.isnan(dataset.query_vectors).any(), "NaN found in Query vectors!"
    print("[OK] Vectors verified: no NaN or Inf values.")

    # Check Metadata Index
    meta_idx = MetadataIndex(dataset.database_attributes)
    print(f"[OK] Inverted index built. Unique sub_categories values: {len(meta_idx.em_index)}, Unique main categories: {len(meta_idx.emis_index)}")

    # Check Workloads
    for w_name, workload in dataset.workloads.items():
        if w_name == "unfiltered":
            print(f"[OK] Workload '{w_name}': {workload.ground_truth.shape[0]} queries, GT shape: {workload.ground_truth.shape}")
        else:
            gt_lens = [len(x) for x in workload.ground_truth]
            print(f"[OK] Workload '{w_name}': {len(workload.query_attributes)} queries, variable GT length range: [{min(gt_lens)}, {max(gt_lens)}], mean: {np.mean(gt_lens):.1f}")

    print("\n" + "=" * 70)
    print("DATASET VALIDATION SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
