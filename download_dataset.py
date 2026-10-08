from huggingface_hub import hf_hub_download
from pathlib import Path

DATASET_ID = "SPCL/arxiv-for-fanns-small"

OUTPUT_DIR = Path("data/arxiv-for-fanns-small")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "database_vectors.fvecs",
    "database_attributes.jsonl",
    "query_vectors.fvecs",
    "ground_truth.ivecs",
    "em_query_attributes.jsonl",
    "ground_truth_em.ivecs",
    "r_query_attributes.jsonl",
    "ground_truth_r.ivecs",
    "emis_query_attributes.jsonl",
    "ground_truth_emis.ivecs",
]

print("=" * 60)
print("ARXIV-FOR-FANNS SMALL DATASET DOWNLOADER")
print("=" * 60)

print(f"\nDataset: {DATASET_ID}")
print(f"Output:  {OUTPUT_DIR.resolve()}")

for filename in FILES:

    print("\n" + "-" * 60)
    print(f"Downloading: {filename}")

    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(OUTPUT_DIR),
    )

    print(f"Saved: {path}")

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print("\nFiles downloaded:")

for file in sorted(OUTPUT_DIR.iterdir()):
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"{file.name:<35} {size_mb:>10.2f} MB")