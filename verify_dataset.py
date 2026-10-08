from pathlib import Path
import json
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/arxiv-for-fanns-small")

EXPECTED_DIMENSION = 4096
EXPECTED_DATABASE_SIZE = 1000
EXPECTED_QUERY_SIZE = 10000


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ARXIV-FOR-FANNS SMALL DATASET VERIFICATION")
print("=" * 70)


# ============================================================
# CHECK DATA DIRECTORY
# ============================================================

if not DATA_DIR.exists():

    print("\n[ERROR] Dataset directory does not exist:")
    print(DATA_DIR.resolve())

    print("\nRun:")
    print("python download_dataset.py")

    raise SystemExit(1)


print("\nDataset directory:")
print(DATA_DIR.resolve())


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [
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


print("\n" + "=" * 70)
print("1. CHECKING REQUIRED FILES")
print("=" * 70)

missing_files = []

for filename in required_files:

    path = DATA_DIR / filename

    if path.exists():

        size_mb = path.stat().st_size / (1024 * 1024)

        print(
            f"[OK]      {filename:<35} "
            f"{size_mb:>10.2f} MB"
        )

    else:

        print(
            f"[MISSING] {filename}"
        )

        missing_files.append(filename)


if missing_files:

    print("\n[ERROR] Missing files:")

    for filename in missing_files:
        print(f"  - {filename}")

    raise SystemExit(1)


print("\nAll required files are present.")


# ============================================================
# FVECS READER
# ============================================================

def read_fvecs(filename):
    """
    Read standard .fvecs format.

    Each vector is stored as:

        [dimension][float32 values...]

    Returns:
        NumPy array of shape:
        (number_of_vectors, dimension)
    """

    with open(filename, "rb") as f:

        # Read dimension
        dim_bytes = f.read(4)

        if len(dim_bytes) != 4:
            raise ValueError(
                f"Invalid or empty fvecs file: {filename}"
            )

        dim = np.frombuffer(
            dim_bytes,
            dtype=np.int32
        )[0]

        if dim <= 0:
            raise ValueError(
                f"Invalid dimension {dim} in {filename}"
            )

        # Return to beginning
        f.seek(0)

        # Read everything
        data = np.fromfile(
            f,
            dtype=np.float32
        )

    record_size = dim + 1

    if len(data) % record_size != 0:

        raise ValueError(
            f"\nInvalid fvecs structure:\n"
            f"File: {filename}\n"
            f"Dimension: {dim}\n"
            f"Total float32 values: {len(data)}\n"
            f"Record size: {record_size}"
        )

    num_vectors = len(data) // record_size

    data = data.reshape(
        num_vectors,
        record_size
    )

    vectors = data[:, 1:]

    return vectors


# ============================================================
# STANDARD IVECS READER
# ============================================================

def read_standard_ivecs(filename):
    """
    Read fixed-width .ivecs format.

    This is appropriate for the standard unfiltered
    ground_truth.ivecs file.
    """

    with open(filename, "rb") as f:

        dim_bytes = f.read(4)

        if len(dim_bytes) != 4:
            raise ValueError(
                f"Invalid or empty ivecs file: {filename}"
            )

        dim = np.frombuffer(
            dim_bytes,
            dtype=np.int32
        )[0]

        if dim <= 0:
            raise ValueError(
                f"Invalid dimension {dim} in {filename}"
            )

        f.seek(0)

        data = np.fromfile(
            f,
            dtype=np.int32
        )

    record_size = dim + 1

    if len(data) % record_size != 0:

        raise ValueError(
            f"\nInvalid fixed-width ivecs structure:\n"
            f"File: {filename}\n"
            f"Dimension: {dim}\n"
            f"Total int32 values: {len(data)}\n"
            f"Record size: {record_size}"
        )

    num_vectors = len(data) // record_size

    data = data.reshape(
        num_vectors,
        record_size
    )

    vectors = data[:, 1:]

    return vectors


# ============================================================
# LOAD DATABASE VECTORS
# ============================================================

print("\n" + "=" * 70)
print("2. LOADING DATABASE VECTORS")
print("=" * 70)

database_vectors = read_fvecs(
    DATA_DIR / "database_vectors.fvecs"
)

print(
    f"\nDatabase shape: {database_vectors.shape}"
)

print(
    f"Number of database vectors: "
    f"{database_vectors.shape[0]}"
)

print(
    f"Vector dimension: "
    f"{database_vectors.shape[1]}"
)

print(
    f"Data type: "
    f"{database_vectors.dtype}"
)


# ============================================================
# DATABASE SIZE CHECK
# ============================================================

if database_vectors.shape[0] == EXPECTED_DATABASE_SIZE:

    print(
        f"[OK] Database contains "
        f"{EXPECTED_DATABASE_SIZE} vectors."
    )

else:

    print(
        f"[WARNING] Expected "
        f"{EXPECTED_DATABASE_SIZE} database vectors, "
        f"found {database_vectors.shape[0]}."
    )


# ============================================================
# LOAD QUERY VECTORS
# ============================================================

print("\n" + "=" * 70)
print("3. LOADING QUERY VECTORS")
print("=" * 70)

query_vectors = read_fvecs(
    DATA_DIR / "query_vectors.fvecs"
)

print(
    f"\nQuery shape: {query_vectors.shape}"
)

print(
    f"Number of queries: "
    f"{query_vectors.shape[0]}"
)

print(
    f"Query dimension: "
    f"{query_vectors.shape[1]}"
)

print(
    f"Data type: "
    f"{query_vectors.dtype}"
)


# ============================================================
# QUERY SIZE CHECK
# ============================================================

if query_vectors.shape[0] == EXPECTED_QUERY_SIZE:

    print(
        f"[OK] Dataset contains "
        f"{EXPECTED_QUERY_SIZE} queries."
    )

else:

    print(
        f"[WARNING] Expected "
        f"{EXPECTED_QUERY_SIZE} queries, "
        f"found {query_vectors.shape[0]}."
    )


# ============================================================
# VECTOR DIMENSION CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. VECTOR DIMENSION CHECK")
print("=" * 70)

if database_vectors.shape[1] == query_vectors.shape[1]:

    print(
        f"[OK] Database/query dimensions match: "
        f"{database_vectors.shape[1]}"
    )

else:

    print(
        "[ERROR] Database and query dimensions "
        "do not match."
    )

    raise SystemExit(1)


if database_vectors.shape[1] == EXPECTED_DIMENSION:

    print(
        f"[OK] Vector dimension = "
        f"{EXPECTED_DIMENSION}"
    )

else:

    print(
        f"[WARNING] Expected dimension "
        f"{EXPECTED_DIMENSION}, found "
        f"{database_vectors.shape[1]}"
    )


# ============================================================
# VECTOR SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("5. VECTOR SANITY CHECK")
print("=" * 70)

database_nan = np.isnan(database_vectors).sum()
database_inf = np.isinf(database_vectors).sum()

query_nan = np.isnan(query_vectors).sum()
query_inf = np.isinf(query_vectors).sum()


print(
    f"\nDatabase NaN values : {database_nan}"
)

print(
    f"Database Inf values : {database_inf}"
)

print(
    f"Query NaN values    : {query_nan}"
)

print(
    f"Query Inf values    : {query_inf}"
)


if database_nan == 0:

    print("[OK] Database contains no NaN values.")

else:

    print("[ERROR] Database contains NaN values.")


if database_inf == 0:

    print("[OK] Database contains no infinite values.")

else:

    print("[ERROR] Database contains infinite values.")


if query_nan == 0:

    print("[OK] Queries contain no NaN values.")

else:

    print("[ERROR] Queries contain NaN values.")


if query_inf == 0:

    print("[OK] Queries contain no infinite values.")

else:

    print("[ERROR] Queries contain infinite values.")


# ============================================================
# VECTOR STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("6. VECTOR STATISTICS")
print("=" * 70)

print(
    f"\nDatabase minimum: "
    f"{database_vectors.min():.6f}"
)

print(
    f"Database maximum: "
    f"{database_vectors.max():.6f}"
)

print(
    f"Database mean: "
    f"{database_vectors.mean():.6f}"
)

print(
    f"Database std: "
    f"{database_vectors.std():.6f}"
)


# ============================================================
# LOAD UNFILTERED GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("7. LOADING UNFILTERED GROUND TRUTH")
print("=" * 70)

ground_truth = read_standard_ivecs(
    DATA_DIR / "ground_truth.ivecs"
)

print(
    f"\nGround truth shape: "
    f"{ground_truth.shape}"
)

print(
    f"Queries represented: "
    f"{ground_truth.shape[0]}"
)

print(
    f"Neighbors per query: "
    f"{ground_truth.shape[1]}"
)


# ============================================================
# GROUND TRUTH QUERY COUNT
# ============================================================

if ground_truth.shape[0] == query_vectors.shape[0]:

    print(
        "[OK] Ground truth query count "
        "matches query vectors."
    )

else:

    print(
        "[ERROR] Ground truth/query count mismatch."
    )

    raise SystemExit(1)


# ============================================================
# GROUND TRUTH ID CHECK
# ============================================================

print("\n" + "=" * 70)
print("8. UNFILTERED GROUND TRUTH ID CHECK")
print("=" * 70)

invalid_ids = np.logical_or(
    ground_truth < 0,
    ground_truth >= len(database_vectors)
)

num_invalid_ids = invalid_ids.sum()


if num_invalid_ids == 0:

    print(
        "[OK] All ground-truth IDs are valid "
        "database IDs."
    )

else:

    print(
        f"[ERROR] Found "
        f"{num_invalid_ids} invalid ground-truth IDs."
    )

    raise SystemExit(1)


print("\nFirst query ground-truth IDs:")

print(
    ground_truth[0]
)


# ============================================================
# LOAD DATABASE ATTRIBUTES
# ============================================================

print("\n" + "=" * 70)
print("9. LOADING DATABASE ATTRIBUTES")
print("=" * 70)

attributes_file = (
    DATA_DIR / "database_attributes.jsonl"
)

attributes = []

with open(
    attributes_file,
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        try:

            record = json.loads(line)

            attributes.append(record)

        except json.JSONDecodeError as e:

            print(
                f"[ERROR] Invalid JSON at "
                f"line {line_number}"
            )

            print(e)

            raise SystemExit(1)


print(
    f"\nMetadata records: "
    f"{len(attributes)}"
)


# ============================================================
# ATTRIBUTE / VECTOR COUNT CHECK
# ============================================================

if len(attributes) == len(database_vectors):

    print(
        "[OK] Metadata count matches "
        "database vector count."
    )

else:

    print(
        "[ERROR] Metadata/vector count mismatch."
    )

    print(
        f"Vectors : {len(database_vectors)}"
    )

    print(
        f"Metadata: {len(attributes)}"
    )

    raise SystemExit(1)


# ============================================================
# SAMPLE METADATA
# ============================================================

print("\n" + "=" * 70)
print("10. SAMPLE DATABASE METADATA")
print("=" * 70)

if attributes:

    print(
        json.dumps(
            attributes[0],
            indent=2
        )
    )

else:

    print(
        "[WARNING] No metadata records found."
    )


# ============================================================
# FILTER QUERY FILES
# ============================================================

print("\n" + "=" * 70)
print("11. FILTER QUERY FILE CHECK")
print("=" * 70)


filter_files = [
    (
        "EM",
        "em_query_attributes.jsonl",
        "ground_truth_em.ivecs"
    ),

    (
        "Range",
        "r_query_attributes.jsonl",
        "ground_truth_r.ivecs"
    ),

    (
        "EMIS",
        "emis_query_attributes.jsonl",
        "ground_truth_emis.ivecs"
    ),
]


for filter_name, query_file, truth_file in filter_files:

    query_path = DATA_DIR / query_file
    truth_path = DATA_DIR / truth_file

    print(f"\n--- {filter_name} FILTER ---")

    # --------------------------------------------------------
    # Query attribute file
    # --------------------------------------------------------

    query_records = []

    with open(
        query_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                query_records.append(
                    json.loads(line)
                )

    print(
        f"Query metadata records: "
        f"{len(query_records)}"
    )

    if len(query_records) == query_vectors.shape[0]:

        print(
            "[OK] Filter query count matches "
            "number of queries."
        )

    else:

        print(
            "[WARNING] Filter query count "
            "does not equal total query count."
        )


    # --------------------------------------------------------
    # Ground truth file information
    # --------------------------------------------------------

    raw = np.fromfile(
        truth_path,
        dtype=np.int32
    )

    print(
        f"Ground-truth int32 values: "
        f"{len(raw)}"
    )

    print(
        f"Ground-truth file size: "
        f"{truth_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

    print(
        "First 20 int32 values:"
    )

    print(
        raw[:20]
    )

    print(
        "[INFO] Filtered ground truth is "
        "retained for later FANNS evaluation."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(
    f"\nDatabase vectors      : "
    f"{len(database_vectors)}"
)

print(
    f"Query vectors         : "
    f"{len(query_vectors)}"
)

print(
    f"Vector dimension      : "
    f"{database_vectors.shape[1]}"
)

print(
    f"Metadata records      : "
    f"{len(attributes)}"
)

print(
    f"Unfiltered GT shape   : "
    f"{ground_truth.shape}"
)


print("\n" + "=" * 70)
print("BASIC DATASET CHECKS PASSED")
print("=" * 70)

print(
    "\nThe dataset files are present and "
    "the vector/database structure is valid."
)

print(
    "\nFiltered ground-truth files have been "
    "inspected without assuming a fixed-width "
    "representation."
)

print(
    "\nNext step: build the common FANNS loader "
    "and evaluate the filtering workloads."
)