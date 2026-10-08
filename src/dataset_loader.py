"""
Dataset loader for Filtered Approximate Nearest Neighbor Search (FANNS).
Supports SPCL/arxiv-for-fanns standard benchmark format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np


def read_fvecs(filename: Union[str, Path]) -> np.ndarray:
    """
    Read standard binary .fvecs vector file.
    Each vector is stored as [dim (int32), components (float32 * dim)].
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"fvecs file not found: {filename}")

    with open(filename, "rb") as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError(f"Empty or corrupted fvecs file: {filename}")
        dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim} in {filename}")
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)

    record_size = dim + 1
    if len(data) % record_size != 0:
        raise ValueError(
            f"Corrupted fvecs file: total elements {len(data)} not divisible by record size {record_size}"
        )
    num_vectors = len(data) // record_size
    reshaped = data.reshape(num_vectors, record_size)
    return np.ascontiguousarray(reshaped[:, 1:], dtype=np.float32)


def read_standard_ivecs(filename: Union[str, Path]) -> np.ndarray:
    """
    Read fixed-width binary .ivecs file (e.g. unfiltered ground truth).
    Each row is stored as [k (int32), indices (int32 * k)].
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"ivecs file not found: {filename}")

    with open(filename, "rb") as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError(f"Empty or corrupted ivecs file: {filename}")
        dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        if dim <= 0:
            raise ValueError(f"Invalid dimension {dim} in {filename}")
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)

    record_size = dim + 1
    if len(data) % record_size != 0:
        raise ValueError(
            f"Corrupted fixed-width ivecs file: total elements {len(data)} not divisible by {record_size}"
        )
    num_vectors = len(data) // record_size
    reshaped = data.reshape(num_vectors, record_size)
    return np.ascontiguousarray(reshaped[:, 1:], dtype=np.int32)


def read_variable_ivecs(filename: Union[str, Path]) -> List[np.ndarray]:
    """
    Read variable-width binary .ivecs file (e.g. filtered ground truths where candidate counts vary).
    Reads length prefix per row followed by that exact number of int32 values.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"variable ivecs file not found: {filename}")

    results: List[np.ndarray] = []
    with open(filename, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes or len(dim_bytes) < 4:
                break
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            if dim == 0:
                results.append(np.empty(0, dtype=np.int32))
            else:
                data_bytes = f.read(dim * 4)
                if len(data_bytes) < dim * 4:
                    raise ValueError(f"Unexpected EOF while reading record in {filename}")
                arr = np.frombuffer(data_bytes, dtype=np.int32)
                results.append(np.ascontiguousarray(arr, dtype=np.int32))
    return results


def load_jsonl(filename: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load JSONL metadata records."""
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"JSONL file not found: {filename}")

    records = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass
class FANNSWorkload:
    """Encapsulates a specific filter query workload with corresponding ground truth."""
    name: str
    filter_type: str  # 'unfiltered', 'EM', 'Range', 'EMIS'
    query_attributes: Optional[List[Dict[str, Any]]]
    ground_truth: Union[np.ndarray, List[np.ndarray]]


@dataclass
class FANNSDataset:
    """Comprehensive dataset representation containing vectors, metadata, and all query workloads."""
    data_dir: Path
    database_vectors: np.ndarray
    query_vectors: np.ndarray
    database_attributes: List[Dict[str, Any]]
    workloads: Dict[str, FANNSWorkload]

    @property
    def num_database(self) -> int:
        return self.database_vectors.shape[0]

    @property
    def num_queries(self) -> int:
        return self.query_vectors.shape[0]

    @property
    def dimension(self) -> int:
        return self.database_vectors.shape[1]

    def get_workload(self, name: str) -> FANNSWorkload:
        if name not in self.workloads:
            raise KeyError(f"Workload '{name}' not found. Available: {list(self.workloads.keys())}")
        return self.workloads[name]


def load_fanns_dataset(data_dir: Union[str, Path] = "data/arxiv-for-fanns-small") -> FANNSDataset:
    """
    Loads all vectors, attributes, and query workloads for the FANNS benchmark dataset.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path.resolve()}")

    db_vectors = read_fvecs(path / "database_vectors.fvecs")
    query_vectors = read_fvecs(path / "query_vectors.fvecs")
    db_attributes = load_jsonl(path / "database_attributes.jsonl")

    # Unfiltered GT
    gt_unfiltered = read_standard_ivecs(path / "ground_truth.ivecs")

    # Filtered Workloads
    em_attrs = load_jsonl(path / "em_query_attributes.jsonl")
    gt_em = read_variable_ivecs(path / "ground_truth_em.ivecs")

    r_attrs = load_jsonl(path / "r_query_attributes.jsonl")
    gt_r = read_variable_ivecs(path / "ground_truth_r.ivecs")

    emis_attrs = load_jsonl(path / "emis_query_attributes.jsonl")
    gt_emis = read_variable_ivecs(path / "ground_truth_emis.ivecs")

    workloads = {
        "unfiltered": FANNSWorkload("unfiltered", "unfiltered", None, gt_unfiltered),
        "EM": FANNSWorkload("EM", "EM", em_attrs, gt_em),
        "Range": FANNSWorkload("Range", "Range", r_attrs, gt_r),
        "EMIS": FANNSWorkload("EMIS", "EMIS", emis_attrs, gt_emis),
    }

    return FANNSDataset(
        data_dir=path,
        database_vectors=db_vectors,
        query_vectors=query_vectors,
        database_attributes=db_attributes,
        workloads=workloads,
    )
