"""
Filter representation and metadata indexing for FANNS.
Provides exact predicate evaluation, inverted indices, and boolean bitsets for high-performance filtering.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np


class FilterPredicate(ABC):
    """Abstract base class for all metadata filter predicates."""

    @abstractmethod
    def matches(self, doc_metadata: Dict[str, Any]) -> bool:
        """Returns True if the document metadata satisfies the filter predicate."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Returns serializable dictionary representation."""
        pass


class EMFilterPredicate(FilterPredicate):
    """
    Exact Match (EM) filter:
    Matches documents where 'number_of_sub_categories' equals target integer label.
    """
    def __init__(self, target_label: int, field: str = "number_of_sub_categories"):
        self.target_label = int(target_label)
        self.field = field

    def matches(self, doc_metadata: Dict[str, Any]) -> bool:
        return doc_metadata.get(self.field) == self.target_label

    def to_dict(self) -> Dict[str, Any]:
        return {"filter_type": "EM", "field": self.field, "label": self.target_label}

    def __repr__(self) -> str:
        return f"EMFilter({self.field} == {self.target_label})"


class RangeFilterPredicate(FilterPredicate):
    """
    Range (R) filter:
    Matches documents where 'update_date' is within [range_start, range_end] (inclusive).
    """
    def __init__(self, range_start: int, range_end: int, field: str = "update_date"):
        self.range_start = int(range_start)
        self.range_end = int(range_end)
        self.field = field

    def matches(self, doc_metadata: Dict[str, Any]) -> bool:
        val = doc_metadata.get(self.field)
        if val is None:
            return False
        return self.range_start <= val <= self.range_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filter_type": "Range",
            "field": self.field,
            "range_start": self.range_start,
            "range_end": self.range_end,
        }

    def __repr__(self) -> str:
        return f"RangeFilter({self.range_start} <= {self.field} <= {self.range_end})"


class EMISFilterPredicate(FilterPredicate):
    """
    Exact Match In Set (EMIS) filter:
    Matches documents where target string label is in 'main_categories' list.
    """
    def __init__(self, target_label: str, field: str = "main_categories"):
        self.target_label = str(target_label)
        self.field = field

    def matches(self, doc_metadata: Dict[str, Any]) -> bool:
        categories = doc_metadata.get(self.field)
        if categories is None:
            return False
        if isinstance(categories, (list, tuple, set)):
            return self.target_label in categories
        return self.target_label == categories

    def to_dict(self) -> Dict[str, Any]:
        return {"filter_type": "EMIS", "field": self.field, "label": self.target_label}

    def __repr__(self) -> str:
        return f"EMISFilter('{self.target_label}' in {self.field})"


def parse_query_filter(filter_type: str, query_attribute: Optional[Dict[str, Any]]) -> Optional[FilterPredicate]:
    """Factory creating FilterPredicate instance from query attribute dict."""
    if query_attribute is None or filter_type.lower() == "unfiltered":
        return None

    if filter_type == "EM":
        return EMFilterPredicate(target_label=query_attribute["label"])
    elif filter_type == "Range":
        return RangeFilterPredicate(
            range_start=query_attribute["range_start"],
            range_end=query_attribute["range_end"]
        )
    elif filter_type == "EMIS":
        return EMISFilterPredicate(target_label=query_attribute["label"])
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


class MetadataIndex:
    """
    High-performance inverted metadata index for fast bitmap candidate generation.
    Pre-indexes database attributes to avoid linear per-document dict scans during search.
    """
    def __init__(self, database_attributes: List[Dict[str, Any]]):
        self.num_docs = len(database_attributes)
        self.raw_attributes = database_attributes

        # Inverted index for EM (number_of_sub_categories -> list of doc_ids)
        self.em_index: Dict[int, np.ndarray] = {}
        em_temp: Dict[int, List[int]] = {}

        # Inverted index for EMIS (category_str -> list of doc_ids)
        self.emis_index: Dict[str, np.ndarray] = {}
        emis_temp: Dict[str, List[int]] = {}

        # Columnar array for Range filter (update_date per doc)
        self.update_dates = np.zeros(self.num_docs, dtype=np.int64)

        for doc_id, attr in enumerate(database_attributes):
            # EM
            n_sub = attr.get("number_of_sub_categories")
            if n_sub is not None:
                em_temp.setdefault(n_sub, []).append(doc_id)

            # EMIS
            cats = attr.get("main_categories", [])
            if isinstance(cats, (list, tuple, set)):
                for c in cats:
                    emis_temp.setdefault(str(c), []).append(doc_id)
            elif cats:
                emis_temp.setdefault(str(cats), []).append(doc_id)

            # Range
            u_date = attr.get("update_date", -1)
            self.update_dates[doc_id] = u_date

        for k, v in em_temp.items():
            self.em_index[k] = np.array(v, dtype=np.int32)

        for k, v in emis_temp.items():
            self.emis_index[k] = np.array(v, dtype=np.int32)

    def get_matching_ids(self, predicate: Optional[FilterPredicate]) -> np.ndarray:
        """Returns sorted array of matching document IDs."""
        if predicate is None:
            return np.arange(self.num_docs, dtype=np.int32)

        if isinstance(predicate, EMFilterPredicate) and predicate.field == "number_of_sub_categories":
            return self.em_index.get(predicate.target_label, np.empty(0, dtype=np.int32))

        if isinstance(predicate, EMISFilterPredicate) and predicate.field == "main_categories":
            return self.emis_index.get(predicate.target_label, np.empty(0, dtype=np.int32))

        if isinstance(predicate, RangeFilterPredicate) and predicate.field == "update_date":
            mask = (self.update_dates >= predicate.range_start) & (self.update_dates <= predicate.range_end)
            return np.flatnonzero(mask).astype(np.int32)

        # Fallback linear scan
        matching = [i for i, attr in enumerate(self.raw_attributes) if predicate.matches(attr)]
        return np.array(matching, dtype=np.int32)

    def get_boolean_mask(self, predicate: Optional[FilterPredicate]) -> np.ndarray:
        """Returns boolean mask of length num_docs."""
        mask = np.zeros(self.num_docs, dtype=bool)
        if predicate is None:
            mask.fill(True)
            return mask

        matching_ids = self.get_matching_ids(predicate)
        if len(matching_ids) > 0:
            mask[matching_ids] = True
        return mask

    def get_selectivity(self, predicate: Optional[FilterPredicate]) -> float:
        """Returns selectivity fraction (matching / total)."""
        if predicate is None:
            return 1.0
        matching = len(self.get_matching_ids(predicate))
        return float(matching) / float(self.num_docs) if self.num_docs > 0 else 0.0
