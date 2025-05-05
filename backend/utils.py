# backend/utils.py
import logging
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

# Example Utility Function: Safely get value from nested dict
def safe_get(data: dict, keys: list, default: Any = None) -> Any:
    """Safely get a value from a nested dictionary."""
    d = data
    try:
        for key in keys:
            d = d[key]
        return d
    except (KeyError, TypeError, IndexError):
        return default

# Example Utility Function: Basic data type inference beyond pandas
def infer_semantic_type(series: pd.Series) -> str:
    """Infers a more semantic type for a pandas Series (basic example)."""
    if pd.api.types.is_numeric_dtype(series):
        # Could add checks for integer vs float, or if it looks like an ID
        if series.nunique() < 20: # Arbitrary threshold for potential category ID
             return "numeric_id_candidate"
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        # Further checks for object type
        unique_ratio = series.nunique() / len(series.dropna())
        if unique_ratio > 0.9: # High cardinality, might be text or unique ID
            # Check average string length maybe?
            avg_len = series.astype(str).str.len().mean()
            if avg_len > 50: return "text_long"
            return "text_or_id_high_cardinality"
        else: # Lower cardinality, likely categorical
             return "categorical"
    elif pd.api.types.is_bool_dtype(series):
         return "boolean"
    else:
        return "unknown"

# Add other general-purpose helper functions needed across the backend modules here.
# For example: formatting functions, specific validation routines, etc.