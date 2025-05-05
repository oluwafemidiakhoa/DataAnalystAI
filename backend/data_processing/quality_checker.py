# backend/data_processing/quality_checker.py
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
import re
from sqlalchemy.orm import Session # <--- ADD THIS IMPORT

# Assuming DB access might be needed for complex rules or logging violations
# from sqlalchemy.orm import Session
# from backend.database import crud, models

logger = logging.getLogger(__name__)

# --- Constants for Rule Types ---
# Match rule types potentially used in models.py or suggested by LLM
RULE_NOT_NULL = "not_null"
RULE_IS_UNIQUE = "is_unique"
RULE_MIN_VALUE = "min_value"
RULE_MAX_VALUE = "max_value"
RULE_REGEX_MATCH = "regex_match"
RULE_ENUM_VALUES = "enum_values"
RULE_MIN_LENGTH = "min_length"
RULE_MAX_LENGTH = "max_length"
# Add custom SQL rule type?

def run_single_quality_check(
    df: pd.DataFrame,
    rule_type: str,
    column_name: Optional[str] = None, # Required for most rules
    params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
    """
    Executes a single data quality rule against a DataFrame.

    Args:
        df: The DataFrame to check.
        rule_type: The type of rule to execute (e.g., 'not_null', 'is_unique').
        column_name: The column to apply the rule to (if applicable).
        params: Dictionary of parameters specific to the rule type
                (e.g., {'min': 0}, {'pattern': '...'}, {'allowed': [...]}).

    Returns:
        A dictionary containing the check results:
        {'status': 'passed' | 'failed', 'violation_count': int, 'details': str}
    """
    if df is None or df.empty:
        return {'status': 'skipped', 'violation_count': 0, 'details': 'DataFrame is empty.'}

    params = params or {} # Ensure params is a dict
    result = {'status': 'passed', 'violation_count': 0, 'details': f"Rule '{rule_type}' passed."}
    failing_indices = None # Store index of failing rows

    try:
        # --- Column Validation ---
        if column_name:
            if column_name not in df.columns:
                raise KeyError(f"Column '{column_name}' not found in DataFrame.")
            series = df[column_name] # The target column series
        elif rule_type not in ['table_row_count']: # Check rules needing columns
             raise ValueError(f"Rule type '{rule_type}' requires a 'column_name'.")


        # --- Rule Execution Logic ---
        if rule_type == RULE_NOT_NULL:
            if not column_name: raise ValueError("not_null rule requires column_name.")
            failing_indices = series.isnull()
            result['violation_count'] = int(failing_indices.sum())

        elif rule_type == RULE_IS_UNIQUE:
            if not column_name: raise ValueError("is_unique rule requires column_name.")
            duplicates = series.duplicated(keep=False) # Mark all duplicates as True
            failing_indices = duplicates
            # Count unique values that are duplicated
            result['violation_count'] = int(series[duplicates].nunique()) if failing_indices.any() else 0
            if result['violation_count'] > 0:
                 result['details'] = f"{result['violation_count']} non-unique values found."

        elif rule_type == RULE_MIN_VALUE:
            if not column_name: raise ValueError("min_value rule requires column_name.")
            min_val = params.get('min')
            if min_val is None: raise ValueError("'min' parameter required for min_value rule.")
            if not pd.api.types.is_numeric_dtype(series): raise TypeError("min_value rule requires numeric column.")
            failing_indices = pd.to_numeric(series, errors='coerce') < float(min_val)
            result['violation_count'] = int(failing_indices.sum())

        elif rule_type == RULE_MAX_VALUE:
            if not column_name: raise ValueError("max_value rule requires column_name.")
            max_val = params.get('max')
            if max_val is None: raise ValueError("'max' parameter required for max_value rule.")
            if not pd.api.types.is_numeric_dtype(series): raise TypeError("max_value rule requires numeric column.")
            failing_indices = pd.to_numeric(series, errors='coerce') > float(max_val)
            result['violation_count'] = int(failing_indices.sum())

        elif rule_type == RULE_REGEX_MATCH:
            if not column_name: raise ValueError("regex_match rule requires column_name.")
            pattern = params.get('pattern')
            if not pattern: raise ValueError("'pattern' parameter required for regex_match rule.")
            # Ensure series is string, handle NaN safely
            failing_indices = ~series.astype(str).str.match(pattern, na=False)
            result['violation_count'] = int(failing_indices.sum())

        elif rule_type == RULE_ENUM_VALUES:
            if not column_name: raise ValueError("enum_values rule requires column_name.")
            allowed_values = params.get('allowed')
            if not isinstance(allowed_values, list): raise ValueError("'allowed' parameter (list) required for enum_values rule.")
            failing_indices = ~series.isin(allowed_values) & series.notna() # Fail if not in list and not NaN
            result['violation_count'] = int(failing_indices.sum())

        elif rule_type == RULE_MIN_LENGTH:
            if not column_name: raise ValueError("min_length rule requires column_name.")
            min_len = params.get('min')
            if min_len is None: raise ValueError("'min' parameter required for min_length rule.")
            failing_indices = series.astype(str).str.len() < int(min_len)
            result['violation_count'] = int(failing_indices.sum())

        elif rule_type == RULE_MAX_LENGTH:
            if not column_name: raise ValueError("max_length rule requires column_name.")
            max_len = params.get('max')
            if max_len is None: raise ValueError("'max' parameter required for max_length rule.")
            failing_indices = series.astype(str).str.len() > int(max_len)
            result['violation_count'] = int(failing_indices.sum())

        # Add more rules here...
        # e.g., check for whitespace, check date ranges, custom SQL checks against DB...

        else:
            raise ValueError(f"Unsupported rule type: {rule_type}")

        # --- Finalize Result ---
        if result['violation_count'] > 0:
            result['status'] = 'failed'
            result['details'] = f"Rule '{rule_type}' failed for column '{column_name}' on {result['violation_count']} rows."
            # Optional: Add sample failing values
            if failing_indices is not None and failing_indices.any():
                 sample_failures = series[failing_indices].unique()[:5] # Get first 5 unique failing values
                 result['sample_failures'] = [str(v) for v in sample_failures] # Convert to string for JSON safety


    except (KeyError, ValueError, TypeError) as e:
         logger.error(f"Error executing rule '{rule_type}' on column '{column_name}': {e}", exc_info=True)
         result = {'status': 'error', 'violation_count': None, 'details': f"Error executing rule: {e}"}
    except Exception as e:
         logger.error(f"Unexpected error during rule '{rule_type}' execution: {e}", exc_info=True)
         result = {'status': 'error', 'violation_count': None, 'details': f"Unexpected error: {e}"}

    return result


def run_quality_checks(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]], # List of rule definition dicts
    db: Optional[Session] = None, # Optional DB session for logging violations
    rule_model_ids: Optional[Dict[int, int]] = None # Optional mapping if rules came from DB {index: rule_db_id}
    ) -> List[Dict[str, Any]]:
    """
    Runs a list of data quality rules against a DataFrame and optionally logs violations.

    Args:
        df: The DataFrame to check.
        rules: A list of rule definition dictionaries, each must contain
               'rule_type', 'column_name' (optional), and 'rule_parameters'.
               Can also include 'rule_name' and 'rule_id' if loaded from DB.
        db: Optional SQLAlchemy Session for logging violations.
        rule_model_ids: Optional map of list index to DB rule ID for logging.

    Returns:
        A list of result dictionaries, one for each rule executed.
    """
    if not rules:
        logger.info("No data quality rules provided to run.")
        return []

    logger.info(f"Running {len(rules)} data quality checks...")
    all_results = []

    for i, rule in enumerate(rules):
        rule_type = rule.get('rule_type')
        column_name = rule.get('column_name')
        params = rule.get('rule_parameters')
        rule_name = rule.get('rule_name', f"Rule_{i+1}") # Use provided name or generate one

        if not rule_type:
            logger.warning(f"Skipping rule #{i+1} ('{rule_name}') due to missing 'rule_type'.")
            all_results.append({'rule_name': rule_name, 'status': 'skipped', 'details': 'Missing rule_type.'})
            continue

        logger.debug(f"Executing Rule #{i+1}: '{rule_name}' (Type: {rule_type}, Col: {column_name}, Params: {params})")
        result = run_single_quality_check(df, rule_type, column_name, params)
        result['rule_name'] = rule_name # Add name to result for clarity
        all_results.append(result)

        # --- Log Violation to DB (#2) ---
        if result['status'] in ['failed', 'error'] and db and DB_AVAILABLE:
            rule_db_id = rule.get('id') or (rule_model_ids.get(i) if rule_model_ids else None)
            if rule_db_id:
                try:
                    crud.log_quality_violation(
                        db=db,
                        rule_id=rule_db_id,
                        status=result['status'],
                        count=result.get('violation_count'),
                        details=result # Log the full result dict as details? Or just sample failures?
                    )
                except Exception as log_e:
                    logger.error(f"Failed to log violation to DB for rule ID {rule_db_id}: {log_e}")
            else:
                logger.warning(f"Cannot log violation for rule '{rule_name}': DB Rule ID not provided.")

    logger.info(f"Finished running {len(rules)} quality checks.")
    return all_results