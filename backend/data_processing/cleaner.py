# backend/data_processing/cleaner.py
import pandas as pd
import numpy as np # Import numpy for numeric checks
import logging
from typing import List, Dict, Any, Optional, Tuple

# Assuming LLM utils are available for suggestion generation
try:
    # Assumes functions for suggesting structured steps and explaining actions exist
    from backend.llm.gemini_utils import suggest_structured_cleaning_steps_llm, explain_action_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    # Define mocks if needed
    def suggest_structured_cleaning_steps_llm(profile, schema): return [] # Return empty list
    def explain_action_llm(action_type, params): return f"Mock explanation for {action_type}."

logger = logging.getLogger(__name__)

# --- Constants for Cleaning Step Types ---
# Use constants for step types to avoid typos
DROP_DUPLICATES = "drop_duplicates"
IMPUTE_MISSING = "impute_missing"
CONVERT_TYPE = "convert_type"
RENAME_COLUMN = "rename_column"
DROP_COLUMN = "drop_column"
STRIP_WHITESPACE = "strip_whitespace"
CHANGE_CASE = "change_case"
# Add more constants as needed

def suggest_cleaning_steps(df: pd.DataFrame, profile_summary: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Analyzes DataFrame and profile summary to suggest structured cleaning steps.

    Args:
        df: The DataFrame to analyze.
        profile_summary: Optional dictionary containing profiling results (e.g., from profiler.py).

    Returns:
        A list of dictionaries, each representing a suggested cleaning step with
        keys like 'step_type', 'params', 'rationale', 'details'.
    """
    if df is None or df.empty:
        return [] # Return empty list for no data

    logger.info(f"Generating structured cleaning suggestions for DataFrame shape {df.shape}")
    suggestions = []

    # --- Heuristic Checks (Generate structured suggestions) ---

    # 1. Missing Values
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            perc = (count / len(df)) * 100
            details = f"Column '{col}' has {count} missing values ({perc:.1f}%)."
            # Suggest a default strategy based on type (can be enhanced)
            default_strategy = 'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
            suggestions.append({
                "step_type": IMPUTE_MISSING,
                "params": {"column": col, "strategy": default_strategy}, # Provide default params
                "issue": "Missing Values",
                "details": details,
                "rationale": f"Suggesting '{default_strategy}' imputation for '{col}'. Consider alternatives like mean, constant, or dropping.",
                "ui_options": {"strategy": ["mean", "median", "mode", "constant", "drop_rows"]} # Hint for UI dropdown
            })

    # 2. Duplicate Rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
         suggestions.append({
            "step_type": DROP_DUPLICATES,
            "params": {}, # No params needed for dropping all duplicates
            "issue": "Duplicate Rows",
            "details": f"Found {duplicates} complete duplicate rows ({duplicates / len(df) * 100:.1f}%).",
            "rationale": "Dropping duplicates ensures each record is unique.",
        })

    # 3. Data Type Issues (Suggest conversion)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Try converting a sample to numeric/datetime to see if it works
        sample = df[col].dropna().sample(min(100, len(df[col].dropna())), random_state=42)
        try:
            pd.to_numeric(sample)
            # If conversion works for sample, suggest it
            suggestions.append({
                "step_type": CONVERT_TYPE,
                "params": {"column": col, "target_type": "numeric"},
                "issue": "Potential Numeric Type",
                "details": f"Column '{col}' (object type) might be convertible to numeric.",
                "rationale": "Converting to numeric enables calculations and analysis.",
            })
            continue # Move to next col if numeric suggestion made
        except (ValueError, TypeError):
            pass # Cannot convert sample to numeric

        try:
            pd.to_datetime(sample)
            suggestions.append({
                "step_type": CONVERT_TYPE,
                "params": {"column": col, "target_type": "datetime"},
                "issue": "Potential Datetime Type",
                "details": f"Column '{col}' (object type) might be convertible to datetime.",
                "rationale": "Converting to datetime enables time-based analysis.",
            })
        except (ValueError, TypeError):
             # If not numeric or datetime, suggest checking if it should be category or needs string cleaning
             if df[col].nunique() / len(df[col].dropna()) < 0.5: # Heuristic for categorical potential
                 suggestions.append({
                    "step_type": CONVERT_TYPE,
                    "params": {"column": col, "target_type": "category"},
                    "issue": "Potential Categorical Type",
                    "details": f"Column '{col}' (object) has relatively few unique values.",
                    "rationale": "Converting to 'category' can save memory and improve performance.",
                 })
             else:
                  suggestions.append({
                    "step_type": STRIP_WHITESPACE, # Suggest basic string cleaning
                    "params": {"column": col},
                    "issue": "Potential String Cleaning Needed",
                    "details": f"Column '{col}' is object type. Check for leading/trailing spaces or inconsistent casing.",
                    "rationale": "Standardizing strings ensures consistency.",
                    "ui_options": {"step_type": [STRIP_WHITESPACE, CHANGE_CASE]} # Allow user to choose case change too
                 })


    # --- LLM Integration (Placeholder for suggesting structured steps) ---
    # if LLM_AVAILABLE and profile_summary:
    #    try:
    #        logger.info("Calling LLM for additional structured cleaning suggestions...")
    #        schema_context = str({col: str(dtype) for col, dtype in df.dtypes.items()}) # Simple schema
    #        llm_structured_suggestions = suggest_structured_cleaning_steps_llm(profile_summary, schema_context)
    #        # Validate LLM suggestions structure before extending
    #        valid_llm_suggestions = [s for s in llm_structured_suggestions if isinstance(s, dict) and 'step_type' in s and 'params' in s]
    #        suggestions.extend(valid_llm_suggestions)
    #    except Exception as e:
    #        logger.warning(f"LLM structured cleaning suggestion failed: {e}")

    if not suggestions:
        logger.info("No specific cleaning suggestions generated from heuristics.")
        # Optionally return an empty list or a "seems clean" message dict
        # return [{"issue": "Initial Scan Clean", "details": "..."}]


    # --- Generate Explanations for Suggestions ---
    if LLM_AVAILABLE:
        logger.info(f"Generating AI explanations for {len(suggestions)} suggestions...")
        for sugg in suggestions:
             if "rationale" not in sugg or not sugg["rationale"]: # Only generate if missing
                 try:
                      sugg["rationale"] = explain_action_llm(sugg["step_type"], sugg["params"])
                 except Exception as e:
                      logger.warning(f"Failed to get explanation for step {sugg['step_type']}: {e}")
                      sugg["rationale"] = "(AI explanation failed)"

    logger.info(f"Finished generating {len(suggestions)} structured cleaning suggestions.")
    return suggestions


def apply_cleaning_step(df: pd.DataFrame, step_type: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a specific, predefined cleaning step safely using structured parameters.

    Args:
        df: The DataFrame to clean.
        step_type: The type of cleaning action (e.g., 'drop_duplicates', 'impute_missing'). Use constants.
        params: Dictionary of parameters for the step (e.g., {'column': 'col_name', 'value': 0}).

    Returns: The modified DataFrame.
    Raises: ValueError, KeyError, TypeError on invalid inputs or operations.
    """
    logger.info(f"Applying SAFE cleaning step '{step_type}' with params: {params}")
    df_cleaned = df.copy() # IMPORTANT: Always work on a copy

    try:
        column = params.get("column") # Common parameter

        # Validate column exists if specified for most operations
        if column and column not in df_cleaned.columns:
             raise KeyError(f"Column '{column}' not found in DataFrame for step '{step_type}'.")

        # --- Apply Step Based on Type ---
        if step_type == DROP_DUPLICATES:
            subset = params.get("subset")
            keep = params.get("keep", 'first')
            count_before = len(df_cleaned)
            df_cleaned.drop_duplicates(subset=subset, keep=keep, inplace=True)
            count_after = len(df_cleaned)
            logger.info(f"Dropped {count_before - count_after} duplicate rows.")

        elif step_type == IMPUTE_MISSING:
            if not column: raise ValueError(f"Missing 'column' parameter for {step_type}")
            strategy = params.get("strategy", "mode") # Default strategy
            fill_value = params.get("value") # Required only for 'constant'

            original_missing = df_cleaned[column].isnull().sum()
            if original_missing == 0:
                logger.info(f"No missing values found in '{column}'. Skipping imputation.")
                return df_cleaned # Return unchanged df

            imputed_value = None
            if strategy == "drop_rows":
                 df_cleaned.dropna(subset=[column], inplace=True)
            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(df_cleaned[column]): imputed_value = df_cleaned[column].mean(); df_cleaned[column].fillna(imputed_value, inplace=True)
                else: raise TypeError(f"Cannot impute mean on non-numeric column '{column}'.")
            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df_cleaned[column]): imputed_value = df_cleaned[column].median(); df_cleaned[column].fillna(imputed_value, inplace=True)
                else: raise TypeError(f"Cannot impute median on non-numeric column '{column}'.")
            elif strategy == "mode":
                 # Handle potential multiple modes - consistently take the first one
                 modes = df_cleaned[column].mode()
                 if not modes.empty: imputed_value = modes[0]; df_cleaned[column].fillna(imputed_value, inplace=True)
                 else: logger.warning(f"Could not determine mode for column '{column}' (maybe all NaN?). Skipping imputation."); return df_cleaned # Or fill with a default?
            elif strategy == "constant":
                 if fill_value is None: raise ValueError("Fill value ('value' parameter) required for 'constant' strategy.")
                 imputed_value = fill_value; df_cleaned[column].fillna(imputed_value, inplace=True)
            else: raise ValueError(f"Unknown imputation strategy: {strategy}")

            logger.info(f"Imputed {original_missing} missing values in '{column}' using strategy '{strategy}' (value: {imputed_value}).")

        elif step_type == CONVERT_TYPE:
            if not column: raise ValueError(f"Missing 'column' parameter for {step_type}")
            target_type = params.get("target_type")
            if not target_type: raise ValueError("Missing 'target_type' parameter.")

            original_type = str(df_cleaned[column].dtype)
            logger.info(f"Attempting conversion of '{column}' from {original_type} to {target_type}")
            if target_type == 'numeric': df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            elif target_type == 'datetime': df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
            elif target_type == 'string': df_cleaned[column] = df_cleaned[column].astype(str)
            elif target_type == 'category': df_cleaned[column] = df_cleaned[column].astype('category')
            else: raise ValueError(f"Unsupported target type for conversion: {target_type}")
            logger.info(f"Converted column '{column}' to type '{target_type}'. NaN count after: {df_cleaned[column].isnull().sum()}")

        elif step_type == RENAME_COLUMN:
             if not column: raise ValueError(f"Missing 'column' (old name) parameter for {step_type}")
             new_name = params.get("new_name")
             if not new_name: raise ValueError("Missing 'new_name' parameter.")
             if column == new_name: logger.warning(f"Old and new names are the same ('{column}'). Skipping rename."); return df_cleaned
             df_cleaned.rename(columns={column: new_name}, inplace=True)
             logger.info(f"Renamed column '{column}' to '{new_name}'.")

        elif step_type == DROP_COLUMN:
             if not column: raise ValueError(f"Missing 'column' parameter for {step_type}")
             df_cleaned.drop(columns=[column], inplace=True)
             logger.info(f"Dropped column '{column}'.")

        elif step_type == STRIP_WHITESPACE:
             if not column: raise ValueError(f"Missing 'column' parameter for {step_type}")
             if pd.api.types.is_object_dtype(df_cleaned[column]) or pd.api.types.is_string_dtype(df_cleaned[column]):
                 df_cleaned[column] = df_cleaned[column].str.strip()
                 logger.info(f"Stripped leading/trailing whitespace from column '{column}'.")
             else: logger.warning(f"Cannot strip whitespace from non-string column '{column}'. Skipping.")

        elif step_type == CHANGE_CASE:
             if not column: raise ValueError(f"Missing 'column' parameter for {step_type}")
             case_type = params.get("case", "lower") # 'lower', 'upper', 'title'
             if pd.api.types.is_object_dtype(df_cleaned[column]) or pd.api.types.is_string_dtype(df_cleaned[column]):
                 if case_type == "lower": df_cleaned[column] = df_cleaned[column].str.lower()
                 elif case_type == "upper": df_cleaned[column] = df_cleaned[column].str.upper()
                 elif case_type == "title": df_cleaned[column] = df_cleaned[column].str.title()
                 else: raise ValueError(f"Unsupported case type: {case_type}")
                 logger.info(f"Converted column '{column}' to {case_type} case.")
             else: logger.warning(f"Cannot change case of non-string column '{column}'. Skipping.")

        # --- Add more predefined step types here ---

        else:
            raise ValueError(f"Unsupported cleaning step type: '{step_type}'")

        # --- Log Lineage (Placeholder - requires lineage backend) (#3) ---
        # try:
        #     from backend.database.crud import log_lineage_step # Example import
        #     log_lineage_step(
        #         process_type='cleaning',
        #         process_details={'step': step_type, 'params': params},
        #         # Pass input/output datasource IDs or identifiers
        #         input_source_id = df.attrs.get('datasource_id'),
        #         output_source_id = df_cleaned.attrs.get('datasource_id') # Might be same ID
        #     )
        # except Exception as lineage_err:
        #     logger.warning(f"Failed to log lineage for step '{step_type}': {lineage_err}")

        return df_cleaned

    except (KeyError, ValueError, TypeError) as e:
        # Catch specific errors related to bad parameters or operations
        logger.error(f"Error applying cleaning step '{step_type}': {e}", exc_info=True)
        raise ValueError(f"Failed to apply step '{step_type}': {e}") from e
    except Exception as e:
        # Catch other unexpected errors
        logger.error(f"Unexpected error during cleaning step '{step_type}': {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during cleaning: {e}") from e