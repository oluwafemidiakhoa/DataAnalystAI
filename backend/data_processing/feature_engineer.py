# backend/data_processing/feature_engineer.py
import pandas as pd
import numpy as np
import logging
import io
from typing import List, Dict, Any, Optional

# --- Optional Dependency Imports ---
# Attempt to import libraries needed for specific steps, set flags
try:
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PolynomialFeatures
    from sklearn.impute import SimpleImputer # Could be used for target encoding later
    # from category_encoders import TargetEncoder # Example for target encoding
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed (`pip install scikit-learn`). Some feature engineering steps (binning, encoding, polynomial) will be unavailable.")

# --- LLM Utility Imports ---
# Assuming LLM utils are available (import path might differ)
LLM_AVAILABLE = False
try:
    # Assumes functions exist for suggesting structured steps and explaining actions
    from backend.llm.gemini_utils import (
        suggest_structured_feature_steps_llm, # Ideal: Suggests {'step_type': '...', 'params': {...}}
        explain_code_llm # Re-use for explaining structured steps
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger = logging.getLogger(__name__) # Define logger if import fails early
    logger.error("LLM utility functions not found for feature engineer.", exc_info=True)
    # Define mocks if needed for standalone testing of this module
    def suggest_structured_feature_steps_llm(schema, goal): return []
    def explain_code_llm(step_type, params_str): return f"Mock explanation for {step_type} with {params_str}."

logger = logging.getLogger(__name__) # Get logger instance for this module

# --- Constants for Feature Engineering Step Types ---
DATE_EXTRACTION = "date_extraction"
NUMERIC_INTERACTION = "numeric_interaction"
GROUPED_AGGREGATION = "grouped_aggregation"
CATEGORICAL_ENCODING = "categorical_encoding" # Params should specify method ('one-hot', 'target', etc.)
BINNING = "binning" # Discretizing numeric features
POLYNOMIAL_FEATURES = "polynomial_features"
# Add more types like: TEXT_FEATURE_EXTRACTION, LOG_TRANSFORM, etc.


# --- Feature Suggestion ---
def suggest_feature_engineering(df: pd.DataFrame, goal: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Suggests potential new features based on the DataFrame and an optional goal,
    as structured steps suitable for apply_feature_engineering.
    Includes advanced feature suggestions (#5).

    Args:
        df: The input DataFrame.
        goal: Optional user goal (e.g., "predict churn", "understand sales drivers").

    Returns:
        A list of dictionaries, each representing a suggested feature engineering step.
        Structure: {'step_type': str, 'params': dict, 'rationale': str, 'details': str}
    """
    if df is None or df.empty:
        return []

    logger.info(f"Generating feature engineering suggestions for DataFrame shape {df.shape}. Goal: {goal or 'General'}")
    suggestions = []
    # Prepare schema context for LLM
    buffer = io.StringIO(); df.info(buf=buffer, max_cols=100, verbose=False); schema_context = f"Schema:\n{buffer.getvalue()}"

    # --- Heuristic Suggestions (Generate structured steps) ---
    date_cols = df.select_dtypes(include=['datetime64', 'datetime', 'datetime64[ns]']).columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 1. Date Features
    if date_cols:
        col = date_cols[0]
        params = {"column": col, "components": ["year", "month", "dayofweek", "hour"]}
        suggestions.append({
             "step_type": DATE_EXTRACTION, "params": params,
             "details": f"Extract time components from '{col}'.",
             "rationale": "Temporal patterns are often predictive or cyclical."
         })

    # 2. Numeric Interactions
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        params = {"column1": col1, "column2": col2, "operation": "multiply"}
        suggestions.append({
             "step_type": NUMERIC_INTERACTION, "params": params,
             "details": f"Create interaction: '{col1}' * '{col2}'.",
             "rationale": "May capture synergistic effects.",
             "ui_options": {"operation": ["multiply", "divide", "add", "subtract"]}
         })

    # 3. Grouped Aggregations
    if numeric_cols and categorical_cols:
         num_col, cat_col = numeric_cols[0], categorical_cols[0]
         params = {"group_by_column": cat_col, "agg_column": num_col, "agg_func": "mean"}
         suggestions.append({
              "step_type": GROUPED_AGGREGATION, "params": params,
              "details": f"Aggregate '{num_col}' grouped by '{cat_col}'.",
              "rationale": "Captures group-level behavior.",
              "ui_options": {"agg_func": ["mean", "median", "sum", "std", "count", "nunique"]}
          })

    # 4. Binning Numeric Features
    if numeric_cols and SKLEARN_AVAILABLE: # Check if library available
        col = numeric_cols[0]
        params = {"column": col, "bins": 5, "strategy": "quantile"}
        suggestions.append({
            "step_type": BINNING, "params": params,
            "details": f"Discretize '{col}' into bins.",
            "rationale": "Can capture non-linear relationships.",
            "ui_options": {"strategy": ["uniform", "quantile", "kmeans"], "bins": "number_input"}
        })

    # 5. Categorical Encoding
    if categorical_cols and SKLEARN_AVAILABLE: # Check if library available
         params = {"columns": [categorical_cols[0]], "method": "one-hot"}
         suggestions.append({
              "step_type": CATEGORICAL_ENCODING, "params": params,
              "details": f"Encode categorical: '{categorical_cols[0]}'.",
              "rationale": "Needed for many ML models.",
              "ui_options": {"method": ["one-hot"]} # Add more if implemented
          })

    # 6. Polynomial Features
    if len(numeric_cols) >= 2 and SKLEARN_AVAILABLE: # Check if library available
        cols_to_use = numeric_cols[:min(len(numeric_cols), 3)] # Example: Limit default columns
        params = {"columns": cols_to_use, "degree": 2, "interaction_only": False, "include_bias": False}
        suggestions.append({
            "step_type": POLYNOMIAL_FEATURES, "params": params,
            "details": f"Generate polynomial features (degree {params['degree']}) for {len(cols_to_use)} column(s).",
            "rationale": "Capture non-linear interactions between numeric features."
        })

    # --- LLM Integration (Suggest structured steps) ---
    if LLM_AVAILABLE:
        try:
            logger.info("Calling LLM for advanced structured feature suggestions...")
            llm_structured_suggestions = suggest_structured_feature_steps_llm(schema_context, goal)
            valid_llm_suggestions = [s for s in llm_structured_suggestions if isinstance(s, dict) and 'step_type' in s and 'params' in s]
            if valid_llm_suggestions:
                 logger.info(f"Adding {len(valid_llm_suggestions)} suggestions from LLM.")
                 existing_types_params = {(s['step_type'], tuple(sorted(s['params'].items()))) for s in suggestions}
                 for llm_sugg in valid_llm_suggestions:
                      s_key = (llm_sugg['step_type'], tuple(sorted(llm_sugg['params'].items())))
                      if s_key not in existing_types_params: suggestions.append(llm_sugg); existing_types_params.add(s_key)
        except Exception as e: logger.warning(f"LLM structured feature suggestion failed: {e}")


    # --- Generate Explanations for All Suggestions ---
    if LLM_AVAILABLE:
        logger.info(f"Generating AI explanations for {len(suggestions)} feature suggestions...")
        for sugg in suggestions:
             if "rationale" not in sugg or not sugg["rationale"]: # Generate if missing
                 try: sugg["rationale"] = explain_code_llm(f"Feature Step: {sugg['step_type']}", str(sugg.get("params",{})))
                 except Exception as e: logger.warning(f"Failed explanation for FE step {sugg['step_type']}: {e}"); sugg["rationale"] = "(AI explanation failed)"

    if not suggestions:
        suggestions.append({"step_type": "No Suggestions", "details": "Could not generate suggestions."})

    logger.info(f"Finished generating {len(suggestions)} feature suggestions.")
    return suggestions


# --- Safe Feature Application ---
def apply_feature_engineering(df: pd.DataFrame, step_type: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a specific, predefined feature engineering step safely using structured parameters. (#6 Safe Execution)

    Args:
        df: The DataFrame to modify.
        step_type: The type of feature engineering action (use constants).
        params: Dictionary of parameters for the step.

    Returns: The DataFrame with the new feature(s).
    Raises: ValueError, KeyError, TypeError, ImportError on invalid inputs or operations or missing libs.
    """
    logger.info(f"Applying SAFE feature engineering step '{step_type}' with params: {params}")
    if df is None or df.empty: raise ValueError("Input DataFrame cannot be empty.")

    df_engineered = df.copy() # IMPORTANT: Always work on a copy
    new_column_names = [] # Track newly created columns

    try:
        column = params.get("column") # Common parameter, might be None
        columns = params.get("columns") # For steps operating on multiple columns

        # --- Step Execution Logic ---
        if step_type == DATE_EXTRACTION:
            if not column or column not in df.columns: raise KeyError(f"Date column '{column}' not found.")
            if not pd.api.types.is_datetime64_any_dtype(df[column]): raise TypeError(f"Column '{column}' is not datetime.")
            components = params.get("components", ["year", "month", "dayofweek", "hour"])
            for comp in components:
                new_col_name = f"{column}_{comp}"; new_column_names.append(new_col_name)
                datetime_col = pd.to_datetime(df_engineered[column], errors='coerce') # Ensure datetime before dt accessor
                if comp == 'year': df_engineered[new_col_name] = datetime_col.dt.year
                elif comp == 'month': df_engineered[new_col_name] = datetime_col.dt.month
                elif comp == 'day': df_engineered[new_col_name] = datetime_col.dt.day
                elif comp == 'dayofweek': df_engineered[new_col_name] = datetime_col.dt.dayofweek
                elif comp == 'hour': df_engineered[new_col_name] = datetime_col.dt.hour
                else: logger.warning(f"Unsupported date component '{comp}'.")
            logger.info(f"Extracted components {components} from '{column}'.")

        elif step_type == NUMERIC_INTERACTION:
            col1 = params.get("column1"); col2 = params.get("column2"); op = params.get("operation", "multiply")
            if not col1 or col1 not in df.columns: raise KeyError(f"Column1 '{col1}' not found.")
            if not col2 or col2 not in df.columns: raise KeyError(f"Column2 '{col2}' not found.")
            val1 = pd.to_numeric(df_engineered[col1], errors='coerce'); val2 = pd.to_numeric(df_engineered[col2], errors='coerce') # Coerce ensures numeric operation works
            if val1.isnull().all() or val2.isnull().all(): raise TypeError(f"One or both interaction columns ('{col1}', '{col2}') non-numeric or all null after coercion.")
            new_col_name = f"{col1}_{op[:3]}_{col2}"; new_column_names.append(new_col_name)
            if op == "multiply": df_engineered[new_col_name] = val1 * val2
            elif op == "divide": df_engineered[new_col_name] = val1.divide(val2.replace(0, np.nan))
            elif op == "add": df_engineered[new_col_name] = val1 + val2
            elif op == "subtract": df_engineered[new_col_name] = val1 - val2
            else: raise ValueError(f"Unsupported numeric operation: {op}")
            logger.info(f"Applied '{op}' between '{col1}' and '{col2}' into '{new_col_name}'.")

        elif step_type == GROUPED_AGGREGATION:
            group_col = params.get("group_by_column"); agg_col = params.get("agg_column"); agg_func = params.get("agg_func", "mean")
            if not group_col or group_col not in df.columns: raise KeyError(f"Group-by column '{group_col}' not found.")
            if not agg_col or agg_col not in df.columns: raise KeyError(f"Aggregation column '{agg_col}' not found.")
            valid_agg_funcs = ["mean", "median", "sum", "std", "count", "nunique", "min", "max"]
            if agg_func not in valid_agg_funcs: raise ValueError(f"Unsupported agg func: {agg_func}. Use one of {valid_agg_funcs}")
            if agg_func not in ["count", "nunique"] and not pd.api.types.is_numeric_dtype(df[agg_col]): raise TypeError(f"Agg column '{agg_col}' must be numeric for '{agg_func}'.")
            new_col_name = f"{agg_col}_{agg_func}_by_{group_col}"; new_column_names.append(new_col_name)
            logger.info(f"Applying grouped agg '{agg_func}' on '{agg_col}' by '{group_col}' into '{new_col_name}'.")
            # Ensure group column is not accidentally converted if numeric-like object
            if pd.api.types.is_object_dtype(df_engineered[group_col]) or pd.api.types.is_categorical_dtype(df_engineered[group_col]):
                df_engineered[new_col_name] = df_engineered.groupby(group_col, observed=False)[agg_col].transform(agg_func)
            else: # Avoid grouping by float/high cardinality numeric unless intended
                 logger.warning(f"Grouping by numeric column '{group_col}'. Ensure this is intended.")
                 df_engineered[new_col_name] = df_engineered.groupby(group_col)[agg_col].transform(agg_func)


        elif step_type == BINNING:
            if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn needed for Binning.")
            column = params.get("column"); bins = params.get("bins", 5); strategy = params.get("strategy", "quantile")
            if not column or column not in df.columns: raise KeyError(f"Column '{column}' not found.")
            if not pd.api.types.is_numeric_dtype(df[column]): raise TypeError(f"Cannot bin non-numeric '{column}'.")
            if not isinstance(bins, int) or bins < 2: raise ValueError("Bins must be integer >= 2.")
            valid_strategies = ['uniform', 'quantile', 'kmeans']
            if strategy not in valid_strategies: raise ValueError(f"Unsupported strategy: {strategy}. Use one of {valid_strategies}")
            new_col_name = f"{column}_binned_{strategy}"; new_column_names.append(new_col_name)
            data_to_bin = df_engineered[[column]].dropna()
            if data_to_bin.empty: logger.warning(f"No data to bin in '{column}'."); return df_engineered
            est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy, subsample=None)
            binned_data = est.fit_transform(data_to_bin); df_engineered.loc[data_to_bin.index, new_col_name] = binned_data.flatten(); df_engineered[new_col_name] = df_engineered[new_col_name].astype('category')
            logger.info(f"Applied binning ({strategy}, {bins} bins) on '{column}' into '{new_col_name}'.")

        elif step_type == CATEGORICAL_ENCODING:
            if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn needed for Encoding.")
            columns = params.get("columns", []); method = params.get("method", "one-hot")
            if not columns or not isinstance(columns, list): raise ValueError("Invalid 'columns' list.")
            missing_cols = [col for col in columns if col not in df.columns];
            if missing_cols: raise KeyError(f"Columns not found for encoding: {', '.join(missing_cols)}")
            if method == "one-hot":
                 logger.info(f"Applying One-Hot Encoding to: {columns}")
                 ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=params.get('drop', None)) # Add drop option
                 encoded_data = ohe.fit_transform(df_engineered[columns])
                 new_cols = ohe.get_feature_names_out(columns)
                 encoded_df = pd.DataFrame(encoded_data, columns=new_cols, index=df_engineered.index)
                 df_engineered = df_engineered.drop(columns=columns)
                 df_engineered = pd.concat([df_engineered, encoded_df], axis=1)
                 new_column_names.extend(new_cols)
            # Add LabelEncoder, TargetEncoder (requires y) etc.
            else: raise ValueError(f"Unsupported encoding method: {method}")

        elif step_type == POLYNOMIAL_FEATURES:
            if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn needed for Polynomial Features.")
            columns = params.get("columns", []); degree = params.get("degree", 2); interaction_only = params.get("interaction_only", False); include_bias = params.get("include_bias", False)
            if not columns or not isinstance(columns, list): raise ValueError("Invalid 'columns' list.")
            missing_cols = [col for col in columns if col not in df.columns];
            if missing_cols: raise KeyError(f"Columns not found: {', '.join(missing_cols)}")
            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric: raise TypeError(f"Columns must be numeric: {', '.join(non_numeric)}")
            logger.info(f"Generating Polynomial Features (degree={degree}) for: {columns}")
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
            # Impute NaNs simply before applying polynomial features (could be configurable)
            data_for_poly = df_engineered[columns].fillna(df_engineered[columns].mean())
            poly_features = poly.fit_transform(data_for_poly)
            poly_feature_names = poly.get_feature_names_out(columns)
            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_engineered.index)
            # Drop redundant columns ('1' if bias=false, originals if degree>1)
            if not include_bias: poly_df = poly_df.drop(columns=['1'], errors='ignore')
            cols_to_drop_from_poly = columns if degree > 1 else []
            poly_df = poly_df.drop(columns=cols_to_drop_from_poly, errors='ignore')
            # Add new features
            df_engineered = pd.concat([df_engineered.drop(columns=cols_to_drop_from_poly, errors='ignore'), poly_df], axis=1) # Drop originals if degree > 1
            new_column_names.extend(poly_df.columns)


        # --- Add more predefined, safe feature engineering steps here ---


        else:
            raise ValueError(f"Unsupported feature engineering step type: '{step_type}'")


        # --- Log Lineage (#3) ---
        try:
            # from backend.database.crud import log_lineage_step # Example import
            lineage_details = {'step': step_type, 'params': params, 'new_columns': new_column_names}
            # log_lineage_step(...)
            logger.info(f"Lineage logged for feature step '{step_type}'. New columns: {new_column_names}")
        except Exception as lineage_err:
            logger.warning(f"Failed to log lineage for feature step '{step_type}': {lineage_err}")


        return df_engineered

    except (KeyError, ValueError, TypeError, ImportError) as e:
        logger.error(f"Error applying feature step '{step_type}': {e}", exc_info=False) # Log less verbose traceback for config errors
        raise ValueError(f"Failed feature step '{step_type}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during feature step '{step_type}': {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during feature engineering: {e}") from e

# --- Fallback for code execution (legacy/mocking) ---
# This should NOT be the primary way features are applied due to security risks.
def apply_feature_engineering_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Applies feature engineering code to a DataFrame.
    WARNING: THIS IS LIKELY A MOCK / UNSAFE. Use apply_feature_engineering instead.
    """
    logger.warning(f"Applying feature engineering code via unsafe fallback (MOCK or Sandbox Needed): \n{code}")
    # This should point to the mock_apply_code or a future safe_execute_code
    # Re-using mock_apply_code logic here for consistency if mocks are used
    try:
        from pages._mocks import mock_apply_code # Try to import mock if needed
        return mock_apply_code(df, code)
    except ImportError:
         logger.error("Fallback mock_apply_code not found!")
         raise RuntimeError("Cannot apply feature code: execution mechanism missing.")