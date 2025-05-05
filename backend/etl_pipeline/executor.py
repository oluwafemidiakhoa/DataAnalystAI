# backend/etl_pipeline/executor.py
# Logic to execute pipelines defined visually or programmatically

import pandas as pd
import logging
from typing import List, Dict, Any, Optional

# Import the specific step functions needed from other modules
# This assumes cleaner, transformer, feature_engineer expose functions
# that accept a DataFrame and parameters dict.
try:
    from backend.data_processing.cleaner import apply_cleaning_step
    from backend.data_processing.transformer import apply_transformation # Or apply_structured_transformation
    from backend.data_processing.feature_engineer import apply_feature_engineering
    # Import functions for data loading if pipelines start from source
    # from backend.data_processing.loader import load_dataframe_from_file, ...
    # from backend.analysis.query_executor import execute_query
    BACKEND_STEPS_AVAILABLE = True
except ImportError:
    BACKEND_STEPS_AVAILABLE = False
    logging.error("Could not import necessary step functions for ETL pipeline executor.", exc_info=True)
    # Define dummy functions if needed for basic structure testing
    def apply_cleaning_step(df, *args, **kwargs): logging.warning("MOCK apply_cleaning_step"); return df
    def apply_transformation(df, *args, **kwargs): logging.warning("MOCK apply_transformation"); return df
    def apply_feature_engineering(df, *args, **kwargs): logging.warning("MOCK apply_feature_engineering"); return df


logger = logging.getLogger(__name__)

# Mapping from step names (used in pipeline definition) to actual functions
# This allows the executor to be data-driven
STEP_FUNCTION_MAP = {
    # From cleaner.py (use constants if defined there)
    "drop_duplicates": apply_cleaning_step,
    "impute_missing": apply_cleaning_step,
    "convert_type": apply_cleaning_step,
    "rename_column": apply_cleaning_step,
    "drop_column": apply_cleaning_step,
    "strip_whitespace": apply_cleaning_step,
    "change_case": apply_cleaning_step,
    # From transformer.py (use structured version if available)
    "apply_nl_transform": apply_transformation, # Assumes this takes df, code
    # "combine_columns": apply_structured_transformation, # Example structured
    # "calculate_column": apply_structured_transformation, # Example structured
    # From feature_engineer.py
    "date_extraction": apply_feature_engineering,
    "numeric_interaction": apply_feature_engineering,
    "grouped_aggregation": apply_feature_engineering,
    "binning": apply_feature_engineering,
    "categorical_encoding": apply_feature_engineering,
    # Add more steps as implemented
}


def execute_pipeline(
    initial_data: Union[pd.DataFrame, Dict[str, Any]], # Can be DF or connection info
    pipeline_steps: List[Dict[str, Any]],
    initial_data_load_func: Optional[callable] = None # Function to load initial data if not DF
    ) -> pd.DataFrame:
    """
    Executes a defined sequence of data processing steps (pipeline).

    Args:
        initial_data: The starting DataFrame or connection info to load data from.
        pipeline_steps: A list of step dictionaries, e.g.,
                        [{'step_type': 'impute_missing', 'params': {'column': 'A', 'strategy': 'mean'}},
                         {'step_type': 'convert_type', 'params': {'column': 'B', 'target_type': 'numeric'}},
                         ...]
        initial_data_load_func: Optional function to load initial data if connection info is passed.

    Returns:
        The final DataFrame after executing all pipeline steps.

    Raises:
        ValueError: If a step type is unsupported or parameters are invalid.
        RuntimeError: For unexpected execution errors.
    """
    if not BACKEND_STEPS_AVAILABLE:
         raise RuntimeError("Cannot execute pipeline: Required backend step functions are not available.")
    if not pipeline_steps:
         logger.warning("Pipeline has no steps.")
         if isinstance(initial_data, pd.DataFrame): return initial_data
         else: raise ValueError("Cannot return result: Pipeline is empty and initial data is not a DataFrame.")

    logger.info(f"Executing pipeline with {len(pipeline_steps)} steps...")

    # --- Load initial data if needed ---
    if isinstance(initial_data, pd.DataFrame):
        current_df = initial_data.copy() # Start with a copy
    elif isinstance(initial_data, dict) and initial_data_load_func:
        try:
            logger.info("Loading initial data using provided function...")
            current_df = initial_data_load_func(initial_data) # Call loader function
            if not isinstance(current_df, pd.DataFrame):
                 raise TypeError("Initial data load function did not return a Pandas DataFrame.")
            logger.info(f"Initial data loaded. Shape: {current_df.shape}")
        except Exception as load_err:
             logger.error(f"Failed to load initial data for pipeline: {load_err}", exc_info=True)
             raise RuntimeError(f"Pipeline failed: Could not load initial data: {load_err}") from load_err
    else:
        raise ValueError("Invalid initial_data provided. Must be DataFrame or connection info with a load function.")


    # --- Execute steps sequentially ---
    total_steps = len(pipeline_steps)
    for i, step in enumerate(pipeline_steps):
        step_type = step.get("step_type")
        params = step.get("params", {})
        step_name = step.get("name", f"Step {i+1}: {step_type}") # Optional name for logging

        logger.info(f"Executing Step {i+1}/{total_steps}: '{step_name}' (Type: {step_type})")

        if not step_type:
            logger.warning(f"Skipping step {i+1} due to missing 'step_type'.")
            continue

        # Find the corresponding function
        if step_type not in STEP_FUNCTION_MAP:
            logger.error(f"Unsupported step type '{step_type}' in pipeline.")
            raise ValueError(f"Pipeline failed: Unsupported step type '{step_type}'")

        step_func = STEP_FUNCTION_MAP[step_type]

        try:
            # Call the function associated with the step_type
            # Pass df, step_type, and params if needed by the function signature
            # Adjust based on actual function signatures (e.g., some might just need df, code)
            if step_func in [apply_cleaning_step, apply_feature_engineering]: # Assumes these take step_type and params
                 current_df = step_func(df=current_df, step_type=step_type, params=params)
            elif step_func == apply_transformation: # Assumes this takes df and generated_code (or NL command?)
                 # This needs refinement - how is code generated or passed in a pipeline?
                 # Maybe pipeline step includes 'nl_command' and we generate code here? Risky.
                 # Or maybe it includes pre-generated 'code'? Still risky.
                 # Sticking to structured steps is safer.
                 logger.warning(f"Executing arbitrary transformation step '{step_type}' requires careful handling/sandboxing (using mock apply).")
                 code_to_run = params.get("code", f"# No code provided for step {step_name}") # Example param
                 current_df = step_func(df=current_df, generated_code=code_to_run)
            else:
                 # Default call signature if unknown (might fail)
                 current_df = step_func(df=current_df, params=params)

            logger.debug(f"Step {i+1} completed. DataFrame shape: {current_df.shape}")

            # --- Log Lineage for each step (Placeholder) ---
            # log_lineage_step(process_type='pipeline_step', process_details=step, ...)

        except Exception as step_err:
            logger.error(f"Pipeline execution failed at Step {i+1} ('{step_name}'): {step_err}", exc_info=True)
            raise RuntimeError(f"Pipeline failed at step '{step_name}': {step_err}") from step_err


    logger.info(f"Pipeline execution finished successfully. Final shape: {current_df.shape}")
    return current_df