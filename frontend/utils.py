# frontend/utils.py
import streamlit as st
import logging
import pandas as pd
from typing import Any, Optional, Union, List, Dict # Import necessary types

logger = logging.getLogger(__name__)

# --- Styling ---

def load_css(file_path: str):
    """Loads a CSS file into the Streamlit app."""
    try:
        with open(file_path, "r", encoding="utf-8") as f: # Specify encoding
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        logger.info(f"Successfully loaded CSS from {file_path}")
    except FileNotFoundError:
        logger.error(f"CSS file not found at {file_path}")
    except Exception as e:
        logger.error(f"Error loading CSS from {file_path}: {e}", exc_info=True)


# --- Formatting ---

def format_value(
    value: Union[float, int, str, None],
    value_type: str = 'number', # 'number', 'currency', 'percentage', 'string'
    precision: int = 2
    ) -> str:
    """
    Formats a value based on its intended type with error handling.

    Args:
        value: The value to format.
        value_type: The type of formatting ('number', 'currency', 'percentage', 'string').
        precision: Number of decimal places for numeric types.

    Returns:
        Formatted string representation.
    """
    if value is None:
        return "N/A" # Consistent representation for None

    try:
        if value_type == 'currency':
            return f"${float(value):,.{precision}f}"
        elif value_type == 'percentage':
            # Assumes input value is a decimal (e.g., 0.05 for 5%)
            return f"{float(value):.{precision}%}"
        elif value_type == 'number':
            # Distinguish between int and float for default formatting
            if isinstance(value, int):
                 return f"{value:,}" # Integer with comma
            else:
                 return f"{float(value):,.{precision}f}" # Float with comma and precision
        elif value_type == 'string':
            return str(value)
        else: # Fallback for unknown type
            logger.warning(f"Unknown format value_type: {value_type}. Returning as string.")
            return str(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format value '{value}' as {value_type}: {e}")
        return str(value) # Return original string representation on error


# --- Session State Management ---

def initialize_session_state(defaults: Dict[str, Any]):
    """
    Initializes multiple session state keys from a dictionary of defaults
    if they don't already exist.

    Args:
        defaults (Dict[str, Any]): Dictionary where keys are session state variable
                                   names and values are their default values.
    """
    initialized_keys = []
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            initialized_keys.append(key)
    if initialized_keys:
        logger.debug(f"Initialized session state keys: {', '.join(initialized_keys)}")

def reset_session_state(keys_to_reset: List[str], defaults: Dict[str, Any]):
    """
    Resets specified session state keys back to their default values.

    Args:
        keys_to_reset: A list of session state keys to reset.
        defaults: The dictionary of default values used for initialization.
    """
    logger.info(f"Resetting session state keys: {', '.join(keys_to_reset)}")
    reset_count = 0
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = defaults.get(key) # Get default value from dict
            reset_count += 1
        else:
            logger.warning(f"Attempted to reset non-existent session state key: {key}")
    logger.debug(f"Reset {reset_count} session state keys.")


# --- Data Handling Helpers ---

def get_dataframe_summary(df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Generates a quick summary dictionary for a DataFrame."""
    if df is None or df.empty:
        return None
    try:
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2)
        }
    except Exception as e:
        logger.error(f"Failed to generate DataFrame summary: {e}")
        return None

def get_column_options(df: Optional[pd.DataFrame], col_type: str = 'all') -> List[str]:
    """
    Safely gets column names from a DataFrame, optionally filtering by type.

    Args:
        df: The Pandas DataFrame.
        col_type: 'all', 'numeric', 'categorical', 'datetime', 'object'.

    Returns:
        A list of column names, or an empty list if df is None or empty.
    """
    if df is None or df.empty:
        return []
    try:
        if col_type == 'numeric':
            return df.select_dtypes(include=np.number).columns.tolist()
        elif col_type == 'categorical':
            # Include object and category types
            return df.select_dtypes(include=['object', 'category']).columns.tolist()
        elif col_type == 'datetime':
            return df.select_dtypes(include=['datetime64', 'datetime', 'datetime64[ns]']).columns.tolist()
        elif col_type == 'object':
             return df.select_dtypes(include=['object']).columns.tolist()
        elif col_type == 'all':
            return df.columns.tolist()
        else:
            logger.warning(f"Unsupported col_type '{col_type}' in get_column_options. Returning all.")
            return df.columns.tolist()
    except Exception as e:
        logger.error(f"Error getting column options: {e}")
        return []


# --- Example Usage in a Page Script ---
# import streamlit as st
# from frontend.utils import load_css, initialize_session_state, format_value, get_column_options

# # Define default state for this page at the top
# PAGE_STATE_DEFAULTS = {
#     'my_page_variable': None,
#     'user_selection': "Option A",
# }
# initialize_session_state(PAGE_STATE_DEFAULTS) # Ensures keys exist

# # Load CSS
# load_css("frontend/styles/style.css")

# # Use formatter
# price = 199.99
# st.write(f"Formatted Price: {format_value(price, 'currency')}")
# completion = 0.75
# st.write(f"Formatted Percentage: {format_value(completion, 'percentage', precision=0)}")

# # Get columns for a selectbox
# if 'cleaned_dataframe' in st.session_state and st.session_state.cleaned_dataframe is not None:
#     numeric_cols = get_column_options(st.session_state.cleaned_dataframe, 'numeric')
#     selected_col = st.selectbox("Select Numeric Column:", options=numeric_cols)