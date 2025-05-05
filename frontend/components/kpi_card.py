# frontend/components/kpi_card.py
import streamlit as st
from typing import Optional, Union

def display_kpi(
    label: str,
    value: Union[float, int, str, None],
    description: str = "",
    target: Optional[Union[float, int]] = None,
    delta: Optional[Union[float, int]] = None,
    delta_label: str = "vs previous",
    delta_color: str = "normal", # "normal", "inverse", "off"
    show_arrow: bool = True,
    value_prefix: str = "", # e.g., "$"
    value_suffix: str = "", # e.g., "%"
    value_format: str = ",.2f", # Default formatting
    help_text: Optional[str] = None
    ):
    """
    Displays an enhanced KPI metric card using Streamlit elements.

    Includes value, optional target, delta, description, trend arrows, and formatting.

    Args:
        label (str): The main name/title of the KPI.
        value (Union[float, int, str, None]): The current value of the KPI. Can be numeric or string. None handled.
        description (str, optional): A short description displayed below the KPI name. Defaults to "".
        target (Optional[Union[float, int]], optional): A target value to display. Defaults to None.
        delta (Optional[Union[float, int]], optional): Change value (e.g., vs previous period, vs target). Defaults to None.
        delta_label (str, optional): Label accompanying the delta value. Defaults to "vs previous".
        delta_color (str, optional): Color for the delta ('normal', 'inverse', 'off'). Defaults to "normal".
        show_arrow (bool, optional): Whether to show trend arrow based on delta sign. Defaults to True.
        value_prefix (str, optional): Prefix for the main value (e.g., '$'). Defaults to "".
        value_suffix (str, optional): Suffix for the main value (e.g., '%'). Defaults to "".
        value_format (str, optional): Format specifier for numeric values (e.g., ",.2f", ".0%"). Defaults to ",.2f".
        help_text (Optional[str], optional): Tooltip text shown on hover over the metric area. Defaults to None.
    """

    # Use a container with border for visual separation and styling
    with st.container(border=True):

        # --- Display Label and Description ---
        st.markdown(f"**{label}**") # Make label bold
        if description:
            st.caption(description) # Use caption for description below label

        # --- Format Main Value ---
        if value is None:
            formatted_value = "N/A"
            numeric_value = None
        elif isinstance(value, (int, float)):
            numeric_value = value
            try:
                 # Apply prefix, formatting, and suffix
                 formatted_value = f"{value_prefix}{value:{value_format}}{value_suffix}"
            except (ValueError, TypeError):
                 formatted_value = f"{value_prefix}{value}{value_suffix}" # Fallback if format fails
                 st.warning(f"Could not apply format '{value_format}' to value '{value}' for KPI '{label}'.", icon="⚠️")
        else: # Handle string values
            formatted_value = str(value)
            numeric_value = None # Cannot calculate % change vs target easily

        # --- Format Delta ---
        formatted_delta = None
        delta_suffix = ""
        if delta is not None and isinstance(delta, (int, float)):
             # Optionally show delta as percentage of previous value (needs previous value passed in)
             # For now, just format the delta number itself
            try:
                formatted_delta = f"{delta:{value_format}}" # Use same format as main value? Or specific delta format?
            except (ValueError, TypeError):
                 formatted_delta = f"{delta}" # Fallback

            # Add arrow based on sign if requested
            if show_arrow:
                 if delta > 0: formatted_delta = f"⬆️ {formatted_delta}"
                 elif delta < 0: formatted_delta = f"⬇️ {formatted_delta}"
                 # else: no arrow for zero delta
            delta_suffix = f" ({delta_label})" # Add label to delta string

        # --- Display using st.metric ---
        # st.metric is good for the core number + delta display
        st.metric(
            label="", # Label is handled above by st.markdown
            value=formatted_value,
            delta=f"{formatted_delta}{delta_suffix}" if formatted_delta else None,
            delta_color=delta_color,
            help=help_text if help_text else description # Use description as help text if no specific help provided
        )

        # --- Display Target (Optional) ---
        if target is not None and isinstance(target, (int, float)):
             formatted_target = f"{value_prefix}{target:{value_format}}{value_suffix}"
             # Display target and potentially % difference from target
             target_text = f"Target: {formatted_target}"
             if numeric_value is not None and target != 0: # Avoid division by zero
                 try:
                    percent_to_target = ((numeric_value - target) / target)
                    target_text += f" ({percent_to_target:+.1%})" # Show percentage difference with sign
                 except Exception: pass # Ignore errors in % calculation
             elif numeric_value is not None and numeric_value == target:
                  target_text += " (✅ Achieved)"

             st.caption(target_text)


# --- Example Usage ---
if __name__ == '__main__':
    st.set_page_config(layout="centered")
    st.header("Enhanced KPI Card Examples")

    st.subheader("Basic Examples")
    col1, col2 = st.columns(2)
    with col1:
        display_kpi(
            label="Total Sales",
            value=125430.50,
            description="Total revenue this period.",
            delta=1230.45,
            delta_label="vs last month",
            value_prefix="$"
        )
    with col2:
         display_kpi(
            label="Avg. Order Value",
            value=85.75,
            delta=-2.10,
            delta_color="inverse",
            value_prefix="$"
         )

    st.subheader("With Targets & Different Formatting")
    col3, col4 = st.columns(2)
    with col3:
         display_kpi(
            label="New Users",
            value=512,
            delta=50,
            target=500,
            show_arrow=True,
            value_format=",.0f", # Format as integer
            help_text="Total unique new users acquired."
         )
    with col4:
         display_kpi(
            label="Churn Rate",
            value=0.0525,
            delta=0.011,
            delta_color="inverse", # Higher churn is bad
            target=0.04,
            description="Monthly customer churn.",
            value_suffix="%",
            value_format=".2%" # Format as percentage
         )

    st.subheader("Handling N/A")
    col5, col6 = st.columns(2)
    with col5:
        display_kpi(label="API Success Rate", value=None, description="Data currently unavailable.")
    with col6:
        display_kpi(label="Error Count", value=15, delta=None, value_format=",.0f")