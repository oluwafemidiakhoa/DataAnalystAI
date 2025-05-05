<<<<<<< HEAD
# frontend/pages/3_🛠️_Explore_Engineer.py
import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import time

# Placeholder for backend imports
# from backend.reporting.visualizer import create_plotly_chart, suggest_visualization # Example
# from backend.data_processing.feature_engineer import suggest_feature_engineering, apply_feature_engineering # Example
# from backend.llm.gemini_utils import generate_code_from_nl # Example

# --- Mock/Placeholder Backend Functions ---
def mock_suggest_visualization(df, user_goal=None):
    # Basic mock based on column types
    time.sleep(1)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    suggestions = []

    if len(numeric_cols) >= 2:
        suggestions.append({"chart_type": "scatter", "x_column": numeric_cols[0], "y_column": numeric_cols[1], "rationale": f"Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}."})
    if categorical_cols and numeric_cols:
        suggestions.append({"chart_type": "bar", "x_column": categorical_cols[0], "y_column": numeric_cols[0], "rationale": f"Compare {numeric_cols[0]} across categories in {categorical_cols[0]}."})
    if numeric_cols:
        suggestions.append({"chart_type": "histogram", "x_column": numeric_cols[0], "y_column": None, "rationale": f"Show distribution of {numeric_cols[0]}."})
    if not suggestions:
         return {"error": "Could not generate mock suggestions for this data."}

    # Select one suggestion for simplicity in mock
    import random
    return random.choice(suggestions)

def mock_create_plotly_chart(df, suggestion):
    # Basic mock chart creation
    chart_type = suggestion.get("chart_type")
    x = suggestion.get("x_column")
    y = suggestion.get("y_column")
    title = f"Mock {chart_type.capitalize()} Chart: {y} by {x}" if y else f"Mock Distribution of {x}"
    try:
        if chart_type == "scatter" and x and y:
            fig = px.scatter(df, x=x, y=y, title=title)
        elif chart_type == "bar" and x and y:
            # Aggregate if necessary for bar chart (basic sum)
            grouped_df = df.groupby(x)[y].sum().reset_index()
            fig = px.bar(grouped_df, x=x, y=y, title=title)
        elif chart_type == "histogram" and x:
             fig = px.histogram(df, x=x, title=title)
        else:
            # Fallback placeholder
             fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Mock Fallback Chart")
        return fig
    except Exception as e:
        logger.error(f"Mock chart generation failed: {e}")
        # Return an empty figure on error
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(title_text=f"Chart Error: {e}")
        return fig


def mock_suggest_feature_engineering(df, goal=None):
    suggestions = []
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if date_cols:
         suggestions.append({
             "type": "Date/Time Extraction",
             "details": f"Extract month, year, day of week from '{date_cols[0]}'.",
             "mock_code": f"df['{date_cols[0]}_month'] = df['{date_cols[0]}'].dt.month\ndf['{date_cols[0]}_dayofweek'] = df['{date_cols[0]}'].dt.dayofweek"
         })
    if len(numeric_cols) >= 2:
         suggestions.append({
             "type": "Interaction Term",
             "details": f"Create interaction term between '{numeric_cols[0]}' and '{numeric_cols[1]}'.",
             "mock_code": f"df['{numeric_cols[0]}_x_{numeric_cols[1]}'] = df['{numeric_cols[0]}'] * df['{numeric_cols[1]}']"
         })
    if not suggestions:
         suggestions.append({"type": "No suggestions", "details": "Could not generate mock feature suggestions."})
    time.sleep(1)
    return suggestions

# Mock apply_code can be reused from page 2

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Page Specific Configuration ---
st.set_page_config(page_title="Explore & Engineer", layout="wide")
st.title("3. 🛠️ Explore & Engineer Features")
st.markdown("Visualize your data and create new features to enhance analysis.")

# --- Initialize Session State ---
if 'cleaned_dataframe' not in st.session_state: st.session_state.cleaned_dataframe = None
if 'engineered_dataframe' not in st.session_state: st.session_state.engineered_dataframe = None # Result of this step
if 'current_viz_suggestion' not in st.session_state: st.session_state.current_viz_suggestion = None
if 'current_plot' not in st.session_state: st.session_state.current_plot = None
if 'feature_suggestions' not in st.session_state: st.session_state.feature_suggestions = None
if 'applied_feature_steps' not in st.session_state: st.session_state.applied_feature_steps = []

# --- Check if Data is Available ---
if st.session_state.cleaned_dataframe is None:
    st.warning("⚠️ Please connect and optionally clean/transform data in the previous steps first.")
    st.stop()

# --- Use cleaned_dataframe as starting point if engineered_dataframe doesn't exist ---
if st.session_state.engineered_dataframe is None:
    st.session_state.engineered_dataframe = st.session_state.cleaned_dataframe.copy()
    st.session_state.applied_feature_steps = ["Initial load from Clean/Transform step."] # Reset steps


# --- Main Layout: Exploration | Feature Engineering ---
explore_col, engineer_col = st.columns(2)

with explore_col:
    st.subheader("Data Exploration & Visualization")
    st.dataframe(st.session_state.engineered_dataframe.head(), use_container_width=True)
    st.caption(f"Current Data Shape: {st.session_state.engineered_dataframe.shape}")

    exploration_goal = st.text_input(
        "What do you want to explore or visualize?",
        placeholder="e.g., relationship between sales and marketing spend",
        key="explore_goal_input"
    )

    if st.button("Suggest Visualization", key="suggest_explore_viz_btn"):
        with st.spinner("Generating visualization suggestion..."):
            try:
                # Replace with actual backend call
                suggestion = mock_suggest_visualization(st.session_state.engineered_dataframe, exploration_goal)
                st.session_state.current_viz_suggestion = suggestion
                # Clear previous plot
                st.session_state.current_plot = None
                if "error" in suggestion:
                     st.warning(f"Could not get suggestion: {suggestion['error']}")
                else:
                     st.success("Suggestion received.")
                     # Attempt to generate plot immediately based on suggestion
                     try:
                         st.session_state.current_plot = mock_create_plotly_chart(
                             st.session_state.engineered_dataframe,
                             suggestion
                         )
                     except Exception as plot_e:
                         st.error(f"Failed to automatically generate plot: {plot_e}")
                         logger.error(f"Auto-plot generation failed: {plot_e}", exc_info=True)

            except Exception as e:
                st.error(f"Failed to get visualization suggestion: {e}")
                logger.error(f"Viz suggestion failed: {e}", exc_info=True)

    # Display suggestion and plot
    if st.session_state.current_viz_suggestion and "error" not in st.session_state.current_viz_suggestion:
        st.caption("Suggestion:")
        st.json(st.session_state.current_viz_suggestion)
        if st.session_state.current_plot:
            st.plotly_chart(st.session_state.current_plot, use_container_width=True)
            rationale = st.session_state.current_viz_suggestion.get('rationale', '')
            if rationale:
                st.caption(f"Rationale: {rationale}")
        # Optionally add button to regenerate plot if needed
        # if st.button("Generate Plot from Suggestion", key="regen_plot_btn"): ...


with engineer_col:
    st.subheader("Feature Engineering")
    feature_goal = st.text_input(
        "What is your goal for feature engineering?",
        placeholder="e.g., Prepare features for sales forecasting",
        key="feature_goal_input"
    )

    if st.button("Suggest New Features", key="suggest_features_btn"):
        with st.spinner("Analyzing data for feature engineering..."):
            try:
                 # Replace with actual backend call
                suggestions = mock_suggest_feature_engineering(st.session_state.engineered_dataframe, feature_goal)
                st.session_state.feature_suggestions = suggestions
            except Exception as e:
                 st.error(f"Failed to get feature suggestions: {e}")
                 logger.error(f"Feature suggestion failed: {e}", exc_info=True)

    if st.session_state.feature_suggestions:
        st.markdown("**Suggested Features:**")
        for i, feature_sugg in enumerate(st.session_state.feature_suggestions):
            with st.expander(f"**{feature_sugg['type']}**: {feature_sugg.get('details', 'N/A')}", expanded=i==0):
                mock_code = feature_sugg.get('mock_code')
                if mock_code:
                    st.code(mock_code, language='python')
                    if st.button(f"Generate Feature(s) {i+1}", key=f"apply_feat_{i}"):
                        with st.spinner("Generating feature(s)..."):
                             try:
                                 # Replace with actual backend call to apply safe transformations
                                 df_before = st.session_state.engineered_dataframe
                                 # Re-use mock apply code (Needs safe implementation in real app)
                                 st.session_state.engineered_dataframe = mock_apply_code(df_before.copy(), mock_code)
                                 st.session_state.applied_feature_steps.append(f"Applied feature: {feature_sugg['type']}")
                                 st.success(f"Generated: {feature_sugg['type']}")
                                 st.rerun() # Update preview
                             except Exception as e:
                                 st.error(f"Failed to generate feature: {e}")
                                 logger.error(f"Apply feature failed: {e}", exc_info=True)


    # --- Display Applied Steps ---
    st.markdown("---")
    st.subheader("Applied Feature Engineering Steps")
    if len(st.session_state.applied_feature_steps) > 1: # More than just initial load message
        st.dataframe(pd.Series(st.session_state.applied_feature_steps[1:], name="Action"), use_container_width=True)
    else:
        st.info("No feature engineering steps applied yet.")

    # --- Reset Button ---
    if st.button("Reset to Cleaned Data", key="reset_engineer_btn"):
        st.session_state.engineered_dataframe = st.session_state.cleaned_dataframe.copy()
        st.session_state.applied_feature_steps = ["Reset to cleaned data."]
        st.session_state.feature_suggestions = None # Clear suggestions too
        st.success("Data reset to state after Clean/Transform step.")
        st.rerun()

# --- Next Step Hint ---
st.success("➡️ Proceed to **4_📊_Analyze_Insight** to query your prepared data and find insights.")
=======
# pages/3_🛠️_Explore_Engineer.py
# Note: This file should be in the 'pages/' directory at the project root.

import streamlit as st
import pandas as pd
import numpy as np # Import numpy for numeric checks
import plotly.express as px
import plotly.graph_objects as go # Import go for empty figures
import logging
import time
import io # Import io for schema context if needed
from typing import List, Dict, Any, Optional # For type hints


# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Backend Function Imports ---
# Assumes 'backend' is importable and mocks are in pages/_mocks.py
BACKEND_AVAILABLE = False
# Initialize function variables to None or dummy functions first
suggest_visualization_func = None # Initialize
create_plotly_chart_func = None
suggest_feature_engineering_func = None
apply_feature_engineering_func = None # Should point to structured apply if available
explain_code_func = None

try:
    # Import REAL backend functions
    from backend.reporting.visualizer import create_plotly_chart, suggest_visualization
    from backend.data_processing.feature_engineer import suggest_feature_engineering, apply_feature_engineering # Import structured apply function
    from backend.llm.gemini_utils import explain_code_llm # Assuming this exists for explaining structured steps/code

    # Assign real functions
    suggest_visualization_func = suggest_visualization # Assign REAL function
    create_plotly_chart_func = create_plotly_chart
    suggest_feature_engineering_func = suggest_feature_engineering
    apply_feature_engineering_func = apply_feature_engineering # Assign REAL structured apply function
    explain_code_func = explain_code_llm

    BACKEND_AVAILABLE = True
    logger.info("Backend modules imported successfully in Explore_Engineer page.")

except ImportError as e:
    logger.error(f"Backend import failed in Explore_Engineer: {e}", exc_info=True)
    st.error(f"CRITICAL: Backend modules not found. Using MOCK functions. Error: {e}", icon="🚨")
    # Import MOCK functions as fallback
    try:
        from ._mocks import (
            mock_suggest_visualization,
            mock_create_plotly_chart,
            mock_suggest_feature_engineering,
            # Use mock_apply_code as the implementation for the structured apply mock for now
            mock_apply_code as mock_apply_feature_engineering,
            mock_explain_code
        )
        # Assign MOCK functions
        suggest_visualization_func = mock_suggest_visualization # Assign MOCK function
        create_plotly_chart_func = mock_create_plotly_chart
        suggest_feature_engineering_func = mock_suggest_feature_engineering
        apply_feature_engineering_func = mock_apply_feature_engineering # Assign MOCK function
        explain_code_func = mock_explain_code
        logger.info("Loaded MOCK functions for Explore_Engineer.")
    except ImportError as mock_e:
         logger.critical(f"Failed to import mock functions from pages._mocks! Error: {mock_e}", exc_info=True)
         st.error(f"CRITICAL ERROR: Could not load fallback functions: {mock_e}", icon="🚨"); st.stop()
    except Exception as general_mock_e:
         logger.critical(f"Unexpected error importing mocks: {general_mock_e}", exc_info=True); st.error(f"Unexpected critical error during setup: {general_mock_e}", icon="🚨"); st.stop()

except Exception as e:
     logger.critical(f"Unexpected error during backend import: {e}", exc_info=True); st.error(f"A critical error occurred during setup: {e}", icon="🚨"); st.stop()


# --- Safety Check ---
# Verify all functions needed by this page are assigned
if not all([suggest_visualization_func, create_plotly_chart_func, suggest_feature_engineering_func, apply_feature_engineering_func, explain_code_func]):
    logger.critical("One or more essential functions (real or mock) failed to load for Explore/Engineer.")
    st.error("Critical application setup error for this page. Check logs.", icon="🚨")
    st.stop()


# --- Page Title and Introduction ---
st.header("3. 🛠️ Explore & Engineer Features")
st.markdown("""
Interactively explore your prepared data through visualizations using AI suggestions.
Generate potentially insightful new features based on your analysis goals, review the
suggested actions and AI explanations, and apply them safely to enhance your dataset.
""")
st.divider()

# --- Initialize Session State ---
keys_needed = ['cleaned_dataframe', 'engineered_dataframe', 'current_viz_suggestion',
               'current_plot', 'feature_suggestions', 'applied_feature_steps']
# Add keys for explanations tied to feature suggestions dynamically later
for key in keys_needed:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'applied_feature_steps' else []

# --- Check if Data is Available ---
if 'cleaned_dataframe' not in st.session_state or st.session_state.cleaned_dataframe is None:
    st.warning("⚠️ Please complete the 'Connect & Profile' and 'Clean & Transform' steps first.")
    st.stop()

# --- Initialize or Reset Engineered Data ---
if st.session_state.engineered_dataframe is None:
    logger.info("Initializing engineered_dataframe from cleaned_dataframe.")
    st.session_state.engineered_dataframe = st.session_state.cleaned_dataframe.copy()
    st.session_state.applied_feature_steps = ["Initial load from Clean/Transform step."]


# --- Display Current Data State ---
st.subheader("Current Data Preview (Engineered)")
preview_container = st.container(border=True)
with preview_container:
    display_df = st.session_state.engineered_dataframe
    st.dataframe(display_df.head(), use_container_width=True)
    st.caption(f"Shape: {display_df.shape}")
st.divider()

# --- Main Layout: Exploration Column | Feature Engineering Column ---
explore_col, engineer_col = st.columns([0.5, 0.5]) # Equal columns

# --- Exploration Column ---
with explore_col:
    st.subheader("🔎 Data Exploration")
    explore_container = st.container(border=True)
    with explore_container:
        exploration_goal = st.text_input(
            "What do you want to explore or visualize?",
            placeholder="e.g., relationship between sales and marketing spend",
            key="explore_goal_input",
            help="Describe what pattern or insight you are looking for."
        )

        if st.button("📊 Suggest & Generate Visualization", key="suggest_explore_viz_btn", use_container_width=True, type="primary"):
            with st.spinner("Generating visualization suggestion and plot..."):
                try:
                    df_to_analyze = st.session_state.engineered_dataframe
                    # ** FIX: Call the correctly assigned function variable **
                    suggestion = suggest_visualization_func(df_to_analyze, exploration_goal)
                    st.session_state.current_viz_suggestion = suggestion
                    st.session_state.current_plot = None # Clear previous plot

                    if isinstance(suggestion, dict) and "error" in suggestion:
                        st.warning(f"Could not get suggestion: {suggestion['error']}")
                    elif isinstance(suggestion, dict) and suggestion.get("chart_type"):
                        st.toast("Suggestion received. Generating plot...", icon="📊")
                        try:
                             # ** FIX: Call the correctly assigned function variable **
                            st.session_state.current_plot = create_plotly_chart_func(df_to_analyze, suggestion)
                            # No rerun needed, display happens below based on state
                        except Exception as plot_e:
                            st.error(f"Failed to generate plot from suggestion: {plot_e}")
                            logger.error(f"Plot generation failed: {plot_e}", exc_info=True)
                    else:
                         st.info("AI could not determine a suitable visualization suggestion.")

                except Exception as e:
                    st.error(f"Failed to get visualization suggestion: {e}")
                    logger.error(f"Viz suggestion failed: {e}", exc_info=True)

        # --- Display Suggestion ---
        if st.session_state.current_viz_suggestion and isinstance(st.session_state.current_viz_suggestion, dict) and "error" not in st.session_state.current_viz_suggestion:
            st.markdown("**AI Suggestion:**")
            suggestion = st.session_state.current_viz_suggestion
            sug_col1, sug_col2 = st.columns(2); sug_col3, sug_col4 = st.columns(2)
            with sug_col1: st.metric("Chart Type", suggestion.get("chart_type", "N/A"))
            with sug_col2: st.metric("X-Axis", suggestion.get("x_column", "N/A"))
            with sug_col3: st.metric("Y-Axis", suggestion.get("y_column", "N/A"))
            with sug_col4: st.metric("Color By", suggestion.get("color_column", "N/A"))
            st.caption(f"Rationale: {suggestion.get('rationale', 'N/A')}")
            st.divider()

        # --- Display Plot ---
        if st.session_state.current_plot:
             st.markdown("**Generated Visualization:**")
             st.plotly_chart(st.session_state.current_plot, use_container_width=True)
        else:
             # Show only if a suggestion was made but plot failed or hasn't been generated
             if st.session_state.current_viz_suggestion and "error" not in st.session_state.current_viz_suggestion:
                 st.warning("Plot generation failed or is pending.")
             else:
                 st.info("Suggest a visualization to generate a plot.")


# --- Feature Engineering Column ---
with engineer_col:
    st.subheader("✨ Feature Engineering")
    engineer_container = st.container(border=True)
    with engineer_container:
        feature_goal = st.text_input(
            "Goal for new features?",
            placeholder="e.g., Create time-based features for forecasting",
            key="feature_goal_input",
            help="Helps AI suggest relevant features (optional)."
        )

        if st.button("💡 Suggest New Features", key="suggest_features_btn", use_container_width=True):
            with st.spinner("Analyzing data for feature engineering..."):
                try:
                    # ** FIX: Call the correctly assigned function variable **
                    suggestions = suggest_feature_engineering_func(st.session_state.engineered_dataframe, feature_goal)
                    st.session_state.feature_suggestions = suggestions or [] # Ensure list
                    st.toast(f"Generated {len(suggestions)} feature suggestions.", icon="💡")
                except Exception as e:
                     st.error(f"Failed to get feature suggestions: {e}")
                     logger.error(f"Feature suggestion failed: {e}", exc_info=True)

        # --- Display Feature Suggestions ---
        if st.session_state.feature_suggestions:
            st.markdown("**Suggested Features:**")
            if not st.session_state.feature_suggestions or st.session_state.feature_suggestions[0].get("type") == "No Data" or st.session_state.feature_suggestions[0].get("type") == "No Suggestions":
                 st.info("No feature suggestions available or generated.")
            else:
                for i, feature_sugg in enumerate(st.session_state.feature_suggestions):
                    step_type_key = feature_sugg.get('step_type','Sugg').replace(' ','_')
                    unique_key_base = f"feat_{i}_{step_type_key}"
                    expander_label = f"**{feature_sugg.get('type', step_type_key)}**: {feature_sugg.get('details', 'N/A')}"

                    with st.expander(expander_label, expanded=i < 1):
                        st.caption(f"Rationale: {feature_sugg.get('rationale', '_N/A_')}")

                        step_type = feature_sugg.get('step_type')
                        params = feature_sugg.get('params', {})
                        suggested_code = feature_sugg.get('suggested_code') # Fallback if structured not generated

                        # Display parameters or code
                        display_content = None
                        if step_type and params:
                            display_content = params; st.markdown("**Action Parameters:**"); st.json(params)
                        elif suggested_code:
                             display_content = suggested_code; st.markdown("**Suggested Code:**"); st.code(suggested_code, language='python')
                        else: st.info("Suggestion provided without specific parameters or code.")

                        # --- Explain Button ---
                        if display_content:
                            explain_key = f"{unique_key_base}_explanation"
                            if st.button("Explain Action/Code", key=f"{unique_key_base}_explain"):
                                with st.spinner("AI explaining..."):
                                     try:
                                         content_to_explain = f"Action: {step_type}, Params: {params}" if step_type and params else suggested_code
                                         # ** FIX: Call the correctly assigned function variable **
                                         explanation = explain_code_func(content_to_explain, "python action/code")
                                         st.session_state[explain_key] = explanation
                                     except Exception as e: st.warning(f"Could not get explanation: {e}"); st.session_state[explain_key] = f"Error explaining: {e}"
                            explanation = st.session_state.get(explain_key)
                            if explanation: st.caption(explanation)
                            else: st.caption("_Click 'Explain' for AI description._")

                        # --- Confirmation and Apply ---
                        st.warning("⚠️ Review action/code AND explanation carefully before applying!")
                        if st.button("✅ Apply Feature Generation", key=f"{unique_key_base}_apply", type="primary"):
                            if not step_type and not suggested_code: st.error("Cannot apply suggestion without step/code.")
                            else:
                                 with st.spinner("Applying feature generation..."):
                                     try:
                                         df_before = st.session_state.engineered_dataframe
                                         # ** FIX: Call the correctly assigned function variable **
                                         # Prioritize structured apply function if step_type and params exist
                                         if step_type and params:
                                             st.session_state.engineered_dataframe = apply_feature_engineering_func(df_before.copy(), step_type, params)
                                         elif suggested_code: # Fallback ONLY if needed and using a mock/sandbox function
                                             logger.warning(f"Applying feature '{expander_label}' using CODE EXECUTION (potentially unsafe).")
                                             # Assumes apply_feature_engineering_func might handle code via mock/sandbox for now
                                             st.session_state.engineered_dataframe = apply_feature_engineering_func(df_before.copy(), suggested_code) # Requires mock to accept code
                                         else: raise ValueError("No valid action to apply.")

                                         step_desc = f"Applied feature: {feature_sugg.get('type', step_type)}"
                                         new_cols = set(st.session_state.engineered_dataframe.columns) - set(df_before.columns)
                                         if new_cols: step_desc += f" (New cols: {', '.join(new_cols)})"
                                         st.session_state.applied_feature_steps.append(step_desc)
                                         st.toast(f"Generated: {feature_sugg.get('type', step_type)}", icon="✨")
                                         st.rerun()
                                     except Exception as e: st.error(f"Apply Failed: {e}"); logger.error(f"Apply feature failed: {e}", exc_info=True)
                        st.markdown("---")


    # --- Display Applied Feature Steps & Reset ---
    st.markdown("---")
    steps_col, reset_col = st.columns([0.7, 0.3])
    with steps_col:
        st.subheader("📜 Applied Feature Steps")
        steps_container = st.container(height=150, border=True)
        with steps_container:
            steps_to_show = st.session_state.applied_steps[1:] if len(st.session_state.applied_steps) > 1 else []
            if steps_to_show:
                for i, step in enumerate(reversed(steps_to_show)): st.text(f"{len(steps_to_show)-i}. {step}")
            else: st.info("No feature steps applied yet.")

    with reset_col:
        st.subheader("Reset")
        if st.button("⏮️ Reset to Cleaned Data", key="reset_engineer_btn", use_container_width=True):
            if 'cleaned_dataframe' in st.session_state and st.session_state.cleaned_dataframe is not None:
                st.session_state.engineered_dataframe = st.session_state.cleaned_dataframe.copy()
                st.session_state.applied_feature_steps = ["Reset to cleaned data."]
                st.session_state.feature_suggestions = None
                keys_to_clear = [k for k in st.session_state if k.startswith('feat_') and k.endswith('_explanation')]
                for k in keys_to_clear: del st.session_state[k]
                st.toast("Data reset to state after Clean/Transform.", icon="🔄"); time.sleep(1); st.rerun()
            else: st.error("Cleaned data not available.")

st.divider()
# --- Next Step Hint ---
st.success("➡️ Data explored and features engineered! Proceed to **4_📊_Analyze_Insight**.")
>>>>>>> 946a937 (Add application file)
