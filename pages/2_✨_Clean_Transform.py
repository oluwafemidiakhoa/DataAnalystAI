<<<<<<< HEAD
# frontend/pages/2_✨_Clean_Transform.py
import streamlit as st
import pandas as pd
import logging
import time # For simulating processes

# Placeholder for backend imports
# from backend.data_processing.cleaner import suggest_cleaning_steps, apply_cleaning # Example
# from backend.data_processing.transformer import apply_transformation # Example
# from backend.llm.gemini_utils import generate_code_from_nl # Example

# --- Mock/Placeholder Backend Functions ---
def mock_suggest_cleaning_steps(df):
    suggestions = []
    # Missing values
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    if not missing_cols.empty:
        suggestions.append({
            "issue": "Missing Values",
            "details": f"Columns with missing values: {', '.join(missing_cols.index.tolist())} (Counts: {missing_cols.to_dict()})",
            "recommendation": "Consider imputation (mean, median, mode) or dropping rows/columns based on percentage missing.",
            "mock_code": f"# Example: df['{missing_cols.index[0]}'].fillna(df['{missing_cols.index[0]}'].median(), inplace=True)"
        })
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
         suggestions.append({
            "issue": "Duplicate Rows",
            "details": f"Found {duplicates} duplicate rows.",
            "recommendation": "Recommend dropping duplicate rows.",
            "mock_code": "df.drop_duplicates(inplace=True)"
        })
    # Data Types (Example: find object columns that might be numeric/date)
    object_cols = df.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        suggestions.append({
            "issue": "Potential Data Type Issues",
            "details": f"Object columns found: {', '.join(object_cols)}. Review if any should be numeric, datetime, etc.",
            "recommendation": f"Use pd.to_numeric() or pd.to_datetime().",
            "mock_code": f"# Example: df['{object_cols[0]}'] = pd.to_numeric(df['{object_cols[0]}'], errors='coerce')"
        })
    if not suggestions:
        suggestions.append({"issue": "No obvious issues found", "details": "Data looks initially clean.", "recommendation": "Proceed or apply custom transformations."})
    time.sleep(2)
    return suggestions

def mock_generate_code_from_nl(nl_command, df_schema_context):
    # Very basic mock - real implementation needs Gemini call
    time.sleep(1)
    if "combine" in nl_command and "first" in nl_command and "last" in nl_command:
        return "df['full_name'] = df['first_name'] + ' ' + df['last_name']"
    elif "calculate profit" in nl_command:
        return "df['profit'] = df['revenue'] - df['cost']"
    elif "convert" in nl_command and "numeric" in nl_command:
        col = nl_command.split("'")[1] if "'" in nl_command else "some_column"
        return f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
    else:
        return f"# Mock code for: {nl_command}\nprint('Transformation applied (mock)')"

def mock_apply_code(df, code):
    # !!! WARNING: In a real app, executing arbitrary code is a HUGE security risk!
    # This mock just simulates it. Real implementation needs sandboxing or
    # a library that parses steps safely, or Gemini Function Calling.
    # For this example, we'll just print and return the original df.
    print(f"--- MOCK EXECUTION START ---\n{code}\n--- MOCK EXECUTION END ---")
    time.sleep(1)
    # Simulate a change for demonstration if code seems valid
    if 'df[' in code and '=' in code:
         try:
              # Try a simple heuristic to apply change if possible (VERY basic)
              if "'full_name'" in code and "'first_name'" in df.columns and "'last_name'" in df.columns:
                   df['full_name'] = df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)
              elif "'profit'" in code and "'revenue'" in df.columns and "'cost'" in df.columns:
                   df['profit'] = pd.to_numeric(df['revenue'], errors='coerce') - pd.to_numeric(df['cost'], errors='coerce')
              else:
                   print("Mock applied, but no specific change simulated for this code.")
         except Exception as e:
              print(f"Mock apply simulation error: {e}")
              # Return original df on error
    return df.copy() # Return a copy to simulate modification


# --- Logging ---
logger = logging.getLogger(__name__)

# --- Page Specific Configuration ---
st.set_page_config(page_title="Clean & Transform", layout="wide")
st.title("2. ✨ Clean & Transform Data")
st.markdown("Apply cleaning steps and transformations to prepare your data for analysis.")

# --- Initialize Session State Keys for this Page ---
if 'raw_dataframe' not in st.session_state: st.session_state.raw_dataframe = None # Loaded in previous step
if 'cleaned_dataframe' not in st.session_state: st.session_state.cleaned_dataframe = None # Result of this step
if 'cleaning_suggestions' not in st.session_state: st.session_state.cleaning_suggestions = None
if 'applied_steps' not in st.session_state: st.session_state.applied_steps = [] # Track applied actions

# --- Check if Data is Loaded ---
if st.session_state.raw_dataframe is None:
    st.warning("⚠️ Please connect to a file-based data source or query a database in **1_🔗_Connect_Profile** first.")
    # TODO: Add logic here to load data if connected to a DB (e.g., query a selected table)
    st.stop()

# --- Use raw_dataframe as starting point if cleaned_dataframe doesn't exist ---
if st.session_state.cleaned_dataframe is None:
    st.session_state.cleaned_dataframe = st.session_state.raw_dataframe.copy()
    st.session_state.applied_steps = ["Initial load from source."] # Reset steps

# --- Display Current DataFrame ---
st.subheader("Current Data Preview")
st.dataframe(st.session_state.cleaned_dataframe.head(), use_container_width=True)
st.caption(f"Shape: {st.session_state.cleaned_dataframe.shape}")

# --- Main Layout: Cleaning Suggestions | Transformations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Automated Cleaning Suggestions")
    if st.button("Analyze Data for Cleaning Steps", key="analyze_cleaning_btn"):
        with st.spinner("Analyzing data and generating suggestions..."):
            try:
                # Replace with actual backend call
                suggestions = mock_suggest_cleaning_steps(st.session_state.cleaned_dataframe)
                st.session_state.cleaning_suggestions = suggestions
            except Exception as e:
                st.error(f"Failed to get cleaning suggestions: {e}")
                logger.error(f"Cleaning suggestion failed: {e}", exc_info=True)

    if st.session_state.cleaning_suggestions:
        for i, suggestion in enumerate(st.session_state.cleaning_suggestions):
            with st.expander(f"**{suggestion['issue']}**: {suggestion['recommendation']}", expanded=i==0):
                st.markdown(f"**Details:** {suggestion.get('details', 'N/A')}")
                mock_code = suggestion.get('mock_code')
                if mock_code:
                    st.code(mock_code, language='python')
                    if st.button(f"Apply Suggestion {i+1}", key=f"apply_sugg_{i}"):
                        with st.spinner("Applying suggestion..."):
                             try:
                                 # Replace with actual backend call to apply safe transformations
                                 df_before = st.session_state.cleaned_dataframe
                                 st.session_state.cleaned_dataframe = mock_apply_code(df_before.copy(), mock_code)
                                 st.session_state.applied_steps.append(f"Applied suggestion: {suggestion['issue']}")
                                 st.success(f"Applied: {suggestion['issue']}")
                                 st.rerun() # Update preview
                             except Exception as e:
                                 st.error(f"Failed to apply suggestion: {e}")
                                 logger.error(f"Apply suggestion failed: {e}", exc_info=True)


with col2:
    st.subheader("Apply Transformations (Natural Language)")
    nl_transform_command = st.text_area(
        "Describe the transformation you want to apply:",
        placeholder="e.g., Combine 'first_name' and 'last_name' into 'full_name'",
        key="nl_transform_input"
    )

    if st.button("Generate & Apply Transformation Code", key="apply_transform_btn"):
        if not nl_transform_command:
            st.warning("Please describe the transformation.")
        else:
            with st.spinner("Generating and applying transformation code..."):
                try:
                    # Get schema context for the LLM
                    df = st.session_state.cleaned_dataframe
                    schema_context = f"DataFrame columns: {', '.join(df.columns)}" # Basic context
                    # Replace with actual backend calls
                    generated_code = mock_generate_code_from_nl(nl_transform_command, schema_context)
                    st.code(generated_code, language='python') # Show the code

                    # --- Apply Code (Needs user confirmation or safe execution) ---
                    # WARNING: Direct execution is risky. This is just a placeholder.
                    # In a real app, you might show the code and have an "Apply" button
                    # that calls a secure backend execution environment.
                    st.session_state.cleaned_dataframe = mock_apply_code(df.copy(), generated_code)
                    st.session_state.applied_steps.append(f"Applied NL transform: {nl_transform_command[:50]}...")
                    st.success("Transformation applied (mock execution).")
                    st.rerun() # Update preview

                except Exception as e:
                    st.error(f"Failed to apply transformation: {e}")
                    logger.error(f"NL Transformation failed: {e}", exc_info=True)

# --- Display Applied Steps ---
st.subheader("Applied Steps")
if st.session_state.applied_steps:
    st.dataframe(pd.Series(st.session_state.applied_steps, name="Action"), use_container_width=True)
else:
    st.info("No cleaning or transformation steps applied yet.")

# --- Reset Button ---
if st.button("Reset to Raw Data", key="reset_clean_btn"):
    st.session_state.cleaned_dataframe = st.session_state.raw_dataframe.copy()
    st.session_state.applied_steps = ["Reset to raw data."]
    st.session_state.cleaning_suggestions = None # Clear suggestions too
    st.success("Data reset to its original state.")
    st.rerun()


# --- Next Step Hint ---
st.success("➡️ Proceed to **3_🛠️_Explore_Engineer** to visualize and create new features.")
=======
# pages/2_✨_Clean_Transform.py
# Note: This file should be in the 'pages/' directory at the project root.

import streamlit as st
import pandas as pd
import logging
import io # Import io for schema generation
import time
from typing import List, Dict, Any, Optional # For type hints, added Optional

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Backend Function Imports ---
# Assumes 'backend' is importable and mocks are in pages/_mocks.py
BACKEND_AVAILABLE = False
# Initialize function variables to None or dummy functions first
suggest_cleaning_steps_func = None
apply_cleaning_step_func = None # This should ideally point to a function accepting structured params
generate_transformation_code_func = None # Correct initialization name
apply_transformation_code_func = None # Correct initialization name
explain_code_func = None

try:
    # Import REAL backend functions
    from backend.data_processing.cleaner import suggest_cleaning_steps, apply_cleaning_step
    from backend.data_processing.transformer import generate_transformation_code, apply_transformation # Changed import
    # ** NEW: Import function to explain code **
    from backend.llm.gemini_utils import explain_code_llm # Assuming this exists
    # Placeholder for safe execution function (replace mock_apply_code)
    # from backend.sandbox.executor import safe_execute_code # Example of safe execution import

    # Assign real functions if import succeeds
    suggest_cleaning_steps_func = suggest_cleaning_steps
    apply_cleaning_step_func = apply_cleaning_step # Use the function accepting structured params
    generate_transformation_code_func = generate_transformation_code # Correct assignment name
    apply_transformation_code_func = apply_transformation # Correct assignment name
    explain_code_func = explain_code_llm

    BACKEND_AVAILABLE = True
    logger.info("Backend modules imported successfully in Clean_Transform page.")

except ImportError as e:
    logger.error(f"Backend import failed in Clean_Transform: {e}", exc_info=True)
    st.error(f"CRITICAL: Backend modules could not be loaded. Using MOCK functions. Error: {e}", icon="🚨")
    # Import MOCK functions as fallback
    try:
        from ._mocks import (
            mock_suggest_cleaning_steps,
            mock_apply_code as mock_apply_structured_step, # Map mock_apply_code to the structured apply for now
            mock_generate_code_from_nl,
            mock_apply_code as mock_apply_transformation_code, # Use mock_apply_code for transformation mock
            mock_explain_code
        )
        # Assign MOCK functions
        suggest_cleaning_steps_func = mock_suggest_cleaning_steps
        apply_cleaning_step_func = mock_apply_structured_step # Pointing to mock_apply_code
        generate_transformation_code_func = mock_generate_code_from_nl # Correct assignment name
        apply_transformation_code_func = mock_apply_transformation_code # Correct assignment name
        explain_code_func = mock_explain_code
        logger.info("Loaded MOCK functions for Clean_Transform.")
    except ImportError as mock_e:
         logger.critical(f"Failed to import mock functions from pages._mocks! Error: {mock_e}", exc_info=True)
         st.error(f"CRITICAL ERROR: Could not load fallback functions: {mock_e}", icon="🚨"); st.stop()
    except Exception as general_mock_e:
         logger.critical(f"Unexpected error importing mocks: {general_mock_e}", exc_info=True); st.error(f"Unexpected critical error during setup: {general_mock_e}", icon="🚨"); st.stop()

except Exception as e:
     logger.critical(f"Unexpected error during backend import: {e}", exc_info=True); st.error(f"A critical error occurred during setup: {e}", icon="🚨"); st.stop()

# --- Safety Check ---
# Ensure functions are assigned before proceeding
# Using the correct variable name in the check
if not all([suggest_cleaning_steps_func, apply_cleaning_step_func, generate_transformation_code_func, apply_transformation_code_func, explain_code_func]):
    logger.critical("One or more essential functions (real or mock) failed to load for Clean/Transform.")
    st.error("Critical application setup error for this page. Check logs.", icon="🚨")
    st.stop()


# --- Page Title and Introduction ---
st.header("2. ✨ Clean & Transform Data")
st.markdown("""
Review AI-suggested cleaning actions or describe transformations using natural language.
Generated code (where applicable) will be explained by AI. **Confirm carefully before applying any changes.**
""")
st.divider()

# --- Initialize Session State ---
keys_needed = ['raw_dataframe', 'cleaned_dataframe', 'cleaning_suggestions', 'applied_steps',
               'current_transform_code', 'current_transform_explanation'] # Added keys
for key in keys_needed:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'applied_steps' else "" if key in ['current_transform_code', 'current_transform_explanation'] else None

# --- Check if Data is Loaded ---
data_loaded = False
if 'raw_dataframe' in st.session_state and st.session_state.raw_dataframe is not None:
    data_loaded = True
    if st.session_state.cleaned_dataframe is None:
        logger.info("Initializing cleaned_dataframe from raw_dataframe.")
        st.session_state.cleaned_dataframe = st.session_state.raw_dataframe.copy()
        st.session_state.applied_steps = ["Initial load from source."]
# Add DB loading logic here...
elif 'connection_info' in st.session_state and st.session_state.connection_info is not None:
     st.info(f"Connected to {st.session_state.connection_info.get('type')}. Add UI to load data.")
     st.warning("⚠️ File-based data needed for current Clean/Transform steps.")
     st.stop()

if not data_loaded or st.session_state.cleaned_dataframe is None:
    st.warning("⚠️ Please connect to a file in **1_🔗_Connect_Profile** first.")
    st.stop()


# --- Display Current Data State ---
st.subheader("Current Data Preview (Cleaned/Transformed)")
preview_container = st.container(border=True)
with preview_container:
    st.dataframe(st.session_state.cleaned_dataframe.head(), use_container_width=True)
    st.caption(f"Shape: {st.session_state.cleaned_dataframe.shape}")
st.divider()

# --- Main Layout: Suggestions & History | Transformations ---
col1, col2 = st.columns([0.5, 0.5])

# --- Column 1: Automated Cleaning Suggestions & History ---
with col1:
    st.subheader("🤖 Automated Cleaning Suggestions")
    analyze_btn_placeholder = st.empty()
    suggestions_container = st.container(border=True)

    with suggestions_container:
        # --- Button to generate suggestions ---
        if st.session_state.cleaning_suggestions is None:
            if analyze_btn_placeholder.button("🔍 Analyze for Cleaning Steps", key="analyze_cleaning_btn", use_container_width=True):
                with st.spinner("Analyzing data and generating suggestions..."):
                    try:
                        suggestions = suggest_cleaning_steps_func(st.session_state.cleaned_dataframe) # Use selected func
                        st.session_state.cleaning_suggestions = suggestions or [] # Ensure list
                        st.rerun()
                    except Exception as e: st.error(f"Suggestion generation failed: {e}"); logger.error(f"Cleaning suggestion failed: {e}", exc_info=True)
        else:
             analyze_btn_placeholder.empty() # Hide button after analysis

             # --- Display Suggestions ---
             if not st.session_state.cleaning_suggestions or st.session_state.cleaning_suggestions[0].get("issue") == "No Data": st.info("No cleaning suggestions available.")
             elif st.session_state.cleaning_suggestions[0].get("issue") == "Initial Scan Clean": st.success("✅ Initial scan found no obvious data quality issues.")
             else:
                 st.markdown("**Review suggested actions:**")
                 for i, suggestion in enumerate(st.session_state.cleaning_suggestions):
                     # Create a more unique key based on content if possible
                     unique_key_base = f"sugg_{i}_{suggestion.get('step_type','unknown')}_{suggestion.get('params',{}).get('column','all')}"
                     expander_label = f"**{suggestion.get('issue', suggestion.get('step_type', 'Suggestion'))}**: {suggestion.get('recommendation', suggestion.get('details', 'N/A'))}"

                     with st.expander(expander_label, expanded=i < 1):
                         step_type = suggestion.get('step_type')
                         params = suggestion.get('params', {})
                         rationale = suggestion.get('rationale', '')
                         suggested_code = suggestion.get('suggested_code') # Might be present from heuristics/old mocks

                         st.markdown(f"**Details:** {suggestion.get('details', 'N/A')}")
                         if rationale: st.caption(f"Rationale: {rationale}")

                         # --- Display Action Parameters/Code ---
                         # Display structured params if available, else show code if provided
                         action_explanation = ""
                         if step_type and params:
                             st.write("Action Parameters:")
                             st.json(params) # Display structured parameters clearly
                             # Button to get explanation for structured step
                             if st.button("Explain Action", key=f"{unique_key_base}_explain_struct"):
                                 with st.spinner("AI explaining..."):
                                     try: action_explanation = explain_code_func(f"Action: {step_type}", str(params)) # Pass params as context
                                     except Exception as e: st.warning(f"Explanation Error: {e}")
                         elif suggested_code:
                              st.code(suggested_code, language='python')
                              if st.button("Explain Code", key=f"{unique_key_base}_explain_code"):
                                  with st.spinner("AI explaining..."):
                                      try: action_explanation = explain_code_func(suggested_code, "python")
                                      except Exception as e: st.warning(f"Explanation Error: {e}")
                         if action_explanation: st.caption(action_explanation)


                         # --- Confirmation and Apply ---
                         st.warning("⚠️ Review action carefully before applying!")
                         if st.button("✅ Apply Suggestion", key=f"{unique_key_base}_apply", type="primary"):
                             with st.spinner("Applying suggestion..."):
                                 try:
                                     df_before = st.session_state.cleaned_dataframe
                                     # ** Call the appropriate apply function **
                                     if step_type and params: # Prefer structured apply
                                         st.session_state.cleaned_dataframe = apply_cleaning_step_func(df_before.copy(), step_type, params) # Assumes function accepts step_type, params
                                     elif suggested_code: # Fallback to code apply (likely mock/unsafe)
                                          logger.warning(f"Applying suggestion '{suggestion['issue']}' using code execution (potentially unsafe).")
                                          # Assuming apply_cleaning_step_func might point to mock_apply_code which expects positional args
                                          st.session_state.cleaned_dataframe = apply_cleaning_step_func(df_before.copy(), suggested_code)
                                     else:
                                          raise ValueError("Suggestion has neither structured parameters nor code to apply.")

                                     step_desc = f"Applied suggestion: {suggestion.get('issue', step_type)}"
                                     st.session_state.applied_steps.append(step_desc)
                                     st.toast(f"Applied: {suggestion.get('issue', step_type)}", icon="✨")
                                     # Clear suggestions after applying? Causes immediate re-analysis if button shown. Better to leave them.
                                     st.rerun() # Update preview & history
                                 except Exception as e: st.error(f"Apply failed: {e}"); logger.error(f"Apply suggestion failed: {e}", exc_info=True)
                         st.markdown("---") # Separator within expander

    # --- Applied Steps History ---
    st.divider()
    st.subheader("📜 Applied Steps History")
    steps_container = st.container(height=200, border=True)
    with steps_container:
        steps_to_show = st.session_state.applied_steps[1:] if len(st.session_state.applied_steps) > 1 else []
        if steps_to_show:
            for i, step in enumerate(reversed(steps_to_show)): st.text(f"{len(steps_to_show)-i}. {step}")
        else: st.info("No steps applied yet.")

    # --- Reset Button ---
    st.subheader("⏮️ Reset Data")
    if st.button("Reset to Raw Data", key="reset_clean_btn", use_container_width=True):
        if st.session_state.raw_dataframe is not None:
            # Reset relevant state variables
            st.session_state.cleaned_dataframe = st.session_state.raw_dataframe.copy()
            st.session_state.applied_steps = ["Reset to raw data."]
            st.session_state.cleaning_suggestions = None
            st.session_state.current_transform_code = ""
            st.session_state.current_transform_explanation = ""
            st.toast("Data reset to original state.", icon="🔄"); time.sleep(1); st.rerun()
        else: st.error("Raw data not available.")


# --- Column 2: Natural Language Transformations ---
with col2:
    st.subheader("✍️ Custom Transformations")
    transform_container = st.container(border=True)
    with transform_container:
        with st.form("nl_transform_form"):
            nl_transform_command = st.text_area(
                "Describe the transformation:",
                placeholder="e.g., Combine 'first_name' and 'last_name' into 'full_name'", key="nl_transform_input",
                height=100, help="Describe the data manipulation."
            )
            # Add option: Generate structured steps (safer) vs Generate code (more flexible, less safe)
            # output_type = st.radio("Generation Type:", ("Code Snippet", "Structured Steps (Safer - WIP)"), horizontal=True, key="transform_gen_type")
            submitted = st.form_submit_button("⚡ Generate Code", use_container_width=True) # Keep as code gen for now

            if submitted:
                if not nl_transform_command: st.warning("Please describe the transformation.")
                else:
                    with st.spinner("Generating transformation code..."):
                        try:
                            df = st.session_state.cleaned_dataframe
                            buffer = io.StringIO(); df.info(buf=buffer, max_cols=100, verbose=False); schema_context = f"Schema:\n{buffer.getvalue()}"
                            # Generate code using selected func
                            generated_code = generate_transformation_code_func(schema_context, nl_transform_command) # Use correct name
                            st.session_state.current_transform_code = generated_code
                            # Auto-generate explanation
                            explanation = explain_code_func(generated_code, "python") # Use selected func
                            st.session_state.current_transform_explanation = explanation
                            st.toast("Code generated.", icon="💡")
                        except NameError as ne: # Catch the specific error if it somehow still occurs
                             st.error(f"Code generation failed: {ne}. Check function names."); logger.error(f"NL Transform code gen failed due to NameError: {ne}", exc_info=True)
                             st.session_state.current_transform_code = f"# Error: {ne}"; st.session_state.current_transform_explanation = ""
                        except Exception as e:
                            st.error(f"Code generation failed: {e}"); logger.error(f"NL Transform code gen failed: {e}", exc_info=True)
                            st.session_state.current_transform_code = f"# Error: {e}"; st.session_state.current_transform_explanation = ""

        # --- Display Generated Code and Explanation ---
        if st.session_state.current_transform_code:
            st.markdown("**Generated Code & AI Explanation:**")
            code_col, explain_col = st.columns([0.6, 0.4])
            with code_col:
                st.code(st.session_state.current_transform_code, language='python')
            with explain_col:
                if st.session_state.current_transform_explanation: st.caption(st.session_state.current_transform_explanation)
                else: st.caption("_Explanation pending or failed._")

            # --- Confirmation and Apply ---
            st.warning("⚠️ **Security Risk:** Applying arbitrary code can be unsafe. Review carefully!")
            if st.button("✅ Apply Transformation", key="confirm_apply_transform_btn", type="primary", use_container_width=True):
                 with st.spinner("Applying transformation (using potentially unsafe method)..."):
                    try:
                        code_to_apply = st.session_state.current_transform_code
                        if not code_to_apply or code_to_apply.startswith("# Error"): raise ValueError("No valid code generated to apply.")

                        df_before = st.session_state.cleaned_dataframe

                        # **** ADDED DEBUG LOG LINE ****
                        logger.critical(f"@@@ DEBUG ApplyTransform: Applying code. Function name: '{apply_transformation_code_func.__name__}', Code value: '{code_to_apply[:50]}...'")

                        # ** CRITICAL: Replace with safe execution **
                        # Use selected apply function (currently points to mock/unsafe one OR real one)
                        # Pass code using the keyword argument expected by the *real* function
                        st.session_state.cleaned_dataframe = apply_transformation_code_func(
                            df_before.copy(),
                            generated_code=code_to_apply # Pass code as keyword argument 'generated_code'
                        )

                        step_desc = f"Applied NL transform: {nl_transform_command[:50]}..." if nl_transform_command else "Applied generated transformation."
                        st.session_state.applied_steps.append(step_desc)
                        st.toast("Transformation applied.", icon="🪄"); time.sleep(1)
                        # Clear generated code after applying?
                        # st.session_state.current_transform_code = ""
                        # st.session_state.current_transform_explanation = ""
                        st.rerun()
                    except Exception as e: st.error(f"Apply Failed: {e}"); logger.error(f"Apply transform failed: {e}", exc_info=True)
        else:
             st.info("Describe a transformation above and click 'Generate Code'.")


# --- Next Step Hint ---
st.divider()
st.success("➡️ Data prepared! Proceed to **3_🛠️_Explore_Engineer** in the sidebar.")
>>>>>>> 946a937 (Add application file)
