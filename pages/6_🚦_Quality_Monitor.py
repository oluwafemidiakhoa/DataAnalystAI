# pages/6_🚦_Quality_Monitor.py
# Note: This file should be in the 'pages/' directory at the project root.

import streamlit as st
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Backend Function Imports ---
# Assumes backend functions for fetching rules, running checks, getting history exist
BACKEND_AVAILABLE = False
DB_AVAILABLE = False
try:
    from backend.database import crud, models # For rule definitions and violations
    from backend.database.session import get_db_session # For DB interaction
    from backend.data_processing.quality_checker import run_quality_checks # To run checks on demand
    # Function to load data for a specific datasource (needed for running checks)
    # from backend.data_processing.loader import load_data_for_datasource # Needs implementation
    BACKEND_AVAILABLE = True
    DB_AVAILABLE = True # Assume DB is available if CRUD loads
    logger.info("Backend modules loaded for Quality Monitor page.")
except ImportError as e:
    logger.error(f"Backend/DB import failed in Quality Monitor: {e}", exc_info=True)
    st.error(f"CRITICAL: Backend/DB modules not found. Quality monitoring unavailable. Error: {e}", icon="🚨")
    # Define dummy functions to prevent crash
    class crud:
        @staticmethod
        def get_all_datasources(*args, **kwargs): return []
        @staticmethod
        def get_rules_for_datasource(*args, **kwargs): return []
        @staticmethod
        def get_violation_history(*args, **kwargs): return pd.DataFrame(columns=['Rule Name', 'Timestamp', 'Status', 'Count', 'Details'])
    def run_quality_checks(*args, **kwargs): logger.warning("Mock run_quality_checks called."); return [{"rule_name": "Mock Rule", "status": "skipped", "details": "Backend unavailable"}]
    def get_db_session(): logger.error("Dummy DB session context used!"); yield None
    # Define dummy load function
    # def load_data_for_datasource(*args, **kwargs): logger.warning("Using mock data loader."); return pd.DataFrame({'A':[1]})

except Exception as e:
     logger.critical(f"Unexpected error during backend import in Quality Monitor: {e}", exc_info=True)
     st.error(f"A critical error occurred during setup: {e}", icon="🚨")
     st.stop()


# --- Page Config & Title ---
st.header("6. 🚦 Data Quality Monitor")
st.markdown("Define, manage, and monitor data quality rules across your connected data sources. View violation history and run checks on demand.")
st.divider()

# --- Check Backend Availability ---
if not BACKEND_AVAILABLE or not DB_AVAILABLE:
    st.error("Data Quality Monitoring requires backend database and processing functions to be available.")
    st.stop()


# --- Initialize Session State ---
keys_needed = ['dq_selected_datasource_id', 'dq_rules', 'dq_violations', 'dq_run_results']
for key in keys_needed:
    if key not in st.session_state:
        st.session_state[key] = None


# --- Data Source Selection ---
st.subheader("Select Data Source to Monitor")
datasource_options: Dict[int, str] = {} # Store as {id: name}
try:
    with get_db_session() as db:
        # Assuming a CRUD function exists to get datasources suitable for quality checks
        # datasources = crud.get_all_datasources(db, filter_types=['postgresql', 'mysql', ...]) # Filter as needed
        # Mock data if DB call fails or not implemented
        datasources = crud.get_all_datasources(db) if DB_AVAILABLE else []
        if not datasources: # Use mock data if no datasources found or DB unavailable
             datasources = [models.DataSource(id=1, name="Mock Sales DB (Postgres)", source_type='postgresql'), models.DataSource(id=2, name="Mock User File (CSV)", source_type='file')]
        datasource_options = {ds.id: f"{ds.name} ({ds.source_type})" for ds in datasources}

except Exception as e:
    st.error(f"Failed to load data sources: {e}")
    logger.error(f"Failed to load data sources for quality monitor: {e}", exc_info=True)

# Store the selected ID in session state
selected_ds_id = st.selectbox(
    "Choose a data source:",
    options=list(datasource_options.keys()),
    format_func=lambda x: datasource_options.get(x, "Unknown"),
    key="dq_selected_datasource_selectbox",
    index=None, # Default to no selection
    placeholder="Select a source..."
)

# Update session state only if selection changes
if selected_ds_id != st.session_state.get('dq_selected_datasource_id'):
    logger.info(f"Data source selection changed to ID: {selected_ds_id}")
    st.session_state.dq_selected_datasource_id = selected_ds_id
    # Clear previous rules and results when source changes
    st.session_state.dq_rules = None
    st.session_state.dq_violations = None
    st.session_state.dq_run_results = None
    st.rerun() # Rerun immediately to load data for the new source

st.divider()

# --- Main Content Area (Only if a data source is selected) ---
if st.session_state.dq_selected_datasource_id is not None:
    selected_ds_name = datasource_options.get(st.session_state.dq_selected_datasource_id, "Selected Source")
    st.subheader(f"Quality Rules for: `{selected_ds_name}`")

    # --- Load Rules for Selected Source ---
    if st.session_state.dq_rules is None: # Load only once per selection
        logger.info(f"Loading rules for DS ID: {st.session_state.dq_selected_datasource_id}")
        try:
            with get_db_session() as db:
                # Use actual CRUD function
                rules_orm = crud.get_rules_for_datasource(db, data_source_id=st.session_state.dq_selected_datasource_id)
                # Convert ORM objects to simple list of dicts for state/UI
                st.session_state.dq_rules = [
                    {"id": r.id, "rule_name": r.rule_name, "rule_type": r.rule_type, "column_name": r.column_name, "rule_parameters": r.rule_parameters, "is_active": r.is_active}
                    for r in rules_orm
                ] if rules_orm else [] # Ensure it's a list even if empty
                logger.info(f"Loaded {len(st.session_state.dq_rules)} rules.")
        except Exception as e:
            st.error(f"Failed to load quality rules: {e}")
            logger.error(f"Failed loading rules for DS ID {st.session_state.dq_selected_datasource_id}: {e}", exc_info=True)
            st.session_state.dq_rules = [] # Set to empty list on error


    # --- Display & Manage Rules ---
    rules_container = st.container(border=True)
    with rules_container:
        if not st.session_state.dq_rules:
            st.info("No quality rules defined for this data source yet.")
        else:
            st.markdown(f"**{len(st.session_state.dq_rules)} Rule(s) Defined:**")
            # Display rules in a table or list
            rules_df = pd.DataFrame(st.session_state.dq_rules)
            # Select columns to display - convert params dict to string for display
            rules_df['Parameters'] = rules_df['rule_parameters'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
            st.dataframe(
                rules_df[['rule_name', 'rule_type', 'column_name', 'Parameters', 'is_active']],
                use_container_width=True,
                hide_index=True
            )
            # TODO: Add buttons for Edit Rule / Delete Rule / Toggle Active

        # --- Add New Rule Section (Placeholder Form) ---
        with st.expander("➕ Define New Quality Rule"):
            with st.form("new_rule_form"):
                 st.text_input("Rule Name*", key="new_rule_name")
                 st.selectbox("Rule Type*", options=["not_null", "is_unique", "min_value", "max_value", "regex_match", "enum_values"], key="new_rule_type")
                 st.text_input("Column Name (if applicable)", key="new_rule_column")
                 st.text_area("Parameters (JSON format)*", key="new_rule_params", placeholder='e.g., {"min": 0} or {"allowed": ["A", "B"]}')
                 new_rule_submitted = st.form_submit_button("💾 Add Rule")
                 if new_rule_submitted:
                      st.warning("Adding new rules via UI not yet implemented.")
                      # Add logic here to parse params, call crud.create_quality_rule, update state, rerun


    # --- Run Checks On Demand ---
    st.subheader("⚡ Run Quality Checks Now")
    run_container = st.container(border=True)
    with run_container:
        if st.button("🚦 Run All Active Rules", key="run_dq_now", type="primary", use_container_width=True):
            st.session_state.dq_run_results = None # Clear previous run results
            with st.spinner(f"Running quality checks on '{selected_ds_name}'..."):
                 try:
                     # 1. Load data for the selected source (Needs implementation)
                     # datasource_obj = crud.get_data_source(db, st.session_state.dq_selected_datasource_id) # Get source info
                     # df_to_check = load_data_for_datasource(datasource_obj) # Call loader
                     df_to_check = st.session_state.get('raw_dataframe') # TEMPORARY: Use loaded file data if available
                     if df_to_check is None: raise ValueError("Could not load data for selected source to run checks.")

                     # 2. Get active rules for this source (already loaded in st.session_state.dq_rules)
                     active_rules = [r for r in st.session_state.dq_rules if r.get('is_active', True)]
                     if not active_rules: raise ValueError("No active rules found for this source.")

                     # 3. Run checks using the backend function
                     with get_db_session() as db: # Get session for logging violations
                          results = run_quality_checks(df=df_to_check, rules=active_rules, db=db) # Pass DB session
                     st.session_state.dq_run_results = results
                     st.toast(f"Checks complete. {len([r for r in results if r['status']=='failed'])} failures.", icon="🚦")
                     st.rerun() # Update results display

                 except Exception as e:
                     st.error(f"Failed to run quality checks: {e}")
                     logger.error(f"On-demand quality check failed: {e}", exc_info=True)

        # --- Display Last Run Results ---
        if st.session_state.dq_run_results:
             st.markdown("**Last Run Results:**")
             results_df = pd.DataFrame(st.session_state.dq_run_results)
             # Style the results based on status?
             st.dataframe(results_df[['rule_name', 'status', 'violation_count', 'details']], use_container_width=True, hide_index=True)


    # --- Violation History ---
    st.subheader("📜 Violation History")
    history_container = st.container(border=True, height=300)
    with history_container:
        # Load history only when needed or source changes
        if st.session_state.dq_violations is None:
            logger.info(f"Loading violation history for DS ID: {st.session_state.dq_selected_datasource_id}")
            try:
                 with get_db_session() as db:
                     # Add crud function like get_violation_history(db, datasource_id, limit=100)
                     # For now, use placeholder or mock
                     violations = crud.get_violation_history(db, data_source_id=st.session_state.dq_selected_datasource_id, limit=50) # Example
                     st.session_state.dq_violations = pd.DataFrame([
                         # Convert ORM objects to dicts/series for display
                         {"Timestamp": v.check_timestamp, "Rule": v.rule.rule_name, "Status": v.status, "Count": v.violation_count, "Details": str(v.violation_details)}
                         for v in violations
                     ]) if violations else pd.DataFrame(columns=['Timestamp', 'Rule', 'Status', 'Count', 'Details']) # Empty df if no violations
                 logger.info(f"Loaded {len(st.session_state.dq_violations)} violation history records.")
            except Exception as e:
                 st.error(f"Failed to load violation history: {e}")
                 logger.error(f"Failed loading violation history: {e}", exc_info=True)
                 st.session_state.dq_violations = pd.DataFrame(columns=['Timestamp', 'Rule', 'Status', 'Count', 'Details']) # Empty df on error


        if st.session_state.dq_violations is not None:
             if st.session_state.dq_violations.empty:
                  st.info("No violation history found for this data source.")
             else:
                  st.dataframe(st.session_state.dq_violations, use_container_width=True, hide_index=True)
        else:
            st.info("Loading violation history...")


else:
    st.info("Select a data source above to view its quality rules and history.")


st.divider()
st.caption("_Data Quality Monitoring under development. Rule editing and advanced checks coming soon._")