<<<<<<< HEAD
# frontend/pages/1_🔗_Connect_Profile.py
=======
# pages/1_🔗_Connect_Profile.py
# Note: This file should be in the 'pages/' directory at the project root.

>>>>>>> 946a937 (Add application file)
import streamlit as st
import pandas as pd
import logging
import time # For simulating processes
<<<<<<< HEAD

# Placeholder for backend imports - Replace with actual imports later
# from backend.database.connectors import get_sql_engine, get_mongo_client # Example
# from backend.data_processing.profiler import generate_profile_report, get_schema_details # Example
# from backend.llm.gemini_utils import get_connection_summary # Example

# --- Mock/Placeholder Backend Functions (Remove once backend is implemented) ---
def mock_get_sql_engine(db_type, user, pwd, host, port, database):
    if not database: raise ValueError("Mock DB requires a database name")
    time.sleep(1) # Simulate connection time
    # Return a simple object indicating success for now
    return {"type": db_type, "db": database, "status": "mock connected"}

def mock_get_mongo_client(uri):
    if not uri: raise ValueError("Mock Mongo requires URI")
    time.sleep(1)
    return {"uri": uri, "status": "mock connected"}

def mock_load_dataframe_from_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        # Make sure to install openpyxl: pip install openpyxl
        return pd.read_excel(uploaded_file, engine='openpyxl')
    elif uploaded_file.name.lower().endswith(".json"):
        return pd.read_json(uploaded_file)
    else:
        raise ValueError("Unsupported file type for mock loader.")

def mock_get_schema_details(source_type, source_obj, db_name=None):
    schema_info = "Mock Schema:\n"
    if source_type == "file" and isinstance(source_obj, pd.DataFrame):
        schema_info += f"- Source: File ({source_obj.attrs.get('filename', 'N/A')})\n"
        schema_info += "- Columns:\n"
        for col, dtype in source_obj.dtypes.items():
            schema_info += f"  - {col}: {dtype}\n"
    elif source_type in ["postgresql", "mysql", "sqlite"]:
        schema_info += f"- Source: {source_type.capitalize()} DB ({source_obj.get('db', 'N/A')})\n"
        schema_info += "- Tables (Mock): ['users', 'orders', 'products']\n"
        schema_info += "- Columns (orders Mock): [order_id (INT), user_id (INT), order_date (DATE), amount (FLOAT)]"
    elif source_type == "mongodb":
         schema_info += f"- Source: MongoDB ({db_name})\n"
         schema_info += "- Collections (Mock): ['customers', 'transactions']\n"
         schema_info += "- Fields (transactions Mock): [_id (ObjectID), customer_id (STRING), timestamp (DATETIME), items (ARRAY), total (NUMBER)]"
    else:
        schema_info = "Could not generate mock schema."
    return schema_info

def mock_generate_profile_report(df):
    # In a real scenario, this would call ydata-profiling or similar
    time.sleep(3) # Simulate profiling time
    report = {
        "overview": {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
        },
        "variables": {col: {"type": str(dtype)} for col, dtype in df.dtypes.items()},
        "gemini_summary": f"Mock Gemini Summary: The dataset has {len(df)} rows and {len(df.columns)} columns. Found {df.isnull().sum().sum()} missing values and {df.duplicated().sum()} duplicate rows. Key columns appear to be: {', '.join(df.columns[:3])}..."
    }
    return report # Return a dictionary simulating profiling output

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Page Specific Configuration ---
st.set_page_config(page_title="Connect & Profile", layout="wide")
st.title("1. 🔗 Connect & Profile Data")
st.markdown("Connect to your data source and get an initial profile.")

# --- Initialize Session State Keys for this Page ---
if 'connection_info' not in st.session_state: st.session_state.connection_info = None
if 'raw_dataframe' not in st.session_state: st.session_state.raw_dataframe = None # Store initially loaded DF here
if 'data_profile' not in st.session_state: st.session_state.data_profile = None
if 'connection_error' not in st.session_state: st.session_state.connection_error = None

# --- Helper to reset state ---
def reset_connection_state():
    st.session_state.connection_info = None
    st.session_state.raw_dataframe = None
    st.session_state.data_profile = None
    st.session_state.connection_error = None
    # Also reset states from subsequent pages if necessary
    if 'cleaned_dataframe' in st.session_state: st.session_state.cleaned_dataframe = None
    if 'engineered_dataframe' in st.session_state: st.session_state.engineered_dataframe = None
    # Add resets for other states as needed

# --- Data Source Selection ---
data_source_type = st.radio(
    "Select Data Source Type:",
    options=["CSV/Excel/JSON File", "PostgreSQL", "MySQL", "SQLite", "MongoDB"],
    key="connect_data_source_type_radio",
    horizontal=True,
    on_change=reset_connection_state # Reset if type changes after connecting
)

# --- Connection Input Area ---
connection_form = st.container(border=True)
connection_details = {}

with connection_form:
    if data_source_type == "CSV/Excel/JSON File":
        uploaded_file = st.file_uploader(
            "Upload Data File", type=['csv', 'xlsx', 'xls', 'json'],
            key="connect_file_uploader"
        )
        connection_details["uploaded_file"] = uploaded_file
    elif data_source_type == "MongoDB":
        mongo_uri = st.text_input("MongoDB Connection URI*", key="connect_mongo_uri")
        mongo_db_name = st.text_input("Database Name*", key="connect_mongo_db_name")
        connection_details["mongo_uri"] = mongo_uri
        connection_details["mongo_db_name"] = mongo_db_name
    else: # SQL
        db_name_label = "Database File Path*" if data_source_type == "SQLite" else "Database Name*"
        db_name = st.text_input(db_name_label, key="connect_db_name")
        connection_details["database"] = db_name
        if data_source_type != "SQLite":
            db_host = st.text_input("Host*", key="connect_db_host")
            db_port = st.text_input("Port*", key="connect_db_port")
            db_user = st.text_input("Username*", key="connect_db_user")
            db_password = st.text_input("Password", type="password", key="connect_db_password")
            connection_details.update({"host": db_host, "port": db_port, "user": db_user, "password": db_password})

    # --- Connect Button Logic ---
    if st.button("Connect and Profile", key="connect_profile_btn", type="primary"):
        reset_connection_state() # Clear previous attempt
        is_valid_input = True
        conn_info_temp = None
        df_temp = None
        profile_temp = None
        schema_temp = "Schema generation pending..."

        with st.spinner(f"Connecting to {data_source_type} and generating profile..."):
            try:
                # --- File Logic ---
                if data_source_type == "CSV/Excel/JSON File":
                    file_obj = connection_details.get("uploaded_file")
                    if file_obj is None:
                        raise ValueError("Please upload a file.")
                    df_temp = mock_load_dataframe_from_file(file_obj)
                    df_temp.attrs['filename'] = file_obj.name # Store filename
                    schema_temp = mock_get_schema_details("file", df_temp)
                    profile_temp = mock_generate_profile_report(df_temp)
                    conn_info_temp = {
                        "type": "file", "filename": file_obj.name,
                        "schema": schema_temp, "status": "connected"
                    }

                # --- MongoDB Logic ---
                elif data_source_type == "MongoDB":
                    uri = connection_details.get("mongo_uri")
                    db_name = connection_details.get("mongo_db_name")
                    if not uri or not db_name: raise ValueError("MongoDB URI and Database Name are required.")
                    client_obj = mock_get_mongo_client(uri)
                    schema_temp = mock_get_schema_details("mongodb", client_obj, db_name)
                    conn_info_temp = {
                        "type": "mongodb", "db_name": db_name,
                        "schema": schema_temp, "status": "connected",
                        # "client": client_obj # Store real client later
                    }
                    # Profiling/Preview for Mongo requires selecting a collection - handle in later steps
                    df_temp = None # No initial DataFrame for DBs
                    profile_temp = {"overview": {"message": "Connect to MongoDB successful. Select a collection in Tab 2 to load and profile data."}, "gemini_summary": "Please proceed to the next step to select a collection."}


                # --- SQL Logic ---
                elif data_source_type in ["PostgreSQL", "MySQL", "SQLite"]:
                    db_type_lower = data_source_type.lower()
                    user, password, host, port, database = (
                        connection_details.get("user"), connection_details.get("password"),
                        connection_details.get("host"), connection_details.get("port"),
                        connection_details.get("database")
                    )
                    if db_type_lower == "sqlite" and not database: raise ValueError("DB Path required.")
                    if db_type_lower != "sqlite" and not all([host, port, database, user]): raise ValueError("Host, Port, DB Name, User required.")

                    engine_obj = mock_get_sql_engine(db_type_lower, user, password, host, port, database)
                    schema_temp = mock_get_schema_details(db_type_lower, engine_obj)
                    conn_info_temp = {
                        "type": db_type_lower, "db_name": database,
                        "schema": schema_temp, "status": "connected",
                        # "engine": engine_obj # Store real engine later
                    }
                    # Profiling/Preview requires running queries - handle in later steps
                    df_temp = None # No initial DataFrame for DBs
                    profile_temp = {"overview": {"message": f"Connect to {data_source_type} successful. Proceed to Tab 2 to query and profile data."}, "gemini_summary": "Please proceed to the next step to query data."}

                else:
                    raise NotImplementedError(f"Connection type '{data_source_type}' not fully implemented.")

                # --- Update Session State on Success ---
                st.session_state.connection_info = conn_info_temp
                st.session_state.raw_dataframe = df_temp # Might be None for DBs initially
                st.session_state.data_profile = profile_temp
                st.session_state.connection_error = None
                st.success("✅ Connection successful! Profile generated (or ready for query).")

            except Exception as e:
                st.session_state.connection_error = f"Error: {e}"
                logger.error(f"Connection/Profiling failed: {e}", exc_info=True)
                st.error(f"❌ {st.session_state.connection_error}")


# --- Display Connection Status and Results ---
st.divider()
results_cols = st.columns([0.5, 0.5])

with results_cols[0]:
    st.subheader("Schema & Connection Status")
    if st.session_state.connection_info:
        st.success(f"Status: Connected to {st.session_state.connection_info['type']}")
        st.text_area("Schema Details", value=st.session_state.connection_info.get('schema', 'N/A'), height=250, disabled=True)
    elif st.session_state.connection_error:
        st.error(f"Status: Failed\n{st.session_state.connection_error}")
    else:
        st.info("Connect to a data source using the form above.")

with results_cols[1]:
    st.subheader("Data Profile Summary")
    if st.session_state.data_profile:
        profile_overview = st.session_state.data_profile.get("overview", {})
        gemini_summary = st.session_state.data_profile.get("gemini_summary", "No summary generated.")

        # Display key overview stats if available
        for key, value in profile_overview.items():
            if key != 'message': # Don't display the message key here
                 st.metric(label=key.replace('_', ' ').title(), value=value)
        if 'message' in profile_overview: # Display info message separately
            st.info(profile_overview['message'])

        st.markdown("**AI Summary:**")
        st.markdown(gemini_summary)
        # In a real app, you might add a button to view the full HTML profile report
        # if st.button("View Full Profile Report"):
        #    display_full_profile(st.session_state.data_profile) # Function to render HTML
    else:
        st.info("Connect and profile data to see the summary.")

# --- Data Preview ---
st.subheader("Raw Data Preview (First 5 Rows)")
if st.session_state.raw_dataframe is not None:
    st.dataframe(st.session_state.raw_dataframe.head(), use_container_width=True)
else:
    st.info("Data preview will appear here after connecting to a file or querying a database.")

# --- Next Step Hint ---
if st.session_state.connection_info:
    st.success("➡️ Proceed to **2_✨_Clean_Transform** to prepare your data.")
=======
import io # For schema generation
from pathlib import Path # Use pathlib
from typing import Dict, Any # For type hints

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Backend Function Imports ---
# Assumes 'backend' is importable and mocks are in pages/_mocks.py
BACKEND_AVAILABLE = False
try:
    # Correct paths based on defined structure
    from backend.database.connectors import get_sql_engine, get_mongo_client
    from backend.data_processing.profiler import generate_profile_report, get_schema_details, load_dataframe_from_file
    # from backend.llm.gemini_utils import get_connection_summary # Example

    BACKEND_AVAILABLE = True
    logger.info("Backend modules imported successfully in Connect_Profile page.")

except ImportError as e:
    logger.error(f"Backend import failed in Connect_Profile: {e}", exc_info=True)
    st.error(f"CRITICAL: Backend modules could not be loaded. Using MOCK functions. Error: {e}", icon="🚨")
    BACKEND_AVAILABLE = False
    try:
        from pages._mocks import (
            mock_get_sql_engine as get_sql_engine,
            mock_get_mongo_client as get_mongo_client,
            mock_load_dataframe_from_file as load_dataframe_from_file,
            mock_get_schema_details as get_schema_details,
            mock_generate_profile_report as generate_profile_report
        )
        logger.info("Loaded MOCK functions as fallback.")
    except ImportError as mock_e:
         logger.critical(f"Failed to import mock functions from pages._mocks! Error: {mock_e}", exc_info=True)
         st.error(f"CRITICAL ERROR: Could not load fallback functions: {mock_e}", icon="🚨"); st.stop()
    except Exception as general_mock_e:
         logger.critical(f"Unexpected error importing mocks: {general_mock_e}", exc_info=True); st.error(f"Unexpected critical error during setup: {general_mock_e}", icon="🚨"); st.stop()

except Exception as e:
     logger.critical(f"Unexpected error during backend import in Connect_Profile: {e}", exc_info=True); st.error(f"A critical error occurred during setup: {e}", icon="🚨"); st.stop()


# --- Use the selected functions (real or mock) ---
get_sql_engine_func = get_sql_engine
get_mongo_client_func = get_mongo_client
load_dataframe_func = load_dataframe_from_file
get_schema_details_func = get_schema_details
generate_profile_report_func = generate_profile_report

# --- Page Title and Introduction ---
st.header("1. 🔗 Connect & Profile Data")
st.markdown("Connect to your data source (file or database) and generate an initial automated profile to understand its structure and quality.")
st.divider()

# --- Initialize Session State Keys ---
# Ensure keys are initialized (ideally in root app.py)
keys_needed = ['connection_info', 'raw_dataframe', 'data_profile', 'connection_error',
               'cleaned_dataframe', 'engineered_dataframe', 'analysis_results',
               'analysis_plot', 'deep_insights', 'generated_report', 'defined_kpis',
               'tracked_kpi_data', 'generated_recommendations', 'mongo_collection_to_analyze']
for key in keys_needed:
    if key not in st.session_state:
        # Initialize with appropriate defaults based on expected type
        st.session_state[key] = {} if key in ['defined_kpis', 'tracked_kpi_data', 'recommendation_feedback','data_profile'] else \
                               [] if key in ['applied_steps', 'applied_feature_steps', 'generated_recommendations', 'cleaning_suggestions', 'feature_suggestions'] else \
                               "" if key in ['analysis_nl_query', 'analysis_generated_sql', 'deep_insights', 'mongo_collection_to_analyze', 'connection_error', 'generated_report_data'] else \
                               None # Default for dataframes, connection_info, plot objects etc.

# --- Helper to reset state ---
def reset_connection_state():
    """Clears connection info and downstream analysis results from session state."""
    logger.info("Resetting connection and downstream session state...")
    # Re-initialize keys to ensure clean state
    for key in keys_needed:
        st.session_state[key] = {} if key in ['defined_kpis', 'tracked_kpi_data', 'recommendation_feedback','data_profile'] else \
                               [] if key in ['applied_steps', 'applied_feature_steps', 'generated_recommendations', 'cleaning_suggestions', 'feature_suggestions'] else \
                               "" if key in ['analysis_nl_query', 'analysis_generated_sql', 'deep_insights', 'mongo_collection_to_analyze', 'connection_error', 'generated_report_data'] else \
                               None
    st.toast("Cleared previous connection and results.", icon="🧹")


# --- Layout ---
input_col, display_col = st.columns([0.45, 0.55])

# --- Input Column ---
with input_col:
    st.subheader("Select Data Source")
    # Add all supported types
    data_source_options = ["CSV/Excel/JSON File", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "MS SQL Server", "Oracle"]
    data_source_type_key = "connect_data_source_type_radio"
    # Get index based on current state or default
    current_ds_type = st.session_state.get(data_source_type_key, data_source_options[0])
    try: current_index = data_source_options.index(current_ds_type)
    except ValueError: current_index = 0 # Default to first option if state value is invalid

    data_source_type = st.radio(
        "Source Type:",
        options=data_source_options,
        key=data_source_type_key,
        horizontal=True,
        label_visibility="collapsed",
        index=current_index,
        on_change=reset_connection_state
    )

    # --- Connection Form ---
    with st.form(key="connection_form"):
        connection_details = {}
        st.markdown(f"**Configure {data_source_type} Connection:**")

        if data_source_type == "CSV/Excel/JSON File":
            uploaded_file = st.file_uploader("Upload File (.csv, .xlsx, .xls, .json)", type=['csv', 'xlsx', 'xls', 'json'], key="connect_file_uploader", accept_multiple_files=False, label_visibility="visible")
            connection_details["uploaded_file"] = uploaded_file
        elif data_source_type == "MongoDB":
            mongo_uri = st.text_input("Connection URI*", key="connect_mongo_uri", placeholder="mongodb://user:pass@host:port/db?options...")
            mongo_db_name = st.text_input("Database Name*", key="connect_mongo_db_name", help="Database containing collections.")
            connection_details["mongo_uri"] = mongo_uri; connection_details["mongo_db_name"] = mongo_db_name
        elif data_source_type == "Oracle": # Oracle UI
            st.info("Provide either Oracle SID or Service Name.")
            db_host = st.text_input("Host*", key="connect_db_host", placeholder="e.g., oracle.example.com")
            db_port = st.text_input("Port*", key="connect_db_port", value="1521", placeholder="Default: 1521")
            db_user = st.text_input("Username*", key="connect_db_user")
            db_password = st.text_input("Password", type="password", key="connect_db_password")
            db_sid = st.text_input("Oracle SID (Use this OR Service Name)", key="connect_db_name", help="The System ID of the database.")
            db_service_name = st.text_input("Service Name (Use this OR SID)", key="connect_db_service_name", help="The Service Name alias.")
            connection_details.update({"host": db_host, "port": db_port, "user": db_user, "password": db_password, "database": db_sid, "service_name": db_service_name})
        else: # SQL Databases (PostgreSQL, MySQL, SQLite, MSSQL)
            db_type_lower = data_source_type.lower().replace(" ", "")
            db_name_label = "Database File Path*" if db_type_lower == "sqlite" else "Database Name*"
            db_name = st.text_input(db_name_label, key="connect_db_name")
            connection_details["database"] = db_name
            if db_type_lower != "sqlite":
                col_a, col_b = st.columns(2)
                with col_a:
                    host_label = "Host/Server*" if db_type_lower != "mssql" else "Server Name/IP*"
                    db_host = st.text_input(host_label, key="connect_db_host")
                    db_user = st.text_input("Username*", key="connect_db_user")
                with col_b:
                    default_port = {"postgresql": "5432", "mysql": "3306", "mssql": "1433"}.get(db_type_lower,"")
                    db_port = st.text_input(f"Port ({default_port})", key="connect_db_port", value=default_port, placeholder="Optional")
                    db_password = st.text_input("Password", type="password", key="connect_db_password")
                connection_details.update({"host": db_host, "port": db_port or None, "user": db_user, "password": db_password}) # Pass None if port empty
                if db_type_lower == "mssql":
                    connection_details["driver"] = st.text_input("ODBC Driver (Optional)", key="connect_mssql_driver", placeholder="e.g., ODBC Driver 17 for SQL Server", help="Leave blank to try defaults.")


        st.markdown("---") # Separator before button
        # --- Submit Button ---
        submitted = st.form_submit_button(label="🔗 Connect and Profile", type="primary", use_container_width=True)

        if submitted:
            # Clear previous error state before attempting connection
            st.session_state.connection_error = ""
            conn_info_temp = None; df_temp = None; profile_temp = None; schema_details_dict = None
            success = False; error_message = ""

            with st.spinner(f"Connecting to {data_source_type} and generating profile..."):
                try:
                    logger.info(f"Connect button submitted for type: {data_source_type}")
                    # --- File Logic ---
                    if data_source_type == "CSV/Excel/JSON File":
                        file_obj = connection_details.get("uploaded_file")
                        if file_obj is None: raise ValueError("Please upload a file.")
                        df_temp = load_dataframe_func(file_obj)
                        if df_temp is None or df_temp.empty: raise ValueError("Failed to load data or file is empty.")
                        schema_details_dict = get_schema_details_func("file", df_temp)
                        profile_temp = generate_profile_report_func(df_temp, source_name=file_obj.name)
                        conn_info_temp = {"type": "file", "filename": file_obj.name, "schema": schema_details_dict.get("schema_string", "N/A"), "status": "connected"}
                        success = True; logger.info(f"File '{file_obj.name}' loaded & profiled.")

                    # --- MongoDB Logic ---
                    elif data_source_type == "MongoDB":
                        uri = connection_details.get("mongo_uri"); db_name = connection_details.get("mongo_db_name")
                        if not uri or not db_name: raise ValueError("MongoDB URI and Database Name required.")
                        client_obj = get_mongo_client_func(uri)
                        schema_details_dict = get_schema_details_func("mongodb", client_obj, db_name)
                        conn_info_temp = {"type": "mongodb", "db_name": db_name, "schema": schema_details_dict.get("schema_string", "N/A"), "status": "connected", "client": client_obj if BACKEND_AVAILABLE else None}
                        df_temp = None; profile_temp = {"overview": {"message": "Connect successful."}, "llm_summary": "Proceed to query."}
                        success = True; logger.info(f"Connected to MongoDB '{db_name}'.")

                    # --- SQL Logic (All SQL Types) ---
                    elif data_source_type in ["PostgreSQL", "MySQL", "SQLite", "MS SQL Server", "Oracle"]:
                        db_type_map = {"PostgreSQL":"postgresql", "MySQL":"mysql", "SQLite":"sqlite", "MS SQL Server":"mssql", "Oracle":"oracle"}
                        db_type_lower = db_type_map.get(data_source_type)
                        # Gather all potential args from connection_details
                        conn_args = {k: v for k, v in connection_details.items() if v is not None and v != ''} # Filter out None/empty
                        conn_args['db_type'] = db_type_lower # Add db_type for the function

                        engine_obj = get_sql_engine_func(**conn_args) # Pass relevant args using **

                        schema_details_dict = get_schema_details_func(db_type_lower, engine_obj, db_name=conn_args.get('database') or conn_args.get('service_name'))
                        conn_info_temp = {
                            "type": db_type_lower, "db_name": conn_args.get('database') or conn_args.get('service_name'),
                            "schema": schema_details_dict.get("schema_string", "N/A"), "status": "connected",
                            "engine": engine_obj if BACKEND_AVAILABLE else None,
                            "join_key_candidates": schema_details_dict.get("join_key_candidates", []) # Store potential keys
                            # Store other details like host if needed: "host": conn_args.get('host')
                        }
                        df_temp = None; profile_temp = {"overview": {"message": f"Connect to {data_source_type} successful."}, "llm_summary": "Proceed to query data."}
                        success = True
                        logger.info(f"Connected to {data_source_type} DB '{conn_args.get('database') or conn_args.get('service_name')}'.")
                    else:
                        raise NotImplementedError(f"Connection type '{data_source_type}' logic missing.")

                except Exception as e: success = False; error_message = f"Error: {e}"; logger.error(f"Connection/Profiling failed: {e}", exc_info=True)

            # --- Update Session State Post-Attempt ---
            if success:
                # Reset downstream first
                keys_to_clear = ['cleaned_dataframe', 'engineered_dataframe', 'analysis_results', 'analysis_plot', 'deep_insights', 'generated_report_data', 'defined_kpis', 'tracked_kpi_data', 'generated_recommendations', 'recommendation_feedback', 'applied_steps', 'applied_feature_steps', 'cleaning_suggestions', 'feature_suggestions']
                for key in keys_to_clear: st.session_state[key] = {} if key in ['defined_kpis', 'tracked_kpi_data', 'recommendation_feedback'] else [] if key in ['applied_steps', 'applied_feature_steps', 'generated_recommendations', 'cleaning_suggestions', 'feature_suggestions'] else None

                # Set new state
                st.session_state.connection_info = conn_info_temp
                st.session_state.raw_dataframe = df_temp
                st.session_state.data_profile = profile_temp
                st.session_state.connection_error = "" # Clear error on success
                st.toast("Connection successful!", icon="✅")
                st.rerun() # Rerun to update display immediately
            else:
                # Clear connection info on failure, keep error
                st.session_state.connection_info = None; st.session_state.raw_dataframe = None; st.session_state.data_profile = None
                st.session_state.connection_error = error_message
                # Error displayed below outside the form

    # Display connection error (if any) after form submission attempt fails
    if st.session_state.connection_error:
        st.error(f"❌ {st.session_state.connection_error}")


# --- Display Column ---
with display_col:
    st.subheader("ℹ️ Status & Schema")
    status_container = st.container(border=True, height=250)
    with status_container:
        if st.session_state.connection_info:
            conn = st.session_state.connection_info
            conn_type = conn.get('type', 'N/A').capitalize()
            conn_name = conn.get('filename') or conn.get('db_name') or 'Unknown Source'
            st.success(f"Status: Connected to **{conn_type}**: `{conn_name}`")
            st.markdown("**Schema Details:**")
            st.code(conn.get('schema', 'Schema not available.'), language='text')
            # Display Join Key Candidates if available
            join_keys = conn.get('join_key_candidates')
            if join_keys:
                 with st.expander("Potential Join Keys"):
                      for key_info in join_keys: st.caption(f"- `{key_info[0]}` <=> `{key_info[1]}` ({key_info[2]})")
        elif st.session_state.connection_error:
             st.error(f"Status: Connection Failed\n{st.session_state.connection_error}") # Show error again here if needed
        else:
            st.info("Connect to a data source using the form on the left.")

    st.subheader("📊 Data Profile Summary")
    profile_container = st.container(border=True)
    with profile_container:
        if st.session_state.data_profile:
            # (Profile display logic - same as before) ...
            profile = st.session_state.data_profile; overview = profile.get("overview", {}); ai_summary = profile.get("llm_summary", "_AI summary failed or disabled._")
            if "error" in profile: st.error(f"Profiling Error: {profile['error']}")
            else:
                stats_cols = st.columns(3)
                rows=overview.get('rows'); cols=overview.get('columns'); missing=overview.get('missing_cells'); dups=overview.get('duplicate_rows'); mem=overview.get('memory_usage'); duration=overview.get('profiling_duration_sec')
                if rows is not None: stats_cols[0].metric("Rows", f"{rows:,}")
                if cols is not None: stats_cols[1].metric("Columns", cols)
                if missing is not None: stats_cols[2].metric("Missing Cells", f"{missing:,}")
                if dups is not None: stats_cols[0].metric("Duplicate Rows", f"{dups:,}")
                if mem is not None: stats_cols[1].metric("Memory Size", mem)
                if duration is not None: stats_cols[2].metric("Profile Time", f"{duration:.2f}s")
                if 'message' in overview: st.info(f"ℹ️ {overview['message']}")
                st.markdown("**🤖 AI Summary:**"); st.caption(ai_summary)
                # Add display for suggested rules/tags if generated
                if profile.get("ai_tags"): st.caption(f"**AI Tags:** {' '.join(f'`{t}`' for t in profile['ai_tags'])}")
                if profile.get("ai_quality_rules"):
                     with st.expander("AI Suggested Quality Rules"):
                          for rule in profile["ai_quality_rules"][:5]: # Show first 5
                               st.caption(f"- **{rule.get('rule_name')}**: {rule.get('rule_type')} on `{rule.get('column_name')}` ({rule.get('rationale', '')[:50]}...)")
        else: st.info("Connect and profile data to see the summary.")

    st.subheader("📄 Raw Data Preview")
    preview_container = st.container(border=True, height=250)
    with preview_container:
        # (Preview display logic - same as before) ...
        if st.session_state.raw_dataframe is not None and not st.session_state.raw_dataframe.empty:
            st.dataframe(st.session_state.raw_dataframe.head(), use_container_width=True); st.caption(f"Showing first 5 of {st.session_state.raw_dataframe.shape[0]} rows.")
        else: st.info("Preview appears here for files or after querying DB.")

# --- Guidance ---
st.divider()
if st.session_state.connection_info:
    st.success("➡️ Data connected! Proceed to **2_✨_Clean_Transform** in the sidebar.")
else:
    st.caption("Connect to your data source above to begin the analysis lifecycle.")
>>>>>>> 946a937 (Add application file)
