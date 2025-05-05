<<<<<<< HEAD
# frontend/pages/4_📊_Analyze_Insight.py
import streamlit as st
import pandas as pd
import plotly.express as px # Use Plotly for results visualization too
import logging
import time

# Placeholder for backend imports
# from backend.analysis.nlq_processor import generate_sql_from_nl # If specific function used
# from backend.analysis.query_executor import execute_query # Handles SQL, Pandas, Mongo
# from backend.analysis.insight_generator import generate_deep_insights # Example
# from backend.reporting.visualizer import create_plotly_chart, suggest_visualization # Reuse visualizer

# --- Mock/Placeholder Backend Functions ---
def mock_generate_sql_from_nl(nl_query, schema_context):
    time.sleep(1.5)
    if "total sales by region" in nl_query.lower():
        return "SELECT region, SUM(sales_amount) AS total_sales\nFROM sales_data\nGROUP BY region\nORDER BY total_sales DESC;"
    elif "customer count" in nl_query.lower():
        return "SELECT COUNT(DISTINCT customer_id) AS unique_customers\nFROM customer_data;"
    elif "average order value" in nl_query.lower():
        return "SELECT AVG(order_total) AS average_order_value\nFROM orders WHERE status = 'completed';"
    else:
        return f"-- Mock SQL Query for: {nl_query}\nSELECT * FROM your_table LIMIT 10;"

def mock_execute_query(conn_info, query_type, query, dataframe=None):
    # Mock execution - in reality, this calls SQLAlchemy, PyMongo, or Pandas
    time.sleep(2)
    print(f"--- MOCK QUERY EXECUTION ({query_type}) ---")
    print(query)
    print("--- END MOCK ---")
    # Return a sample DataFrame based on query type for demo
    if query_type == "sql":
        if "total_sales" in query:
            return pd.DataFrame({'region': ['North', 'South', 'East', 'West'], 'total_sales': [15000, 12000, 18000, 9000]})
        elif "unique_customers" in query:
            return pd.DataFrame({'unique_customers': [1250]})
        else:
             # Return head of engineered data if available, else sample
             if dataframe is not None: return dataframe.head(10).copy()
             else: return pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B'], 'info': ['Mock SQL Result']})
    elif query_type == "pandas": # Placeholder for future NLQ->Pandas
         if dataframe is not None: return dataframe.head(10).copy()
         else: return pd.DataFrame({'col1': [3, 4], 'col2': ['C', 'D'], 'info': ['Mock Pandas Result']})
    elif query_type == "mongo": # Placeholder
         return pd.DataFrame({'_id': ['id1', 'id2'], 'value': [100, 200], 'info': ['Mock Mongo Result']})
    return pd.DataFrame()


def mock_generate_deep_insights(df, nl_query=None):
    time.sleep(3)
    if df is None or df.empty:
        return "No data to analyze for deep insights."

    insight = f"**Mock AI Insights for query '{nl_query or 'general data'}':**\n\n"
    insight += f"- The analysis produced {len(df)} results.\n"
    # Try to find numeric columns for basic insights
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        col = numeric_cols[0]
        insight += f"- The average value for '{col}' is {df[col].mean():.2f} with a standard deviation of {df[col].std():.2f}.\n"
        insight += f"- There might be potential outliers in '{col}' if the max value ({df[col].max()}) is significantly higher than the mean.\n"
    else:
        insight += "- No clear numeric columns found for statistical insights in this mock.\n"
    insight += "- Further analysis could involve time series decomposition (if date columns exist) or customer segmentation."
    return insight

# Reuse mock suggest_visualization and create_plotly_chart from Page 3 if needed

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Page Specific Configuration ---
st.set_page_config(page_title="Analyze & Insight", layout="wide")
st.title("4. 📊 Analyze & Insight")
st.markdown("Query your prepared data using natural language and uncover deeper insights with AI.")

# --- Initialize Session State ---
# Required states from previous steps
if 'engineered_dataframe' not in st.session_state: st.session_state.engineered_dataframe = None
if 'connection_info' not in st.session_state: st.session_state.connection_info = None
# States for this page
if 'analysis_nl_query' not in st.session_state: st.session_state.analysis_nl_query = ""
if 'analysis_generated_sql' not in st.session_state: st.session_state.analysis_generated_sql = ""
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None # Stores result DataFrame
if 'analysis_plot' not in st.session_state: st.session_state.analysis_plot = None
if 'deep_insights' not in st.session_state: st.session_state.deep_insights = ""

# --- Check if Data is Available ---
can_analyze = False
data_source_context = None # DataFrame or Connection Info

if st.session_state.engineered_dataframe is not None:
    # If we have an engineered DataFrame (from file or previous steps), prioritize it
    can_analyze = True
    data_source_context = st.session_state.engineered_dataframe
    st.info("Using data prepared in previous steps (Explore & Engineer).")
elif st.session_state.cleaned_dataframe is not None:
    # Fallback to cleaned data if engineering step was skipped/reset
    can_analyze = True
    data_source_context = st.session_state.cleaned_dataframe
    st.info("Using data prepared in previous steps (Clean & Transform).")
elif st.session_state.connection_info and st.session_state.connection_info.get('type') != 'file':
    # If connected to DB but no DataFrame loaded yet
    can_analyze = True
    data_source_context = st.session_state.connection_info # Use connection for DB queries
    st.info(f"Connected to {st.session_state.connection_info['type']} database. Ready to query.")
else:
    st.warning("⚠️ Please connect to a data source and prepare data in the previous steps first.")
    st.stop()


# --- Main Layout: Query Input | Results & Insights ---
query_col, results_col = st.columns([0.4, 0.6])

with query_col:
    st.subheader("Ask Your Data")
    nl_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the total sales by region?",
        key="analysis_nl_query_input",
        value=st.session_state.analysis_nl_query
    )
    st.session_state.analysis_nl_query = nl_query # Update state continuously

    # --- SQL Generation (if applicable) ---
    is_sql_db = st.session_state.connection_info and st.session_state.connection_info['type'] in ["postgresql", "mysql", "sqlite"]
    schema_context_for_llm = st.session_state.connection_info.get('schema', 'Schema unavailable.') if st.session_state.connection_info else "Schema unavailable."

    if is_sql_db:
        if st.button("Generate SQL Query", key="analysis_gen_sql_btn"):
            if not nl_query:
                st.warning("Please enter a question.")
            else:
                 with st.spinner("Generating SQL Query..."):
                    try:
                        # Replace with actual backend call
                        generated_sql = mock_generate_sql_from_nl(nl_query, schema_context_for_llm)
                        st.session_state.analysis_generated_sql = generated_sql
                        st.success("SQL Generated.")
                        st.rerun()
                    except Exception as e:
                         st.error(f"Failed to generate SQL: {e}")
                         logger.error(f"SQL Generation failed: {e}", exc_info=True)

        if st.session_state.analysis_generated_sql:
            st.code(st.session_state.analysis_generated_sql, language='sql')

    # --- Execute Analysis Button ---
    if st.button("Run Analysis Query", key="run_analysis_query_btn", type="primary"):
        # Reset previous results
        st.session_state.analysis_results = None
        st.session_state.analysis_plot = None
        st.session_state.deep_insights = ""

        if not nl_query:
             st.warning("Please enter a question first.")
        else:
             with st.spinner("Executing analysis..."):
                try:
                    query_type = "sql" if is_sql_db else "pandas" # Or "mongo" if implemented
                    query_to_run = st.session_state.analysis_generated_sql if is_sql_db else nl_query # Use NLQ for pandas/mongo mock

                    if is_sql_db and not query_to_run:
                        raise ValueError("No SQL query generated or provided.")

                    current_df_context = data_source_context if isinstance(data_source_context, pd.DataFrame) else None
                    conn_info_context = data_source_context if isinstance(data_source_context, dict) else None

                    # Replace with actual backend call
                    results_df = mock_execute_query(conn_info_context, query_type, query_to_run, dataframe=current_df_context)
                    st.session_state.analysis_results = results_df

                    if results_df is None or results_df.empty:
                        st.warning("Query executed successfully, but returned no results.")
                    else:
                        st.success(f"Analysis successful! Found {len(results_df)} results.")
                        # Attempt to auto-visualize results
                        try:
                            # Use simplified suggestion for results - just try a bar/scatter/histogram
                            if len(results_df.columns) >= 2:
                                mock_viz_sugg = {"chart_type": "bar", "x_column": results_df.columns[0], "y_column": results_df.columns[1]}
                                if pd.api.types.is_numeric_dtype(results_df[results_df.columns[0]]) and pd.api.types.is_numeric_dtype(results_df[results_df.columns[1]]):
                                     mock_viz_sugg["chart_type"] = "scatter"
                            elif len(results_df.columns) == 1 and pd.api.types.is_numeric_dtype(results_df[results_df.columns[0]]):
                                 mock_viz_sugg = {"chart_type": "histogram", "x_column": results_df.columns[0]}
                            else:
                                 mock_viz_sugg = None # Cannot suggest basic viz

                            if mock_viz_sugg:
                                st.session_state.analysis_plot = mock_create_plotly_chart(results_df, mock_viz_sugg)
                                st.info("Generated a quick visualization for the results.")
                        except Exception as viz_e:
                             st.warning(f"Could not auto-visualize results: {viz_e}")

                    # Optionally generate deep insights immediately
                    if results_df is not None and not results_df.empty:
                         try:
                              st.session_state.deep_insights = mock_generate_deep_insights(results_df, nl_query)
                         except Exception as insight_e:
                              st.warning(f"Could not generate deep insights: {insight_e}")

                except Exception as e:
                     st.error(f"Analysis Failed: {e}")
                     logger.error(f"Analysis execution failed: {e}", exc_info=True)

# --- Results and Insights Column ---
with results_col:
    st.subheader("Analysis Results")
    if st.session_state.analysis_results is not None:
        st.dataframe(st.session_state.analysis_results, use_container_width=True)
    else:
        st.info("Results will appear here after running an analysis query.")

    st.subheader("Quick Visualization")
    if st.session_state.analysis_plot:
        st.plotly_chart(st.session_state.analysis_plot, use_container_width=True)
    else:
        st.info("A relevant visualization may appear here after analysis.")

    st.subheader("AI Deep Insights")
    if st.button("Generate Deep Insights", key="gen_deep_insights_btn"):
         if st.session_state.analysis_results is None or st.session_state.analysis_results.empty:
              st.warning("No analysis results available to generate insights from.")
         else:
              with st.spinner("Generating deep insights..."):
                   try:
                        # Replace with actual backend call
                        st.session_state.deep_insights = mock_generate_deep_insights(
                            st.session_state.analysis_results,
                            st.session_state.analysis_nl_query
                        )
                   except Exception as e:
                        st.error(f"Failed to generate insights: {e}")
                        logger.error(f"Deep insight generation failed: {e}", exc_info=True)


    if st.session_state.deep_insights:
        st.markdown(st.session_state.deep_insights)
    else:
        st.info("Click 'Generate Deep Insights' after running analysis.")


# --- Next Step Hint ---
if st.session_state.analysis_results is not None:
    st.success("➡️ Proceed to **5_📈_Report_Recommend** to build dashboards, reports, and get recommendations.")
=======
# pages/4_📊_Analyze_Insight.py
# Note: This file should be in the 'pages/' directory at the project root.

import streamlit as st
import pandas as pd
import numpy as np # Ensure numpy is imported
import plotly.express as px
import plotly.graph_objects as go
import logging
import time
import io
import json
from typing import List, Dict, Any, Optional, Union

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Backend Function Imports ---
BACKEND_AVAILABLE = False
ADVANCED_BACKEND_AVAILABLE = False
# Initialize function variables
process_nlq_to_sql_func=None; execute_query_func=None; generate_insights_func=None;
create_chart_func=None; generate_forecast_func=None; find_segments_func=None;
generate_rca_hints_func=None;
# ** NEW: Function placeholder for NLQ -> Pandas **
process_nlq_to_pandas_func = None # This needs implementation in backend

try:
    from backend.reporting.visualizer import create_plotly_chart
    from backend.analysis.nlq_processor import process_nlq_to_sql
    # ** NEW: Import the NLQ -> Pandas processor **
    from backend.analysis.nlq_processor import process_nlq_to_pandas # Assume this will exist
    from backend.analysis.query_executor import execute_query
    from backend.analysis.insight_generator import generate_deep_insights, generate_root_cause_hints
    BACKEND_AVAILABLE = True
    try: # Advanced Features
         from backend.analysis.forecaster import generate_forecast
         from backend.analysis.segmenter import find_segments
         ADVANCED_BACKEND_AVAILABLE = True
         logger.info("Standard & Advanced backend modules loaded for Analyze page.")
    except ImportError as adv_e: logger.warning(f"Advanced backend modules unavailable: {adv_e}")
    # Assign standard functions
    process_nlq_to_sql_func = process_nlq_to_sql
    process_nlq_to_pandas_func = process_nlq_to_pandas # Assign real function
    execute_query_func = execute_query
    generate_insights_func = generate_deep_insights
    create_chart_func = create_plotly_chart
    generate_rca_hints_func = generate_root_cause_hints
    # Assign advanced functions if available
    if ADVANCED_BACKEND_AVAILABLE: generate_forecast_func = generate_forecast; find_segments_func = find_segments
    else: # Define dummies
        def _dummy_raise(*args, **kwargs): raise NotImplementedError(f"{kwargs.get('name','Feature')} unavailable")
        generate_forecast_func = lambda *a, name="Forecasting", **kw: _dummy_raise(name=name)
        find_segments_func = lambda *a, name="Segmentation", **kw: _dummy_raise(name=name)

except ImportError as e:
    logger.error(f"Core backend import failed in Analyze: {e}", exc_info=True); st.error(f"CRITICAL: Backend not found. Using MOCK. Error: {e}", icon="🚨")
    try: # Import Mocks
        from pages._mocks import (mock_process_nlq_to_sql, mock_execute_query, mock_generate_deep_insights, mock_create_plotly_chart, mock_generate_forecast, mock_find_segments, mock_generate_rca_hints, mock_process_nlq_to_pandas) # Import NLQ->Pandas mock
        process_nlq_to_sql_func=mock_process_nlq_to_sql; process_nlq_to_pandas_func=mock_process_nlq_to_pandas; execute_query_func=mock_execute_query; generate_insights_func=mock_generate_deep_insights; create_chart_func=mock_create_plotly_chart; generate_forecast_func=mock_generate_forecast; find_segments_func=mock_find_segments; generate_rca_hints_func=mock_generate_rca_hints
        logger.info("Loaded MOCK functions for Analyze page.")
    except ImportError as mock_e: logger.critical(f"Failed mock import: {mock_e}", exc_info=True); st.error(f"CRITICAL ERROR: Fallback failed: {mock_e}", icon="🚨"); st.stop()
    except Exception as general_mock_e: logger.critical(f"Mock import error: {general_mock_e}", exc_info=True); st.error(f"Setup error: {general_mock_e}", icon="🚨"); st.stop()
except Exception as e: logger.critical(f"Backend import error: {e}", exc_info=True); st.error(f"Setup error: {e}", icon="🚨"); st.stop()

# --- Safety Check ---
if not all([process_nlq_to_sql_func, execute_query_func, generate_insights_func, create_chart_func, generate_forecast_func, find_segments_func, generate_rca_hints_func, process_nlq_to_pandas_func]):
    logger.critical("Essential functions failed load for Analyze page."); st.error("App setup error.", icon="🚨"); st.stop()

# --- Page Title and Introduction ---
st.header("4. 📊 Analyze & Insight")
st.markdown("Query data via NLQ, run analyses, forecast trends, find segments, investigate root causes, and generate AI insights.")
st.divider()

# --- Initialize Session State ---
# (Keep state initialization as before) ...
keys_needed = ['engineered_dataframe', 'cleaned_dataframe', 'connection_info', 'analysis_nl_query', 'analysis_generated_sql', 'analysis_results', 'analysis_plot', 'deep_insights', 'mongo_collection_to_analyze', 'forecast_params', 'forecast_results', 'segmentation_params', 'segmentation_results', 'rca_conversation']
for key in keys_needed:
    if key not in st.session_state: st.session_state[key] = "" if key in ['analysis_nl_query','analysis_generated_sql','deep_insights','mongo_collection_to_analyze'] else [] if key=='rca_conversation' else {} if key in ['forecast_params','forecast_results','segmentation_params','segmentation_results'] else None


# --- Check if Data is Available & Determine Context ---
can_analyze = False; data_source_context = None; active_dataframe = None; source_description = "No data loaded."
info_container = st.container(border=True)
with info_container: # (Data context determination logic using .get() - same as before) ...
    if st.session_state.get('engineered_dataframe') is not None: active_dataframe = st.session_state.engineered_dataframe; data_source_context = active_dataframe; source_description = f"✅ Using 'Explore & Engineer' data (`{active_dataframe.shape}`)."; can_analyze = True; st.success(source_description)
    elif st.session_state.get('cleaned_dataframe') is not None: active_dataframe = st.session_state.cleaned_dataframe; data_source_context = active_dataframe; source_description = f"✅ Using 'Clean & Transform' data (`{active_dataframe.shape}`)."; can_analyze = True; st.success(source_description)
    elif st.session_state.get('connection_info') and st.session_state.connection_info.get('type') != 'file': can_analyze = True; data_source_context = st.session_state.connection_info; conn_type = st.session_state.connection_info['type']; conn_name = st.session_state.connection_info.get('db_name', 'N/A'); source_description = f"✅ Connected to **{conn_type}** (`{conn_name}`). Ready to query."; st.success(source_description)
    else: st.warning("⚠️ Please connect/prepare data first."); st.stop()


# --- Main Layout: Tabs ---
st.subheader("Select Analysis Type")
tab_query, tab_forecast, tab_segment, tab_rca = st.tabs(["❓ NLQ & Insights", "📈 Forecasting", "🧩 Segmentation", "🕵️ Root Cause"])

# --- Tab 1: NLQ & Insights ---
with tab_query:
    query_col, results_col = st.columns([0.45, 0.55])
    with query_col:
        st.markdown("**Ask Your Data (NLQ)**")
        query_input_container = st.container(border=True)
        with query_input_container:
            nl_query = st.text_area("Enter question:", key="analysis_nl_query_input_tab1", value=st.session_state.analysis_nl_query, height=100)
            st.session_state.analysis_nl_query = nl_query

            # Determine context type
            is_sql_db = isinstance(data_source_context, dict) and data_source_context.get('type') in ["postgresql", "mysql", "sqlite", "mssql", "oracle"]
            is_mongo_db = isinstance(data_source_context, dict) and data_source_context.get('type') == 'mongodb'
            is_dataframe = isinstance(data_source_context, pd.DataFrame)

            schema_context_for_llm = ""
            if isinstance(data_source_context, dict): schema_context_for_llm = data_source_context.get('schema', '')
            elif isinstance(data_source_context, pd.DataFrame):
                buffer = io.StringIO(); data_source_context.info(buf=buffer, max_cols=100, verbose=False); schema_context_for_llm = f"Schema:\n{buffer.getvalue()}"

            # Generate SQL button (only if SQL DB)
            if is_sql_db:
                if st.button("⚙️ Generate SQL", key="analysis_gen_sql_btn_tab1"):
                    if not nl_query: st.warning("Enter question.")
                    else:
                        with st.spinner("Generating..."):
                            try: generated_sql = process_nlq_to_sql_func(nl_query, connection_info=data_source_context); st.session_state.analysis_generated_sql = generated_sql; st.toast("SQL Generated.", icon="⚙️")
                            except Exception as e: st.error(f"SQL Gen Failed: {e}"); logger.error(f"SQL Gen failed: {e}", exc_info=True)
                if st.session_state.analysis_generated_sql: st.code(st.session_state.analysis_generated_sql, language='sql')
                else: st.caption("_Generate SQL from question._")

            # Mongo inputs
            if is_mongo_db:
                 mongo_collection = st.text_input("Collection Name*", key="mongo_query_collection", value=st.session_state.mongo_collection_to_analyze); st.session_state.mongo_collection_to_analyze = mongo_collection
                 st.button("Generate Mongo Query (WIP)", disabled=True); st.caption("_NLQ->Mongo WIP. Run uses find({})._")

            st.markdown("---")
            # Run Analysis Button
            if st.button("🚀 Run Analysis", key="run_analysis_query_btn_tab1", type="primary", use_container_width=True):
                st.session_state.analysis_results = None; st.session_state.analysis_plot = None; st.session_state.deep_insights = ""
                query_type = None; query_to_run = None; exec_connection_obj = None; db_name_for_exec = None; collection_name_for_exec = None

                try: # Determine execution params
                    if is_sql_db:
                        query_type="sql"; query_to_run=st.session_state.analysis_generated_sql;
                        if not query_to_run or query_to_run.startswith("Error:"): raise ValueError("Valid SQL required.")
                        exec_connection_obj=data_source_context.get('engine');
                        if exec_connection_obj is None: raise ValueError("DB engine missing.")
                    elif is_mongo_db:
                         query_type="mongodb"; query_to_run={}; collection_name_for_exec=st.session_state.mongo_collection_to_analyze
                         if not collection_name_for_exec: raise ValueError("Enter Collection Name.")
                         exec_connection_obj=data_source_context.get('client'); db_name_for_exec=data_source_context.get('db_name')
                         if not all([exec_connection_obj, db_name_for_exec]): raise ValueError("MongoDB details missing.")
                    elif is_dataframe:
                         # ** FIX: Generate Pandas instructions from NLQ **
                         query_type="pandas"; exec_connection_obj=active_dataframe
                         if not nl_query: raise ValueError("Enter question for Pandas analysis.")
                         with st.spinner("Processing NLQ for DataFrame..."):
                              # Call the new NLQ->Pandas function (real or mock)
                              query_to_run = process_nlq_to_pandas_func(nl_query, schema_context_for_llm)
                              # query_to_run might be generated code string or structured steps dict/list
                              if not query_to_run: raise ValueError("Could not translate question to Pandas operation.")
                              logger.info(f"NLQ processed for Pandas. Resulting query/instructions: {str(query_to_run)[:100]}...")
                    else: raise ValueError("Cannot determine analysis type.")
                except ValueError as e: st.warning(str(e)); st.stop()
                except Exception as e: st.error(f"NLQ Processing Error: {e}"); logger.error(f"NLQ Processing failed: {e}"); st.stop()

                # Execute Query (Pass generated code/steps for pandas)
                if query_type and exec_connection_obj is not None: # query_to_run can be dict/list/str
                    with st.spinner(f"Executing {query_type}..."):
                        try:
                            exec_args = {"query_type": query_type, "query": query_to_run, "connection_obj": exec_connection_obj, "db_name": db_name_for_exec, "collection_name": collection_name_for_exec}
                            results_df = execute_query_func(**exec_args) # execute_query needs to handle pandas query_to_run
                            st.session_state.analysis_results = results_df
                            # ... (Toast, Auto-viz, Auto-insight logic - same as before) ...
                            if results_df is None or results_df.empty: st.toast("Query ran, no results.", icon="🤷")
                            else: st.toast(f"Complete ({len(results_df)} rows).", icon="✅")
                            try: # Auto-viz
                                viz_sugg = None; cols = results_df.columns
                                if len(cols)>=2: c0n=pd.api.types.is_numeric_dtype(results_df.iloc[:,0]); c1n=pd.api.types.is_numeric_dtype(results_df.iloc[:,1]); viz_sugg = {"chart_type":"scatter","x_column":cols[0],"y_column":cols[1]} if c0n and c1n else {"chart_type":"bar","x_column":cols[0],"y_column":cols[1]} if not c0n and c1n else None
                                elif len(cols)==1 and pd.api.types.is_numeric_dtype(results_df.iloc[:,0]): viz_sugg = {"chart_type":"histogram","x_column":cols[0]}
                                if viz_sugg: st.session_state.analysis_plot = create_chart_func(results_df, viz_sugg)
                            except Exception as viz_e: logger.error(f"Auto-viz fail: {viz_e}", exc_info=True); st.warning(f"Viz Error: {type(viz_e).__name__}")
                            if results_df is not None and not results_df.empty: # Auto-insight
                                try: st.session_state.deep_insights = generate_insights_func(results_df, nl_query)
                                except Exception as insight_e: st.warning(f"Insight Gen Error: {insight_e}")
                            st.rerun()
                        except Exception as e: st.error(f"Analysis Failed: {e}"); logger.error(f"Analysis exec failed: {e}", exc_info=True)
                else: st.error("Analysis setup failed.")

    with results_col: # Display Results, Viz, Insights (same as before)
        st.subheader("📈 Results & Insights")
        results_container = st.container(border=True); viz_container = st.container(border=True); insights_container = st.container(border=True)
        with results_container: # Results Display
            st.markdown("**Query Results**");
            if st.session_state.analysis_results is not None:
                if st.session_state.analysis_results.empty: st.info("Query returned no data.")
                else: st.dataframe(st.session_state.analysis_results, use_container_width=True); st.caption(f"{len(st.session_state.analysis_results)} results.")
            else: st.info("Results appear here.")
        with viz_container: # Viz Display
             st.markdown("**Quick Visualization**");
             if st.session_state.analysis_plot: st.plotly_chart(st.session_state.analysis_plot, use_container_width=True)
             else: st.info("Visualization appears here.")
        with insights_container: # Insights Display
            st.markdown("**💡 AI Deep Insights**");
            if st.session_state.deep_insights: st.markdown(st.session_state.deep_insights)
            else:
                if st.session_state.analysis_results is not None and not st.session_state.analysis_results.empty:
                    if st.button("Generate Deep Insights", key="gen_deep_insights_btn_tab1"):
                        with st.spinner("Generating..."):
                             try: st.session_state.deep_insights = generate_insights_func(st.session_state.analysis_results, st.session_state.analysis_nl_query); st.rerun()
                             except Exception as e: st.error(f"Insight Gen Error: {e}"); logger.error(f"Insight gen failed: {e}", exc_info=True)
                else: st.info("Run analysis for insights.")

# --- Tab 2: Forecasting (#7) ---
with tab_forecast:
    st.subheader("📈 Automated Forecasting")
    forecast_container = st.container(border=True)
    with forecast_container:
        # ** FIX: Ensure correct checks and imports **
        if not ADVANCED_BACKEND_AVAILABLE or generate_forecast_func is None: st.warning("Forecasting backend unavailable."); st.stop()
        if active_dataframe is None: st.warning("Forecasting needs DataFrame data."); st.stop()

        # Check for required column types *after* verifying dataframe exists
        date_cols = active_dataframe.select_dtypes(include=['datetime64', 'datetime', 'datetime64[ns]']).columns.tolist()
        numeric_cols = active_dataframe.select_dtypes(include=np.number).columns.tolist() # Requires numpy import

        if not date_cols or not numeric_cols:
            st.warning("Forecasting requires the DataFrame to have at least one datetime column and one numeric column. Please convert types in the 'Clean Transform' step if necessary.")
            st.dataframe(active_dataframe.dtypes.reset_index().rename(columns={'index':'Column', 0:'DataType'}), hide_index=True) # Show types
            st.stop()

        # (Forecasting UI and Logic - same as before, uses generate_forecast_func) ...
        with st.form("forecast_form"):
            st.markdown("**Configure Forecast:**"); time_col = st.selectbox("Time Column:",date_cols, key="fc_time"); target_col = st.selectbox("Target Column:",numeric_cols, key="fc_target"); periods = st.number_input("Periods Ahead:", 1, value=12, key="fc_periods"); submitted = st.form_submit_button("🔮 Generate Forecast", type="primary")
            if submitted:
                st.session_state.forecast_params={"time_col":time_col, "target_col":target_col, "periods":periods}; st.session_state.forecast_results=None
                with st.spinner(f"Forecasting '{target_col}'..."):
                    try: fc_df, fc_fig = generate_forecast_func(df=active_dataframe, time_col=time_col, target_col=target_col, periods=periods); st.session_state.forecast_results={"data":fc_df, "fig":fc_fig}; st.toast("Forecast done!", icon="🔮")
                    except Exception as e: st.error(f"Forecast Failed: {e}"); logger.error(f"Forecast failed: {e}", exc_info=True)
        # (Display Forecast Results - same as before) ...
        if st.session_state.forecast_results:
             st.markdown("**Forecast Plot:**");
             if st.session_state.forecast_results.get("fig"): st.plotly_chart(st.session_state.forecast_results["fig"], use_container_width=True)
             else: st.warning("Plot unavailable.")
             st.markdown("**Forecast Data (Last Periods):**");
             if st.session_state.forecast_results.get("data") is not None: st.dataframe(st.session_state.forecast_results["data"].tail(st.session_state.forecast_params['periods']), use_container_width=True)
             else: st.warning("Data unavailable.")

# --- Tab 3: Segmentation (#9) ---
with tab_segment:
    st.subheader("🧩 Automated Segmentation")
    segment_container = st.container(border=True)
    with segment_container:
        # ** FIX: Ensure correct checks and imports **
        if not ADVANCED_BACKEND_AVAILABLE or find_segments_func is None: st.warning("Segmentation backend unavailable."); st.stop()
        if active_dataframe is None: st.warning("Segmentation needs DataFrame data."); st.stop()

        # Check for required column types
        numeric_cols = active_dataframe.select_dtypes(include=np.number).columns.tolist() # Requires numpy import

        if len(numeric_cols) < 2:
            st.warning("Segmentation requires the DataFrame to have at least two numeric features. Please convert types in 'Clean Transform' or create features in 'Explore Engineer'.")
            st.dataframe(active_dataframe.dtypes.reset_index().rename(columns={'index':'Column', 0:'DataType'}), hide_index=True) # Show types
            st.stop()

        # (Segmentation UI and Logic - same as before, uses find_segments_func) ...
        with st.form("segment_form"):
             st.markdown("**Configure Segmentation:**"); features_to_use = st.multiselect("Features:", numeric_cols, default=numeric_cols[:min(len(numeric_cols), 4)], key="seg_features"); n_clusters = st.slider("Segments:", 2, 10, 3, key="seg_clusters"); submitted = st.form_submit_button("💡 Find Segments", type="primary")
             if submitted:
                  if len(features_to_use)<2: st.warning("Select >= 2 features.")
                  else:
                      st.session_state.segmentation_params={"features":features_to_use, "n_clusters":n_clusters}; st.session_state.segmentation_results=None
                      with st.spinner(f"Finding {n_clusters} segments..."):
                          try: seg_df, seg_sum, seg_fig = find_segments_func(df=active_dataframe, feature_cols=features_to_use, n_clusters=n_clusters); st.session_state.segmentation_results={"data":seg_df, "summary":seg_sum, "fig":seg_fig}; st.toast("Segments found!", icon="💡")
                          except Exception as e: st.error(f"Segmentation Failed: {e}"); logger.error(f"Segmentation failed: {e}", exc_info=True)
        # (Display Segmentation Results - same as before) ...
        if st.session_state.segmentation_results:
             st.markdown("**Segmentation Plot (PCA):**");
             if st.session_state.segmentation_results.get("fig"): st.plotly_chart(st.session_state.segmentation_results["fig"], use_container_width=True)
             else: st.warning("Plot unavailable.")
             st.markdown("**Segment Summary:**"); summary=st.session_state.segmentation_results.get("summary")
             if summary:
                 for seg_id, seg_info in summary.items():
                      with st.expander(f"**{seg_info.get('label', f'Segment {seg_id}')}** ({seg_info.get('size_percentage', 'N/A')})"): st.markdown(seg_info.get('description', 'N/A')); st.write("Characteristics:", seg_info.get('characteristics', {}))
             else: st.warning("Summary unavailable.")
             st.markdown("**Data w/ Labels:**");
             if st.session_state.segmentation_results.get("data") is not None: st.dataframe(st.session_state.segmentation_results["data"].head(), use_container_width=True)
             else: st.warning("Segmented data unavailable.")

# --- Tab 4: Root Cause Analysis (#8) ---
with tab_rca:
    st.subheader("🕵️ Root Cause Analysis (Chat)")
    rca_container = st.container(border=True)
    with rca_container:
        st.info("Ask about anomalies or changes (e.g., 'Why did sales drop?') to get investigation hints.", icon="💡")
        # (Chat Interface Logic - same as before, uses generate_rca_hints_func) ...
        if "rca_messages" not in st.session_state: st.session_state.rca_messages = [{"role": "assistant", "content": "What issue to investigate?"}]
        # Display chat history
        for message in st.session_state.rca_messages:
             with st.chat_message(message["role"]): st.markdown(message["content"])
        # Get user input
        if prompt := st.chat_input("Describe issue or ask follow-up"):
             st.session_state.rca_messages.append({"role": "user", "content": prompt})
             with st.chat_message("user"): st.markdown(prompt)
             # Get assistant response
             with st.chat_message("assistant"):
                  with st.spinner("AI investigating..."):
                      try:
                           rca_data_context = active_dataframe if active_dataframe is not None else data_source_context
                           if rca_data_context is None: raise ValueError("Data context not available for RCA.")
                           hints = generate_rca_hints_func(df=rca_data_context, issue_description=prompt, conversation_history=st.session_state.rca_messages)
                           response = "- " + "\n- ".join(hints) if hints else "Cannot determine specific hints."
                      except Exception as e: response = f"Error: {e}"; logger.error(f"RCA failed: {e}", exc_info=True)
                      st.markdown(response); st.session_state.rca_messages.append({"role": "assistant", "content": response})


# --- Next Step Hint ---
st.divider()
# (Next step hint logic - same as before) ...
analysis_done = False; analysis_keys = ['analysis_results', 'forecast_results', 'segmentation_results']
for key in analysis_keys:
    res = st.session_state.get(key);
    if isinstance(res, pd.DataFrame) and not res.empty: analysis_done=True; break
    if isinstance(res, dict) and res.get("data") is not None and not res["data"].empty: analysis_done=True; break
if analysis_done: st.success("➡️ Analysis complete! Proceed to **5_📈_Report_Recommend**.")
else: st.caption("Perform analysis (NLQ, Forecast, Segment) to generate results.")
>>>>>>> 946a937 (Add application file)
