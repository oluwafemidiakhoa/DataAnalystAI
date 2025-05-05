# pages/_mocks.py
# Centralized mock functions for frontend testing when backend is unavailable

import pandas as pd
import time
import logging
import io
import plotly.graph_objects as go
import plotly.express as px
import random
import numpy as np
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__) # Get logger for mock functions themselves

# --- Mocks from Page 1 (Connect & Profile) ---
def mock_get_sql_engine(db_type, user, pwd, host, port, database, **kwargs):
    logger.debug(f"MOCK get_sql_engine called for {db_type}"); time.sleep(0.5); return {"type": db_type, "db": database, "status": "mock connected", "_is_mock": True}
def mock_get_mongo_client(uri):
    logger.debug("MOCK get_mongo_client called"); time.sleep(0.5); return {"uri": uri, "status": "mock connected", "_is_mock": True}
def mock_load_dataframe_from_file(uploaded_file_or_path):
    filename = getattr(uploaded_file_or_path, 'name', str(uploaded_file_or_path)); logger.debug(f"MOCK load_dataframe_from_file: {filename}"); time.sleep(0.5);
    try: # Simplified logic
        if filename.lower().endswith(".csv"): return pd.read_csv(uploaded_file_or_path)
        elif filename.lower().endswith((".xlsx", ".xls")): return pd.read_excel(uploaded_file_or_path, engine='openpyxl')
        elif filename.lower().endswith(".json"): return pd.read_json(uploaded_file_or_path)
        else: raise ValueError("Unsupported mock file type.")
    except Exception as e: logger.error(f"Mock file load failed: {e}"); raise
def mock_get_schema_details(source_type: str, source_obj: Any, db_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    logger.debug(f"MOCK get_schema_details: {source_type}"); schema_info = f"**Mock Schema ({source_type})**\n```text\n"; join_candidates = []
    try: # Simplified logic
        if source_type == "file": schema_info += f"File: {getattr(source_obj,'attrs',{}).get('filename','N/A')}\n" + source_obj.head().to_string()
        else: schema_info += "Mock DB Schema text..."
    except Exception as e: schema_info += f"\nError: {e}"
    schema_info += "\n```"; return {"schema_string": schema_info, "join_key_candidates": join_candidates}
def mock_generate_profile_report(df: pd.DataFrame, source_name: str = "DataFrame", **kwargs) -> Dict[str, Any]:
    logger.debug(f"MOCK generate_profile_report: {source_name}"); time.sleep(1); # Simplified logic
    return {"overview": {"rows": len(df) if df is not None else 0}, "llm_summary": "**Mock Profile Summary**"}

# --- Mocks from Page 2 (Clean & Transform) ---
def mock_suggest_cleaning_steps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    logger.debug("MOCK suggest_cleaning_steps"); time.sleep(1); return [{"issue": "Mock Missing", "step_type":"impute_missing", "params":{"column":"A"}, "rationale":"Mock rationale"}]
def mock_generate_code_from_nl(nl_command, df_schema_context):
    logger.debug(f"MOCK generate_code_from_nl: {nl_command}"); time.sleep(0.5); return f"# Mock code for: {nl_command}"
def mock_apply_code(df, code_or_step_type, params=None):
    logger.warning(f"MOCK apply_code called (UNSAFE MOCK)"); time.sleep(0.5); print(f"--- MOCK EXECUTION ---"); return df.copy()
# Mock explanation function
def mock_explain_code(code_or_action: str, context_or_language: str = "python") -> str:
     logger.debug(f"MOCK explain_code called for: {code_or_action[:100]}..."); time.sleep(0.7)
     return f"**Mock Explanation:** This likely involves '{str(code_or_action)[:50]}...'."

# --- Mocks from Page 3 (Explore & Engineer) ---
def mock_suggest_visualization(df: pd.DataFrame, user_goal: Optional[str] = None) -> Dict[str, Any]:
    logger.debug("MOCK suggest_visualization"); time.sleep(0.5); return {"chart_type": "bar", "x_column": df.columns[0] if len(df.columns)>0 else 'A', "y_column": df.columns[1] if len(df.columns)>1 else 'B', "rationale": "Mock bar suggestion"}
def mock_create_plotly_chart(df: pd.DataFrame, suggestion: Dict[str, Any]) -> go.Figure:
    logger.warning("Using MOCK create_plotly_chart"); fig = go.Figure(); fig.update_layout(title_text=f"Mock Chart: {suggestion.get('chart_type')}"); return fig
def mock_suggest_feature_engineering(df: pd.DataFrame, goal: Optional[str] = None) -> List[Dict[str, Any]]:
    logger.debug("MOCK suggest_feature_engineering"); time.sleep(1); return [{"step_type": "mock_feature", "params": {"col":"A"}, "details":"Mock details", "rationale":"Mock rationale"}]
# apply_feature_engineering uses mock_apply_code

# --- Mocks from Page 4 (Analyze & Insight) ---
# ** FIX: Define the missing mock function **
def mock_process_nlq_to_sql(natural_language_query, connection_info, schema_context=None):
    logger.debug(f"MOCK process_nlq_to_sql: {natural_language_query}")
    time.sleep(1)
    return f"-- Mock SQL for: {natural_language_query}\nSELECT * FROM mock_table;"

def mock_execute_query(query_type: str, query: Any, connection_obj: Any, db_name: Optional[str] = None, collection_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    logger.debug(f"MOCK execute_query: {query_type}"); time.sleep(1); return pd.DataFrame({'MockResult': [1, 2]})
def mock_generate_deep_insights(df: pd.DataFrame, nl_query: Optional[str] = None) -> str:
    logger.debug("MOCK generate_deep_insights"); time.sleep(1.5); return "**Mock Insights:** Data looks fine."
# Add mocks for forecast, segment, rca
def mock_generate_forecast(*args, **kwargs): logger.warning("MOCK generate_forecast"); time.sleep(1); return pd.DataFrame({'ds':[], 'yhat':[]}), go.Figure()
def mock_find_segments(*args, **kwargs): logger.warning("MOCK find_segments"); time.sleep(1); return pd.DataFrame({'segment':[0,1]}), {'0':{'label':'Mock Seg A'}}, go.Figure()
def mock_generate_rca_hints(*args, **kwargs): logger.warning("MOCK generate_rca_hints"); time.sleep(1); return ["Mock RCA Hint 1"]

# --- Mocks from Page 5 (Report & Recommend) ---
def mock_define_kpi(kpi_name: str, **kwargs) -> Dict[str, Any]:
    logger.debug(f"MOCK define_kpi: {kpi_name}"); time.sleep(0.2); return {"name": kpi_name, "id": random.randint(1000, 9999)}
def mock_track_kpi(kpi_id: int, data_context: Any) -> Dict[str, Any]:
    logger.debug(f"MOCK track_kpi ID {kpi_id}"); time.sleep(0.1); return {"value": random.uniform(10,100), "trend": random.choice([-1,0,1]), "delta": random.uniform(-5,5)}
def mock_track_all_active_kpis(data_context: Any, db: Any = None, defined_kpis: Optional[Dict] = None) -> Dict[str, Dict[str, Any]]:
    logger.debug("MOCK track_all_active_kpis"); tracked = {}; kpis_to_track = defined_kpis or {"Mock KPI 1": {"id": 1}}
    for name, definition in kpis_to_track.items(): tracked[name] = mock_track_kpi(definition.get("id"), data_context)
    return tracked
# ** FIX: Update mock signature to accept 'kpis' **
def mock_build_narrative_report(report_goal: str, analysis_results: pd.DataFrame | None, insights: str | None, kpis: Dict | None, visualizations: Optional[List[go.Figure]]=None, include_images: bool=False, **kwargs) -> Dict[str, Any]:
    logger.debug(f"MOCK build_narrative_report: {report_goal}"); time.sleep(1)
    narrative = f"## Mock Report: {report_goal}\n"; narrative += f"Insights: {insights}\n" if insights else ""; narrative += f"KPIs: {list(kpis.keys())}\n" if kpis else ""; narrative += "Report built."
    return {"title": report_goal, "narrative": narrative, "visualizations": [], "kpi_summary": str(kpis), "images":[], "export_available": {'pdf': False, 'pptx': False}}
def mock_generate_recommendations(insights: str | None, **kwargs) -> List[Dict[str, Any]]:
    logger.debug("MOCK generate_recommendations"); time.sleep(1); return [{"recommendation": "Mock Rec.", "rationale": "Mock.", "confidence": "Low", "estimated_impact": "Low", "priority": "Low"}]