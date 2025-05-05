# pages/project_dashboard.py
# Displays overview for a specific project

import streamlit as st
import logging
import pandas as pd # Example for displaying lists
from typing import List, Dict, Any, Optional # Import necessary types

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Backend/DB Imports ---
DB_AVAILABLE = False
# Initialize crud functions to None or dummies
get_all_datasources_func = lambda *a, **k: []
get_reports_for_project_func = lambda *a, **k: [] # Define dummy immediately
get_active_kpis_func = lambda *a, **kw: []
get_project_func = lambda *a, **kw: None # Dummy for get_project
get_db_session = None
# Define dummy Project model if needed for type hints when DB fails
class Project: pass # Dummy Class
class DataSource: pass
class Report: pass
class KPI: pass


try:
    from backend.database import crud, models
    from backend.database.session import get_db_session # Example session management
    # Assign real functions
    get_all_datasources_func = crud.get_all_datasources
    # Use getattr for potentially missing CRUD function with fallback
    get_reports_for_project_func = getattr(crud, "get_reports_for_project", lambda *a, **kw: []) # Safely get function or provide dummy
    get_active_kpis_func = crud.get_active_kpis
    get_project_func = crud.get_project # Assign real get_project
    # Assign real models for type hints
    Project = models.Project
    DataSource = models.DataSource
    Report = models.Report
    KPI = models.KPI

    DB_AVAILABLE = True
    logger.info("Database components loaded for Project Dashboard.")
except ImportError as e:
    logger.warning(f"DB components not found for Project Dashboard: {e}. Displaying placeholder data.")
    # Fallback dummies already assigned basically
except AttributeError as e:
     logger.warning(f"Attribute error during DB import (likely missing function in crud.py): {e}")
     # Ensure dummies are assigned if getattr failed implicitly above
     if get_reports_for_project_func is None: get_reports_for_project_func = lambda *a, **kw: []
     if get_project_func is None: get_project_func = lambda *a, **kw: None
except Exception as e:
     logger.error(f"Unexpected error importing DB components: {e}", exc_info=True)
     st.error(f"Error setting up page dependencies: {e}")


# --- Page Config & Title ---
# Page config set in root app.py
st.header("📁 Project Dashboard")
st.markdown("Overview of the currently selected project's resources and status.")
st.divider()

# --- Check if a Project is Selected ---
# Access session state safely using .get()
current_project_info = st.session_state.get('current_project')

if not current_project_info:
    st.warning("⚠️ Please select or create a project using the sidebar first.")
    # Optional: Add button to create one? Needs navigation logic if not in sidebar
    # if st.button("Go to Project Creation"): st.switch_page("app.py") # Go back to main?
    st.stop() # Stop rendering if no project context

# --- Load Project Data ---
project_id = current_project_info.get('id')
project_name = current_project_info.get('name', 'Unnamed Project')
project_description = None # Initialize

st.subheader(f"Project: {project_name}")

# Fetch full project details (like description) if needed
if DB_AVAILABLE and project_id and get_project_func and get_db_session:
     try:
          with get_db_session() as db:
               project_details = get_project_func(db=db, project_id=project_id)
               if project_details:
                    project_description = project_details.description
     except Exception as e:
          logger.error(f"Could not fetch project description for ID {project_id}: {e}")

if project_description:
     st.caption(f"Description: {project_description}")

# Initialize lists for resources
data_sources_list: List[DataSource] = [] # Use imported or dummy model type
reports_list: List[Report] = []
kpis_list: List[KPI] = []
error_loading = None

# Fetch related resources only if DB is available and project_id is valid
if DB_AVAILABLE and project_id is not None and get_db_session:
    try:
        # Use a single session for all queries for this page load
        with get_db_session() as db:
            logger.info(f"Loading resources for Project ID: {project_id}")
            # Call the assigned functions (real or dummy/lambda)
            data_sources_list = get_all_datasources_func(db=db, project_id=project_id)
            reports_list = get_reports_for_project_func(db=db, project_id=project_id)
            kpis_list = get_active_kpis_func(db=db, project_id=project_id)
            logger.info(f"Loaded {len(data_sources_list)} sources, {len(reports_list)} reports, {len(kpis_list)} KPIs for project {project_id}.")
    except Exception as e:
         error_loading = f"Error loading project resources: {e}"
         logger.error(f"Failed loading resources for project {project_id}: {e}", exc_info=True)
elif not DB_AVAILABLE:
     error_loading = "Database connection not available."
elif project_id is None: # Should have been caught by the check above, but defensive
     error_loading = "Project ID is missing."
elif get_db_session is None:
     error_loading = "Database session handler is unavailable."


if error_loading:
     st.error(error_loading)

st.divider()

# --- Display Project Contents ---
st.markdown("#### Project Resources Overview")
# ** FIX: Complete the st.columns call **
col1, col2, col3 = st.columns(3)
# ** END FIX **

with col1:
    st.markdown("**🔗 Data Sources**")
    ds_container = st.container(border=True, height=250) # Give fixed height
    with ds_container:
        if not data_sources_list: st.caption("_No data sources linked._")
        else:
            for ds in data_sources_list:
                 # Check attributes exist before accessing
                 ds_name_display = getattr(ds, 'name', 'Unnamed')
                 ds_type_display = getattr(ds, 'source_type', 'N/A')
                 st.markdown(f"- `{ds_name_display}` (`{ds_type_display}`)")
        if st.button("Manage Sources", key="manage_sources_btn", disabled=True): pass # Placeholder

with col2:
    st.markdown("**📄 Reports/Dashboards**")
    report_container = st.container(border=True, height=250)
    with report_container:
        if not reports_list: st.caption("_No reports saved._")
        else:
            for r in reports_list:
                r_name_display = getattr(r, 'name', 'Unnamed')
                r_type_display = getattr(r, 'report_type', 'N/A')
                st.markdown(f"- `{r_name_display}` (`{r_type_display}`)")
        if st.button("Manage Reports", key="manage_reports_btn", disabled=True): pass

with col3:
    st.markdown("**🎯 Tracked KPIs**")
    kpi_container = st.container(border=True, height=250)
    with kpi_container:
        if not kpis_list: st.caption("_No KPIs defined._")
        else:
            for kpi in kpis_list:
                 kpi_name_display = getattr(kpi, 'name', 'Unnamed')
                 st.markdown(f"- `{kpi_name_display}`")
        if st.button("Manage KPIs", key="manage_kpis_btn", disabled=True): pass


st.divider()
st.caption("_Project dashboard under development. More details and actions coming soon._")