<<<<<<< HEAD
# app.py (Absolute Minimum)
import streamlit as st
import time

st.set_page_config(page_title="Bare Minimum Test")
st.title("Hello Hugging Face!")
st.write("If this stays running, the core environment is okay.")

# Optional loop just to keep it explicitly alive if needed
# while True:
#     print("Bare minimum app alive...", flush=True) # Print directly, flush buffer
#     time.sleep(30)
=======
# app.py (Located in the project root directory)
# Main entry point for the AI-Native Analytics Workspace

import streamlit as st
import logging
from pathlib import Path
import sys
import os # Import os
import time # Import time for delays
from typing import List, Dict, Any, Optional # For type hints

# --- Early Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("--- Starting DataVisionAI Workspace ---")

# --- Add frontend directory to path for utility import ---
frontend_dir = Path(__file__).parent / 'frontend'
if str(frontend_dir) not in sys.path:
    sys.path.append(str(frontend_dir))
    logger.info(f"Added {frontend_dir} to sys.path")

# --- Utility Function Import ---
CSS_LOADER_AVAILABLE = False
try:
    from utils import load_css # Assumes utils.py is in frontend/
    logger.info("Successfully imported load_css from frontend.utils")
    CSS_LOADER_AVAILABLE = True
except ImportError:
    logger.warning("Could not import load_css utility from frontend/utils.py. CSS will not be loaded.")
    def load_css(path): pass # Dummy function
except Exception as e:
     logger.error(f"An unexpected error occurred during utility import: {e}", exc_info=True)
     def load_css(path): pass

# --- Backend Imports for DB Setup & Projects ---
DB_AVAILABLE = False
# Define dummy crud initially
class crud:
    @staticmethod
    def get_projects_for_user(*args, **kwargs): logger.warning("Using dummy get_projects_for_user."); return []
    @staticmethod
    def create_project(*args, **kwargs): logger.error("DB unavailable - using dummy create_project."); return None
    # Add other dummies if needed elsewhere in this file
settings = None
db_engine = None
def create_db_and_tables(*args): logger.error("DB Cannot be initialized."); return None
def get_db_session(): logger.error("Dummy DB session used."); yield None

try:
    from backend.core.config import settings
    from backend.database.session import get_db_session, engine as db_engine # Import engine directly
    from backend.database.models import create_db_and_tables
    from backend.database import crud, models # Import REAL crud and models
    DB_AVAILABLE = True
    logger.info("Core backend and DB components imported successfully.")
except ImportError as import_err:
     logger.error(f"Failed initial import of DB/Config components: {import_err}", exc_info=True)
     # Keep using dummy crud defined above
except Exception as setup_err:
     logger.critical(f"Critical error during initial backend imports: {setup_err}", exc_info=True)
     st.error(f"Critical application setup error: {setup_err}. Cannot start.")
     st.stop() # Stop if essential backend components fail early


# --- Page Configuration ---
APP_TITLE = "DataVisionAI Workspace"
try:
    st.set_page_config(
        page_title=APP_TITLE, page_icon="✨", layout="wide", initial_sidebar_state="expanded",
        menu_items={ 'Get Help': None, 'Report a bug': None, 'About': f"## {APP_TITLE} 💡\nAI-Native Analytics by DataVisionAI." }
    )
    logger.info("Streamlit page config set.")
except Exception as e: logger.error(f"Failed set page config: {e}", exc_info=True); st.error(f"Fatal Error: Config failed. {e}"); st.stop()

# --- Load Custom CSS ---
if CSS_LOADER_AVAILABLE:
    css_file_path = frontend_dir / "styles" / "style.css"
    if css_file_path.is_file(): load_css(str(css_file_path))
    else: logger.warning(f"CSS file not found: {css_file_path}")


# --- Session State Initialization ---
def initialize_global_state():
    logger.info("Running global session state initialization...")
    state_defaults = {
        # Core Data Flow
        'connection_info': None, 'raw_dataframe': None, 'data_profile': {},
        'connection_error': "", 'cleaned_dataframe': None, 'applied_steps': [],
        'cleaning_suggestions': None, 'engineered_dataframe': None, 'applied_feature_steps': [],
        'analysis_nl_query': "", 'analysis_generated_sql': "", 'analysis_results': None,
        'analysis_plot': None, 'deep_insights': "", 'defined_kpis': {},
        'tracked_kpi_data': {}, 'generated_report_data': None,
        'generated_recommendations': [], 'recommendation_feedback': {}, 'mongo_collection_to_analyze': "",
        'current_plot': None,
        # Project/User state
        'current_user': None, # Will be set by login process or mock
        'current_project': None, # Dict {'id': ..., 'name': ...} or None
        # Advanced Feature Placeholders
        'data_catalog': {}, 'data_lineage': {}, 'quality_rules': [], 'quality_alerts': [],
        'active_notifications': [], 'etl_pipeline_def': None, 'forecast_params': {},
        'forecast_results': {}, 'segmentation_params': {}, 'segmentation_results': {},
        # Internal Flags
        'app_initialized': False,
        'db_initialized': False,
    }
    keys_initialized = []
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            import copy; st.session_state[key] = copy.deepcopy(default_value); keys_initialized.append(key)
    if keys_initialized: logger.debug(f"Initialized state keys: {', '.join(keys_initialized)}")
    else: logger.debug("Session state already initialized.")

if not st.session_state.get('app_initialized', False):
     initialize_global_state()
     st.session_state.app_initialized = True
     logger.info("Global session state initialization complete for this session.")


# --- Database Initialization (Run Once Per Process Start) ---
# Use session state flag to ensure it runs only once
if DB_AVAILABLE and not st.session_state.get('db_initialized', False):
    logger.info("Checking and initializing application database setup...")
    DB_SETUP_SUCCESS = False
    try:
        # Imports already attempted at the top, check if engine is available
        if settings and settings.database_url and db_engine is not None:
            logger.info(f"Using database engine for URL: {settings.database_url[:30]}...")
            create_db_and_tables(db_engine) # Call the imported function
            logger.info("Database tables verified/created successfully.")
            DB_SETUP_SUCCESS = True
        elif db_engine is None: logger.error("DB engine not created (check session.py/config). Skipping table creation.")
        else: logger.warning("DB URL not configured. Skipping DB table creation.")
    except Exception as db_setup_err:
        logger.error(f"CRITICAL: Failed DB setup during initialization: {db_setup_err}", exc_info=True)
        # Display error in UI if possible during startup (might not show immediately)
        st.error(f"Error initializing application database ({type(db_setup_err).__name__}). Persistence unavailable.")

    st.session_state.db_initialized = DB_SETUP_SUCCESS # Store True only on success
    logger.info(f"Database initialization attempt finished. Success: {DB_SETUP_SUCCESS}")
else:
     # Log status if DB wasn't available or already initialized
     status = "initialized" if st.session_state.get('db_initialized') else "unavailable" if not DB_AVAILABLE else "init check skipped"
     logger.debug(f"Database status: {status}.")


# --- Sidebar Content ---
with st.sidebar:
    logger.info("Setting up sidebar...")
    # --- Logo & Title ---
    logo_path = frontend_dir / "assets" / "Logo.png"
    if logo_path.is_file(): st.image(str(logo_path), width=150)
    else: st.markdown(f"## {APP_TITLE}")
    st.title("DataVisionAI"); st.caption("AI-Native Analytics"); st.divider()

    # --- Project Selection/Creation ---
    st.subheader("📂 Projects")
    # Authentication Placeholder - NEED LOGIN TO SET THIS
    if 'current_user' not in st.session_state or st.session_state.current_user is None:
        # Set a dummy user FOR LOCAL TESTING ONLY if no user is logged in
        if settings and settings.environment == 'development':
             st.session_state.current_user = {'id': 1, 'email': 'dev_user@local.test', 'name': 'Dev User'} # Dummy user
             st.caption("⚠️ Using DEV User (ID: 1)")
        else:
             st.warning("Login required for Projects.")
             st.caption("_(Use Auth Login page)_")

    current_user = st.session_state.get('current_user')
    project_list: List[models.Project] = []
    project_dict: Dict[int, str] = {} # {project_id: project_name}
    error_loading_projects = None

    # Load projects only if user and DB are available
    if current_user and DB_AVAILABLE:
        try:
            with get_db_session() as db: # Use the imported session manager
                user_projects = crud.get_projects_for_user(db=db, user_id=current_user['id']) # Use real crud
                project_list = sorted([p for p in user_projects if p], key=lambda p: p.name)
                project_dict = {p.id: p.name for p in project_list}
        except Exception as e:
            error_loading_projects = f"Failed to load projects: {e}"
            logger.error(f"Error loading projects for user {current_user.get('id')}: {e}", exc_info=True)
    elif not current_user: error_loading_projects = "Login required to manage projects."
    else: error_loading_projects = "DB unavailable for projects." # DB Check failed

    if error_loading_projects: st.caption(f"_{error_loading_projects}_")

    # --- Project Selector ---
    # ** FIX: Handle case where current_project might be None before getting 'id' **
    current_project_data = st.session_state.get('current_project') # Get the dict or None
    current_project_id = current_project_data.get('id') if isinstance(current_project_data, dict) else None

    options_with_none = [None] + list(project_dict.keys())
    format_func = lambda x: "Select a Project..." if x is None else project_dict.get(x, f"ID: {x}")
    current_selection_index = 0 # Default to "Select..."
    if current_project_id is not None and current_project_id in project_dict:
         try: current_selection_index = options_with_none.index(current_project_id)
         except ValueError: current_selection_index = 0

    selected_project_id = st.selectbox(
        "Current Project:", options=options_with_none, format_func=format_func,
        index=current_selection_index, key="project_selector_sb",
        label_visibility="collapsed", help="Select the project to work on."
    )

    # Update session state if selection changed
    if selected_project_id != current_project_id: # Compare IDs
        if selected_project_id is None: st.session_state.current_project = None; logger.info("Project deselected.")
        else:
            selected_project_obj = next((p for p in project_list if p.id == selected_project_id), None)
            if selected_project_obj: st.session_state.current_project = {"id": selected_project_obj.id, "name": selected_project_obj.name}; logger.info(f"Project selected: {selected_project_obj.name}")
            else: st.session_state.current_project = None; logger.error(f"Selected project ID {selected_project_id} not found.")
        st.rerun()

    # --- Create New Project Form ---
    with st.expander("➕ Create New Project"):
        if not current_user: st.caption("_Login to create projects._")
        elif not DB_AVAILABLE: st.caption("_DB unavailable._")
        else:
            with st.form("new_project_form"):
                new_project_name = st.text_input("New Project Name*")
                new_project_desc = st.text_area("Description (Optional)")
                submitted = st.form_submit_button("Create Project")
                if submitted:
                    if not new_project_name: st.warning("Project Name is required.")
                    else:
                        with st.spinner("Creating project..."):
                            try:
                                with get_db_session() as db:
                                    new_project = crud.create_project(db=db, name=new_project_name, description=new_project_desc, owner_id=current_user.get('id'))
                                    # TODO: Add current user to ProjectUser association table
                                    st.success(f"Project '{new_project.name}' created!")
                                    st.session_state.current_project = {"id": new_project.id, "name": new_project.name}
                                    time.sleep(1); st.rerun()
                            except Exception as e: st.error(f"Failed: {e}"); logger.error(f"Error creating project: {e}", exc_info=True)
    st.divider()

    # --- Core Navigation ---
    st.success("Select a stage from the lifecycle above.")
    st.caption("_Core data workflow pages._"); st.divider()

    # --- Workspace Tools ---
    st.subheader("Workspace Tools");
    # Comment out links until pages exist
    # st.page_link("pages/0_📚_Data_Catalog.py", label="Data Catalog", icon="📚")
    # st.page_link("pages/6_🚦_Quality_Monitor.py", label="Data Quality Monitor", icon="🚦")
    # st.page_link("pages/7_⚙️_Settings.py", label="Settings", icon="⚙️")
    st.caption("_Advanced tools placeholder._"); st.divider()

    # --- Other Sidebar Elements ---
    st.caption("_(Notifications placeholder)_"); st.divider()
    st.caption("_(User login placeholder)_")
    logger.info("Sidebar setup complete.")


# --- Main Area Content (Welcome Message) ---
logger.info("Rendering main welcome area...")
st.title(f"🚀 Welcome to the {APP_TITLE}!")
st.markdown(f"""Your intelligent partner""")
st.info("👈 **Select or Create a Project, then choose 'Connect & Profile' from the sidebar to begin!**") # Updated guidance
logger.info("Main welcome area rendered.")

# --- Optional: Debug Expander ---
# with st.expander("DEBUG: Show Full Session State"): st.write(st.session_state)

logger.info("Main app.py execution finished.")
>>>>>>> 946a937 (Add application file)
