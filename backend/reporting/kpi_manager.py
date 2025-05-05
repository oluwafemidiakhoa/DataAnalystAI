# backend/reporting/kpi_manager.py
import pandas as pd
import logging
from typing import Optional, Dict, Any, List, Union
import datetime
from sqlalchemy.orm import Session # Import Session for type hinting DB interactions

# Assuming CRUD utils are available for persistence
try:
    from backend.database import crud, models # Import models for type hints
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logging.warning("Database CRUD/models not found. KPI persistence/history will be disabled.")
    # Define dummy crud functions if needed for structure, though logic will mostly disable
    class DummyKPI: id = None; name = None; description = None; calculation_logic = None; target_value = None; is_active = True
    class DummyKpiValue: pass
    class crud:
         @staticmethod
         def get_kpi_by_name(*args, **kwargs): return None
         @staticmethod
         def create_or_update_kpi(*args, **kwargs): return DummyKPI() # Return dummy object
         @staticmethod
         def get_active_kpis(*args, **kwargs): return []
         @staticmethod
         def save_kpi_value(*args, **kwargs): return DummyKpiValue()
         @staticmethod
         def get_latest_kpi_value(*args, **kwargs): return None

# Assuming query executor might be needed to calculate KPIs from source data
try:
    from backend.analysis.query_executor import execute_query
    QUERY_EXECUTOR_AVAILABLE = True
except ImportError:
     QUERY_EXECUTOR_AVAILABLE = False
     logging.warning("Query executor not found. KPI calculation logic might be limited.")
     def execute_query(*args, **kwargs): raise NotImplementedError("Query Executor not available")


logger = logging.getLogger(__name__)

# --- KPI Definition ---

def define_kpi(
    name: str,
    description: Optional[str] = None,
    calculation_logic: Optional[str] = None, # SQL, Pandas expression, or NL description for AI
    calculation_type: str = 'pandas', # 'sql', 'pandas', 'mongo' - How to interpret logic
    target: Optional[float] = None,
    alert_upper: Optional[float] = None,
    alert_lower: Optional[float] = None,
    project_id: Optional[int] = None,
    created_by_id: Optional[int] = None,
    db: Optional[Session] = None # Pass DB Session for saving
    ) -> Dict[str, Any]:
    """
    Defines a new KPI and saves it to the database if a session is provided.

    Args:
        name: The unique name of the KPI.
        description: A description of the KPI.
        calculation_logic: How the KPI is calculated (SQL, Pandas expression, etc.).
        calculation_type: How to interpret the calculation_logic ('sql', 'pandas', 'mongo').
        target: An optional target value for the KPI.
        alert_upper: Optional upper alert threshold.
        alert_lower: Optional lower alert threshold.
        project_id: Optional project ID to associate with.
        created_by_id: Optional user ID of the creator.
        db: Optional database session for persistence.

    Returns:
        A dictionary representing the defined KPI, including its database ID if saved.
    """
    logger.info(f"Defining KPI: '{name}' (Type: {calculation_type})")
    if not name: raise ValueError("KPI name cannot be empty.")
    if not calculation_logic: logger.warning(f"KPI '{name}' defined without calculation logic.")
    if not calculation_type: raise ValueError("KPI calculation_type is required.")

    kpi_data_for_db = {
        "name": name, "description": description, "calculation_logic": calculation_logic,
        "target_value": target, "alert_threshold_upper": alert_upper, "alert_threshold_lower": alert_lower,
        "is_active": True, "project_id": project_id, "created_by_id": created_by_id
        # Add calculation_type to model if needed, or infer from logic? For now, not saving type directly.
    }

    saved_kpi_id = None
    if db and DB_AVAILABLE:
        try:
            # Use create_or_update to handle existing KPIs
            db_kpi = crud.create_or_update_kpi(db=db, **kpi_data_for_db)
            saved_kpi_id = db_kpi.id
            logger.info(f"KPI '{name}' saved/updated in database with ID {saved_kpi_id}.")
        except Exception as e:
            logger.error(f"Failed to save KPI '{name}' to database: {e}", exc_info=True)
            # Decide whether to raise or just log and continue without ID
            st.error(f"Database Error: Could not save KPI '{name}'.") # Show error in UI if session passed from Streamlit
            # raise # Optional: re-raise if saving is critical

    kpi_definition = kpi_data_for_db.copy()
    kpi_definition["id"] = saved_kpi_id # Add the ID if it was saved
    kpi_definition["calculation_type"] = calculation_type # Include type in returned dict

    return kpi_definition


# --- KPI Calculation (#7, #9 Logic Implementation) ---

def calculate_kpi_value(
    kpi_definition: Dict[str, Any],
    data_context: Union[pd.DataFrame, Dict[str, Any]] # DataFrame or Connection Info dict
    ) -> Optional[float]:
    """
    Calculates the current value of a KPI based on its logic and the data context.
    Attempts to execute SQL or evaluate Pandas expressions.
    """
    kpi_name = kpi_definition.get("name", "Unknown KPI")
    logic = kpi_definition.get("calculation_logic", "")
    calc_type = kpi_definition.get("calculation_type", "pandas") # Default to pandas if type missing

    logger.info(f"Calculating value for KPI: '{kpi_name}' (Type: {calc_type})")

    if not logic:
        logger.warning(f"No calculation logic defined for KPI '{kpi_name}'. Cannot calculate.")
        return None

    calculated_value = None
    try:
        # --- SQL Calculation ---
        if calc_type == "sql":
            if not isinstance(data_context, dict) or not data_context.get("engine"):
                raise ValueError(f"SQL calculation for KPI '{kpi_name}' requires DB connection info with engine.")
            if not QUERY_EXECUTOR_AVAILABLE:
                 raise RuntimeError("Query Executor not available for SQL KPI calculation.")

            logger.debug(f"Executing SQL logic for KPI '{kpi_name}': {logic}")
            # Assume the SQL logic returns a single row with a single numeric column
            result_df = execute_query(query_type="sql", query=logic, connection_obj=data_context["engine"])
            if result_df.empty or len(result_df.columns) == 0:
                 logger.warning(f"SQL for KPI '{kpi_name}' returned no results.")
            else:
                 # Attempt to convert the first cell to float
                 calculated_value = float(result_df.iloc[0, 0])

        # --- Pandas Calculation ---
        elif calc_type == "pandas":
            if not isinstance(data_context, pd.DataFrame):
                 raise ValueError(f"Pandas calculation for KPI '{kpi_name}' requires a DataFrame.")
            if data_context.empty:
                 logger.warning(f"DataFrame is empty for KPI '{kpi_name}'. Cannot calculate.")
            else:
                 logger.debug(f"Evaluating Pandas expression for KPI '{kpi_name}': {logic}")
                 # ** VERY Basic & Unsafe Evaluation ** - Use a proper expression evaluator in production
                 # This attempts common pandas aggregations based on simple strings
                 df = data_context # Alias for clarity
                 if logic.lower() == "count": calculated_value = float(len(df))
                 elif logic.lower().startswith("count_unique"): col = logic.split("'")[1]; calculated_value = float(df[col].nunique())
                 elif logic.lower().startswith("sum"): col = logic.split("'")[1]; calculated_value = float(df[col].sum())
                 elif logic.lower().startswith("mean"): col = logic.split("'")[1]; calculated_value = float(df[col].mean())
                 elif logic.lower().startswith("median"): col = logic.split("'")[1]; calculated_value = float(df[col].median())
                 # Add more specific parsing or use a safe evaluation library like 'numexpr' or 'asteval'
                 else:
                      logger.warning(f"Pandas logic '{logic}' for KPI '{kpi_name}' is not directly supported by basic evaluator. Trying generic eval (unsafe).")
                      try:
                           # UNSAFE EVALUATION - REPLACE WITH SAFER METHOD
                           calculated_value = pd.eval(logic, engine='python', local_dict={'df': df, 'pd': pd, 'np': np})
                           # Ensure it's a scalar float
                           if isinstance(calculated_value, (pd.Series, pd.DataFrame)):
                                raise ValueError("Pandas expression did not evaluate to a single scalar value.")
                           calculated_value = float(calculated_value)
                      except Exception as eval_err:
                           logger.error(f"Failed to evaluate pandas expression '{logic}' for KPI '{kpi_name}': {eval_err}")
                           raise ValueError(f"Invalid Pandas expression for KPI calculation: {eval_err}") from eval_err

        # --- Add MongoDB Calculation Logic Here ---
        # elif calc_type == "mongodb":
        #    if not isinstance(data_context, dict) or not data_context.get("client"): ...
        #    # Construct aggregation pipeline from 'logic' string? Complex.
        #    # result = execute_query(query_type='mongodb', ...) ...

        else:
             raise ValueError(f"Unsupported calculation_type '{calc_type}' for KPI '{kpi_name}'.")

        if calculated_value is not None:
            logger.info(f"Calculated value for '{kpi_name}': {calculated_value:.4f}") # Log with precision
            return float(calculated_value) # Ensure float
        else:
             logger.warning(f"Calculation for KPI '{kpi_name}' resulted in None.")
             return None

    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as e: # Catch expected errors
        logger.error(f"Failed to calculate KPI '{kpi_name}': {e}", exc_info=True)
        # Return None to indicate calculation failure without crashing
        return None
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error calculating KPI '{kpi_name}': {e}", exc_info=True)
        return None


# --- KPI Tracking & History ---

def track_all_active_kpis(
    data_context: Union[pd.DataFrame, Dict[str, Any]],
    project_id: Optional[int] = None, # Optional project filter
    db: Optional[Session] = None # DB Session required for history/definitions
    ) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves active KPI definitions (from DB), calculates their current values,
    determines trends based on historical data (from DB), and saves the new value.

    Args:
        data_context: The data source (DataFrame or connection info) for calculation.
        project_id: Optional project ID to filter KPIs.
        db: Database session is REQUIRED for loading definitions and history.

    Returns:
        Dict {kpi_name: {'value': float|None, 'trend': int|None, 'delta': float|None}}.
    """
    logger.info(f"Tracking all active KPIs (Project: {project_id or 'All'})...")
    if not db or not DB_AVAILABLE:
        logger.error("Database session required for tracking KPIs (loading definitions & history).")
        return {} # Cannot track without DB access

    tracked_data = {}
    active_kpis: List[models.KPI] = []

    # --- Load Active KPI Definitions from DB ---
    try:
        active_kpis = crud.get_active_kpis(db, project_id=project_id)
        if not active_kpis:
            logger.info("No active KPI definitions found in the database for tracking.")
            return {}
        logger.info(f"Loaded {len(active_kpis)} active KPI definitions from database.")
    except Exception as e:
        logger.error(f"Failed to load KPI definitions from database: {e}", exc_info=True)
        return {} # Cannot proceed without definitions

    # --- Calculate Value, Trend, Delta for each KPI ---
    now = datetime.datetime.now(datetime.timezone.utc) # Consistent timestamp for this run
    for kpi in active_kpis:
        kpi_def_dict = { # Convert model to dict for calculate function
            "id": kpi.id, "name": kpi.name, "description": kpi.description,
            "calculation_logic": kpi.calculation_logic, "target": kpi.target_value,
            # Assuming calculation_type stored or inferred? Add to model if needed.
            "calculation_type": "pandas" if isinstance(data_context, pd.DataFrame) else "sql" # Basic inference
        }
        current_value = calculate_kpi_value(kpi_def_dict, data_context)

        previous_value = None
        trend = None # None if cannot determine
        delta = None

        # Fetch latest historical value for trend/delta calculation (#11 Logic)
        try:
            latest_historical = crud.get_latest_kpi_value(db, kpi_id=kpi.id)
            if latest_historical:
                 previous_value = latest_historical.value
                 logger.debug(f"KPI '{kpi.name}': Current={current_value}, Previous={previous_value}")
            else:
                 logger.debug(f"KPI '{kpi.name}': Current={current_value}, No previous value found.")

            # Calculate Trend and Delta
            if current_value is not None and previous_value is not None:
                delta = current_value - previous_value
                if delta > 0: trend = 1
                elif delta < 0: trend = -1
                else: trend = 0
            elif current_value is not None and previous_value is None:
                 trend = None # Cannot determine trend on first value
                 delta = None # No delta on first value

        except Exception as e:
             logger.error(f"Failed to retrieve/calculate trend for KPI '{kpi.name}': {e}", exc_info=True)
             # Proceed without trend/delta

        tracked_data[kpi.name] = {"value": current_value, "trend": trend, "delta": delta}

        # --- Save Current Value to History (#11 Logic) ---
        if current_value is not None:
            try:
                crud.save_kpi_value(db=db, kpi_id=kpi.id, value=current_value) # Timestamp defaults to now()
            except Exception as e:
                logger.error(f"Failed to save historical value for KPI '{kpi.name}': {e}", exc_info=True)
                # Continue tracking other KPIs even if saving fails

    logger.info(f"Finished tracking {len(tracked_data)} KPIs.")
    return tracked_data