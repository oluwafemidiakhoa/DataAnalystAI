# backend/analysis/query_executor.py
import pandas as pd
import numpy as np # Import numpy for eval context
import logging
from sqlalchemy.engine import Engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError, ServerSelectionTimeoutError
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MONGO_LIMIT = 100000

# --- Placeholder for Safe Execution ---
# Ideally, import and use this:
# from backend.sandbox.executor import safe_execute_pandas_code
USE_SAFE_EXECUTION = False # Set to True when sandbox is ready

def execute_query(
    query_type: str,
    query: Union[str, Dict[str, Any], List[Dict[str, Any]]], # SQL str, Mongo filter/pipeline, Pandas code str
    connection_obj: Union[Engine, MongoClient, pd.DataFrame],
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None
    ) -> pd.DataFrame:
    """
    Executes a query based on type and connection object.
    Handles SQL via SQLAlchemy Engine.
    Handles basic Mongo find/aggregate.
    Handles Pandas via executing generated code string (UNSAFE MOCK/SANDBOX NEEDED).
    """
    logger.info(f"Executing query of type '{query_type}'...")
    start_time = pd.Timestamp.now()
    df_result = pd.DataFrame()

    try:
        # --- SQL Execution ---
        if query_type == "sql":
            # (SQL logic remains the same as previous version)
            if not isinstance(connection_obj, Engine): raise ValueError(f"SQL needs Engine, got {type(connection_obj)}");
            if not isinstance(query, str) or not query.strip(): raise ValueError("SQL query empty.");
            sql_query = query.strip(); dialect = connection_obj.dialect.name; logger.debug(f"SQL ({dialect}): {sql_query[:300]}...")
            with connection_obj.connect() as conn: result_proxy = conn.execute(text(sql_query))
            if result_proxy.returns_rows: df_result = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys()); logger.info(f"SQL success. Rows: {len(df_result)}")
            else: rows_affected = result_proxy.rowcount; df_result = pd.DataFrame({'rows_affected': [rows_affected]}); logger.info(f"SQL action success. Rows affected: {rows_affected}")

        # --- MongoDB Execution ---
        elif query_type == "mongodb":
            # (Mongo logic remains the same as previous version)
            if not isinstance(connection_obj, MongoClient): raise ValueError(f"Mongo needs MongoClient, got {type(connection_obj)}");
            if not db_name or not collection_name: raise ValueError("Mongo needs db/collection name.")
            db=connection_obj[db_name]; collection=db[collection_name]
            if isinstance(query, dict): logger.debug(f"Mongo find on {db_name}/{collection_name}, filter: {query}"); cursor=collection.find(query).limit(DEFAULT_MONGO_LIMIT); df_result=pd.DataFrame(list(cursor)); logger.info(f"Mongo find success. Rows: {len(df_result)}")
            elif isinstance(query, list): logger.debug(f"Mongo aggregate on {db_name}/{collection_name}, pipeline: {query}"); cursor=collection.aggregate(query); df_result=pd.DataFrame(list(cursor)); logger.info(f"Mongo aggregate success. Rows: {len(df_result)}")
            else: raise ValueError("Mongo query must be dict (filter) or list (pipeline).")
            if '_id' in df_result.columns: df_result = df_result.drop(columns=['_id'])

        # --- Pandas Execution (Using Generated Code) ---
        elif query_type == "pandas":
             if not isinstance(connection_obj, pd.DataFrame):
                 raise ValueError(f"Pandas execution requires input DataFrame. Got: {type(connection_obj)}")
             if not isinstance(query, str) or not query.strip():
                  raise ValueError("Pandas execution requires the generated code string.")

             pandas_code = query.strip()
             logger.info("Attempting to execute generated Pandas code...")
             logger.debug(f"Pandas Code:\n{pandas_code}")

             if pandas_code.startswith("# Error:"):
                  logger.error(f"Cannot execute Pandas code due to generation error: {pandas_code}")
                  raise ValueError(f"AI failed to generate valid Pandas code: {pandas_code.split(':',1)[-1].strip()}")

             df = connection_obj.copy() # Work on a copy

             # --- Choose Execution Method ---
             if USE_SAFE_EXECUTION:
                  # ** IDEAL: Call the safe sandbox **
                  # logger.info("Using SAFE SANDBOX execution...")
                  # try:
                  #      df_result = safe_execute_pandas_code(code=pandas_code, input_df=df)
                  #      logger.info("Safe Pandas execution successful.")
                  # except Exception as sandbox_err:
                  #      logger.error(f"Safe sandbox execution failed: {sandbox_err}", exc_info=True)
                  #      raise RuntimeError(f"Error executing generated Pandas code in sandbox: {sandbox_err}") from sandbox_err
                  logger.error("Safe execution flag is True, but sandbox function call is commented out!")
                  raise NotImplementedError("Safe Pandas code execution sandbox is not implemented.")

             else:
                  # ** WARNING: UNSAFE EXECUTION USING exec() - DEVELOPMENT/TESTING ONLY **
                  logger.warning("Executing Pandas code using UNSAFE 'exec'. Replace with sandbox for production!")
                  local_vars = {'df': df, 'pd': pd, 'np': np} # Provide context
                  global_vars = {'pd': pd, 'np': np} # Allow imports if needed by generated code? Risky.
                  try:
                      exec(pandas_code, global_vars, local_vars) # Execute the generated code
                      # Assume the result is either modifying 'df' in place or assigned to 'result_df'
                      if 'result_df' in local_vars and isinstance(local_vars['result_df'], pd.DataFrame):
                           df_result = local_vars['result_df']
                           logger.info("Pandas code executed successfully (result in 'result_df').")
                      elif 'df' in local_vars and isinstance(local_vars['df'], pd.DataFrame):
                           df_result = local_vars['df'] # Get potentially modified df
                           logger.info("Pandas code executed successfully (result in 'df').")
                      else:
                           # Maybe the code printed something? Or returned a scalar?
                           logger.warning("Pandas code executed, but no DataFrame named 'df' or 'result_df' found in local scope.")
                           # Return the modified df even if no explicit result_df, as inplace ops might have happened
                           df_result = local_vars.get('df', pd.DataFrame()) # Default to empty if df was somehow removed
                  except Exception as exec_err:
                       logger.error(f"Error executing generated Pandas code: {exec_err}", exc_info=True)
                       raise RuntimeError(f"Error executing generated Pandas code: {exec_err}") from exec_err

             logger.info(f"Pandas code execution result shape: {df_result.shape}")


        else:
            raise ValueError(f"Unsupported query type: '{query_type}'")

        end_time = pd.Timestamp.now(); duration = (end_time - start_time).total_seconds()
        logger.info(f"Query execution type '{query_type}' completed in {duration:.3f} seconds.")
        return df_result

    # (Exception handling remains the same)
    except (OperationalError) as conn_err: logger.error(f"DB conn error exec {query_type}: {conn_err}", exc_info=True); raise ConnectionError(f"DB conn issue: {conn_err}") from conn_err
    except (SQLAlchemyError) as sql_err: logger.error(f"SQL DB error exec: {sql_err}", exc_info=True); raise RuntimeError(f"SQL exec error: {sql_err}") from sql_err
    except (OperationFailure, PyMongoError, ServerSelectionTimeoutError) as mongo_err: logger.error(f"Mongo error exec: {mongo_err}", exc_info=True); raise RuntimeError(f"Mongo exec error: {mongo_err}") from mongo_err
    except (KeyError, ValueError, TypeError) as config_err: logger.error(f"Config/param error exec: {config_err}", exc_info=True); raise ValueError(f"Invalid input for exec: {config_err}") from config_err
    except Exception as e: logger.error(f"Unexpected error exec {query_type}: {e}", exc_info=True); raise RuntimeError(f"Unexpected error during {query_type} exec: {e}") from e