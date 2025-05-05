# backend/database/connectors.py
import logging
from sqlalchemy import create_engine, inspect, text, exc as sqlalchemy_exc
from sqlalchemy.engine import Engine
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, OperationFailure
import pandas as pd
from typing import Optional, Dict, Any, Union
import urllib
import os # Import os for Oracle DSN construction if needed

# ... (settings import and logger setup) ...
logger = logging.getLogger(__name__)

# --- Relational Database Connection ---
def get_sql_engine(
    db_type: str,
    user: Optional[str],
    password: Optional[str],
    host: Optional[str], # Server name/IP/Hostname
    port: Optional[Union[str, int]], # Port number
    database: Optional[str], # Database name (SQL Server, PG, MySQL) OR SID/ServiceName (Oracle)
    service_name: Optional[str] = None, # Optional: Explicit Oracle Service Name
    driver: Optional[str] = None # Optional driver specification (mainly for MSSQL)
    ) -> Engine:
    """
    Creates SQLAlchemy engine for PostgreSQL, MySQL, SQLite, MS SQL Server, Oracle.

    Args:
        db_type: 'postgresql', 'mysql', 'sqlite', 'mssql', 'oracle'.
        user: DB username.
        password: DB password.
        host: DB host address/server name.
        port: DB port.
        database: DB name OR Oracle SID. Use 'service_name' arg for Oracle Service Name if needed.
        service_name: Explicit Oracle Service Name (takes precedence over 'database' if provided for Oracle connection string).
        driver: ODBC driver name for MSSQL.

    Returns: SQLAlchemy Engine.
    Raises: ValueError, ConnectionError.
    """
    db_url = None
    connection_args = {}
    default_ports = {"postgresql": 5432, "mysql": 3306, "mssql": 1433, "oracle": 1521}
    port = port or default_ports.get(db_type)
    safe_password = urllib.parse.quote_plus(password) if password else ""

    try:
        if db_type == "postgresql":
            # ... (as before) ...
            if not all([user, host, port, database]): raise ValueError("PostgreSQL: user, host, port, database required.")
            pwd_part = f":{safe_password}" if password else ""; db_url = f"postgresql+psycopg2://{user}{pwd_part}@{host}:{port}/{database}"
        elif db_type == "mysql":
             # ... (as before) ...
            if not all([user, host, port, database]): raise ValueError("MySQL: user, host, port, database required.")
            pwd_part = f":{safe_password}" if password else ""; db_url = f"mysql+mysqlconnector://{user}{pwd_part}@{host}:{port}/{database}"
        elif db_type == "sqlite":
             # ... (as before) ...
             if not database: raise ValueError("SQLite: database file path required.")
             db_url = f"sqlite:///{database}"
        elif db_type == "mssql":
            # ... (as before, ensure driver logic is robust) ...
            if not all([user, host, port, database]): raise ValueError("MSSQL: user, host, port, database required.")
            driver_to_use = driver or "ODBC Driver 17 for SQL Server"; logger.info(f"Using MSSQL driver: {driver_to_use}")
            pwd_part = f":{safe_password}" if password else ""; driver_enc = urllib.parse.quote_plus(driver_to_use)
            db_url = f"mssql+pyodbc://{user}{pwd_part}@{host}:{port}/{database}?driver={driver_enc}"
            # Handle Windows Auth if needed: db_url = f"mssql+pyodbc://@{host}:{port}/{database}?driver={driver_enc}&trusted_connection=yes"

        # --- ADD ORACLE SUPPORT ---
        elif db_type == "oracle":
            if not all([user, host, port]) or not (database or service_name):
                raise ValueError("Oracle: user, host, port, and EITHER database (SID) OR service_name required.")

            # Construct DSN (Data Source Name) for Oracle
            # Prioritize service_name if provided
            if service_name:
                # Thin mode DSN format (recommended if Instant Client not available/needed)
                # Example: host:port/service_name
                dsn = f"{host}:{port}/{service_name}"
                # Alternative using TNSNAMES style within DSN (less common directly):
                # dsn = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={host})(PORT={port}))(CONNECT_DATA=(SERVICE_NAME={service_name})))"
                logger.info(f"Using Oracle Service Name: {service_name}")
            elif database: # Assume 'database' arg is the SID
                 dsn = f"{host}:{port}/{database}" # SID format
                 logger.info(f"Using Oracle SID: {database}")
            else:
                 # This case shouldn't be reached due to initial check, but defensively:
                 raise ValueError("Oracle connection requires either SID (in database field) or Service Name.")

            # Construct the SQLAlchemy URL: oracle+oracledb://user:password@dsn
            pwd_part = f":{safe_password}" if password else ""
            # Add mode=oracledb.THIN? This should be the default for oracledb driver >= 1.0
            db_url = f"oracle+oracledb://{user}{pwd_part}@{dsn}"
            logger.info("Constructed Oracle DB URL (Thin mode assumed).")
            # For Thick mode (requires Instant Client configured in PATH/LD_LIBRARY_PATH):
            # db_url = f"oracle+oracledb://{user}{pwd_part}@{dsn}?mode=thick"

        else:
            raise ValueError(f"Unsupported SQL database type: {db_type}")

        logger.info(f"Attempting to create engine for {db_type} target: '{host or database}'")
        engine = create_engine(db_url, pool_pre_ping=True, connect_args=connection_args) # Add connect_args if needed

        # Test connection
        with engine.connect() as connection:
            logger.info(f"Successfully connected to {db_type} identified by '{host or database}'. Engine created.")
        return engine

    # --- Exception Handling ---
    except sqlalchemy_exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error connecting to {db_type} target '{host or database}': {e}", exc_info=True)
        guidance = f"Check credentials, connection details ({host}, {port}, {database or service_name}), network access."
        if db_type == "mssql": guidance += " Ensure ODBC driver is installed/configured."
        if db_type == "oracle": guidance += " Ensure Oracle DB is reachable, credentials are valid, and SID/Service Name is correct. For Thick mode, check Instant Client setup."
        raise ConnectionError(f"Failed to connect to {db_type}: {guidance}") from e
    except ValueError as e: logger.error(f"Config error for {db_type}: {e}"); raise
    except ImportError as e: logger.error(f"Import error (missing driver?) for {db_type}: {e}", exc_info=True); raise ConnectionError(f"Missing required library for {db_type}: {e}.") from e
    except Exception as e: logger.error(f"Unexpected error creating engine for {db_type}: {e}", exc_info=True); raise ConnectionError(f"Unexpected error connecting to {db_type}: {e}") from e


# --- MongoDB Connection (remains largely the same) ---
def get_mongo_client(uri: str) -> MongoClient:
    # ... (keep existing robust implementation) ...
    if not uri or not uri.startswith("mongodb"): raise ValueError("Invalid MongoDB URI.")
    try:
        timeout = 5000 # Default timeout
        logger.info(f"Attempting MongoDB connection (timeout: {timeout}ms)...")
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout, connectTimeoutMS=timeout, socketTimeoutMS=timeout, retryWrites=True, w='majority')
        client.admin.command('serverStatus'); logger.info("Successfully connected to MongoDB."); return client
    # ... (keep existing exception handling) ...
    except ConfigurationError as e: logger.error(f"MongoDB config error: {e}", exc_info=True); raise ValueError(f"Invalid MongoDB URI/config: {e}") from e
    except OperationFailure as e: logger.error(f"MongoDB operation failure (auth?): {e.details}", exc_info=True); raise ConnectionError(f"MongoDB connection failed (check credentials/permissions?): {e.details}") from e
    except ConnectionFailure as e: logger.error(f"MongoDB connection failure: {e}", exc_info=True); raise ConnectionError(f"Failed to connect to MongoDB server: {e}") from e
    except Exception as e: logger.error(f"Unexpected MongoDB connection error: {e}", exc_info=True); raise ConnectionError(f"Unexpected error connecting to MongoDB: {e}") from e

# --- Cloud Storage Connectors (remain the same) ---
# ... (get_s3_client, get_gcs_client) ...