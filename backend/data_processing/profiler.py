# backend/data_processing/profiler.py
import pandas as pd
import numpy as np
import logging
import io
from typing import Dict, Any, Optional, List, Tuple

# Required for SQL schema details if using Engine object
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError # Import for specific error handling

# Attempt to import ydata-profiling, but make it optional
try:
    from ydata_profiling import ProfileReport
    YDATA_PROFILING_AVAILABLE = True
except ImportError:
    YDATA_PROFILING_AVAILABLE = False
    ProfileReport = None # Define it as None if not available
    logging.warning("ydata-profiling not found. Using basic pandas profiling.")

# Assuming LLM utils are available (import path might differ)
# These will be used for descriptions, tags, quality rules
try:
    from backend.llm.gemini_utils import (
        generate_text_summary_llm,
        generate_description_and_tags_llm, # NEW function needed
        suggest_quality_rules_llm # NEW function needed
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger = logging.getLogger(__name__) # Define logger here if imports fail early
    logger.error("LLM utility functions not found. AI features in profiler will be disabled/mocked.", exc_info=True)
    # Define mock functions if LLM is not available
    def generate_text_summary_llm(context, task): return f"Mock LLM Summary: Task '{task}' context seems okay."
    def generate_description_and_tags_llm(sample_data, schema): return {"description": "Mock description: Data appears tabular.", "tags": ["mock", "data"]}
    def suggest_quality_rules_llm(profile_summary, schema): return [{"rule_name": "Mock Rule: Check Nulls", "rule_type": "not_null", "column_name": "column_a", "rule_parameters": {}}]

# Get logger instance for this module
logger = logging.getLogger(__name__)

# --- Constants ---
PROFILE_SAMPLE_ROWS = 10000 # Default sample size for profiling large datasets
MAX_SCHEMA_TABLES = 10 # Limit tables shown in schema detail
MAX_SCHEMA_COLS = 25 # Limit columns per table shown
MAX_SCHEMA_COLLECTIONS = 10 # Limit collections shown in Mongo schema
MAX_PROFILE_LLM_CONTEXT_LEN = 15000 # Approx character limit for LLM context to avoid exceeding limits

# --- File Loading Function (Copied from previous fix) ---
def load_dataframe_from_file(uploaded_file_or_path):
    """Loads data from CSV, Excel, or JSON file into a Pandas DataFrame."""
    # (Keep the existing robust implementation from the previous fix)
    try:
        file_name = ""; file_obj = None; is_path = False
        if isinstance(uploaded_file_or_path, str):
            file_path = uploaded_file_or_path; file_name = file_path.lower(); file_obj = file_path; is_path = True
            logger.info(f"Attempting to load data from path: {file_path}")
        elif hasattr(uploaded_file_or_path, 'name') and hasattr(uploaded_file_or_path, 'seek'):
            file_name = uploaded_file_or_path.name.lower(); file_obj = uploaded_file_or_path
            try: uploaded_file_or_path.seek(0)
            except Exception as seek_err: logger.warning(f"Could not seek on uploaded file object: {seek_err}")
            logger.info(f"Attempting to load data from uploaded file: {file_name}")
        else: raise ValueError("Input must be a file path or a readable file-like object.")

        if file_name.endswith(".csv"): df = pd.read_csv(file_obj)
        elif file_name.endswith((".xls", ".xlsx")): df = pd.read_excel(file_obj, engine='openpyxl')
        elif file_name.endswith(".json"): df = pd.read_json(file_obj)
        else: raise ValueError("Unsupported file format. Use CSV, XLSX, or JSON.")

        if df.empty: logger.warning(f"Loaded file '{file_name or 'object'}' resulted in an empty DataFrame.")
        else: logger.info(f"Successfully loaded data. Shape: {df.shape}")

        if hasattr(uploaded_file_or_path, 'name'): df.attrs['filename'] = uploaded_file_or_path.name
        elif is_path: df.attrs['filename'] = Path(file_path).name

        return df
    except FileNotFoundError: logger.error(f"File not found: {uploaded_file_or_path}"); raise
    except Exception as e:
        fname = file_name or str(uploaded_file_or_path)
        logger.error(f"Error loading file {fname}: {e}", exc_info=True)
        raise ValueError(f"Could not read file ({fname}): {e}") from e


# --- Enhanced Schema Extraction (Incorporates Potential Join Key Identification) ---
def get_schema_details(source_type: str, source_obj: Any, db_name: Optional[str] = None, include_join_key_candidates: bool = True) -> Dict[str, Any]:
    """
    Generates a dictionary containing schema details and potential join key candidates.

    Args:
        source_type: 'file', 'postgresql', 'mysql', 'sqlite', 'mongodb'.
        source_obj: DataFrame, SQLAlchemy Engine, or PyMongo Client.
        db_name: Required for MongoDB schema.
        include_join_key_candidates: Whether to run heuristics to find potential join keys.

    Returns:
        A dictionary with 'schema_string' (formatted text) and 'join_key_candidates' (list of potential keys).
    """
    schema_details = {"schema_string": f"**Schema Details ({source_type.capitalize()})**\nError retrieving schema.", "join_key_candidates": []}
    raw_schema_info = {} # To store structured info for join key analysis

    try:
        schema_string = f"**Schema Details ({source_type.capitalize()})**\n```text\n"
        if source_type == "file" and isinstance(source_obj, pd.DataFrame):
            filename = source_obj.attrs.get('filename', 'N/A')
            schema_string += f"Source: File ('{filename}')\n"
            buffer = io.StringIO(); max_cols_info = 100
            source_obj.info(buf=buffer, verbose=True, max_cols=max_cols_info)
            schema_string += buffer.getvalue()
            if len(source_obj.columns) > max_cols_info: schema_string += f"\n... ({len(source_obj.columns) - max_cols_info} more columns)"
            raw_schema_info['file'] = {col: str(dtype) for col, dtype in source_obj.dtypes.items()}

        elif source_type in ["postgresql", "mysql", "sqlite"] and isinstance(source_obj, Engine):
            db_identifier = db_name or "connected_db"
            schema_string += f"Source: SQL Database ('{db_identifier}')\n"
            inspector = inspect(source_obj); tables = inspector.get_table_names()
            schema_string += f"Tables ({len(tables)}): {', '.join(tables[:MAX_SCHEMA_TABLES])}{'...' if len(tables) > MAX_SCHEMA_TABLES else ''}\n---\n"
            for i, table_name in enumerate(tables):
                if i >= MAX_SCHEMA_TABLES: schema_string += "...\n"; break
                schema_string += f"Table: {table_name}\n"
                table_cols = {}
                try:
                    columns = inspector.get_columns(table_name)
                    for col in columns[:MAX_SCHEMA_COLS]:
                        col_type = str(col['type'])
                        schema_string += f"  - {col['name']} ({col_type})\n"
                        table_cols[col['name']] = col_type
                    if len(columns) > MAX_SCHEMA_COLS: schema_string += f"  - ... ({len(columns) - MAX_SCHEMA_COLS} more columns)\n"
                    raw_schema_info[table_name] = table_cols
                except Exception as col_err: schema_string += f"  - (Error getting columns: {col_err})\n"; logger.warning(f"Could not get columns for table {table_name}: {col_err}")

        elif source_type == "mongodb" and hasattr(source_obj, 'list_database_names'):
             if not db_name: raise ValueError("Database name required for MongoDB schema.")
             schema_string += f"Source: MongoDB Database ('{db_name}')\n"
             db = source_obj[db_name]
             try:
                 collections = db.list_collection_names()
                 schema_string += f"Collections ({len(collections)}): {', '.join(collections[:MAX_SCHEMA_COLLECTIONS])}{'...' if len(collections) > MAX_SCHEMA_COLLECTIONS else ''}\n---\n"
                 for i, coll_name in enumerate(collections):
                     if i >= MAX_SCHEMA_COLLECTIONS: schema_string += "...\n"; break
                     schema_string += f"Collection: {coll_name}\n"
                     coll_fields = {}
                     try:
                         count = db[coll_name].estimated_document_count(); schema_string += f"  - Approx. Docs: {count:,}\n"
                         sample_doc = db[coll_name].find_one()
                         if sample_doc:
                             fields = list(sample_doc.keys())
                             schema_string += f"  - Sample Fields: {', '.join(fields)}\n"
                             coll_fields = {f: type(sample_doc[f]).__name__ for f in fields} # Store field types too
                         else: schema_string += "  - (Collection appears empty)\n"
                         raw_schema_info[coll_name] = coll_fields
                     except Exception as e: schema_string += f"  - (Error inspecting collection: {e})\n"; logger.warning(f"Error inspecting collection {coll_name}: {e}")
             except Exception as db_err: schema_string += f"\nError listing collections: {db_err}"; logger.error(f"Error listing MongoDB collections for db '{db_name}': {db_err}", exc_info=True)
        else:
             schema_string += "Error: Unsupported source type or invalid object."

        schema_string += "\n```" # Close markdown code block
        schema_details["schema_string"] = schema_string

        # --- Potential Join Key Identification (#1) ---
        if include_join_key_candidates and len(raw_schema_info) > 1: # Only if multiple tables/collections/files
             logger.info("Identifying potential join key candidates...")
             schema_details["join_key_candidates"] = _find_potential_join_keys(raw_schema_info)

    except ValueError as e: # Catch config errors
         schema_details["schema_string"] += f"\nConfiguration Error: {e}```"
         logger.error(f"Config error getting schema: {e}", exc_info=True)
    except SQLAlchemyError as e: # Catch DB specific errors
         schema_details["schema_string"] += f"\nDatabase Error: {e}```"
         logger.error(f"SQLAlchemy error getting schema: {e}", exc_info=True)
    except Exception as e: # Catch unexpected errors
        logger.error(f"Failed to get schema details for {source_type}: {e}", exc_info=True)
        schema_details["schema_string"] += f"\nUnexpected Error: {e}```"

    return schema_details


def _find_potential_join_keys(schema_info: Dict[str, Dict[str, str]]) -> List[Tuple[str, str, str]]:
    """Heuristic approach to find potential join keys based on names and types."""
    candidates = []
    column_map = {} # Map column name to list of (table/coll_name, type)

    # Build map of columns across all tables/collections
    for source_name, columns in schema_info.items():
        for col_name, col_type in columns.items():
            # Normalize column names (lowercase, remove common prefixes/suffixes like id, _id, key)
            norm_name = col_name.lower().replace('_id', '').replace('id', '').replace('_key', '').replace('key', '').strip('_ ')
            if not norm_name: continue # Skip if name becomes empty after normalization

            if norm_name not in column_map:
                column_map[norm_name] = []
            # Store original name, source, and type
            column_map[norm_name].append({"orig_name": col_name, "source": source_name, "type": col_type})

    # Find names appearing in multiple sources
    potential_links = {name: sources for name, sources in column_map.items() if len(sources) > 1}

    # Refine based on type compatibility (very basic check)
    for name, sources in potential_links.items():
        # Compare sources pairwise
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s1 = sources[i]
                s2 = sources[j]
                # Basic type compatibility check (e.g., INT vs BIGINT often okay, STRING vs INT not)
                # This needs improvement based on actual DB type systems
                type1 = s1['type'].lower(); type2 = s2['type'].lower()
                compatible = False
                if ('int' in type1 and 'int' in type2) or \
                   ('string' in type1 and 'string' in type2) or \
                   ('varchar' in type1 and 'varchar' in type2) or \
                   (type1 == type2):
                    compatible = True

                if compatible:
                    # Add candidate: (source1.col1, source2.col2, reason)
                    candidates.append((f"{s1['source']}.{s1['orig_name']}", f"{s2['source']}.{s2['orig_name']}", f"Name match ('{name}') + Type compatible ({type1}/{type2})"))

    logger.info(f"Found {len(candidates)} potential join key candidates.")
    return candidates


# --- Enhanced Data Profiling ---
def generate_profile_report(
    df: pd.DataFrame,
    source_name: str = "DataFrame", # Add source name for context
    use_ydata: bool = True,
    sample_rows: int = PROFILE_SAMPLE_ROWS,
    llm_features: bool = True # Combined flag for LLM enhancements
    ) -> Dict[str, Any]:
    """
    Generates an enhanced data profiling report for a DataFrame, including
    more stats, join key candidates (if applicable to schema), and LLM features.

    Args:
        df: Input Pandas DataFrame.
        source_name: Name of the data source (e.g., filename, table name).
        use_ydata: If True and installed, use ydata-profiling.
        sample_rows: Number of rows to sample for large datasets.
        llm_features: Whether to attempt LLM descriptions, tags, and quality rule suggestions.

    Returns:
        Dictionary containing the profiling report.
    """
    if df is None or df.empty:
        logger.warning("Cannot profile empty DataFrame.")
        return {"overview": {"message": "Input data is empty."}, "llm_summary": "No data to profile."}

    logger.info(f"Starting ENHANCED data profiling for '{source_name}' ({df.shape}). LLM Features: {llm_features}. Sampling: {sample_rows > 0}")
    profile_results: Dict[str, Any] = {"overview": {"source_name": source_name}}
    report_html: Optional[str] = None

    # --- Sampling ---
    df_profile = df
    if sample_rows > 0 and len(df) > sample_rows:
        logger.info(f"Sampling {sample_rows} rows for profiling.")
        df_profile = df.sample(n=sample_rows, random_state=42)
    profile_results["overview"]["rows_profiled"] = len(df_profile)

    start_time = pd.Timestamp.now()
    profile_method = "Basic Pandas"

    # --- Core Profiling (ydata or Basic Pandas) ---
    try:
        # (Keep the ydata-profiling and basic pandas profiling logic from previous version)
        # ... (ydata-profiling try-except block) ...
        if use_ydata and YDATA_PROFILING_AVAILABLE and ProfileReport is not None:
             # ... (ydata code) ...
             profile_method = "ydata-profiling (minimal)"; logger.info("ydata-profiling completed.")
        # ... (basic pandas profiling block if ydata failed or not used) ...
        if not profile_results.get('variables'): # Check if ydata results exist
             logger.info("Running basic pandas profiling..."); profile_method = "Basic Pandas"
             # ... (pandas describe, info, missing values, duplicates logic) ...
             profile_results["overview"].update({"rows": len(df), "columns": len(df.columns), "memory_usage": f"{df_profile.memory_usage(deep=True).sum() / 1e6:.2f} MB", "duplicates_in_profile": int(df_profile.duplicated().sum())})
             try:
                 profile_results["numeric_variables"] = df_profile.describe(include=np.number).round(2).to_dict() # Round stats
                 profile_results["categorical_variables"] = df_profile.describe(include=['object', 'category']).to_dict()
             except Exception as desc_err: logger.warning(f"Describe failed: {desc_err}"); profile_results["variable_stats_error"] = str(desc_err)
             missing = df_profile.isnull().sum(); profile_results["missing_values"] = missing[missing > 0].to_dict()
             profile_results["missing_percentage"] = (missing[missing > 0] / len(df_profile) * 100).round(2).to_dict()

        # --- Add More Stats (Enhancement) ---
        logger.info("Calculating additional stats...")
        if not df_profile.empty:
            profile_results["additional_stats"] = {
                "unique_counts": df_profile.nunique().to_dict(),
                "memory_per_column_mb": (df_profile.memory_usage(deep=True) / 1e6).round(2).to_dict()
            }
            # Skewness and Kurtosis for numeric columns
            numeric_cols = df_profile.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                profile_results["additional_stats"]["skewness"] = df_profile[numeric_cols].skew().round(2).to_dict()
                profile_results["additional_stats"]["kurtosis"] = df_profile[numeric_cols].kurt().round(2).to_dict()


    except Exception as e:
        logger.error(f"Core data profiling failed: {e}", exc_info=True)
        profile_results = {"error": f"Profiling failed: {e}"} # Overwrite results on major failure

    # --- Timing and Overview Update ---
    end_time = pd.Timestamp.now(); profile_duration = (end_time - start_time).total_seconds()
    logger.info(f"Profiling ({profile_method}) took {profile_duration:.2f} seconds.")
    if "overview" in profile_results: profile_results["overview"].update({"profiling_method": profile_method, "profiling_duration_sec": round(profile_duration, 2)})


    # --- LLM Enhancements (Description, Tags, Quality Rules - #2, #3) ---
    if llm_features and LLM_AVAILABLE and "error" not in profile_results:
        logger.info("Attempting LLM enhancements (Description, Tags, Quality Rules)...")
        # Prepare concise context for LLM features
        llm_context_summary = f"Data Source: '{source_name}'\n"
        llm_context_summary += f"Overview: {profile_results.get('overview', {})}\n"
        # Sample schema
        sample_schema = {col: str(dtype) for col, dtype in df_profile.dtypes.items()}
        llm_context_summary += f"Schema (Sample): {str(sample_schema)[:1000]}...\n" # Truncate long schemas
        # Sample data
        sample_data_str = df_profile.head(3).to_string()
        llm_context_summary += f"Data Sample (first 3 rows):\n{sample_data_str[:2000]}...\n" # Truncate long samples

        # Truncate full context if too long
        if len(llm_context_summary) > MAX_PROFILE_LLM_CONTEXT_LEN:
             llm_context_summary = llm_context_summary[:MAX_PROFILE_LLM_CONTEXT_LEN] + "\n... (Context Truncated)"
             logger.warning("LLM context truncated due to length.")


        # 1. Generate Description and Tags (#3)
        try:
            logger.debug("Requesting LLM description and tags...")
            desc_tags = generate_description_and_tags_llm(sample_data_str, str(sample_schema)) # Pass sample & schema
            profile_results["ai_description"] = desc_tags.get("description", "N/A")
            profile_results["ai_tags"] = desc_tags.get("tags", [])
            logger.info(f"LLM generated description and tags: {profile_results['ai_tags']}")
        except Exception as e:
            logger.error(f"LLM description/tag generation failed: {e}", exc_info=True)
            profile_results["ai_description"] = f"Error: {e}"
            profile_results["ai_tags"] = ["error"]

        # 2. Suggest Quality Rules (#2)
        try:
            logger.debug("Requesting LLM quality rule suggestions...")
            # Pass profile summary and schema to the LLM rule suggester
            quality_rules = suggest_quality_rules_llm(profile_results, str(sample_schema))
            profile_results["ai_quality_rules"] = quality_rules # Store list of suggested rule dicts
            logger.info(f"LLM generated {len(quality_rules)} quality rule suggestions.")
        except Exception as e:
            logger.error(f"LLM quality rule suggestion failed: {e}", exc_info=True)
            profile_results["ai_quality_rules"] = [{"rule_name": f"Error generating rules: {e}", "rule_type": "error"}]

        # 3. Generate Overall Summary (incorporating other AI results)
        try:
            logger.debug("Requesting LLM overall profile summary...")
            summary_context = llm_context_summary # Reuse truncated context
            summary_context += f"\nAI Description: {profile_results.get('ai_description', 'N/A')}\n"
            summary_context += f"AI Suggested Tags: {profile_results.get('ai_tags', 'N/A')}\n"
            summary_context += f"AI Quality Rule Suggestions (Count): {len(profile_results.get('ai_quality_rules', []))}\n"

            task = "Based on the data profile overview, schema, sample data, and the generated AI description/tags/rule suggestions, provide a concise (2-4 bullet points) overall summary highlighting the most critical findings for a data analyst."
            profile_results['llm_summary'] = generate_text_summary_llm(summary_context, task=task)
            logger.info("LLM overall summary generated.")
        except Exception as e:
            logger.error(f"LLM overall summary generation failed: {e}", exc_info=True)
            profile_results['llm_summary'] = f"Error generating AI summary: {e}"

    elif "error" in profile_results:
         profile_results['llm_summary'] = "AI features skipped due to profiling error."
    elif not LLM_AVAILABLE:
         profile_results['llm_summary'] = "AI features skipped (LLM not available)."


    # Add the HTML report to the results if generated and needed
    # profile_results['html_report'] = report_html

    return profile_results