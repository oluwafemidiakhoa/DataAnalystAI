# backend/analysis/nlq_processor.py
import logging
from typing import Optional, Dict, Any, Union, List # Added Union, List

# Assuming LLM utils are available
LLM_AVAILABLE = False
try:
    # Import both SQL and Pandas code generators
    from backend.llm.gemini_utils import generate_sql_from_nl_llm, generate_pandas_code_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("LLM utility functions not found for nlq_processor.", exc_info=True)
    # Define mocks
    def generate_sql_from_nl_llm(nlq, schema, dialect): logger.warning("MOCK SQL gen"); return f"-- Mock SQL ({dialect}) for: {nlq}\nSELECT 1;"
    def generate_pandas_code_llm(schema, nlq, task): logger.warning("MOCK Pandas gen"); return f"# Mock Pandas for: {nlq}\ndf.head(5)"

logger = logging.getLogger(__name__)


def process_nlq_to_sql(
    natural_language_query: str,
    connection_info: Dict[str, Any],
    schema_context: Optional[str] = None
    ) -> str:
    """Processes NLQ to generate SQL, inferring dialect."""
    # (Keep existing implementation from previous fix)
    if not natural_language_query or not natural_language_query.strip(): raise ValueError("NLQ empty.")
    if not connection_info or not isinstance(connection_info, dict): raise ValueError("connection_info dict required.")
    db_type = connection_info.get("type")
    supported_sql = ["postgresql", "mysql", "sqlite", "mssql", "oracle"]
    if not db_type or db_type not in supported_sql: raise ValueError(f"Unsupported DB type '{db_type}' for SQL.")
    schema_for_llm = schema_context or connection_info.get("schema") or "/* No schema provided. */"
    dialect_map = {"postgresql": "PostgreSQL", "mysql": "MySQL", "sqlite": "SQLite", "mssql": "Transact-SQL", "oracle": "Oracle"}
    dialect_hint = dialect_map.get(db_type, "standard SQL")
    logger.info(f"Generating SQL for NLQ: '{natural_language_query[:100]}...' dialect: '{dialect_hint}'")
    if not LLM_AVAILABLE: raise RuntimeError("LLM SQL generator unavailable.")
    try:
        sql_query = generate_sql_from_nl_llm(natural_language_query, str(schema_for_llm), dialect_hint)
        if not sql_query or not sql_query.strip() or sql_query.strip().startswith("Error:"): raise RuntimeError(sql_query or "AI failed to generate SQL.")
        logger.info("SQL query generated successfully."); return sql_query.strip()
    except Exception as e: logger.error(f"NLQ->SQL failed: {e}", exc_info=True); raise RuntimeError(f"NLQ->SQL processing failed: {e}") from e


# --- NEW: NLQ to Pandas Processor ---
def process_nlq_to_pandas(
    natural_language_query: str,
    schema_context: str # Schema context is crucial for Pandas generation
    ) -> str: # Returns generated code string (or error comment)
    """
    Processes a natural language query to generate Pandas code using an LLM.

    Args:
        natural_language_query: The user's question in plain English.
        schema_context: String describing the DataFrame schema (e.g., from df.info()).

    Returns:
        The generated Pandas code string, or a string starting with '# Error:' on failure.

    Raises:
        ValueError: If the natural language query or schema is empty.
        RuntimeError: If the LLM call fails unexpectedly.
    """
    if not natural_language_query or not natural_language_query.strip():
        raise ValueError("Natural language query cannot be empty.")
    if not schema_context or not schema_context.strip():
        logger.warning("Generating Pandas code without schema context. Results unreliable.")
        schema_context = "# Schema context unavailable"

    logger.info(f"Generating Pandas code for NLQ: '{natural_language_query[:100]}...'")

    if not LLM_AVAILABLE:
         logger.error("Cannot generate Pandas code: LLM function generate_pandas_code_llm is not available.")
         # Return error comment instead of raising, allows execute_query to handle it
         return "# Error: AI Code Generation service is unavailable."

    try:
        # Call the specific LLM function for generating Pandas code
        # It should return either valid code or an error comment
        pandas_code = generate_pandas_code_llm(
            schema=schema_context,
            nl_query=natural_language_query,
            task="Answer the user query or perform the requested operation on the DataFrame 'df'."
        )

        # No need to validate the code string itself here, executor will handle errors.
        # Just log success or failure based on whether it starts with the error marker.
        if pandas_code.strip().startswith("# Error:"):
             logger.warning(f"LLM failed to generate valid Pandas code: {pandas_code}")
        else:
             logger.info("Pandas code generated successfully by LLM.")

        return pandas_code # Return the code string or the error comment

    except Exception as e:
        logger.error(f"NLQ to Pandas processing failed: {e}", exc_info=True)
        # Return error comment on unexpected failure
        return f"# Error: Failed to process natural language query into Pandas code: {e}"