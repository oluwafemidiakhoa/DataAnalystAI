# backend/sandbox/executor.py
# Secure execution environment for untrusted code (e.g., from LLM)
# WARNING: Implementing a truly secure sandbox is complex and requires careful design.
# This file provides placeholders and examples, NOT a production-ready secure sandbox.

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np # Often needed in pandas code

# --- Option 1: Using RestrictedPython (Limited) ---
RESTRICTEDPYTHON_AVAILABLE = False
try:
    # Note: RestrictedPython might need careful compilation setup
    from RestrictedPython import compile_restricted, safe_builtins, limited_builtins
    from RestrictedPython.Guards import guarded_iter_unpack_sequence, guarded_unpack_sequence
    RESTRICTEDPYTHON_AVAILABLE = True
except ImportError:
    logging.warning("RestrictedPython not installed. `pip install RestrictedPython`. Cannot use restricted execution.")

# --- Option 2: Using Docker (More Secure, More Complex Infrastructure) ---
DOCKER_AVAILABLE = False
try:
    import docker
    # Check if docker daemon is running
    try:
         docker_client = docker.from_env()
         docker_client.ping() # Verify connection
         DOCKER_AVAILABLE = True
         logging.info("Docker client connected successfully.")
    except Exception as docker_err:
         logging.warning(f"Docker library installed, but failed to connect to Docker daemon: {docker_err}. Docker sandbox disabled.")
         DOCKER_AVAILABLE = False
except ImportError:
    logging.warning("Docker library not installed (`pip install docker`). Docker sandbox disabled.")


logger = logging.getLogger(__name__)

# --- RestrictedPython Setup (Example) ---
# Define allowed builtins and guarded functions for RestrictedPython
# This needs careful tuning based on allowed Pandas/Numpy operations
RESTRICTED_GLOBALS = {
    '__builtins__': limited_builtins, # Start with limited builtins
    '_getitem_': lambda obj, key: obj[key], # Allow basic item access (use cautiously)
    '_unpack_sequence_': guarded_unpack_sequence,
    '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
    # Add specific safe modules/functions needed by pandas code
    'pd': pd,
    'np': np,
    # Allow specific safe functions explicitly? e.g.:
    # 'safe_mean': pd.Series.mean, # Example - aliasing might be needed
}
# Add more allowed builtins if necessary:
# RESTRICTED_GLOBALS['__builtins__'].update(safe_builtins) # Add more safe builtins


def execute_code_restricted_python(code: str, execution_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Python code within a RestrictedPython environment.
    WARNING: Limited protection, relies heavily on RestrictedPython's capabilities and configuration.
    """
    if not RESTRICTEDPYTHON_AVAILABLE:
        raise RuntimeError("RestrictedPython library is not available.")

    logger.info("Executing code using RestrictedPython...")
    restricted_globals = RESTRICTED_GLOBALS.copy()
    # Add variables from execution_context (like 'df') to the globals
    restricted_globals.update(execution_context)

    try:
        byte_code = compile_restricted(code, filename='<inline code>', mode='exec')
        # Execute in the restricted environment
        exec(byte_code, restricted_globals, restricted_globals) # Use same dict for locals/globals
        logger.info("RestrictedPython execution completed.")
        # Return the modified context (which should include the modified 'df')
        return restricted_globals
    except SyntaxError as e:
         logger.error(f"Syntax error in code for RestrictedPython: {e}", exc_info=True)
         raise ValueError(f"Syntax Error in generated code: {e}") from e
    except Exception as e:
         # Catch potential errors during restricted execution (e.g., disallowed operations)
         logger.error(f"Error during RestrictedPython execution: {e}", exc_info=True)
         raise RuntimeError(f"Execution error in restricted environment: {e}") from e


def execute_code_docker(code: str, input_data: pd.DataFrame, timeout: int = 30) -> pd.DataFrame:
    """
    Executes Python/Pandas code inside a dedicated Docker container.
    Requires Docker daemon running and appropriate image. More secure but complex setup.
    """
    if not DOCKER_AVAILABLE:
        raise RuntimeError("Docker is not available or client failed to connect.")

    # --- Implementation Details ---
    # 1. Serialize input_data (e.g., to CSV/Parquet in memory or temp file)
    # 2. Choose/Build a Docker image with Python, Pandas, Numpy installed.
    # 3. Create the code to run inside the container:
    #    - Load serialized data into 'df'.
    #    - Execute the user's `code`.
    #    - Serialize the resulting DataFrame ('df' after execution).
    #    - Print the serialized result to stdout.
    # 4. Run the container using docker-py:
    #    - Mount/pass the input data (if using temp files).
    #    # - Pass the code script to execute.
    #    - Set resource limits (memory, CPU).
    #    - Set execution timeout.
    #    - Capture stdout.
    # 5. Deserialize the result from stdout back into a DataFrame.
    # 6. Handle errors (timeouts, container crashes, code errors within container).
    # 7. Clean up container/temp files.

    logger.warning("Docker execution logic is complex and NOT fully implemented in this placeholder.")
    # Placeholder implementation:
    logger.info(f"Simulating Docker execution for code:\n{code[:100]}...")
    time.sleep(1) # Simulate runtime
    # In a real scenario, this would involve the steps above.
    # For now, just return the input df as a mock placeholder.
    if "'new_col'" in code: # Basic mock change
        df_result = input_data.copy()
        df_result['new_col_docker_mock'] = 1
        return df_result
    return input_data # Return unchanged df

    # Example using docker-py (HIGHLY SIMPLIFIED):
    # try:
    #     client = docker.from_env()
    #     # Assume input_df_bytes is serialized df, code_script_str contains loading, exec, saving code
    #     container = client.containers.run(
    #         image="your_python_pandas_image:latest", # Pre-built image
    #         command=["python", "-c", code_script_str],
    #         # volumes={ ... mount volumes if needed ... }
    #         # environment={ ... env vars ... }
    #         # mem_limit="512m", # Example limit
    #         # cpu_quota=50000, # Example limit
    #         detach=True,
    #     )
    #     # Wait for container with timeout
    #     exit_code = container.wait(timeout=timeout)
    #     logs = container.logs(stdout=True, stderr=True).decode('utf-8')
    #     container.remove() # Clean up

    #     if exit_code['StatusCode'] != 0:
    #         raise RuntimeError(f"Container execution failed with code {exit_code['StatusCode']}. Logs:\n{logs}")

    #     # Deserialize logs (stdout) back to DataFrame
    #     result_df = ... # pd.read_csv(io.StringIO(logs)) or similar
    #     return result_df

    # except docker.errors.ContainerError as e: ...
    # except docker.errors.ImageNotFound as e: ...
    # except Exception as e: ...


# --- Main Safe Execution Function ---
# This function decides which method to use
def safe_execute_pandas_code(
    code: str,
    input_df: pd.DataFrame,
    preferred_method: str = 'docker' # 'docker', 'restricted_python'
    ) -> pd.DataFrame:
    """
    Executes Pandas code safely using the preferred available method.

    Args:
        code: The Python/Pandas code string to execute.
        input_df: The input DataFrame (will be named 'df' in execution context).
        preferred_method: 'docker' or 'restricted_python'.

    Returns:
        The resulting DataFrame after code execution.

    Raises:
        RuntimeError: If no safe execution method is available or execution fails.
        ValueError: For syntax errors or execution errors within the sandbox.
    """
    logger.info(f"Attempting safe execution using preferred method: {preferred_method}")
    execution_context = {'df': input_df.copy(), 'pd': pd, 'np': np} # Start with a copy

    if preferred_method == 'docker' and DOCKER_AVAILABLE:
        try:
            return execute_code_docker(code, input_df) # Pass original df
        except Exception as e:
            logger.error(f"Docker execution failed: {e}. Falling back if possible.")
            # Fallback to RestrictedPython or raise error? For now, raise.
            raise RuntimeError(f"Docker execution failed: {e}") from e
            # Or: if RESTRICTEDPYTHON_AVAILABLE: preferred_method = 'restricted_python' else: raise ...

    if preferred_method == 'restricted_python' and RESTRICTEDPYTHON_AVAILABLE:
         try:
             result_context = execute_code_restricted_python(code, execution_context)
             if 'df' not in result_context or not isinstance(result_context['df'], pd.DataFrame):
                 raise ValueError("Restricted execution did not return a valid DataFrame named 'df'.")
             return result_context['df']
         except Exception as e:
            logger.error(f"RestrictedPython execution failed: {e}.")
            raise # Re-raise the specific error (RuntimeError or ValueError)

    # If preferred method failed or wasn't available
    logger.error(f"No suitable safe execution method ('{preferred_method}') available or configured correctly.")
    raise RuntimeError("No safe code execution method is available/functional.")