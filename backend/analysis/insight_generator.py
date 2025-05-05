# backend/analysis/insight_generator.py
import pandas as pd
import numpy as np # Import numpy
import logging
import io
import time # For mock delays
from typing import Optional, Dict, Any, List, Tuple

# Assuming LLM utils are available
LLM_AVAILABLE = False
try:
    from backend.llm.gemini_utils import (
        generate_text_summary_llm,
        generate_rca_hints_llm, # NEW function needed
        identify_anomalies_llm # NEW function needed
        )
    LLM_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("LLM utility functions not found for insight generator.", exc_info=True)
    # Define mocks
    def generate_text_summary_llm(c, t): time.sleep(1); return f"Mock Summary for task '{t}'."
    def generate_rca_hints_llm(c, t): time.sleep(1); return ["Mock RCA Hint: Check related table X.", "Mock RCA Hint: Filter by date Y."]
    def identify_anomalies_llm(c, t): time.sleep(1); return [{"anomaly_type": "Outlier", "details": "High value detected in col Z", "severity": "Medium"}]

logger = logging.getLogger(__name__)

# --- Helper Function for Context Preparation ---
def _prepare_llm_context(
    df: pd.DataFrame,
    query_context: Optional[str] = None,
    analysis_goal: Optional[str] = None,
    max_len: int = 15000 # Max characters for context
    ) -> str:
    """Prepares a concise context string for LLM prompts."""
    if df is None or df.empty: return "Context Error: No DataFrame provided."
    try:
        data_sample = df.head(5).to_markdown(index=False, numalign="left", stralign="left")
        schema_buffer = io.StringIO()
        df.info(verbose=True, buf=schema_buffer, max_cols=100) # Limit columns in info
        schema = schema_buffer.getvalue()
        num_rows, num_cols = df.shape

        context = f"Analysis Context:\n"
        if query_context: context += f"- User Query: {query_context}\n"
        if analysis_goal: context += f"- Goal: {analysis_goal}\n"
        context += f"- Data Shape: {num_rows} rows, {num_cols} columns\n"
        context += f"- Schema Summary:\n{schema}\n"
        context += f"- Data Sample (first 5 rows):\n{data_sample}\n"

        if len(context) > max_len:
            logger.warning(f"LLM context length ({len(context)}) exceeded limit ({max_len}). Truncating.")
            context = context[:max_len] + "\n... (Context Truncated)"
        return context
    except Exception as e:
        logger.error(f"Error preparing LLM context: {e}", exc_info=True)
        return f"Context Preparation Error: {e}"


# --- Standard Deep Insights Generation ---
def generate_deep_insights(
    df: pd.DataFrame,
    query_context: Optional[str] = None,
    analysis_goal: Optional[str] = None
    ) -> str:
    """
    Generates general analysis insights from a DataFrame using an LLM.
    Focuses on summarizing findings, trends, patterns based on provided data/query.
    """
    if df is None or df.empty:
        logger.warning("Cannot generate insights from empty DataFrame.")
        return "The analysis returned no data, so no insights can be generated."

    logger.info(f"Generating deep insights for DataFrame shape {df.shape}. Context: {query_context or 'N/A'}")
    prompt_context = _prepare_llm_context(df, query_context, analysis_goal)
    if "Error:" in prompt_context: return prompt_context # Return error if context prep failed

    task_description = ("Analyze the provided data sample, schema, and context. "
                        "Identify key findings, significant trends, interesting patterns, "
                        "or potential outliers/anomalies relevant to the user query or goal (if provided). "
                        "Provide a concise summary as markdown bullet points (3-5 points).")

    try:
        if not LLM_AVAILABLE: raise RuntimeError("LLM utility functions are not available.")
        insights = generate_text_summary_llm(prompt_context, task=task_description)

        if not insights or insights.lower().startswith("error:"):
             logger.warning(f"LLM failed to generate valid insights. Response: {insights}")
             error_message = insights if insights and insights.lower().startswith("error:") else "AI failed to generate insights for this data."
             raise RuntimeError(error_message)

        logger.info("Deep insights generated successfully.")
        return insights.strip()

    except Exception as e:
        logger.error(f"Deep insight generation failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate deep insights: {e}") from e


# --- Root Cause Analysis Assistance (#8) ---
def generate_root_cause_hints(
    df: pd.DataFrame,
    issue_description: str, # e.g., "Sales dropped in West region", "Conversion rate declined"
    conversation_history: Optional[List[Dict]] = None # For multi-turn
    ) -> List[str]:
    """
    Uses LLM to suggest next steps or questions for root cause analysis.
    """
    if df is None or df.empty:
        logger.warning("Cannot generate RCA hints from empty DataFrame.")
        return ["Error: No data available for analysis."]

    logger.info(f"Generating RCA hints for issue: '{issue_description}'")
    prompt_context = _prepare_llm_context(df, query_context=issue_description)
    if "Error:" in prompt_context: return [prompt_context]

    # Add conversation history if available
    if conversation_history:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]]) # Last 5 turns
        prompt_context += f"\n\nPrevious Conversation Turn(s):\n{history_str}\n"

    task_description = (f"The user is investigating an issue: '{issue_description}'. "
                        "Based on the provided data context and conversation history (if any), "
                        "suggest 2-3 specific, actionable next steps or drill-down questions "
                        "the user should investigate to find the root cause. "
                        "Focus on potential drivers visible or inferable from the data schema/sample. "
                        "Return ONLY a list of strings, where each string is a distinct suggestion."
                       )
                       # Example output format hint (optional for LLM): ["Check inventory levels for related products.", "Analyze customer segment performance in that region.", "Compare marketing spend period-over-period for that area."]

    try:
        if not LLM_AVAILABLE: raise RuntimeError("LLM utility functions are not available.")
        # This LLM function needs to be prompted to return a list of strings
        hints = generate_rca_hints_llm(prompt_context, task=task_description) # Assumes this returns List[str]

        if not hints or not isinstance(hints, list):
            logger.warning(f"LLM did not return a valid list of RCA hints. Response: {hints}")
            raise RuntimeError("AI failed to generate valid suggestions for next steps.")

        logger.info(f"Generated {len(hints)} RCA hints.")
        return hints

    except Exception as e:
        logger.error(f"RCA hint generation failed: {e}", exc_info=True)
        # Return error as a hint
        return [f"Error generating investigation hints: {e}"]


# --- Proactive Anomaly Detection (#10) ---
def find_proactive_anomalies(
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame] = None, # Optional: Compare against previous snapshot
    time_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
    """
    Analyzes data (potentially comparing to previous data) to find anomalies or significant changes.
    Can use statistical methods and/or LLM reasoning.

    Args:
        current_df: The current DataFrame snapshot.
        previous_df: An optional previous snapshot for comparison.
        time_column: Optional name of the time column for trend analysis.

    Returns:
        A list of dictionaries, each describing a detected anomaly/insight.
        e.g., {'type': 'Sudden Spike', 'details': 'Sales increased 50% in week 42', 'severity': 'High'}
    """
    if current_df is None or current_df.empty:
        logger.info("Skipping proactive anomaly detection: No current data.")
        return []

    logger.info(f"Running proactive anomaly detection on data shape {current_df.shape}...")
    anomalies = []

    # --- Statistical Checks (Example: Z-score outlier detection) ---
    numeric_cols = current_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if current_df[col].count() < 10: continue # Need enough data points
        mean = current_df[col].mean()
        std = current_df[col].std()
        if std is None or std == 0: continue # Avoid division by zero or constant columns

        z_scores = np.abs((current_df[col] - mean) / std)
        outlier_threshold = 3.0 # Common threshold
        outliers = current_df[z_scores > outlier_threshold]

        if not outliers.empty:
            details = f"{len(outliers)} potential outliers detected in '{col}' (Z-score > {outlier_threshold}). Max value: {current_df[col].max()}, Mean: {mean:.2f}."
            logger.warning(f"Potential statistical outliers found in {col}")
            anomalies.append({"type": "Statistical Outlier", "details": details, "severity": "Medium"})


    # --- Trend Change Detection (Example: Requires previous data or time column) ---
    if time_column and time_column in current_df.columns and pd.api.types.is_datetime64_any_dtype(current_df[time_column]):
        # Example: Check rolling average change for first numeric column
        target_col = numeric_cols[0] if not numeric_cols.empty else None
        if target_col:
            df_sorted = current_df.sort_values(by=time_column)
            rolling_mean = df_sorted[target_col].rolling(window=5).mean() # 5-period rolling mean
            if len(rolling_mean) > 10: # Need enough data
                 # Check recent change vs historical average change
                 recent_change = rolling_mean.iloc[-1] - rolling_mean.iloc[-2] if len(rolling_mean) >= 2 else 0
                 historical_change_std = rolling_mean.diff().std()
                 if historical_change_std > 0 and abs(recent_change) > 2.5 * historical_change_std: # If recent change is > 2.5 std devs
                     direction = "increase" if recent_change > 0 else "decrease"
                     details = f"Significant recent {direction} detected in rolling average for '{target_col}' (Change: {recent_change:.2f}, StdDev: {historical_change_std:.2f})."
                     logger.warning(details)
                     anomalies.append({"type": "Trend Change", "details": details, "severity": "Medium"})


    # --- LLM-Based Anomaly/Insight Identification (Placeholder) ---
    if LLM_AVAILABLE:
        logger.info("Attempting LLM-based proactive analysis...")
        try:
            # Prepare context - might include comparison stats if previous_df exists
            context = _prepare_llm_context(current_df, analysis_goal="Identify significant anomalies or noteworthy patterns compared to typical data.")
            # Add comparison context if available
            # if previous_df is not None: context += "\n\nComparison to Previous Data: [Add summary stats diff]"

            task = ("Analyze the provided data context. Identify any significant anomalies, "
                    "sudden changes, data quality warnings, or interesting patterns that might warrant user attention. "
                    "Focus on actionable or surprising findings. Return a list of findings as dictionaries: "
                    "[{'type': 'Anomaly Type', 'details': 'Specific finding description.', 'severity': 'Low/Medium/High'}]"
                   )

            llm_anomalies = identify_anomalies_llm(context, task) # Assumes this returns List[Dict]

            # Validate and add LLM findings
            valid_llm_anomalies = [a for a in llm_anomalies if isinstance(a, dict) and 'type' in a and 'details' in a]
            if valid_llm_anomalies:
                 logger.info(f"LLM identified {len(valid_llm_anomalies)} potential proactive insights/anomalies.")
                 anomalies.extend(valid_llm_anomalies)

        except Exception as e:
             logger.error(f"LLM proactive anomaly detection failed: {e}", exc_info=True)
             anomalies.append({"type": "LLM Error", "details": f"AI analysis failed: {e}", "severity": "Low"})


    logger.info(f"Proactive analysis found {len(anomalies)} potential items.")
    return anomalies