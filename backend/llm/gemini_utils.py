# backend/llm/gemini_utils.py
import google.generativeai as genai
import logging
import json
import time # For retry delays
import pandas as pd # Needed for type hint in interpret_clusters_llm
import numpy as np # Often needed in generated pandas code context
import os # Import os
from typing import Optional, Dict, Any, List, Union
from pydantic import SecretStr # Import if used in config

# --- Settings Import and Fallback ---
SETTINGS_AVAILABLE = False
try:
    from backend.core.config import settings
    SETTINGS_AVAILABLE = True
    DEFAULT_MODEL_NAME = settings.gemini_default_model
    ADVANCED_MODEL_NAME = settings.gemini_advanced_model
    API_KEY_CONFIGURED = bool(settings and settings.gemini_api_key and settings.gemini_api_key.get_secret_value())
except (ImportError, AttributeError, RuntimeError) as settings_e:
    settings = None
    SETTINGS_AVAILABLE = False
    DEFAULT_MODEL_NAME = "gemini-1.5-flash"
    ADVANCED_MODEL_NAME = "gemini-1.5-pro"
    API_KEY_CONFIGURED = False
    # Configure basic logging IF logger wasn't already configured elsewhere (e.g., core.logging_config)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO)
    logging.warning(f"LLM Utils: Could not import/use settings: {settings_e}. Using defaults/placeholders.")

logger = logging.getLogger(__name__) # Get logger for this module

# --- Configure Gemini API Client ---
GEMINI_API_KEY_VALUE = None
# Try loading from settings first
if SETTINGS_AVAILABLE and settings and settings.gemini_api_key:
    try:
        GEMINI_API_KEY_VALUE = settings.gemini_api_key.get_secret_value()
        if GEMINI_API_KEY_VALUE:
            genai.configure(api_key=GEMINI_API_KEY_VALUE)
            API_KEY_CONFIGURED = True
            logger.info(f"Gemini API configured successfully via settings. Default: {DEFAULT_MODEL_NAME}, Advanced: {ADVANCED_MODEL_NAME}")
        else: logger.error("Gemini API key found in settings but has no value.")
    except Exception as e: logger.error(f"Error configuring Gemini API from settings: {e}", exc_info=True); API_KEY_CONFIGURED = False

# Fallback check for direct environment variable
if not API_KEY_CONFIGURED and 'GEMINI_API_KEY' in os.environ:
     try:
        key_from_env = os.environ['GEMINI_API_KEY']
        if key_from_env:
            genai.configure(api_key=key_from_env)
            API_KEY_CONFIGURED = True
            GEMINI_API_KEY_VALUE = key_from_env
            logger.info("Gemini API configured successfully using direct environment variable.")
        else: logger.error("GEMINI_API_KEY environment variable found but is empty.")
     except Exception as e: logger.error(f"Error configuring Gemini API using direct environment variable: {e}", exc_info=True); API_KEY_CONFIGURED = False

if not API_KEY_CONFIGURED:
     logger.critical("GEMINI API KEY IS NOT CONFIGURED. LLM features requiring API calls will fail.")


# --- Safety Settings ---
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Core Generation Function (Helper) ---
def _call_gemini_api(
    model_name: str,
    prompt: str,
    temperature: float = 0.5,
    max_output_tokens: Optional[int] = 8192,
    expect_json: bool = False,
    max_retries: int = 2
    ) -> Union[str, Dict, List]:
    """
    Internal helper to call Gemini API with configuration, error handling, retries, and JSON parsing.
    """
    if not API_KEY_CONFIGURED:
        raise RuntimeError("Gemini API key is not configured or configuration failed.")

    logger.debug(f"Calling Gemini model '{model_name}'. Expect JSON: {expect_json}. Prompt (start): {prompt[:150]}...")

    try:
        model = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
    except Exception as model_err:
         logger.error(f"Failed to instantiate Gemini model '{model_name}': {model_err}", exc_info=True)
         raise RuntimeError(f"Could not load Gemini model: {model_err}") from model_err

    gen_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if expect_json else "text/plain"
    )

    attempts = 0
    last_exception = None

    while attempts <= max_retries:
        try:
            # --- Make the API Call ---
            response = model.generate_content(prompt, generation_config=gen_config)

            # --- Handle Response ---
            was_blocked = False; block_reason = "Unknown"; safety_ratings_str = "N/A"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 was_blocked = True; block_reason = response.prompt_feedback.block_reason.name; safety_ratings_str = str(response.prompt_feedback.safety_ratings)
            elif not response.candidates: # Check if candidates list is empty
                 was_blocked = True; block_reason = "NO_CANDIDATES"; safety_ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
            elif response.candidates[0].finish_reason.name == "SAFETY":
                 was_blocked = True; block_reason = "SAFETY"; safety_ratings_str = str(response.candidates[0].safety_ratings)

            if was_blocked:
                 logger.error(f"Gemini response blocked (Attempt {attempts + 1}). Reason: {block_reason}. Safety: {safety_ratings_str}")
                 raise RuntimeError(f"Gemini response blocked (Reason: {block_reason}). Check safety settings and prompt content.")

            # Check normal finish reason after checking for empty candidates
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
            if finish_reason not in ["STOP", "MAX_TOKENS"]:
                 logger.warning(f"Gemini finish abnormal (Reason: {finish_reason}). Attempt {attempts+1}/{max_retries+1}.")
                 if attempts == max_retries: raise RuntimeError(f"Gemini abnormal finish after {max_retries+1} attempts. Reason: {finish_reason}")
                 time.sleep(1.5 * (attempts + 1)); attempts += 1; continue
            if finish_reason == "MAX_TOKENS": logger.warning("Gemini response truncated: MAX_TOKENS.")

            # Extract text content
            if not candidate.content or not candidate.content.parts:
                 logger.warning(f"Gemini content empty. Attempt {attempts+1}/{max_retries+1}.")
                 if attempts == max_retries: raise RuntimeError(f"Gemini response empty after {max_retries+1} attempts.")
                 time.sleep(1.5 * (attempts + 1)); attempts += 1; continue
            response_text = "".join(part.text for part in candidate.content.parts).strip()
            logger.debug(f"Gemini raw response (len {len(response_text)}): {response_text[:150]}...")

            # --- Handle JSON ---
            if expect_json:
                logger.debug("Attempting JSON parsing...")
                cleaned_response = response_text.strip().removeprefix("```json").removesuffix("```").strip()
                if not cleaned_response: raise ValueError("Cannot parse empty string as JSON.")
                try:
                    parsed_json = json.loads(cleaned_response)
                    logger.debug("JSON parsing successful.")
                    return parsed_json
                except json.JSONDecodeError as json_err:
                    # ** FIX: Corrected Indentation and Logic **
                    logger.error(f"JSONDecodeError (Attempt {attempts+1}): {json_err}. Snippet: '{cleaned_response[:100]}...'")
                    if attempts == max_retries: # Only raise on the last attempt
                        raise ValueError(f"Failed JSON parsing after retries: {json_err}. Response: {cleaned_response[:500]}...") from json_err
                    # If not the last attempt, fall through to the general exception handler below to retry
                    raise # Re-raise the JSONDecodeError to be caught by the general Exception handler for retry logic
                except Exception as parse_err:
                    logger.error(f"Unexpected JSON parsing error (Attempt {attempts+1}): {parse_err}", exc_info=True)
                    if attempts == max_retries: raise ValueError(f"Unexpected JSON parsing error after retries: {parse_err}") from parse_err
                    raise # Re-raise to be caught by general Exception handler

            else: # Not expecting JSON
                return response_text # Return raw text

        except Exception as e:
            # Catch API errors, blocked content errors, parsing errors that were re-raised, etc.
            logger.warning(f"Gemini call/processing failed (Attempt {attempts + 1}/{max_retries + 1}): {type(e).__name__} - {e}")
            last_exception = e; attempts += 1
            if attempts <= max_retries:
                time.sleep(1.5 * attempts) # Exponential backoff before next retry
            else:
                # Log final failure and re-raise the last exception encountered
                logger.error(f"Gemini API call failed permanently after {max_retries + 1} attempts: {last_exception}", exc_info=True if last_exception else False)
                raise RuntimeError(f"Gemini API call failed: {last_exception}") from last_exception

    # Fallback if loop finishes unexpectedly (shouldn't happen with current logic)
    raise RuntimeError("Gemini content generation failed unexpectedly after retries.")


# =========================================
# --- Specific Task Functions ---
# (Keep all function definitions from the previous version)
# Example:
def generate_sql_from_nl_llm(natural_language_query: str, schema_context: str, dialect: str = "standard") -> str:
    # ... (prompt and call as before) ...
    prompt = f"Schema:\n{schema_context}\n\nUser Query: \"{natural_language_query}\"\n\nTask: Generate a single, valid {dialect} SQL query answering the query using ONLY the schema. Output ONLY the raw SQL. If impossible, output only: Error: Cannot answer query."
    return _call_gemini_api(ADVANCED_MODEL_NAME, prompt, temperature=0.2)

def generate_text_summary_llm(context: str, task: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    # ... (prompt and call as before) ...
    prompt = f"Context:\n---\n{context}\n---\nTask: {task}\n\nOutput only the result using Markdown."
    return _call_gemini_api(model_name, prompt, temperature=0.6)

def suggest_visualization_llm(context: str, available_columns: List[str], supported_types: Optional[List[str]] = None) -> Dict[str, Any]:
    # ... (prompt and call as before, expecting JSON) ...
    supported_types_str = f"Choose 'chart_type' ONLY from: {', '.join(supported_types)}." if supported_types else "Suggest common chart type."
    prompt = f"Context:\n---\n{context}\n---\nAvailable Columns: {available_columns}\nTask: Suggest best visualization. {supported_types_str} Output ONLY valid JSON object {{'chart_type': string|null, 'x_column': string|null, 'y_column': string|null, 'color_column': string|null, 'rationale': string, 'extra_params': object|null}}."
    result = _call_gemini_api(DEFAULT_MODEL_NAME, prompt, expect_json=True, temperature=0.4)
    if not isinstance(result, dict): raise RuntimeError(f"LLM suggestion unexpected type: {type(result)}")
    return result

def generate_report_narrative_llm(context: str, task: str) -> str:
    # ... (prompt and call as before) ...
    prompt = f"Context:\n---\n{context}\n---\nTask: {task}\nInstructions: Generate professional narrative based ONLY on context. Use Markdown headings (##). Synthesize info. Output only narrative."
    return _call_gemini_api(ADVANCED_MODEL_NAME, prompt, temperature=0.7, max_output_tokens=8192)

def generate_recommendations_llm(context: str, task: str, feedback_context: Optional[str] = None) -> List[Dict]:
    # ... (prompt and call as before, expecting JSON) ...
    full_context = context; prompt = ""; task_desc = task
    if feedback_context: full_context += f"\n--- Feedback Context ---\n{feedback_context}\n"
    prompt = f"Context:\n---\n{full_context}\n---\nTask: {task}\nOutput: ONLY valid JSON list of objects: [{'recommendation':'...', 'rationale':'...', 'confidence':'High|Medium|Low', 'estimated_impact':'High|Medium|Low|desc', 'priority':'High|Medium|Low'}]"
    result = _call_gemini_api(ADVANCED_MODEL_NAME, prompt, expect_json=True, temperature=0.5)
    if not isinstance(result, list): raise RuntimeError(f"LLM recommendations unexpected type: {type(result)}")
    return result

def suggest_relationships_llm(schemas: Dict[str, str]) -> List[Dict]:
    # ... (prompt and call as before, expecting JSON) ...
    if not schemas or len(schemas) < 2: return []
    context = "Schemas:\n" + json.dumps(schemas, indent=2)[:5000] + "...\n\n"
    task = ("Identify potential joins between *different* sources based on names/types. "
            "Output ONLY JSON list: [{'source1': s1, 'column1': c1, 'source2': s2, 'column2': c2, 'rationale': reason}]")
    try: return _call_gemini_api(ADVANCED_MODEL_NAME, context + task, expect_json=True, temperature=0.3)
    except Exception as e: logger.error(f"LLM relationship suggestion failed: {e}"); return [{"error": str(e)}]

def generate_description_and_tags_llm(sample_data: str, schema: str) -> Dict:
    # ... (prompt and call as before, expecting JSON) ...
    context = f"Schema:\n{schema}\n\nSample Data:\n{sample_data}\n"
    task = ("1. Generate concise 1-sentence description of dataset.\n2. Suggest 3-7 relevant tags (keywords).\n"
            "Output ONLY JSON: {'description': string, 'tags': list_of_strings}")
    try: return _call_gemini_api(DEFAULT_MODEL_NAME, context + task, expect_json=True, temperature=0.7)
    except Exception as e: logger.error(f"LLM desc/tag failed: {e}"); return {"description": f"Error: {e}", "tags": ["error"]}

def suggest_quality_rules_llm(profile_summary: Dict, schema: str) -> List[Dict]:
    # ... (prompt and call as before, expecting JSON) ...
    context = f"Profile Summary:\n{json.dumps(profile_summary, default=str)[:3000]}...\n\nSchema:\n{schema}\n"
    task = ("Suggest 3-5 specific data quality rules based on profile/schema. "
            "Output ONLY JSON list: [{'rule_name': name, 'rule_type': type, 'column_name': col|null, 'rule_parameters': {{param: val}}, 'rationale': reason}] "
            "Types: 'not_null', 'is_unique', 'min_value', 'regex_match', 'enum_values'.")
    try: return _call_gemini_api(ADVANCED_MODEL_NAME, context + task, expect_json=True, temperature=0.6)
    except Exception as e: logger.error(f"LLM DQ rule suggestion failed: {e}"); return [{"rule_name": f"Error: {e}", "rule_type": "error"}]

def explain_code_llm(code_or_action: str, language: str = "python") -> str:
    # ... (prompt and call as before) ...
    prompt = f"Explain this {language} step-by-step simply. Focus on purpose/logic. Use Markdown.\n---\n{code_or_action}\n---\nExplanation:"
    return _call_gemini_api(DEFAULT_MODEL_NAME, prompt=prompt, temperature=0.5)

def generate_pandas_code_llm(schema: str, command: str, task: str = "dataframe transformation") -> str:
    # ... (prompt and call as before, using 'command' parameter) ...
    prompt = f"Schema:\n{schema}\nUser request: \"{command}\"\nTask: Generate Python code using Pandas (assume df exists, pd imported) for '{task}'. Output ONLY raw Python code. No explanations/markdown. If ambiguous, output '# Error: Ambiguous request.'"
    return _call_gemini_api(ADVANCED_MODEL_NAME, prompt=prompt, temperature=0.3)

def suggest_structured_cleaning_steps_llm(profile: Dict, schema: str) -> List[Dict]:
    # ... (prompt and call as before, expecting JSON) ...
     context = f"Profile: {str(profile)[:2000]}...\nSchema: {schema[:1000]}..."
     task = ("Suggest cleaning steps as JSON list: [{'step_type': '...', 'params': {...}, 'rationale': '...'}]. Valid types: drop_duplicates, impute_missing, convert_type, etc.")
     try: return _call_gemini_api(ADVANCED_MODEL_NAME, context + task, is_json_output=True)
     except Exception as e: logger.error(f"LLM structured clean suggestion failed: {e}"); return []

def suggest_structured_feature_steps_llm(schema: str, goal: Optional[str] = None) -> List[Dict]:
    # ... (prompt and call as before, expecting JSON) ...
    context = f"Schema: {schema[:1000]}...\nGoal: {goal or 'General features'}"
    task = ("Suggest feature steps as JSON list: [{'step_type': '...', 'params': {...}, 'rationale': '...'}]. Valid types: date_extraction, numeric_interaction, grouped_aggregation, binning, etc.")
    try: return _call_gemini_api(ADVANCED_MODEL_NAME, context + task, is_json_output=True)
    except Exception as e: logger.error(f"LLM structured feature suggestion failed: {e}"); return []

def suggest_forecast_model_and_params_llm(time_series_description: str) -> Dict:
    # ... (prompt and call as before, expecting JSON) ...
    context = f"Time series description:\n{time_series_description}"
    task = ("Suggest suitable forecast model ('ARIMA', 'Prophet', 'ExponentialSmoothing') & basic params based on description. Output ONLY JSON: {'model': 'name', 'params': {}, 'rationale': '...'}")
    try: return _call_gemini_api(DEFAULT_MODEL_NAME, context + task, expect_json=True, temperature=0.4)
    except Exception as e: logger.error(f"LLM forecast suggestion failed: {e}"); return {"error": str(e)}

def generate_rca_hints_llm(context: str, task: str) -> List[str]:
    # ... (prompt and call as before, expecting JSON list) ...
    prompt = f"Context:\n{context}\n\nTask: {task}\n\nOutput: ONLY JSON list of strings (questions or actions)."
    try:
        result = _call_gemini_api(ADVANCED_MODEL_NAME, prompt, expect_json=True, temperature=0.6)
        return result if isinstance(result, list) else [f"Error: AI response not a list ({type(result)})."]
    except Exception as e: logger.error(f"LLM RCA hints failed: {e}"); return [f"Error: {e}"]

def interpret_clusters_llm(cluster_profiles: Dict) -> Dict:
    # ... (prompt and call as before, expecting JSON dict) ...
    context = f"Cluster Profiles:\n{json.dumps(cluster_profiles, indent=2, default=str)[:5000]}..."
    n_clusters = len(cluster_profiles)
    task = (f"Interpret {n_clusters} clusters. For each ID ('0' to '{n_clusters-1}'), provide 'label' & 'description'. Output ONLY valid JSON object mapping cluster ID strings to {{'label': ..., 'description': ...}}." )
    try: return _call_gemini_api(DEFAULT_MODEL_NAME, context + "\nTask: " + task, expect_json=True, temperature=0.7)
    except Exception as e: logger.error(f"LLM cluster interpretation failed: {e}"); return {"error": str(e)}

def identify_anomalies_llm(context: str, task: str) -> List[Dict]:
     # ... (prompt and call as before, expecting JSON list) ...
     prompt = f"Context:\n{context}\n\nTask:{task}\n\nOutput: ONLY JSON list of findings. Each object MUST have 'type', 'details', 'severity' ('Low'/'Medium'/'High'). Return [] if none."
     try:
         result = _call_gemini_api(ADVANCED_MODEL_NAME, prompt, expect_json=True, temperature=0.5)
         return result if isinstance(result, list) and all(isinstance(item, dict) for item in result) else []
     except Exception as e: logger.error(f"LLM anomaly identification failed: {e}"); return [{"type": "Error", "details": str(e), "severity": "Low"}]

def generate_image_for_report_llm(prompt: str) -> Optional[bytes]:
    # ... (Placeholder logic remains the same) ...
    logger.warning(f"Image generation requested (NOT IMPLEMENTED): {prompt[:100]}...")
    return None

def explain_llm_output_llm(request: str, output: Union[str, Dict, List], context: Optional[str] = None) -> str:
    # ... (prompt and call as before) ...
    prompt = f"User Request:\n{request}\n\n";
    if context: prompt += f"Context:\n{context[:2000]}...\n\n"
    prompt += f"AI Output:\n{str(output)[:2000]}...\n\n"
    prompt += "Task: Explain step-by-step how 'AI OUTPUT' was likely generated based ONLY on 'User Request' and 'Context'. Use simple terms. If code, explain logic. Use Markdown."
    return _call_gemini_api(DEFAULT_MODEL_NAME, prompt=prompt, temperature=0.5)