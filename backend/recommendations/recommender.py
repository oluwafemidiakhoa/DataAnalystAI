# backend/recommendations/recommender.py
import pandas as pd # <-- Added missing import
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session # For DB interaction type hint

# Assuming LLM utils are available
LLM_AVAILABLE = False
try:
    # Function needs to be prompted for recommendations including impact/priority
    from backend.llm.gemini_utils import generate_recommendations_llm
    LLM_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__) # Define logger if import fails early
    logger.error("LLM utility function generate_recommendations_llm not found.", exc_info=True)
    # Define mock
    def generate_recommendations_llm(context, task, feedback_context=None): # Add feedback_context to mock sig
        logger.warning("Using MOCK generate_recommendations_llm.")
        recs = [{"recommendation": f"Mock Rec based on context ({len(context)} chars)", "rationale": "Mock rationale.", "confidence": "Medium", "estimated_impact": "Low", "priority": "Medium"}]
        if feedback_context: recs.append({"recommendation": "Mock Rec considering feedback.", "rationale": "Feedback was provided.", "confidence": "High", "estimated_impact": "Medium", "priority": "High"})
        return recs

# Assuming CRUD utils are available for feedback
DB_AVAILABLE = False
try:
    from backend.database import crud, models # For retrieving feedback
    DB_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__) # Define logger if not already defined
    logger.warning("Database CRUD/models not found. Recommendation feedback cannot be retrieved/incorporated.")
    # Define dummy crud if needed for structure
    class crud:
         @staticmethod
         def get_recent_recommendation_feedback(*args, **kwargs): return []
         @staticmethod
         def save_recommendation_feedback(*args, **kwargs): logger.warning("Dummy save feedback called."); return False # Indicate failure

logger = logging.getLogger(__name__)

# --- Main Recommendation Function ---
# ** VERIFY THIS FUNCTION NAME IS CORRECT **
def generate_recommendations(
    insights: Optional[str] = None,
    analysis_results: Optional[pd.DataFrame] = None,
    kpi_data: Optional[Dict[str, Any]] = None,
    business_context: Optional[str] = "General Business Analysis", # Default context
    project_id: Optional[int] = None, # To fetch relevant feedback
    db: Optional[Session] = None, # DB session to fetch feedback
    max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
    """
    Generates actionable business recommendations using an LLM, incorporating
    analysis insights, KPIs, context, and recent user feedback from the database.

    Args:
        insights: AI-generated deep insights string.
        analysis_results: Optional DataFrame from the analysis step.
        kpi_data: Optional dictionary of tracked KPI data {kpi_name: {value, trend, delta}}.
        business_context: String describing the user's business/domain.
        project_id: Optional ID of the current project to retrieve relevant feedback.
        db: Optional SQLAlchemy Session to fetch feedback.
        max_recommendations: The maximum number of recommendations to generate.

    Returns:
        A list of dictionaries, each containing a recommendation with keys like
        'recommendation', 'rationale', 'confidence', 'estimated_impact', 'priority'.

    Raises:
        RuntimeError: If the LLM call fails.
        ValueError: If no input context is provided.
    """
    logger.info(f"Generating up to {max_recommendations} recommendations. Project: {project_id or 'N/A'}")

    if not insights and analysis_results is None and not kpi_data:
        logger.warning("Cannot generate recommendations without input (insights, results, or KPIs).")
        return [{"recommendation": "No input data provided.", "rationale": "N/A", "confidence": "N/A", "estimated_impact": "N/A", "priority": "N/A"}]

    if not LLM_AVAILABLE:
         logger.error("LLM function for recommendations is not available.")
         raise RuntimeError("AI Recommendation engine offline (LLM unavailable).")


    # --- Prepare Context for LLM ---
    context = f"Generate actionable business recommendations based on:\nBusiness Context: {business_context}\n\n"
    if insights: context += f"Key Insights:\n{insights}\n\n"
    if kpi_data:
        kpi_summary = "\n".join([f"- {n}: {d.get('value','N/A'):.2f} (Trend={d.get('trend','N/A')}, Delta={d.get('delta','N/A'):.2f})" for n, d in kpi_data.items()])
        context += f"Current KPI Status:\n{kpi_summary}\n\n"
    if analysis_results is not None:
        context += f"Supporting Analysis Data (Shape: {analysis_results.shape}):\n"
        context += f"- Cols: {', '.join(analysis_results.columns)}\n"
        context += f"- Sample:\n{analysis_results.head(3).to_markdown(index=False)}\n\n"

    # --- Incorporate User Feedback (#12) ---
    feedback_context = None
    if db and DB_AVAILABLE and project_id is not None:
        try:
            recent_feedback = crud.get_recent_recommendation_feedback(db=db, project_id=project_id, limit=10)
            if recent_feedback:
                 feedback_summary = "\n".join([f"- Rec ~'{fb.recommendation.recommendation_text[:50]}...': User rated '{fb.rating}'" for fb in recent_feedback if fb.recommendation])
                 feedback_context = f"Recent Feedback:\n{feedback_summary}\n"
                 logger.info(f"Incorporated {len(recent_feedback)} feedback entries.")
        except Exception as e: logger.warning(f"Could not retrieve recommendation feedback: {e}", exc_info=True)
    else: logger.debug("Skipping feedback retrieval (DB/ProjectID missing).")

    # --- Enhanced Task Prompt for LLM (#12 - Impact/Priority) ---
    task = (f"Provide {max_recommendations} concrete, actionable recommendations derived from context. Consider feedback if provided. "
            "For each include: 'recommendation' (action string), 'rationale' (justification string), 'confidence' ('High'/'Medium'/'Low'), 'estimated_impact' ('High'/'Medium'/'Low' or description), 'priority' ('High'/'Medium'/'Low'). "
            "Focus on improving KPIs, addressing trends, or leveraging insights. Format as JSON list of objects."
           )

    # --- Call LLM ---
    try:
        recommendations = generate_recommendations_llm(context, task, feedback_context=feedback_context) # Pass context, task, feedback

        # --- Validate Response Structure ---
        if not isinstance(recommendations, list): raise ValueError(f"AI response was not a list ({type(recommendations)}).")
        validated_recs = []
        required_keys = ["recommendation", "rationale", "confidence", "estimated_impact", "priority"]
        for i, rec in enumerate(recommendations):
            if isinstance(rec, dict) and all(key in rec for key in required_keys): validated_recs.append(rec)
            else: logger.warning(f"Recommendation {i} invalid structure, skipping: {rec}")
        if not validated_recs: raise RuntimeError("AI failed to generate recommendations in expected format.")

        logger.info(f"Generated {len(validated_recs)} valid recommendations.")
        # --- Optional: Save recommendations to DB here ---
        # if db and DB_AVAILABLE:
        #     for rec_data in validated_recs:
        #         try:
        #             # crud.create_recommendation(db=db, ...) # Needs implementation in crud.py
        #             # Add the returned DB ID to rec_data['id'] for feedback linking
        #             pass
        #         except Exception as save_err: logger.error(f"Failed to save recommendation to DB: {save_err}")

        return validated_recs

    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate recommendations: {e}") from e

# --- Function to Save Feedback (called from frontend/API layer) ---
def store_recommendation_feedback(
    recommendation_id: int,
    user_id: int,
    rating: Optional[str], # e.g., 'Helpful', 'Not Helpful'
    comment: Optional[str] = None,
    db: Optional[Session] = None # Requires DB session
    ) -> bool:
    """Stores user feedback for a specific recommendation."""
    if not DB_AVAILABLE:
        logger.error("Cannot store feedback: Database components not available.")
        return False
    if not db:
         logger.error("Cannot store feedback: DB Session not provided.")
         return False
    if not user_id: # Requires user login context
         logger.warning("Cannot store feedback: User ID not provided.")
         return False
    if not recommendation_id: # Requires recommendation to have been saved and have an ID
         logger.warning("Cannot store feedback: Recommendation ID not provided.")
         return False


    logger.info(f"Storing feedback for recommendation ID {recommendation_id} from user ID {user_id}. Rating: {rating}")
    try:
        # Assumes crud.save_recommendation_feedback exists and takes these args
        crud.save_recommendation_feedback(
            db=db, recommendation_id=recommendation_id, user_id=user_id, rating=rating, comment=comment
        )
        logger.info("Feedback stored successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to store recommendation feedback: {e}", exc_info=True)
        return False