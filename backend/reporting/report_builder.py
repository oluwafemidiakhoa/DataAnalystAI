# backend/reporting/report_builder.py
import logging
import json
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

# --- Get Logger ---
# Define logger early to use in except blocks if needed
logger = logging.getLogger(__name__)

# --- LLM Utility Import ---
LLM_AVAILABLE = False
try:
    # Assumes functions exist for narrative and optional image generation
    from backend.llm.gemini_utils import generate_report_narrative_llm, generate_image_for_report_llm
    LLM_AVAILABLE = True
    logger.debug("LLM utils for report builder loaded.")
except ImportError:
    logger.error("LLM utility functions not found for report builder. AI features disabled.", exc_info=True)
    # Define mocks
    def generate_report_narrative_llm(c, t): logger.warning("Using MOCK generate_report_narrative_llm."); return f"Mock Narrative for: {t}"
    def generate_image_for_report_llm(p): logger.warning("Using MOCK generate_image_for_report_llm."); return None

# --- Exporter Function Imports ---
EXPORTERS_AVAILABLE = False
PDF_EXPORT_AVAILABLE = False
PPTX_EXPORT_AVAILABLE = False
# Define dummies first, might be overwritten by imports
def create_pdf_report(*args, **kwargs): raise NotImplementedError("PDF Exporter function unavailable or not loaded.")
def create_pptx_report(*args, **kwargs): raise NotImplementedError("PPTX Exporter function unavailable or not loaded.")

try:
    # Assume exporters live in the same directory for simplicity, adjust if needed
    from .exporters import create_pdf_report, create_pptx_report
    EXPORTERS_AVAILABLE = True # Flag that exporters module was found
    PDF_EXPORT_AVAILABLE = True
    PPTX_EXPORT_AVAILABLE = True
    logger.info("Exporter functions (create_pdf_report, create_pptx_report) loaded successfully.")
except ImportError:
     # ** FIX: Added closing parenthesis to the warning log **
     logger.warning("Exporter functions (PDF/PPTX) not found in exporters.py.")
except Exception as e:
     logger.error(f"Unexpected error importing exporters: {e}", exc_info=True)
     # Keep dummies defined above


# --- Main Report Building Function ---
def build_narrative_report(
    report_goal: str,
    analysis_results: Optional[pd.DataFrame] = None,
    insights: Optional[str] = None,
    kpis: Optional[Dict[str, Any]] = None, # Changed from kpi_data for consistency with page 5 call
    visualizations: Optional[List[go.Figure]] = None,
    include_images: bool = False,
    # db: Optional[Session] = None # Add DB session if saving report metadata
    ) -> Dict[str, Any]:
    """
    Builds a narrative report combining analysis results, insights, KPIs,
    visualizations, and potentially AI-generated images.

    Args:
        report_goal: The user-defined goal or title for the report.
        analysis_results: Optional DataFrame from the analysis step.
        insights: Optional string containing AI-generated deep insights.
        kpis: Optional dictionary of tracked KPI data {kpi_name: {value, trend, delta}}.
        visualizations: Optional list of Plotly Figure objects to include.
        include_images: Whether to attempt generating illustrative images with AI.
        db: Optional SQLAlchemy Session for saving the report metadata.

    Returns:
        A dictionary containing the report components:
        {
            'title': str, 'narrative': str, 'visualizations': list[dict],
            'kpi_summary': str, 'images': list[dict], 'error': str | None,
            'export_available': {'pdf': bool, 'pptx': bool}
        }
    """
    logger.info(f"Building narrative report for goal: '{report_goal}'. Include Images: {include_images}")

    # Initialize report components dictionary
    report_components = {
        "title": report_goal,
        "narrative": "Narrative generation failed or not attempted.", # Default message
        "visualizations": [],
        "kpi_summary": "No KPI data included or available.",
        "images": [],
        "error": None,
        "export_available": {'pdf': PDF_EXPORT_AVAILABLE, 'pptx': PPTX_EXPORT_AVAILABLE} # Use flags set during import
    }
    kpi_summary_for_llm = "" # Initialize

    # --- Prepare Context for LLM Narrative Generation ---
    # Limit context length to avoid exceeding LLM limits
    MAX_CONTEXT_LEN = 15000
    context_parts = []
    context_parts.append(f"Report Goal: {report_goal}\n")

    if analysis_results is not None:
        context_parts.append(f"Analysis Results Summary (Shape: {analysis_results.shape}):\n")
        context_parts.append(f"- Columns: {', '.join(analysis_results.columns)}\n")
        context_parts.append(f"- Sample Data:\n{analysis_results.head(3).to_markdown(index=False)}\n")
    if insights: context_parts.append(f"Key Insights Found:\n{insights}\n")
    if kpis:
        kpi_summary_for_llm = "\n".join([f"- {name}: {data.get('value', 'N/A'):.2f} (Trend: {data.get('trend', 'N/A')}, Delta: {data.get('delta', 'N/A'):.2f})" for name, data in kpis.items()])
        context_parts.append(f"Current KPI Status:\n{kpi_summary_for_llm}\n")
        report_components["kpi_summary"] = kpi_summary_for_llm # Store formatted summary
    if visualizations:
         viz_desc = "\n".join([f"- Chart {i+1}: {fig.layout.title.text if fig.layout.title else 'Untitled Chart'}" for i, fig in enumerate(visualizations)])
         context_parts.append(f"Included Visualizations:\n{viz_desc}\n")

    context = "\n".join(context_parts)
    if len(context) > MAX_CONTEXT_LEN:
         context = context[:MAX_CONTEXT_LEN] + "\n...(Context Truncated)..."
         logger.warning("LLM context truncated for report narrative generation.")

    task = ("Generate a coherent narrative report (using Markdown) based ONLY on the provided context "
            "(goal, data summary, insights, KPIs, included visualizations). Structure logically "
            "(e.g., ## Summary, ## Key Findings, ## KPI Analysis, ## Conclusion). "
            "Explain the key takeaways clearly and professionally. Do not invent information."
           )

    # --- Generate Narrative ---
    try:
        if not LLM_AVAILABLE: raise RuntimeError("LLM unavailable.")
        logger.info("Generating report narrative with LLM...")
        narrative_text = generate_report_narrative_llm(context, task) # Use imported/mock func

        if not narrative_text or narrative_text.lower().startswith("error:"):
             raise RuntimeError(narrative_text or "AI failed to generate narrative.")
        report_components["narrative"] = narrative_text.strip()
        logger.info("Report narrative generated successfully.")
    except Exception as e:
        logger.error(f"Report narrative generation failed: {e}", exc_info=True)
        error_msg = f"Error generating narrative: {e}"
        report_components["narrative"] = error_msg
        report_components["error"] = str(e) # Store overall error


    # --- Process Visualizations (Serialize to JSON) ---
    if visualizations:
        logger.info(f"Serializing {len(visualizations)} visualizations...")
        for fig in visualizations:
            title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Chart"
            try:
                viz_json_str = fig.to_json() # Convert Plotly figure to JSON string
                report_components["visualizations"].append({"title": title, "plotly_json": viz_json_str})
            except Exception as e:
                logger.error(f"Failed to serialize viz '{title}': {e}", exc_info=True)
                report_components["visualizations"].append({"title": title, "error": f"Serialization failed: {e}"})

    # --- Generate Illustrative Images (Optional - #11) ---
    if include_images and LLM_AVAILABLE:
        logger.info("Attempting to generate illustrative images...")
        # Define prompts based on report goal or key insights
        image_prompts = [ f"Abstract image representing successful business growth related to '{report_goal}'" ]
        for prompt in image_prompts:
             img_result = {"prompt": prompt, "image_bytes": None, "error": None}
             try:
                 img_bytes = generate_image_for_report_llm(prompt) # Use imported/mock func
                 if img_bytes: img_result["image_bytes"] = img_bytes; logger.info(f"Generated image for: {prompt[:50]}...")
                 else: img_result["error"] = "AI did not return an image."
             except Exception as e: logger.error(f"AI image gen failed: {e}", exc_info=True); img_result["error"] = str(e)
             report_components["images"].append(img_result)

    # --- Optional: Save Report Metadata to DB ---
    # (Keep DB saving logic commented out or implement with crud.save_report)

    logger.info(f"Report build complete for '{report_goal}'.")
    return report_components


# --- Export Functions ---
# These call the functions imported (or dummied) from exporters.py
def export_report_to_pdf(report_data: Dict[str, Any]) -> BytesIO:
    """Generates a PDF report from the built report components."""
    if not PDF_EXPORT_AVAILABLE:
         logger.error("PDF Exporter function not available.")
         raise NotImplementedError("PDF Exporter backend is not available or failed to import.")
    title = report_data.get('title', 'Untitled Report')
    logger.info(f"Generating PDF for report: {title}")
    try:
        pdf_bytesio = create_pdf_report(report_data) # Call imported/dummy exporter func
        logger.info("PDF generation successful.")
        pdf_bytesio.seek(0) # Rewind buffer before returning
        return pdf_bytesio
    except Exception as e:
        logger.error(f"PDF export failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate PDF report: {e}") from e

def export_report_to_pptx(report_data: Dict[str, Any]) -> BytesIO:
    """Generates a PPTX report from the built report components."""
    if not PPTX_EXPORT_AVAILABLE:
         logger.error("PPTX Exporter function not available.")
         raise NotImplementedError("PPTX Exporter backend is not available or failed to import.")
    title = report_data.get('title', 'Untitled Report')
    logger.info(f"Generating PPTX for report: {title}")
    try:
        pptx_bytesio = create_pptx_report(report_data) # Call imported/dummy exporter func
        logger.info("PPTX generation successful.")
        pptx_bytesio.seek(0) # Rewind buffer before returning
        return pptx_bytesio
    except Exception as e:
        logger.error(f"PPTX export failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate PPTX report: {e}") from e