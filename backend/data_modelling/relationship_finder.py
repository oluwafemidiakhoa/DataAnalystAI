# backend/data_modeling/relationship_finder.py
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd # May be needed if analyzing data values later

# Assuming LLM utils and schema utils are available
try:
    from backend.llm.gemini_utils import suggest_relationships_llm
    from backend.data_processing.profiler import get_schema_details # To get schema if needed
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger = logging.getLogger(__name__) # Define here if import fails
    logger.error("LLM or Profiler utils not found for relationship finder.", exc_info=True)
    # Define mock
    def suggest_relationships_llm(schemas):
        logger.warning("Using MOCK suggest_relationships_llm.")
        # Basic mock based on common ID patterns across two mock tables
        mock_rels = []
        table_names = list(schemas.keys())
        if len(table_names) >= 2:
             if "user_id" in schemas[table_names[0]] and "id" in schemas[table_names[1]]:
                  mock_rels.append({"source1": table_names[0], "column1": "user_id", "source2": table_names[1], "column2": "id", "rationale": "Mock: Common ID pattern."})
        return mock_rels

logger = logging.getLogger(__name__)

def find_potential_relationships(
    schema_info: Dict[str, Dict[str, str]], # Expected: {'source_name': {'col': 'type', ...}}
    use_llm: bool = True
    ) -> List[Dict]:
    """
    Identifies potential relationships (joins) between different data sources based on schema.

    Args:
        schema_info: A dictionary where keys are source names (table/file names)
                     and values are dictionaries mapping column names to their type strings.
        use_llm: Whether to use the LLM for relationship suggestion.

    Returns:
        A list of dictionaries, each representing a potential relationship link.
        e.g., [{'source1': 'orders', 'column1': 'customer_id', 'source2': 'users', 'column2': 'id', 'rationale': '...'}, ...]
    """
    logger.info(f"Finding potential relationships among {len(schema_info)} sources. Using LLM: {use_llm and LLM_AVAILABLE}")

    if len(schema_info) < 2:
        logger.info("Need at least two data sources to find relationships.")
        return []

    if use_llm and LLM_AVAILABLE:
        try:
            suggested_relationships = suggest_relationships_llm(schema_info)
            logger.info(f"LLM suggested {len(suggested_relationships)} potential relationships.")
            # TODO: Add validation for the structure returned by LLM if needed
            return suggested_relationships
        except Exception as e:
            logger.error(f"LLM relationship suggestion failed: {e}. Falling back to heuristics.", exc_info=True)
            # Fall through to heuristic method if LLM fails

    # --- Heuristic Method (Fallback or if use_llm is False) ---
    # This replicates/moves the logic from the profiler's internal helper
    logger.info("Using heuristic method for relationship finding.")
    candidates = []
    column_map = {} # Map normalized column name to list of {'orig_name': str, 'source': str, 'type': str}

    for source_name, columns in schema_info.items():
        for col_name, col_type in columns.items():
            norm_name = col_name.lower().replace('_id', '').replace('id', '').replace('_key', '').replace('key', '').strip('_ ')
            if not norm_name or len(norm_name) < 2: continue # Skip empty or very short normalized names

            if norm_name not in column_map: column_map[norm_name] = []
            column_map[norm_name].append({"orig_name": col_name, "source": source_name, "type": col_type})

    potential_links = {name: sources for name, sources in column_map.items() if len(sources) > 1}

    for name, sources in potential_links.items():
        # Compare sources pairwise
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s1 = sources[i]
                s2 = sources[j]
                # Ensure sources are different
                if s1['source'] == s2['source']: continue

                # Basic type compatibility check (can be improved)
                type1 = str(s1['type']).lower(); type2 = str(s2['type']).lower()
                compatible = False
                numeric_types = {'int', 'float', 'decimal', 'numeric', 'number'}
                string_types = {'str', 'varchar', 'text', 'char', 'string', 'objectid'} # Treat objectid like string for matching keys

                if any(t in type1 for t in numeric_types) and any(t in type2 for t in numeric_types): compatible = True
                elif any(t in type1 for t in string_types) and any(t in type2 for t in string_types): compatible = True
                elif type1 == type2: compatible = True # Exact match

                if compatible:
                    rationale = f"Heuristic: Name match ('{name}') & Type compatible ({type1}/{type2})"
                    candidates.append({
                        "source1": s1['source'], "column1": s1['orig_name'],
                        "source2": s2['source'], "column2": s2['orig_name'],
                        "rationale": rationale
                    })

    logger.info(f"Heuristic method found {len(candidates)} potential relationships.")
    # Remove duplicates? Heuristics might find A->B and B->A
    unique_candidates = []
    seen_pairs = set()
    for cand in candidates:
         pair1 = tuple(sorted((f"{cand['source1']}.{cand['column1']}", f"{cand['source2']}.{cand['column2']}")))
         if pair1 not in seen_pairs:
              unique_candidates.append(cand)
              seen_pairs.add(pair1)

    return unique_candidates