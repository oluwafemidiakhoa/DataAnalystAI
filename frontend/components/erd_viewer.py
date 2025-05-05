# frontend/components/erd_viewer.py
# Placeholder for visualizing Entity Relationship Diagrams

import streamlit as st
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Attempt to import visualization libraries
AGRAPH_AVAILABLE = False
PYVIS_AVAILABLE = False
MERMAID_AVAILABLE = False # For streamlit-mermaid component

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError: logger.info("streamlit-agraph not installed. Cannot use for ERD.")
try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    PYVIS_AVAILABLE = True
except ImportError: logger.info("pyvis not installed. Cannot use for ERD.")
# Check for mermaid component? Requires separate install usually.


def display_erd(
    relationships: List[Dict[str, str]],
    schema_info: Optional[Dict[str, Dict[str, str]]] = None,
    method: str = 'agraph' # 'agraph', 'pyvis', 'mermaid' (requires component)
    ):
    """
    Displays a relationship diagram based on suggested links.

    Args:
        relationships: List of relationship dicts, e.g.,
                       [{'source1': 'orders', 'column1': 'customer_id',
                         'source2': 'users', 'column2': 'id', 'rationale': '...'}]
        schema_info: Optional dict of schemas to extract all tables/columns for nodes.
        method: The library/method to use for rendering.
    """
    logger.info(f"Attempting to display ERD using method: {method}")
    if not relationships:
        st.info("No relationships provided to visualize.")
        return

    if method == 'agraph' and AGRAPH_AVAILABLE:
        nodes = []
        edges = []
        node_set = set()

        # Add nodes and edges from relationships
        for rel in relationships:
            s1, c1 = rel['source1'], rel['column1']
            s2, c2 = rel['source2'], rel['column2']
            # Add nodes if not already added
            if s1 not in node_set: nodes.append(Node(id=s1, label=s1, size=15)); node_set.add(s1)
            if s2 not in node_set: nodes.append(Node(id=s2, label=s2, size=15)); node_set.add(s2)
            # Add edge representing the relationship
            edges.append(Edge(source=s1, target=s2, label=f"{c1} -> {c2}")) # Simple label

        # Optional: Add nodes for tables/sources that have no relationships found yet
        if schema_info:
             for source_name in schema_info.keys():
                  if source_name not in node_set:
                       nodes.append(Node(id=source_name, label=source_name, size=10, color="#cccccc")) # Different size/color maybe

        config = Config(width=750, height=400, directed=True, physics=True, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6", node={'labelProperty':'label'})
        try:
             agraph(nodes=nodes, edges=edges, config=config)
        except Exception as e:
             st.error(f"Failed to render agraph visualization: {e}")
             logger.error("Agraph rendering failed", exc_info=True)

    elif method == 'pyvis' and PYVIS_AVAILABLE:
         net = Network(notebook=True, directed=True, height="400px", width="100%")
         node_set = set()
         for rel in relationships:
             s1, c1 = rel['source1'], rel['column1']
             s2, c2 = rel['source2'], rel['column2']
             if s1 not in node_set: net.add_node(s1, label=s1); node_set.add(s1)
             if s2 not in node_set: net.add_node(s2, label=s2); node_set.add(s2)
             net.add_edge(s1, s2, title=f"{c1} -> {c2}", arrowStrikethrough=True)

         # Optional: Add unconnected nodes
         if schema_info:
              for source_name in schema_info.keys():
                   if source_name not in node_set: net.add_node(source_name, label=source_name, color="#dddddd")

         try:
              net.show("pyvis_graph.html") # Saves a file
              # Embed the HTML file in Streamlit
              with open("pyvis_graph.html", 'r', encoding='utf-8') as f:
                   html_source = f.read()
              components.html(html_source, height=420)
         except Exception as e:
              st.error(f"Failed to render pyvis visualization: {e}")
              logger.error("Pyvis rendering failed", exc_info=True)

    elif method == 'mermaid':
         # Requires streamlit-mermaid component: pip install streamlit-mermaid
         # Construct Mermaid syntax string
         mermaid_string = "graph LR;\n" # Left-to-right graph
         added_links = set()
         for rel in relationships:
              s1, c1 = rel['source1'], rel['column1']
              s2, c2 = rel['source2'], rel['column2']
              link_fwd = f"{s1}-- {c1} -->{s2}({c2});"
              link_bwd = f"{s2}-- {c2} -->{s1}({c1});"
              # Avoid duplicate links in Mermaid string
              if link_fwd not in added_links and link_bwd not in added_links:
                   mermaid_string += f"  {link_fwd}\n"
                   added_links.add(link_fwd)
         # Add unconnected nodes? Mermaid does this automatically if nodes are defined
         # if schema_info:
         #    for source_name in schema_info.keys(): mermaid_string += f"  {source_name};\n" # Define node

         try:
             from streamlit_mermaid import st_mermaid
             st_mermaid(mermaid_string, height=400)
         except ImportError:
             st.warning("`streamlit-mermaid` component not installed. Cannot display Mermaid ERD.")
         except Exception as e:
             st.error(f"Failed to render Mermaid visualization: {e}")
             logger.error("Mermaid rendering failed", exc_info=True)
             st.code(mermaid_string, language="mermaid") # Show code on error

    else:
        st.warning(f"ERD visualization method '{method}' not available or supported. Please install required libraries (streamlit-agraph, pyvis, streamlit-mermaid).")
        st.json(relationships) # Show raw relationships as fallback