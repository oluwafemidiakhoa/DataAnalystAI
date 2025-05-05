# backend/analysis/segmenter.py
# Logic for customer/data segmentation using clustering

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

# Attempt to import clustering libraries
SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score # To evaluate cluster quality
    from sklearn.decomposition import PCA # For visualization
    SKLEARN_AVAILABLE = True
except ImportError:
     logging.warning("scikit-learn not found. Segmentation features disabled. `pip install scikit-learn`")

# Assuming LLM utils for interpretation
try:
    from backend.llm.gemini_utils import interpret_clusters_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    def interpret_clusters_llm(profiles, n): return {str(i): {"label": f"Mock Segment {i+1}", "description": "Mock description."} for i in range(n)}

logger = logging.getLogger(__name__)

def find_segments(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 3,
    scale_data: bool = True,
    use_llm_interpretation: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[go.Figure]]:
    """
    Performs K-Means clustering to segment data based on selected features.

    Args:
        df: Input DataFrame.
        feature_cols: List of numeric column names to use for clustering.
        n_clusters: The number of segments (clusters) to find.
        scale_data: Whether to scale numeric features before clustering (recommended).
        use_llm_interpretation: Whether to use LLM to name/describe clusters.

    Returns:
        A tuple containing:
        - segmented_df (pd.DataFrame): Original DataFrame with an added 'segment' column.
        - segment_summary (Dict): Dictionary describing each segment (size, characteristics, label).
        - segment_fig (go.Figure): Plotly figure visualizing the segments (using PCA).
        Returns (None, None, None) on failure.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn library is required for segmentation.")
    if df is None or df.empty:
        logger.warning("Cannot perform segmentation on empty DataFrame.")
        return None, None, None
    if not feature_cols:
        raise ValueError("At least one feature column must be selected for segmentation.")
    if not all(col in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        raise KeyError(f"Feature columns not found in DataFrame: {', '.join(missing)}")
    if n_clusters < 2:
        raise ValueError("Number of clusters must be at least 2.")

    logger.info(f"Performing K-Means clustering: {n_clusters} clusters on {len(feature_cols)} features: {feature_cols}")

    try:
        # Prepare data: Select features, drop NaNs
        data_to_cluster = df[feature_cols].copy()
        initial_rows = len(data_to_cluster)
        data_to_cluster.dropna(inplace=True)
        dropped_rows = initial_rows - len(data_to_cluster)
        if dropped_rows > 0: logger.warning(f"Dropped {dropped_rows} rows with missing values in feature columns before clustering.")
        if len(data_to_cluster) < n_clusters: raise ValueError(f"Not enough data points ({len(data_to_cluster)}) remaining after dropping NaNs to form {n_clusters} clusters.")

        # Scale data
        if scale_data:
            logger.debug("Scaling features using StandardScaler.")
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_to_cluster)
        else:
            data_scaled = data_to_cluster.values

        # --- K-Means Clustering ---
        logger.debug(f"Fitting KMeans model with n_clusters={n_clusters}...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # Use modern n_init
        cluster_labels = kmeans.fit_predict(data_scaled)

        # Add segment labels back to the original index (handling dropped rows)
        segmented_df = df.copy()
        # Create a series with the cluster labels, using the index from the data used for clustering
        label_series = pd.Series(cluster_labels, index=data_to_cluster.index, name='segment')
        # Join back to the original df - rows dropped during NaN handling will have NaN segment
        segmented_df = segmented_df.join(label_series)
        logger.info("KMeans fitting complete. Segment labels assigned.")

        # --- Generate Segment Summary ---
        logger.debug("Generating segment summary...")
        segment_summary = {}
        # Get cluster centers (in original scale if scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_) if scale_data else kmeans.cluster_centers_
        for i in range(n_clusters):
            segment_df = data_to_cluster[cluster_labels == i] # Use original features for profiling
            cluster_center_original_scale = centers[i]
            profile = {
                 # Size should be from the original DF including NaNs? Or just clustered data? Using clustered data.
                 "size": len(segment_df),
                 "size_percentage": f"{len(segment_df) / len(data_to_cluster) * 100:.1f}%",
                 # Describe characteristics using means/medians of features
                 "characteristics": {col: f"{segment_df[col].mean():.2f} (mean)" for col in feature_cols},
                 "cluster_center": {col: f"{center_val:.2f}" for col, center_val in zip(feature_cols, cluster_center_original_scale)}
            }
            # Add LLM interpretation later
            segment_summary[str(i)] = profile # Use string key for JSON compatibility

        # --- Use LLM to Interpret/Name Clusters ---
        if use_llm_interpretation and LLM_AVAILABLE:
            try:
                logger.info("Interpreting clusters using LLM...")
                # Pass the profile summary (means/centers) to the LLM
                llm_interpretations = interpret_clusters_llm(segment_summary, n_clusters)
                # Add labels/descriptions back to the summary
                for seg_id_str, interpretation in llm_interpretations.items():
                     if seg_id_str in segment_summary and isinstance(interpretation, dict):
                         segment_summary[seg_id_str]["label"] = interpretation.get("label", f"Segment {seg_id_str}")
                         segment_summary[seg_id_str]["description"] = interpretation.get("description", "AI description pending.")
            except Exception as llm_err:
                 logger.error(f"LLM cluster interpretation failed: {llm_err}", exc_info=True)
                 # Add placeholder labels if interpretation fails
                 for seg_id_str in segment_summary:
                     segment_summary[seg_id_str]["label"] = f"Segment {seg_id_str}"
                     segment_summary[seg_id_str]["description"] = "(AI interpretation failed)"
        else:
             # Add basic labels if LLM not used
             for seg_id_str in segment_summary: segment_summary[seg_id_str]["label"] = f"Segment {seg_id_str}"

        # --- Generate Visualization (PCA) ---
        logger.debug("Generating PCA visualization for clusters...")
        segment_fig = None
        if len(feature_cols) >= 2:
             try:
                 pca = PCA(n_components=2)
                 principal_components = pca.fit_transform(data_scaled)
                 pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'], index=data_to_cluster.index)
                 pca_df['segment'] = cluster_labels.astype(str) # Use string for categorical color mapping

                 segment_fig = px.scatter(
                      pca_df, x='PC1', y='PC2', color='segment',
                      title=f'{n_clusters}-Segment Clustering Visualization (PCA Projection)',
                      labels={'color': 'Segment'},
                      hover_data=pca_df.index # Show original index on hover? or other data?
                 )
                 # Add explained variance ratio
                 explained_var = pca.explained_variance_ratio_
                 segment_fig.update_layout(
                      xaxis_title=f"PC1 ({explained_var[0]:.1%})",
                      yaxis_title=f"PC2 ({explained_var[1]:.1%})"
                 )
             except Exception as viz_err:
                  logger.error(f"Failed to generate PCA visualization: {viz_err}", exc_info=True)
                  segment_fig = go.Figure().update_layout(title_text="PCA Visualization Error")
        else:
            logger.warning("Need at least 2 features for PCA visualization.")
            segment_fig = go.Figure().update_layout(title_text="Insufficient features for PCA plot")


        logger.info("Segmentation analysis completed successfully.")
        return segmented_df, segment_summary, segment_fig

    except ImportError as e: raise # Re-raise missing library error
    except (KeyError, ValueError, TypeError) as e: logger.error(f"Data or configuration error during segmentation: {e}", exc_info=True); raise # Re-raise config errors
    except Exception as e: logger.error(f"Unexpected error during segmentation: {e}", exc_info=True); raise RuntimeError(f"Segmentation failed unexpectedly: {e}") from e