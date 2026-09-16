"""
Point Cloud Visualization Utilities for FSCT Wrapper
"""

import logging
import os

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

logger = logging.getLogger(__name__)


def _plot_failed(what, source, error):
    """
    Report a figure that could not be built, then let the caller return None.

    Every builder below returns None on failure so the page can skip the chart
    rather than break. That is the right behaviour, but the handlers used to
    discard the exception entirely, which made "the CSV is missing a column",
    "the file is unreadable" and "plotly raised" all look identical - a blank
    space with no explanation anywhere.
    """
    logger.warning("%s could not be built from %s: %s: %s",
                   what, os.path.basename(str(source)), type(error).__name__, error)


def create_3d_point_cloud_plot(las_file, max_points=100000, color_by='z'):
    """
    Create a 3D plotly visualization of a point cloud
    
    Args:
        las_file: Path to LAS/LAZ file
        max_points: Maximum number of points to display (for performance)
        color_by: Attribute to color by ('z', 'intensity', 'classification')
    
    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import laspy
        las = laspy.read(las_file)
        
        # Sample points if too many
        n_points = len(las.points)
        if n_points > max_points:
            indices = np.random.choice(n_points, max_points, replace=False)
            x = las.x[indices]
            y = las.y[indices]
            z = las.z[indices]
            
            if color_by == 'intensity' and hasattr(las, 'intensity'):
                color_values = las.intensity[indices]
            elif color_by == 'classification' and hasattr(las, 'classification'):
                color_values = las.classification[indices]
            else:
                color_values = z
        else:
            x = las.x
            y = las.y
            z = las.z
            
            if color_by == 'intensity' and hasattr(las, 'intensity'):
                color_values = las.intensity
            elif color_by == 'classification' and hasattr(las, 'classification'):
                color_values = las.classification
            else:
                color_values = z
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=1,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_by.capitalize())
            ),
            hovertemplate='<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Z:</b> %{z:.2f}<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title=f'Point Cloud Visualization ({len(x):,} points)',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    except Exception as error:
        _plot_failed("3D point cloud plot", las_file, error)
        return None


def create_height_distribution_plot(tree_data_csv):
    """Create a histogram of tree heights"""
    try:
        df = pd.read_csv(tree_data_csv)
        
        if 'Height' in df.columns:
            fig = px.histogram(
                df, 
                x='Height',
                nbins=30,
                title='Tree Height Distribution',
                labels={'Height': 'Height (m)', 'count': 'Number of Trees'},
                color_discrete_sequence=['forestgreen']
            )
            
            fig.update_layout(
                xaxis_title='Height (m)',
                yaxis_title='Number of Trees',
                showlegend=False,
                height=400
            )
            
            return fig
    except Exception as error:
        _plot_failed("Height distribution plot", tree_data_csv, error)
        return None


def create_dbh_distribution_plot(tree_data_csv):
    """Create a histogram of DBH values"""
    try:
        df = pd.read_csv(tree_data_csv)
        
        if 'DBH' in df.columns:
            # Filter out invalid DBH values
            df_filtered = df[df['DBH'] > 0]
            
            fig = px.histogram(
                df_filtered,
                x='DBH',
                nbins=30,
                title='DBH Distribution',
                labels={'DBH': 'DBH (cm)', 'count': 'Number of Trees'},
                color_discrete_sequence=['saddlebrown']
            )
            
            fig.update_layout(
                xaxis_title='DBH (cm)',
                yaxis_title='Number of Trees',
                showlegend=False,
                height=400
            )
            
            return fig
    except Exception as error:
        _plot_failed("DBH distribution plot", tree_data_csv, error)
        return None


def create_tree_map_plot(tree_data_csv):
    """Create a 2D map of tree locations colored by height"""
    try:
        df = pd.read_csv(tree_data_csv)

        # FSCT writes the stem base coordinates as x_tree_base / y_tree_base
        # (see tree_data_dict in scripts/measure.py). This used to look for
        # 'X' and 'Y', which tree_data.csv has never contained, so the map
        # silently never rendered.
        x_col = 'x_tree_base' if 'x_tree_base' in df.columns else 'X'
        y_col = 'y_tree_base' if 'y_tree_base' in df.columns else 'Y'

        if not all(col in df.columns for col in [x_col, y_col, 'Height']):
            return None

        has_dbh = 'DBH' in df.columns
        # A zero or negative size column makes plotly raise, so only trees
        # with a real DBH measurement can drive marker size.
        size_col = 'DBH' if has_dbh and (df['DBH'] > 0).all() else None

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color='Height',
            size=size_col,
            title='Tree Location Map',
            labels={x_col: 'X Coordinate (m)', y_col: 'Y Coordinate (m)', 'Height': 'Height (m)'},
            color_continuous_scale='Viridis',
            hover_data=['Height', 'DBH'] if has_dbh else ['Height']
        )

        fig.update_layout(
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            height=600
        )

        fig.update_xaxes(scaleanchor="y", scaleratio=1)

        return fig
    except Exception as error:
        _plot_failed("Tree map plot", tree_data_csv, error)
        return None


def create_dbh_height_scatter(tree_data_csv):
    """Create a scatter plot of DBH vs Height"""
    try:
        df = pd.read_csv(tree_data_csv)
        
        if 'DBH' in df.columns and 'Height' in df.columns:
            # Filter valid values
            df_filtered = df[(df['DBH'] > 0) & (df['Height'] > 0)]
            
            # plotly's 'ols' trendline needs statsmodels, which is not a hard
            # dependency. Fall back to a plain scatter rather than returning
            # None (which rendered nothing at all) when it is absent.
            try:
                import statsmodels  # noqa: F401
                trendline = 'ols'
            except ImportError:
                trendline = None

            fig = px.scatter(
                df_filtered,
                x='DBH',
                y='Height',
                title='DBH vs Height Relationship',
                labels={'DBH': 'DBH (cm)', 'Height': 'Height (m)'},
                opacity=0.6,
                trendline=trendline
            )
            
            fig.update_layout(
                xaxis_title='DBH (cm)',
                yaxis_title='Height (m)',
                height=500
            )
            
            return fig
    except Exception as error:
        _plot_failed("DBH vs height scatter", tree_data_csv, error)
        return None


def create_summary_stats_cards(plot_summary_csv):
    """Extract and format summary statistics"""
    try:
        df = pd.read_csv(plot_summary_csv)
        
        stats = {}

        # These must match the headers written by scripts/preprocessing.py and
        # filled in by scripts/measure.py. The previous mapping used names like
        # 'Number of Trees' and 'Total Basal Area' that FSCT never emits, so
        # almost every card came back empty.
        metric_mapping = {
            'Num Trees in Plot': 'num_trees',
            'Mean Height': 'mean_height',
            'Mean DBH': 'mean_dbh',
            'Stems/ha': 'stems_per_ha',
            'Plot Area': 'plot_area_ha',
            'Total Volume 1': 'total_volume',
        }

        for col_name, key in metric_mapping.items():
            if col_name in df.columns:
                value = df[col_name].values[0]
                if not pd.isna(value):
                    stats[key] = value

        return stats

    except Exception as error:
        _plot_failed("Summary statistics", plot_summary_csv, error)
        return {}
