"""
Custom Matplotlib styling for SignalLab.

This module provides premium dark theme styling and color schemes to ensure
consistent, professional-looking plots across the application.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def set_custom_style(dark_mode=True):
    """
    Apply custom "premium" styling to Matplotlib plots.
    """
    if dark_mode:
        # Dark background style
        plt.style.use('dark_background')
        
        # Custom colors
        text_color = '#E0E0E0'
        grid_color = '#333333'
        accent_color = '#00ADB5' # Teal-ish
        
        mpl.rcParams.update({
            'figure.facecolor': '#0E1117', # Matches Streamlit dark theme often
            'axes.facecolor': '#0E1117',
            'axes.edgecolor': grid_color,
            'axes.labelcolor': text_color,
            'xtick.color': text_color,
            'ytick.color': text_color,
            'text.color': text_color,
            'grid.color': grid_color,
            'grid.alpha': 0.6,
            'lines.linewidth': 2,
            'lines.color': accent_color,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Inter', 'Arial', 'sans-serif'],
        })
    else:
        # Light mode (clean, minimal)
        plt.style.use('seaborn-v0_8-whitegrid')
        mpl.rcParams.update({
             'figure.facecolor': '#FFFFFF',
             'axes.facecolor': '#FFFFFF',
             'grid.alpha': 0.3,
             'lines.linewidth': 2,
        })

def get_plot_colors():
    """Return a list of premium colors for multiple signals."""
    return ['#00ADB5', '#FF2E63', '#FCE38A', '#95E1D3']
