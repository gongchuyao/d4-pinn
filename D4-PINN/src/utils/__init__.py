"""
Utilities
=========
Utility module containing helper functions for:
- Error metrics calculation
- Scientific visualization
- Training helpers
"""

from .metrics import compute_l2_error, compute_linf_error, compute_symmetry_error_2d, compute_training_metrics
from .visualization import set_scientific_plot_style, set_seaborn_plot_style, save_figure, plot_solution_comparison, plot_convergence_curves

__all__ = [
    'compute_l2_error',
    'compute_linf_error',
    'compute_symmetry_error_2d',
    'compute_training_metrics',
    'set_scientific_plot_style',
    'set_seaborn_plot_style',
    'save_figure',
    'plot_solution_comparison',
    'plot_convergence_curves',
]
