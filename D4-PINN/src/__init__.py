"""
D4-PINN: Dihedral Group Symmetry Preserving Physics-Informed Neural Networks
============================================================================
A PyTorch implementation of D4 symmetry preserving PINNs for solving
partial differential equations with square symmetry.

This package provides:
- Multiple model implementations: base group averaging, optimized, equivariant, ESCNN
- Various PDE problems: Poisson, Ginzburg-Landau, Allen-Cahn, 3D Poisson
- Publication-quality visualization tools
- Comprehensive evaluation metrics

Paper: [Your Paper Title Here]
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core models
from .models.base_d4_pinn import BaseD4PINN
from .models.optimized_d4_pinn import OptimizedD4PINN
from .models.d4_pinn_equivariant import D4PINNEquivariant, D4EquivariantLinear
from .models.d4_pinn_escnn_full import D4PINNESCNNFull
from .models.d4_pinn_escnn_standalone import D4PINNESCNNStandalone

# Core functionality
from .core.d4_transforms import d4_transform_2d, d4_transform_3d, get_all_d4_transforms_2d, get_all_d4_transforms_3d
from .core.pde_problems import (
    poisson_2d, ginzburg_landau_2d, allen_cahn_2d, poisson_3d,
    generate_sample_points_2d, generate_sample_points_3d
)

# Utilities
from .utils.metrics import compute_l2_error, compute_linf_error, compute_symmetry_error_2d, compute_training_metrics
from .utils.visualization import set_scientific_plot_style, set_seaborn_plot_style, save_figure, plot_solution_comparison, plot_convergence_curves

__all__ = [
    # Models
    'BaseD4PINN',
    'OptimizedD4PINN',
    'D4PINNEquivariant',
    'D4EquivariantLinear',
    'D4PINNESCNNFull',
    'D4PINNESCNNStandalone',
    
    # Core
    'd4_transform_2d',
    'd4_transform_3d',
    'get_all_d4_transforms_2d',
    'get_all_d4_transforms_3d',
    'poisson_2d',
    'ginzburg_landau_2d',
    'allen_cahn_2d',
    'poisson_3d',
    'generate_sample_points_2d',
    'generate_sample_points_3d',
    
    # Utils
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
