"""
Core Functionality
==================
Core module containing the fundamental building blocks of D4-PINN:
- D4 group transformations
- PDE problem definitions
- Training utilities
"""

from .d4_transforms import d4_transform_2d, d4_transform_3d, get_all_d4_transforms_2d, get_all_d4_transforms_3d
from .pde_problems import (
    poisson_2d, ginzburg_landau_2d, allen_cahn_2d, poisson_3d,
    generate_sample_points_2d, generate_sample_points_3d
)

__all__ = [
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
]
