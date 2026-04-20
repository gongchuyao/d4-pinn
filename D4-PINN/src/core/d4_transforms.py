"""
D4 Group Transformation Functions
===================================
This module implements the 8 symmetry transformations of the D4 dihedral group,
which includes:
- 4 rotations: 0°, 90°, 180°, 270°
- 4 reflections: flip + the above rotations

These transformations are used to enforce symmetry in PINN models,
ensuring that the solution preserves the D4 symmetry of the physical problem.
"""

import torch
import numpy as np

def d4_transform_2d(x: torch.Tensor, op: int) -> torch.Tensor:
    """
    Apply D4 group transformation to 2D input coordinates.
    
    Args:
        x: Input tensor of shape (N, 2), where each row is (x, y) coordinate
        op: Operation index, 0-7 corresponding to the 8 D4 group actions
    
    Returns:
        Transformed coordinates tensor of the same shape as input
    """
    if op == 0: 
        # Identity transformation (no change)
        return x
    elif op == 1: 
        # 90 degree rotation
        return torch.stack([-x[:,1], x[:,0]], dim=1)
    elif op == 2: 
        # 180 degree rotation
        return torch.stack([-x[:,0], -x[:,1]], dim=1)
    elif op == 3: 
        # 270 degree rotation
        return torch.stack([x[:,1], -x[:,0]], dim=1)
    elif op == 4: 
        # Horizontal reflection (flip x axis)
        return torch.stack([-x[:,0], x[:,1]], dim=1)
    elif op == 5: 
        # Horizontal reflection + 90 rotation
        return torch.stack([-x[:,1], -x[:,0]], dim=1)
    elif op == 6: 
        # Horizontal reflection + 180 rotation
        return torch.stack([x[:,0], -x[:,1]], dim=1)
    elif op == 7: 
        # Horizontal reflection + 270 rotation
        return torch.stack([x[:,1], x[:,0]], dim=1)
    else:
        raise ValueError(f"Invalid operation index {op}, must be 0-7")

def d4_transform_3d(x: torch.Tensor, op: int) -> torch.Tensor:
    """
    Apply D4 group transformation to 3D input coordinates.
    The transformation is applied only on the xy plane, z coordinate remains unchanged.
    
    Args:
        x: Input tensor of shape (N, 3), where each row is (x, y, z) coordinate
        op: Operation index, 0-7 corresponding to the 8 D4 group actions
    
    Returns:
        Transformed coordinates tensor of the same shape as input
    """
    if op == 0: 
        return x
    elif op == 1: 
        return torch.stack([-x[:,1], x[:,0], x[:,2]], dim=1)
    elif op == 2: 
        return torch.stack([-x[:,0], -x[:,1], x[:,2]], dim=1)
    elif op == 3: 
        return torch.stack([x[:,1], -x[:,0], x[:,2]], dim=1)
    elif op == 4: 
        return torch.stack([-x[:,0], x[:,1], x[:,2]], dim=1)
    elif op == 5: 
        return torch.stack([-x[:,1], -x[:,0], x[:,2]], dim=1)
    elif op == 6: 
        return torch.stack([x[:,0], -x[:,1], x[:,2]], dim=1)
    elif op == 7: 
        return torch.stack([x[:,1], x[:,0], x[:,2]], dim=1)
    else:
        raise ValueError(f"Invalid operation index {op}, must be 0-7")

def get_all_d4_transforms_2d(x: torch.Tensor) -> list[torch.Tensor]:
    """
    Get all 8 transformed versions of the input 2D coordinates.
    
    Args:
        x: Input tensor of shape (N, 2)
    
    Returns:
        List of 8 transformed tensors, each of shape (N, 2)
    """
    return [d4_transform_2d(x, op) for op in range(8)]

def get_all_d4_transforms_3d(x: torch.Tensor) -> list[torch.Tensor]:
    """
    Get all 8 transformed versions of the input 3D coordinates.
    
    Args:
        x: Input tensor of shape (N, 3)
    
    Returns:
        List of 8 transformed tensors, each of shape (N, 3)
    """
    return [d4_transform_3d(x, op) for op in range(8)]
