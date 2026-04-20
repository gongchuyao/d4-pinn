"""
Metrics and Evaluation Functions
=================================
This module provides common evaluation metrics for PINN models,
including error calculations and symmetry error metrics.
"""

import torch
import numpy as np
from src.core.d4_transforms import d4_transform_2d

def compute_l2_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """
    Compute L2 relative error between prediction and ground truth.
    
    Args:
        u_pred: Predicted solution array
        u_true: Ground truth solution array
    
    Returns:
        L2 relative error: ||u_pred - u_true||_2 / ||u_true||_2
    """
    return np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)

def compute_linf_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """
    Compute L∞ relative error between prediction and ground truth.
    
    Args:
        u_pred: Predicted solution array
        u_true: Ground truth solution array
    
    Returns:
        L∞ relative error: max|u_pred - u_true| / max|u_true|
    """
    return np.max(np.abs(u_pred - u_true)) / np.max(np.abs(u_true))

def compute_symmetry_error_2d(model, x_grid: torch.Tensor, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute symmetry error for D4-PINN model.
    This measures how much the prediction violates D4 symmetry,
    i.e., how different the predictions are after applying symmetry transformations.
    
    Args:
        model: Trained D4-PINN model
        x_grid: Grid coordinates tensor of shape (N*N, 2)
        X: X meshgrid
        Y: Y meshgrid
    
    Returns:
        Symmetry error array of same shape as X/Y
    """
    # Get the prediction
    u_pred = model(x_grid).detach().numpy().reshape(X.shape)
    
    # Compute average error across all 8 transformations
    sym_err = np.zeros_like(u_pred)
    dx = X[0,1] - X[0,0]
    dy = Y[1,0] - Y[0,0]
    
    for i in range(8):
        x_trans = d4_transform_2d(x_grid, i).numpy()
        x_t = x_trans[:,0].reshape(X.shape)
        y_t = x_trans[:,1].reshape(Y.shape)
        
        # Compute indices for interpolation
        idx_x = np.clip(np.round((x_t + 1) / dx).astype(int), 0, X.shape[1]-1)
        idx_y = np.clip(np.round((y_t + 1) / dy).astype(int), 0, Y.shape[0]-1)
        
        u_t = u_pred[idx_y, idx_x]
        sym_err += np.abs(u_t - u_pred)
    
    # Average over all transformations
    sym_err /= 8.0
    
    return sym_err

def compute_training_metrics(model, x_test: torch.Tensor, u_exact_fn) -> tuple[float, float]:
    """
    Compute training metrics: L2 and L∞ errors on test points.
    
    Args:
        model: Trained model
        x_test: Test points tensor
        u_exact_fn: Exact solution function
    
    Returns:
        l2_error: L2 relative error
        linf_error: L∞ relative error
    """
    u_test_pred = model(x_test).detach()
    u_test_true = u_exact_fn(x_test[:,0], x_test[:,1])
    
    l2_err = torch.norm(u_test_pred - u_test_true) / torch.norm(u_test_true)
    linf_err = torch.max(torch.abs(u_test_pred - u_test_true)) / torch.max(torch.abs(u_test_true))
    
    return l2_err.item(), linf_err.item()
