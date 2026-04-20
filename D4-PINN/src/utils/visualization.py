"""
Visualization Utilities
========================
This module provides common visualization utilities for scientific plotting,
including style settings for publication-quality figures.
"""

import matplotlib.pyplot as plt
import numpy as np

def set_scientific_plot_style():
    """
    Set global matplotlib parameters for scientific publication-quality plots.
    This configures fonts, DPI, and other styling options to match SCI journal standards.
    """
    plt.rcParams.update({
        # Font settings: Times New Roman for scientific publications
        'font.family': 'Times New Roman',
        'font.size': 12,
        'mathtext.fontset': 'stix',
        # Figure settings
        'figure.dpi': 300,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        # Tick settings
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        # Legend settings
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

def set_seaborn_plot_style():
    """
    Set seaborn-style plot settings for more modern-looking figures.
    """
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 14,
        'figure.dpi': 150,
    })

def save_figure(fig, filename: str, dpi: int = 300):
    """
    Save figure in both PNG and PDF formats for publication.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename without extension
        dpi: DPI for raster output
    """
    # Save as high-resolution PNG
    fig.savefig(f'{filename}.png', dpi=dpi, bbox_inches='tight')
    # Save as vector PDF for publication
    fig.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_solution_comparison(X: np.ndarray, Y: np.ndarray, u_true: np.ndarray, 
                            u_pred: np.ndarray, error: np.ndarray, 
                            save_path: str = 'solution_comparison'):
    """
    Plot standard solution comparison figure: exact, prediction, error.
    
    Args:
        X: X meshgrid
        Y: Y meshgrid
        u_true: Exact solution
        u_pred: Predicted solution
        error: Absolute error
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Exact solution
    im1 = axes[0].contourf(X, Y, u_true, 50, cmap='jet')
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_title('Exact Solution (D4 Symmetric)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    # Prediction
    im2 = axes[1].contourf(X, Y, u_pred, 50, cmap='jet')
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_title('D4-PINN Predicted Solution')
    axes[1].set_xlabel('x')
    
    # Error
    im3 = axes[2].contourf(X, Y, error, 50, cmap='jet')
    fig.colorbar(im3, ax=axes[2])
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    
    plt.tight_layout()
    save_figure(fig, save_path)

def plot_convergence_curves(loss_history: list, error_history: list,
                           save_path: str = 'convergence_curve'):
    """
    Plot standard convergence curves: loss and error over iterations.
    
    Args:
        loss_history: List of loss values
        error_history: List of error values
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Loss curve
    axes[0].semilogy(loss_history)
    axes[0].set_xlabel('Iteration (×100)')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss Convergence')
    axes[0].grid(True)
    
    # Error curve
    axes[1].semilogy(error_history)
    axes[1].set_xlabel('Iteration (×100)')
    axes[1].set_ylabel('L2 Relative Error')
    axes[1].set_title('Solution Error Convergence')
    axes[1].grid(True)
    
    plt.tight_layout()
    save_figure(fig, save_path)
