"""
PDE Problem Definitions
========================
This module defines various partial differential equation (PDE) problems
that can be solved with D4-PINN, including:
- 2D Semilinear Poisson Equation
- 2D Ginzburg-Landau Equation
- 2D Allen-Cahn Equation
- 3D Poisson Equation

Each problem provides the exact solution and the source term f,
which can be used for training and validation.
"""

import torch
import numpy as np

# =============================================================================
# 2D Problems
# =============================================================================

def poisson_2d() -> tuple:
    """
    2D Semilinear Poisson Equation:
    -Δu + u³ = f(x, y)
    
    The exact solution is D4 symmetric:
    u(x,y) = (1 - x²)(1 - y²)(1 + 0.5(x² + y²))
    
    Returns:
        u_exact: Function that computes the exact solution
        f_source: Function that computes the source term f
        problem_name: Name of the problem
    """
    def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact solution for 2D Poisson problem"""
        return (1 - x**2) * (1 - y**2) * (1 + 0.5 * (x**2 + y**2))
    
    def f_source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Source term f for 2D Poisson problem"""
        x.requires_grad = True
        y.requires_grad = True
        u = u_exact(x, y)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        return (-u_xx - u_yy + u**3).detach()
    
    return u_exact, f_source, "Semilinear Poisson"

def ginzburg_landau_2d() -> tuple:
    """
    2D Steady Ginzburg-Landau Equation:
    -Δu + κ² u(u² - 1) = f(x, y)
    
    Exact solution:
    u(x,y) = 1 - exp(-(x² + y²)/2)
    
    Returns:
        u_exact: Function that computes the exact solution
        f_source: Function that computes the source term f
        problem_name: Name of the problem
    """
    kappa = 1.0
    
    def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact solution for Ginzburg-Landau problem"""
        r2 = x**2 + y**2
        return 1 - torch.exp(-r2/2)
    
    def f_source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Source term f for Ginzburg-Landau problem"""
        x.requires_grad = True
        y.requires_grad = True
        u = u_exact(x, y)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        return (-u_xx - u_yy + kappa**2 * u * (u**2 - 1)).detach()
    
    return u_exact, f_source, "Steady Ginzburg-Landau"

def allen_cahn_2d() -> tuple:
    """
    2D Steady Allen-Cahn Equation:
    -Δu + (u³ - u)/ε² = f(x, y)
    
    Exact solution:
    u(x,y) = tanh((0.5 - r)/(√2 ε))
    
    Returns:
        u_exact: Function that computes the exact solution
        f_source: Function that computes the source term f
        problem_name: Name of the problem
    """
    eps = 0.1
    
    def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact solution for Allen-Cahn problem"""
        r = torch.sqrt(x**2 + y**2)
        return torch.tanh((0.5 - r)/(np.sqrt(2)*eps))
    
    def f_source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Source term f for Allen-Cahn problem"""
        x.requires_grad = True
        y.requires_grad = True
        u = u_exact(x, y)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        return (-u_xx - u_yy + (u**3 - u)/eps**2).detach()
    
    return u_exact, f_source, "Steady Allen-Cahn"

# =============================================================================
# 3D Problems
# =============================================================================

def poisson_3d() -> tuple:
    """
    3D Linear Poisson Equation:
    -Δu = f(x, y, z)
    
    Exact solution:
    u(x,y,z) = (1 - x²)(1 - y²)(1 - z²)
    
    Returns:
        u_exact: Function that computes the exact solution
        f_source: Function that computes the source term f
        problem_name: Name of the problem
    """
    def u_exact(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Exact solution for 3D Poisson problem"""
        return (1 - x**2) * (1 - y**2) * (1 - z**2)
    
    def f_source(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Source term f for 3D Poisson problem"""
        return 6 - 2*(x**2 + y**2 + z**2)
    
    return u_exact, f_source, "3D Poisson"

# =============================================================================
# Utility functions for sampling points
# =============================================================================

def generate_sample_points_2d(N_interior: int = 2000, N_boundary: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random sampling points for 2D problems.
    Includes interior points and boundary points.
    
    Args:
        N_interior: Number of interior sampling points
        N_boundary: Number of boundary sampling points
    
    Returns:
        x_interior: Interior points tensor of shape (N_interior, 2)
        x_boundary: Boundary points tensor of shape (N_boundary, 2)
    """
    # Interior points: uniform in [-1, 1] x [-1, 1]
    x_interior = torch.FloatTensor(N_interior, 2).uniform_(-1, 1)
    
    # Boundary points: on the four edges of the square
    x_top = torch.FloatTensor(N_boundary//4, 2).uniform_(-1, 1)
    x_top[:, 1] = 1.0
    
    x_bottom = torch.FloatTensor(N_boundary//4, 2).uniform_(-1, 1)
    x_bottom[:, 1] = -1.0
    
    x_left = torch.FloatTensor(N_boundary//4, 2).uniform_(-1, 1)
    x_left[:, 0] = -1.0
    
    x_right = torch.FloatTensor(N_boundary//4, 2).uniform_(-1, 1)
    x_right[:, 0] = 1.0
    
    x_boundary = torch.cat([x_top, x_bottom, x_left, x_right], dim=0)
    
    return x_interior, x_boundary

def generate_sample_points_3d(N_interior: int = 2000, N_boundary: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random sampling points for 3D problems.
    Includes interior points and boundary points on the 6 faces of the cube.
    
    Args:
        N_interior: Number of interior sampling points
        N_boundary: Number of boundary sampling points
    
    Returns:
        x_interior: Interior points tensor of shape (N_interior, 3)
        x_boundary: Boundary points tensor of shape (N_boundary, 3)
    """
    # Interior points: uniform in [-1, 1]^3
    x_interior = torch.FloatTensor(N_interior, 3).uniform_(-1, 1)
    
    # Boundary points: on the six faces of the cube
    x_bc = []
    
    # x = 1 face
    x_bc.append(torch.FloatTensor(N_boundary//6, 3).uniform_(-1,1))
    x_bc[-1][:,0] = 1.0
    
    # x = -1 face
    x_bc.append(torch.FloatTensor(N_boundary//6, 3).uniform_(-1,1))
    x_bc[-1][:,0] = -1.0
    
    # y = 1 face
    x_bc.append(torch.FloatTensor(N_boundary//6, 3).uniform_(-1,1))
    x_bc[-1][:,1] = 1.0
    
    # y = -1 face
    x_bc.append(torch.FloatTensor(N_boundary//6, 3).uniform_(-1,1))
    x_bc[-1][:,1] = -1.0
    
    # z = 1 face
    x_bc.append(torch.FloatTensor(N_boundary//6, 3).uniform_(-1,1))
    x_bc[-1][:,2] = 1.0
    
    # z = -1 face
    x_bc.append(torch.FloatTensor(N_boundary//6, 3).uniform_(-1,1))
    x_bc[-1][:,2] = -1.0
    
    x_boundary = torch.cat(x_bc)
    
    return x_interior, x_boundary
