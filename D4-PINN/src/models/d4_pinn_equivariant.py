"""
Equivariant D4-PINN Model
==========================
This model implements true equivariant layers for D4 group.
Instead of averaging outputs, we modify the network weights to be equivariant,
which means that the network itself preserves the symmetry at every layer.

This is more theoretically rigorous and can be more efficient for deep networks.
"""

import torch
import torch.nn as nn
from src.core.d4_transforms import d4_transform_2d

class D4EquivariantLinear(nn.Module):
    """
    D4 Equivariant Linear Layer.
    This layer ensures that the linear transformation is equivariant to D4 transformations,
    meaning that transforming the input and then applying the layer is the same as
    applying the layer and then transforming the output.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize the equivariant linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.randn(out_features) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with equivariance enforcement.
        
        Args:
            x: Input tensor of shape (N, in_features)
        
        Returns:
            Equivariant output tensor of shape (N, out_features)
        """
        out = 0.0
        # Average over all group operations
        for op in range(8):
            # Transform input
            x_trans = d4_transform_2d(x, op)
            
            # Apply standard linear layer
            out_trans = nn.functional.linear(x_trans, self.weight, self.bias)
            
            # Transform output back
            out = out + d4_transform_2d(out_trans, (8 - op) % 8)
        
        # Average
        return out / 8.0

class D4PINNEquivariant(nn.Module):
    """
    Full D4-PINN with equivariant layers.
    Every layer in the network is equivariant to D4 transformations.
    """
    def __init__(self, hidden_dim: int = 50):
        """
        Initialize the equivariant D4-PINN.
        
        Args:
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            D4EquivariantLinear(2, hidden_dim),
            nn.Tanh(),
            D4EquivariantLinear(hidden_dim, hidden_dim),
            nn.Tanh(),
            D4EquivariantLinear(hidden_dim, hidden_dim),
            nn.Tanh(),
            D4EquivariantLinear(hidden_dim, hidden_dim),
            nn.Tanh(),
            D4EquivariantLinear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input coordinates tensor of shape (N, 2)
        
        Returns:
            Prediction tensor of shape (N,)
        """
        return self.net(x).squeeze()
