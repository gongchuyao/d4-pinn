"""
Base D4-PINN Model (Group Averaging)
=====================================
This is the simplest implementation of D4-PINN using group averaging.
The idea is simple:
1. For any input, we generate all 8 D4 transformed versions
2. We pass all transformed inputs through the same standard network
3. We average the outputs to get a D4-invariant prediction

This approach requires NO specialized libraries and works with pure PyTorch.
It automatically enforces D4 symmetry without modifying the network weights.
"""

import torch
import torch.nn as nn
from src.core.d4_transforms import get_all_d4_transforms_2d

class BaseD4PINN(nn.Module):
    """
    Base D4-PINN model using group averaging technique.
    This is the recommended starting point for beginners, as it's:
    - Simple to understand
    - Pure PyTorch, no extra dependencies
    - Highly efficient and easy to debug
    """
    def __init__(self, hidden_dim: int = 100):
        """
        Initialize the model.
        
        Args:
            hidden_dim: Dimension of hidden layers in the MLP
        """
        super().__init__()
        # Standard MLP network - nothing special!
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with group averaging.
        
        Args:
            x: Input coordinates tensor of shape (N, 2)
        
        Returns:
            D4-invariant prediction of shape (N,)
        """
        # Get all 8 transformed versions of the input
        x_transforms = get_all_d4_transforms_2d(x)
        
        # Pass all transformed inputs through the shared network
        outs = []
        for x_trans in x_transforms:
            outs.append(self.net(x_trans))
        
        # Average the outputs - this automatically guarantees D4 invariance!
        return torch.stack(outs).mean(dim=0).squeeze()
