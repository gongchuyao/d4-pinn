"""
Full ESCNN D4-PINN Model
==========================
This model uses the official `escnn` library to implement equivariant CNN layers.
This is the standard implementation from the ESCNN paper.

Note: This requires the `escnn` library to be installed.
Install with: pip install escnn
"""

import torch
import torch.nn as nn
from escnn import gspaces, nn as enn

class D4EquivariantBlock(nn.Module):
    """
    Equivariant block using ESCNN library.
    """
    def __init__(self, hidden_channels: int, r2_act):
        super().__init__()
        # Field types for hidden layers
        self.in_type = enn.FieldType(r2_act, hidden_channels * [r2_act.regular_repr])
        self.out_type = enn.FieldType(r2_act, hidden_channels * [r2_act.regular_repr])
        
        # Equivariant convolution
        self.conv = enn.R2Conv(
            self.in_type, self.out_type, 
            kernel_size=1, padding=0, bias=True
        )
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.activation = enn.ReLU(self.out_type)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class D4PINNESCNNFull(nn.Module):
    """
    Full D4-PINN using official ESCNN library.
    """
    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        # Initialize D4 group action
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        
        # Input and output types
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.out_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # Input layer
        self.input_layer = enn.R2Conv(
            self.in_type, 
            enn.FieldType(self.r2_act, hidden_channels * [self.r2_act.regular_repr]), 
            kernel_size=1, bias=True
        )
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            D4EquivariantBlock(hidden_channels, self.r2_act),
            D4EquivariantBlock(hidden_channels, self.r2_act),
            D4EquivariantBlock(hidden_channels, self.r2_act),
            D4EquivariantBlock(hidden_channels, self.r2_act),
        )
        
        # Output layer
        self.output_layer = enn.R2Conv(
            enn.FieldType(self.r2_act, hidden_channels * [self.r2_act.regular_repr]), 
            self.out_type, kernel_size=1, bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input coordinates tensor of shape (N, 2)
        
        Returns:
            Prediction tensor of shape (N,)
        """
        # Add batch and spatial dimensions for CNN
        x = x.unsqueeze(-1).unsqueeze(-1)
        
        # Wrap into geometric tensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Forward through network
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        # Extract tensor
        return x.tensor.squeeze()
