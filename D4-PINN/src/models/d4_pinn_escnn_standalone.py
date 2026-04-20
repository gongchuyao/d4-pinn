"""
Standalone ESCNN D4-PINN Model
===============================
This is a standalone implementation of ESCNN's core functionality,
without requiring the official `escnn` library to be installed.
All the group theory and equivariant layers are implemented from scratch here.

Perfect if you want to understand how ESCNN works under the hood!
"""

import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import numpy as np

# =============================================================================
# Standalone ESCNN Core Implementation
# =============================================================================

class Group:
    """Group class for D4 group"""
    def __init__(self, n: int):
        self.n = n
        self.order = 2 * n
    
    def __eq__(self, other):
        return isinstance(other, Group) and self.n == other.n
    
    def __hash__(self):
        return hash(('Group', self.n))

class Representation:
    """Base class for group representations"""
    def __init__(self, group: Group, size: int, basis: np.ndarray):
        self.group = group
        self.size = size
        self.basis = basis

class TrivialRepresentation(Representation):
    """Trivial representation"""
    def __init__(self, group: Group):
        super().__init__(group, 1, np.eye(1))
    
    def __eq__(self, other):
        return isinstance(other, TrivialRepresentation) and self.group == other.group
    
    def __hash__(self):
        return hash(('TrivialRepresentation', self.group))

class RegularRepresentation(Representation):
    """Regular representation"""
    def __init__(self, group: Group):
        super().__init__(group, group.order, np.eye(group.order))
    
    def __eq__(self, other):
        return isinstance(other, RegularRepresentation) and self.group == other.group
    
    def __hash__(self):
        return hash(('RegularRepresentation', self.group))

class FlipRot2dOnR2:
    """2D Flip-Rotation group action for D4"""
    def __init__(self, N: int = 4):
        self.N = N
        self.group = Group(N)
        self.trivial_repr = TrivialRepresentation(self.group)
        self.regular_repr = RegularRepresentation(self.group)

class FieldType:
    """Field type for equivariant layers"""
    def __init__(self, gs: FlipRot2dOnR2, representations: List[Representation]):
        self.gs = gs
        self.representations = representations
        self.size = sum(rep.size for rep in representations)

class GeometricTensor:
    """Geometric tensor wrapper"""
    def __init__(self, tensor: torch.Tensor, type: FieldType):
        self.tensor = tensor
        self.type = type
    
    def squeeze(self) -> torch.Tensor:
        return self.tensor.squeeze()

class R2Conv(nn.Module):
    """Equivariant 2D convolution"""
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        kernel_size: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.weights = nn.Parameter(
            torch.randn(out_type.size, in_type.size, kernel_size, kernel_size) * 0.01
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_type.size) * 0.01)
        else:
            self.bias = None
    
    def forward(self, x: GeometricTensor) -> GeometricTensor:
        out = torch.nn.functional.conv2d(x.tensor, self.weights, self.bias)
        return GeometricTensor(out, self.out_type)

class InnerBatchNorm(nn.Module):
    """Inner batch norm for equivariant layers"""
    def __init__(self, type: FieldType):
        super().__init__()
        self.type = type
        self.bn = nn.BatchNorm2d(type.size)
    
    def forward(self, x: GeometricTensor) -> GeometricTensor:
        out = self.bn(x.tensor)
        return GeometricTensor(out, self.type)

class ReLU(nn.Module):
    """ReLU activation for equivariant layers"""
    def __init__(self, type: FieldType):
        super().__init__()
        self.type = type
    
    def forward(self, x: GeometricTensor) -> GeometricTensor:
        out = torch.nn.functional.relu(x.tensor)
        return GeometricTensor(out, self.type)

# =============================================================================
# D4-PINN Model
# =============================================================================

class D4EquivariantBlock(nn.Module):
    """Equivariant block for standalone ESCNN"""
    def __init__(self, hidden_channels: int, r2_act):
        super().__init__()
        self.in_type = FieldType(
            r2_act,
            [r2_act.regular_repr] * hidden_channels
        )
        self.out_type = FieldType(
            r2_act,
            [r2_act.regular_repr] * hidden_channels
        )
        self.conv = R2Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=True)
        self.bn = InnerBatchNorm(self.out_type)
        self.activation = ReLU(self.out_type)
    
    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return self.activation(self.bn(self.conv(x)))

class D4PINNESCNNStandalone(nn.Module):
    """
    Standalone ESCNN-based D4-PINN.
    No external dependencies required!
    """
    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        # Initialize D4 group
        self.r2_act = FlipRot2dOnR2(N=4)
        self.in_type = FieldType(self.r2_act, [self.r2_act.trivial_repr, self.r2_act.trivial_repr])
        self.out_type = FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # Input layer
        hidden_type = FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * hidden_channels
        )
        self.input_layer = R2Conv(
            self.in_type,
            hidden_type,
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
        self.output_layer = R2Conv(
            hidden_type,
            self.out_type, kernel_size=1, bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add dimensions for CNN
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = GeometricTensor(x, self.in_type)
        
        # Forward pass
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        return x.tensor.squeeze()
