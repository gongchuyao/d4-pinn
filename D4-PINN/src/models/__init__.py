"""
D4-PINN Model Implementations
==============================
This module contains various implementations of D4-PINN models:

1. BaseD4PINN: Simplest group averaging approach, pure PyTorch, no dependencies
2. OptimizedD4PINN: Batched group averaging, ~2-3x faster training
3. D4PINNEquivariant: True equivariant layers, theoretically rigorous
4. D4PINNESCNNFull: Full implementation using official ESCNN library
5. D4PINNESCNNStandalone: Standalone ESCNN implementation, no external dependencies
"""

from .base_d4_pinn import BaseD4PINN
from .optimized_d4_pinn import OptimizedD4PINN
from .d4_pinn_equivariant import D4PINNEquivariant, D4EquivariantLinear
from .d4_pinn_escnn_full import D4PINNESCNNFull
from .d4_pinn_escnn_standalone import D4PINNESCNNStandalone

__all__ = [
    'BaseD4PINN',
    'OptimizedD4PINN',
    'D4PINNEquivariant',
    'D4EquivariantLinear',
    'D4PINNESCNNFull',
    'D4PINNESCNNStandalone',
]
