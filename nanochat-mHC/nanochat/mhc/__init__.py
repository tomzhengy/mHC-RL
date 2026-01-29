"""
mHC (manifold-constrained hyper-connections) implementations.

Two variants:
- static: reference implementation with per-layer parameters (stable, proven)
- dynamic: experimental per-token routing via projections (more expressive, harder to train)
"""

from nanochat.mhc.static.mhc import StaticMHC, expand_streams, reduce_streams
from nanochat.mhc.dynamic.mhc import DynamicMHC

# backwards compatibility: default to dynamic (matches original mhc.py)
MHC = DynamicMHC

__all__ = [
    "StaticMHC",
    "DynamicMHC",
    "MHC",
    "expand_streams",
    "reduce_streams",
]
