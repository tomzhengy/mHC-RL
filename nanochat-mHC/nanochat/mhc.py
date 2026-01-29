"""
Backwards compatibility shim - imports from nanochat.mhc.dynamic

For new code, import directly from:
- nanochat.mhc.static for StaticMHC (reference implementation)
- nanochat.mhc.dynamic for DynamicMHC (experimental per-token routing)
"""

# re-export everything from dynamic for backwards compatibility
from nanochat.mhc.dynamic.mhc import (
    DynamicMHC,
    sinkhorn_log,
    expand_streams,
    reduce_streams,
)

# backwards compatibility alias
MHC = DynamicMHC

__all__ = [
    "DynamicMHC",
    "MHC",
    "sinkhorn_log",
    "expand_streams",
    "reduce_streams",
]
