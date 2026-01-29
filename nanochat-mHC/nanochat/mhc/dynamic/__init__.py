"""Dynamic mHC implementation (per-token routing via projections)."""

from nanochat.mhc.dynamic.mhc import DynamicMHC, expand_streams, reduce_streams, sinkhorn_log

__all__ = ["DynamicMHC", "expand_streams", "reduce_streams", "sinkhorn_log"]
