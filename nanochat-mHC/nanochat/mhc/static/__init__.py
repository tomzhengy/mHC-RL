"""Static mHC implementation (reference-style, per-layer parameters)."""

from nanochat.mhc.static.mhc import StaticMHC, expand_streams, reduce_streams, sinkhorn_log

__all__ = ["StaticMHC", "expand_streams", "reduce_streams", "sinkhorn_log"]
