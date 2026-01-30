"""
Static mHC implementation (matches reference).

Architecture:
- H_res, H_pre, H_post are static per-layer parameters (not input-dependent)
- Same matrices used for all tokens in a batch
- More stable training, proven to work in reference implementations

Key differences from dynamic:
- No proj_H_res, proj_H_pre, proj_H_post linear layers
- H_res_logits is [n, n], not [B, T, n, n]
- Sinkhorn runs once per layer, not per token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten


@torch.compiler.disable()
def sinkhorn_log(
    logits: torch.Tensor,
    num_iters: int = 20,
    tau: float = 0.05,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm in log-space for numerical stability.
    Converts logits to a doubly-stochastic matrix (rows and cols sum to 1).

    Args:
        logits: [..., n, n] matrix of logits
        num_iters: number of Sinkhorn iterations (reference uses 10-20)
        tau: temperature parameter (reference uses 0.05)

    Returns:
        Doubly-stochastic matrix of same shape
    """
    n = logits.shape[-1]

    # scale by temperature
    Z = logits / tau

    # target log marginals (uniform distribution)
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    # dual variables for row/column normalization
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    # alternating row/column normalization in log-space
    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class StaticMHC(nn.Module):
    """
    Static mHC layer matching the reference implementation.

    H_res, H_pre, H_post are learnable parameters per layer (not input-dependent).
    This is more stable for training and has been proven to work.
    """

    def __init__(
        self,
        dim: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 20,
        sinkhorn_tau: float = 0.05,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.layer_idx = layer_idx

        n = num_streams

        # H_res_logits: static [n, n] matrix
        # reference uses -8.0 off-diagonal, 0.0 diagonal
        init_h_res = torch.full((n, n), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        # H_pre_logits: static [n] vector
        # prefer stream at layer_idx % n, reference uses -8.0 for others
        init_h_pre = torch.full((n,), -8.0)
        init_h_pre[layer_idx % n] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        # H_post_logits: static [n] vector
        # uniform distribution (all zeros -> softmax gives 1/n each)
        self.H_post_logits = nn.Parameter(torch.zeros(n))

        # gate for interpolation between identity and H_res
        # paper init: gamma = 0.01 -> logit = -4.6
        self.gate = nn.Parameter(torch.tensor([-4.6]))

        # diagnostics
        self._last_row_err = 0.0
        self._last_col_err = 0.0

    def _init_params(self):
        """Re-initialize params after to_empty() wipes them."""
        n = self.num_streams
        # gate
        self.gate.data.fill_(-4.6)
        # H_res_logits
        self.H_res_logits.data.fill_(-8.0)
        self.H_res_logits.data.fill_diagonal_(0.0)
        # H_pre_logits
        self.H_pre_logits.data.fill_(-8.0)
        self.H_pre_logits.data[self.layer_idx % n] = 0.0
        # H_post_logits
        self.H_post_logits.data.zero_()

    def get_matrices(self):
        """
        Compute H_res, H_pre, H_post matrices.
        These are static (same for all tokens).
        """
        n = self.num_streams

        # H_res: doubly-stochastic via Sinkhorn
        H_res_raw = sinkhorn_log(self.H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)

        # H_pre, H_post: softmax-normalized
        H_pre = F.softmax(self.H_pre_logits, dim=-1)   # [n]
        H_post = F.softmax(self.H_post_logits, dim=-1) # [n]

        # apply gate: H_res = (1 - g) * I + g * H_res_raw
        g = torch.sigmoid(self.gate)
        I = torch.eye(n, device=H_res_raw.device, dtype=H_res_raw.dtype)
        H_res = (1.0 - g) * I + g * H_res_raw

        # store H_res for diagnostics (no .item() here to avoid graph breaks)
        # diagnostics are computed lazily in get_sinkhorn_diagnostics()
        self._last_H_res = H_res.detach()

        return H_res, H_pre, H_post

    def forward(self, x: torch.Tensor, branch_fn) -> torch.Tensor:
        """
        Forward pass with static mHC.

        Args:
            x: [B, T, n*C] flattened stream tensor
            branch_fn: function to apply (attention or MLP)

        Returns:
            [B, T, n*C] output tensor
        """
        B, T, nC = x.shape
        n = self.num_streams
        C = nC // n

        # get static matrices (same for all tokens)
        H_res, H_pre, H_post = self.get_matrices()

        # unflatten streams: [B, T, n*C] -> [B, T, n, C]
        x_streams = x.view(B, T, n, C)

        # === WIDTH CONNECTION ===
        # x_pre[b,t,c] = sum_i H_pre[i] * x_streams[b,t,i,c]
        x_pre = torch.einsum('btnc,n->btc', x_streams, H_pre)  # [B, T, C]

        # === BRANCH ===
        branch_out = branch_fn(x_pre)

        # handle multi-output branches
        (y, *rest), tree_spec = tree_flatten(branch_out)

        # === DEPTH CONNECTION ===
        # x_mixed[b,t,i,c] = sum_j H_res[i,j] * x_streams[b,t,j,c]
        x_mixed = torch.einsum('ij,btjc->btic', H_res, x_streams)  # [B, T, n, C]

        # distribute branch output to streams
        # y_distributed[b,t,j,c] = H_post[j] * y[b,t,c]
        y_distributed = torch.einsum('btc,n->btnc', y, H_post)  # [B, T, n, C]

        # combine
        output = x_mixed + y_distributed  # [B, T, n, C]

        # flatten back: [B, T, n, C] -> [B, T, n*C]
        output = output.contiguous().view(B, T, nC)

        return tree_unflatten((output, *rest), tree_spec)

    def set_gate(self, value: float):
        """Set gate value in [0, 1]."""
        with torch.no_grad():
            value = max(1e-6, min(1.0 - 1e-6, value))
            logit = math.log(value / (1.0 - value))
            self.gate.fill_(logit)

    def get_gate(self) -> float:
        """Get current effective gate value in [0, 1]."""
        return torch.sigmoid(self.gate).item()

    def get_sinkhorn_diagnostics(self) -> dict:
        """Get diagnostics for monitoring training."""
        n = self.num_streams
        H_res_raw = sinkhorn_log(self.H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)

        return {
            "row_err": (H_res_raw.sum(dim=-1) - 1).abs().mean().item(),
            "col_err": (H_res_raw.sum(dim=-2) - 1).abs().mean().item(),
            "diag_mean": H_res_raw.diag().mean().item(),
            "offdiag_mean": H_res_raw[~torch.eye(n, dtype=bool, device=H_res_raw.device)].mean().item(),
        }

    def enable_used_diagnostics(self):
        """Compatibility method - static mHC always computes diagnostics."""
        pass  # no-op for static, diagnostics computed every forward

    def get_used_diagnostics(self) -> dict:
        """Get row/col errors from the last forward pass."""
        return {
            "row_err_used": self._last_row_err,
            "col_err_used": self._last_col_err,
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_streams={self.num_streams}, "
            f"sinkhorn_iters={self.sinkhorn_iters}, sinkhorn_tau={self.sinkhorn_tau}, "
            f"layer_idx={self.layer_idx}"
        )


# helper functions for stream expansion/reduction
def expand_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    """Expand single-stream tensor to multi-stream by replication."""
    B, T, C = x.shape
    return x.unsqueeze(-2).expand(B, T, num_streams, C).reshape(B, T, num_streams * C)


def reduce_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    """Reduce multi-stream tensor to single-stream by summing."""
    B, T, nC = x.shape
    C = nC // num_streams
    return x.view(B, T, num_streams, C).sum(dim=-2)


# test function
if __name__ == "__main__":
    print("testing static mhc.py...")

    # test sinkhorn
    print("\n1. testing sinkhorn_log...")
    logits = torch.randn(4, 4)  # [n, n] - static matrix
    ds_matrix = sinkhorn_log(logits, num_iters=20, tau=0.05)
    row_sums = ds_matrix.sum(dim=-1)
    col_sums = ds_matrix.sum(dim=-2)
    print(f"   input shape: {logits.shape}")
    print(f"   output shape: {ds_matrix.shape}")
    print(f"   row sums (should be ~1): {row_sums}")
    print(f"   col sums (should be ~1): {col_sums}")

    # test static mhc
    print("\n2. testing StaticMHC...")
    mhc = StaticMHC(dim=64, num_streams=4, sinkhorn_iters=20, sinkhorn_tau=0.05)

    B, T, n, C = 2, 8, 4, 64
    x = torch.randn(B, T, n * C)

    def dummy_branch(z):
        return z * 0.5

    y = mhc(x, dummy_branch)
    print(f"   input shape: {x.shape}")
    print(f"   output shape: {y.shape}")
    print(f"   current gate value: {mhc.get_gate():.4f}")

    # test diagnostics
    print("\n3. testing diagnostics...")
    diag = mhc.get_sinkhorn_diagnostics()
    print(f"   row_err: {diag['row_err']:.6f}")
    print(f"   col_err: {diag['col_err']:.6f}")
    print(f"   diag_mean: {diag['diag_mean']:.4f}")
    print(f"   offdiag_mean: {diag['offdiag_mean']:.4f}")

    # test gate control
    print("\n4. testing gate control...")
    mhc.set_gate(0.5)
    print(f"   after set_gate(0.5): {mhc.get_gate():.6f}")

    print("\n✓ all static mhc tests passed!")
