"""shared mHC diagnostics functions for use across training scripts."""

import torch


def enable_mhc_diagnostics(model):
    """enable diagnostics capture for next forward pass."""
    if not hasattr(model, 'config') or not model.config.mhc_enabled:
        return
    for block in model.transformer.h:
        if hasattr(block, 'mhc_attn') and block.mhc_attn is not None:
            block.mhc_attn.enable_used_diagnostics()
        if hasattr(block, 'mhc_mlp') and block.mhc_mlp is not None:
            block.mhc_mlp.enable_used_diagnostics()


def collect_mhc_metrics(model):
    """collect mHC-specific metrics for logging."""
    metrics = {}

    # check if mHC is enabled
    if not hasattr(model, 'config') or not model.config.mhc_enabled:
        return metrics

    # stream embedding stats
    if hasattr(model, 'stream_embed') and model.stream_embed is not None:
        se = model.stream_embed.data
        metrics["mhc/stream_embed_norm"] = se.norm().item()
        metrics["mhc/stream_embed_std"] = se.std().item()
        # inter-stream similarity (how different are streams)
        if se.shape[0] > 1:
            se_norm = se / (se.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = se_norm @ se_norm.T
            # average off-diagonal similarity
            n = se.shape[0]
            off_diag = sim_matrix[~torch.eye(n, dtype=bool, device=se.device)]
            metrics["mhc/stream_similarity"] = off_diag.mean().item()

    # mHC parameter stats from blocks
    h_res_norms = []
    h_pre_norms = []
    h_post_norms = []
    h_res_grads = []
    h_pre_grads = []
    h_post_grads = []

    for i, block in enumerate(model.transformer.h):
        if hasattr(block, 'mhc_attn') and block.mhc_attn is not None:
            mhc = block.mhc_attn
            # handle both static (H_res_logits) and dynamic (H_res_base) attribute names
            h_res_param = getattr(mhc, 'H_res_base', None) or getattr(mhc, 'H_res_logits', None)
            h_pre_param = getattr(mhc, 'H_pre_base', None) or getattr(mhc, 'H_pre_logits', None)
            h_post_param = getattr(mhc, 'H_post_base', None) or getattr(mhc, 'H_post_logits', None)
            if h_res_param is not None:
                h_res_norms.append(h_res_param.data.norm().item())
                if h_res_param.grad is not None:
                    h_res_grads.append(h_res_param.grad.norm().item())
            if h_pre_param is not None:
                h_pre_norms.append(h_pre_param.data.norm().item())
                if h_pre_param.grad is not None:
                    h_pre_grads.append(h_pre_param.grad.norm().item())
            if h_post_param is not None:
                h_post_norms.append(h_post_param.data.norm().item())
                if h_post_param.grad is not None:
                    h_post_grads.append(h_post_param.grad.norm().item())

    if h_res_norms:
        metrics["mhc/H_res_norm_mean"] = sum(h_res_norms) / len(h_res_norms)
        metrics["mhc/H_pre_norm_mean"] = sum(h_pre_norms) / len(h_pre_norms)
        metrics["mhc/H_post_norm_mean"] = sum(h_post_norms) / len(h_post_norms)
    if h_res_grads:
        metrics["mhc/H_res_grad_mean"] = sum(h_res_grads) / len(h_res_grads)
    if h_pre_grads:
        metrics["mhc/H_pre_grad_mean"] = sum(h_pre_grads) / len(h_pre_grads)
    if h_post_grads:
        metrics["mhc/H_post_grad_mean"] = sum(h_post_grads) / len(h_post_grads)

    # exploration schedule (first block's mhc_attn is representative)
    first_block = model.transformer.h[0]
    if hasattr(first_block, 'mhc_attn') and first_block.mhc_attn is not None:
        mhc = first_block.mhc_attn
        # _current_explore_prob only exists in DynamicMHC
        if hasattr(mhc, '_current_explore_prob'):
            metrics["mhc/explore_prob"] = mhc._current_explore_prob
        metrics["mhc/gate_value"] = mhc.get_gate()

    # sinkhorn diagnostics: verify doubly-stochastic property
    # - "raw" = base matrix only (H_res_base through Sinkhorn)
    # - "used" = actual H_res from forward pass (base + dynamic + gate interpolation)
    n_blocks = len(model.transformer.h)
    sample_indices = [0, n_blocks // 2, n_blocks - 1]
    row_errs_raw, col_errs_raw, diag_means, offdiag_means = [], [], [], []
    row_errs_used, col_errs_used = [], []

    for idx in sample_indices:
        block = model.transformer.h[idx]
        if hasattr(block, 'mhc_attn') and block.mhc_attn is not None:
            # raw diagnostics (base matrix only)
            diag = block.mhc_attn.get_sinkhorn_diagnostics()
            row_errs_raw.append(diag["row_err"])
            col_errs_raw.append(diag["col_err"])
            diag_means.append(diag["diag_mean"])
            offdiag_means.append(diag["offdiag_mean"])

            # used diagnostics (from actual forward pass)
            used_diag = block.mhc_attn.get_used_diagnostics()
            row_errs_used.append(used_diag["row_err_used"])
            col_errs_used.append(used_diag["col_err_used"])

    if row_errs_raw:
        metrics["mhc/sinkhorn_row_err_raw"] = sum(row_errs_raw) / len(row_errs_raw)
        metrics["mhc/sinkhorn_col_err_raw"] = sum(col_errs_raw) / len(col_errs_raw)
        metrics["mhc/H_res_diag_mean"] = sum(diag_means) / len(diag_means)
        metrics["mhc/H_res_offdiag_mean"] = sum(offdiag_means) / len(offdiag_means)

    if row_errs_used:
        metrics["mhc/sinkhorn_row_err_used"] = sum(row_errs_used) / len(row_errs_used)
        metrics["mhc/sinkhorn_col_err_used"] = sum(col_errs_used) / len(col_errs_used)

    return metrics
