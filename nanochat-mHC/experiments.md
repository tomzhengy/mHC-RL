# mHC Hyperparameter Experiments

## Summary

**Current approach:** Switched to **static mHC** (matching reference implementation) for stable training.

Dynamic mHC experiments showed fundamental tradeoffs that couldn't be resolved:

- Good sinkhorn convergence OR good stream_similarity - not both
- Per-token routing via projections creates unstable training dynamics

Static mHC uses per-layer H_res (no input-dependent projections), which is proven stable.

## Configurations Tested

### v1: Original Settings (Jan 10)

```
H_res_base off-diag: -0.5
sinkhorn_tau: 0.1
sinkhorn_iters: 50
device_batch_size: 4
total_batch_size: 32768
gate_noise: False (mislabeled as "yes gate noise" in wandb)
torch.compile: disabled
```

**Results:**

- stream_similarity: 0.087 (best ever, but not reproducible)
- gate_value: 0.688
- val/bpb: ~1.15

**Notes:** This was a "lucky" run. Subsequent runs with same config got 0.37-0.52 stream_similarity.

---

### v2: Large Batch (Jan 19)

```
H_res_base off-diag: -0.5
sinkhorn_tau: 0.1
sinkhorn_iters: 50
device_batch_size: 16
total_batch_size: 131072
gate_noise: varied
torch.compile: varied
```

**Results:**

- stream_similarity: 0.43-0.77 (high variance)
- gate_value: 0.57-1.0 (often saturated)
- val/bpb: 0.996-1.15

**Notes:** Better val/bpb due to 4x more tokens, but stream_similarity was worse and highly variable.

---

### v3: Optimized Init (Jan 24) - CURRENT

```
H_res_base off-diag: -3.0  (was -0.5)
sinkhorn_tau: 0.05         (was 0.1)
sinkhorn_iters: 20         (was 50)
device_batch_size: 4
total_batch_size: 32768
gate_noise: False
torch.compile: disabled
```

**Results:**

- stream_similarity: 0.163 (consistent, 2.8x better than v2)
- gate_value: 0.338 (stable, not saturating)
- val/bpb: 1.147
- H_res_diag_mean: 0.504 (learned non-trivial routing)

**Notes:**

- More extreme H_res init prevents early commitment to mixing
- Lower tau makes Sinkhorn sharper
- Fewer iterations still converge well (matches paper's 20)
- Reduced initialization sensitivity

---

### v4: Depth=20 Dynamic Experiments (Jan 28-29)

All at depth=20, small batch, dynamic mHC:

**v4a: -3.0 init, tau=0.05**

- stream_sim: 0.61 (bad - streams collapsed)
- Didn't work at deeper depth

**v4b: -5.0 init, tau=0.05**

- stream_sim: 0.278 (good)
- gate: 0.008 (barely learning - too extreme init)
- sinkhorn_err: good

**v4c: -4.0 init, tau=0.05** (run 015210)

- stream_sim: 0.387
- gate: 0.452 (learning!)
- sinkhorn_err: 0.17 (bad convergence)

**v4d: -4.0 init, tau=0.2** (run 232904)

- stream_sim: 0.765 (bad - too much mixing)
- gate: 0.746 (learning)
- sinkhorn_err: 0.006 (good)

**Conclusion:** Dynamic mHC has fundamental tradeoff at depth=20:

- Low tau: sinkhorn struggles, but streams differentiate
- High tau: sinkhorn converges, but streams collapse

---

### v5: Static mHC (Jan 29) - CURRENT

Switched to static H_res (reference implementation style):

```
mode: static (mhc_static=True)
H_res_logits off-diag: -8.0 (reference default)
sinkhorn_tau: 0.05
sinkhorn_iters: 20
gate init: 0.01
depth: 20
```

**Expected:** Stable training matching reference implementation.

---

## Key Findings

### Initialization Sensitivity

- Original -0.5 init with tau=0.1 had high variance across runs
- More extreme -3.0 init with tau=0.05 is more consistent
- Gate can saturate to 1.0 with mild init, stays moderate with extreme init

### Batch Size Effects

- Larger batch = more tokens per step = better val/bpb
- Larger batch alone didn't fix stream_similarity issues
- Small batch was used in original good run

### torch.compile

- Runs with compile enabled had better val/bpb (~0.996 vs ~1.15)
- Likely due to other config differences, not compile itself
- Sinkhorn excluded via @torch.compiler.disable() decorator

### Static vs Dynamic mHC

**Dynamic mHC (our original approach):**

- Per-token H_res via learned projections
- More expressive - different routing for different tokens
- Unstable at depth=20: tradeoff between sinkhorn convergence and stream differentiation
- No one has successfully combined per-token routing with doubly-stochastic constraints

**Static mHC (reference approach):**

- Per-layer H_res (same for all tokens)
- Proven stable in lucidrains and tokenbender implementations
- For RL control later, can make gate input-dependent instead

### Paper Reference Values

- Paper uses gamma init = 0.01 (we match this)
- Paper uses sinkhorn_iters = 20 (we now match this)
- Paper doesn't specify tau or H_res init values
- Reference implementation uses -8.0 off-diag

## Metrics to Watch

| Metric               | Good Value | Bad Value   | Notes                                 |
| -------------------- | ---------- | ----------- | ------------------------------------- |
| stream_similarity    | < 0.2      | > 0.4       | Lower = better stream differentiation |
| gate_value           | 0.3-0.7    | 1.0         | Saturating to 1.0 is bad              |
| sinkhorn_col_err_raw | < 1e-6     | > 0.01      | Sinkhorn convergence                  |
| H_res_diag_mean      | 0.4-0.8    | 1.0 or 0.25 | Should learn non-trivial routing      |
| val/bpb              | < 1.0      | > 1.2       | Model quality                         |
