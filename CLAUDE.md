# mHC-gsm8k

multi-head communication (mHC) controller for GSM8K math problems. trains GPT-2 models with mHC-enhanced residual streams, then uses RL to learn when to route through different streams.

## project structure

```
mHC-gsm8k/
├── nanochat/              # forked GPT-2 training framework with mHC
│   ├── nanochat/          # core library
│   │   ├── gpt.py         # GPT model with mHC integration
│   │   ├── mhc/           # mHC implementations
│   │   │   ├── static/    # reference-style per-layer H_res (recommended)
│   │   │   └── dynamic/   # experimental per-token routing
│   │   ├── muon.py        # Muon optimizer
│   │   └── ...
│   ├── scripts/           # training scripts
│   │   └── base_train.py  # main training entry point
│   ├── configs/           # training configs
│   │   └── static_mhc_d20.yaml  # validated d20 config
│   └── sanity_mhc.sh      # quick sanity check script
├── routing/               # HuggingFace wrapper for multi-stream routing
│   ├── multistream_wrapper.py  # wraps decoder with n residual streams
│   └── mixing_ops.py      # mixing matrix operations
├── envs/                  # gymnasium environments
│   └── gsm8k_env.py       # GSM8K env for RL-controlled routing
├── controller/            # RL policy (planned)
│   ├── policy.py          # (empty - to be implemented)
│   └── features.py        # (empty - to be implemented)
└── configs/               # configuration files
```

## key concepts

### mHC (multi-head communication)
- paper: arxiv 2512.24880
- adds learnable mixing between residual streams using doubly-stochastic matrices
- uses Sinkhorn-Knopp algorithm to enforce row AND column sums = 1
- gate interpolates between identity (g=0) and learned mixing (g=1)
- paper init: g=0.01 (start near identity, learn to mix more)

### key parameters
- `mhc_enabled`: enable mHC (default False)
- `mhc_static`: use static H_res (default True, recommended)
- `mhc_num_streams`: number of residual streams (default 4)
- `mhc_sinkhorn_iters`: iterations for Sinkhorn-Knopp (default 20)
- `mhc_sinkhorn_tau`: temperature for softmax (default 0.05)
- `gate`: learnable parameter controlling mixing strength (init 0.01)

### static vs dynamic mHC

**static (recommended):**
- per-layer H_res parameters (same for all tokens)
- H_res_logits init: -8.0 off-diagonal, 0.0 diagonal
- stable training, proven in reference implementations
- stream differentiation via H_pre/H_post (width connections)

**dynamic (experimental):**
- per-token H_res via learned projections
- more expressive but unstable at depth > 12
- fundamental tradeoff: good sinkhorn convergence OR good stream_similarity, not both

## development

### setup
```bash
cd nanochat
uv sync
source .venv/bin/activate
```

### sanity check (single GPU)
```bash
cd nanochat
./sanity_mhc.sh 5000  # 5000 steps, depth 20, static mHC
```

### full training with config
```bash
cd nanochat
python -m scripts.base_train --config=configs/static_mhc_d20.yaml --run=my-run-name
```

### manual training
```bash
cd nanochat
python -m scripts.base_train \
    --mhc_enabled=True \
    --mhc_static=True \
    --mhc_num_streams=4 \
    --depth=20 \
    --num_iterations=20000
```

### key environment variables
- `TORCH_COMPILE_DISABLE=1` - disable torch.compile (required for mHC compatibility)
- `WANDB_RUN` - custom wandb run name

## wandb metrics

### mHC metrics (logged every 20 steps)
- `mhc/sinkhorn_row_err_raw` - base matrix row error (should be < 1e-6)
- `mhc/sinkhorn_col_err_raw` - base matrix col error (should be < 1e-6)
- `mhc/sinkhorn_row_err_used` - actual H_res row error after gate interpolation
- `mhc/sinkhorn_col_err_used` - actual H_res col error after gate interpolation
- `mhc/H_res_diag_mean` - diagonal dominance (paper expects near-diagonal matrices)
- `mhc/gate_value` - current gate value (starts ~0.01, learns to increase)
- `mhc/stream_similarity` - cosine similarity between streams

## code patterns

### mHC 1D params go to AdamW, not Muon
muon optimizer requires 2D+ tensors. mHC has 1D params (gate, H_pre_base, H_post_base) that must be filtered to AdamW:

```python
# in base_train.py
mhc_1d_params = [p for n, p in model.named_parameters()
                 if 'mhc' in n.lower() and p.ndim == 1]
# these go to adamw_params, not muon_params
```

### gate initialization
```python
# mhc.py - paper init γ=0.01 (near identity)
self.gate = nn.Parameter(torch.tensor([-4.6]))  # sigmoid(-4.6) ≈ 0.01
```

### doubly-stochastic constraint
```python
# H_res = (1-g)*I + g*H_res_sinkhorn
# where H_res_sinkhorn has rows AND cols summing to 1
```

## known issues

- torch.compile disabled globally (Muon optimizer incompatibility)
- cuDNN errors on some systems - may need `pip install torch` reinstall

## tests

```bash
pytest tests/
```

## current state

- nanochat static mHC training: working, validated at depth=20
- nanochat dynamic mHC: implemented but unstable at depth > 12
- routing wrapper: implemented, untested with RL
- gymnasium env: implemented
- controller policy: not yet implemented
- next step: longer training run with static mHC, then RL integration

## implementation comparison: nanochat vs reference

reference repo: lucidrains/hyper-connections (mHC-manifold-constrained-hyper-connections/)

### papers referenced
- original hyper-connections: arxiv 2409.19606
- mHC (manifold-constrained): arxiv 2512.24880

### sinkhorn implementation
both implementations produce doubly-stochastic matrices correctly:
- reference (hyper_connections.py): uses `log_marginal = -log(n)` then `* n` scaling
- reference (hyper_connections_mhc.py): uses `log_marginal = zeros` (matches ours)
- ours (mhc.py): uses `log_marginal = zeros`

no change needed - our sinkhorn matches the mHC-specific reference.

### initialization scale

| aspect | reference | static (current) | dynamic | paper spec |
|--------|-----------|------------------|---------|------------|
| H_res off-diagonal | -8.0 | -8.0 | -4.0 | not specified |
| H_res diagonal | 0.0 | 0.0 | 0.0 | - |
| tau | 0.05 | 0.05 | 0.2 | - |
| gate init | none | 0.01 | 0.01 | 0.01 |
| sinkhorn iters | 10-20 | 20 | 50 | 20 |

static mode now matches reference exactly for stability.

### architecture integration

reference pattern (branch at init):
```python
self.hc_attn = init_hc(branch=self.attn_branch, ...)
x = self.hc_attn(x)  # branch called internally
```

our pattern (branch at forward):
```python
self.mhc_attn = DynamicMHC(...)
x = self.mhc_attn(x, lambda z: self.attn(norm(z), cos_sin, kv_cache))
```

our approach is better for:
- closures capture external state (cos_sin, kv_cache) naturally
- no wrapper classes needed
- explicit data flow is more debuggable
- per-token dynamic matrices require runtime branch binding anyway

### multi-output branches
added tree_flatten/tree_unflatten support (matching reference):
```python
branch_out = branch_fn(x_pre)
(y, *rest), tree_spec = tree_flatten(branch_out)
# apply depth connection to y only
return tree_unflatten((output, *rest), tree_spec)
```
enables branches returning (output, attention_weights) tuples.

### stream expansion/reduction

| aspect | reference | ours |
|--------|-----------|------|
| tensor layout | `(b*s, t, d)` streams in batch | `(b, t, n*d)` streams in features |
| stream embed init | zeros | orthogonal * 0.02 |
| reduction | einops sum | view + sum |

orthogonal init benefits (NeurIPS 2024 research):
- prevents dimensional collapse
- streams start maximally different (zero correlation)
- faster specialization during training
- stable gradient flow

### static vs dynamic matrices

| aspect | reference | static (recommended) | dynamic (experimental) |
|--------|-----------|---------------------|------------------------|
| H_res | static per layer `[n, n]` | static per layer `[n, n]` | dynamic per token `[B, T, n, n]` |
| H_pre | static per layer `[n]` | static per layer `[n]` | dynamic per token `[B, T, n]` |
| H_post | static per layer `[n]` | static per layer `[n]` | dynamic per token `[B, T, n]` |

**static (mhc_static=True):**
- matches reference implementation
- stable training at all depths
- stream differentiation via H_pre/H_post
- for RL: make gate input-dependent later

**dynamic (mhc_static=False):**
- per-token routing via projections
- unstable at depth > 12 (tradeoff between sinkhorn convergence and stream differentiation)
- no one has successfully combined per-token routing with doubly-stochastic constraints

### features we have that reference doesn't
- learnable gate with paper init (g=0.01)
- per-token dynamic matrices
- gate noise for RL robustness
- exploration schedule API
- diagnostics collection

### features reference has that we don't (not needed)
- num_fracs: frac-connections paper support
- num_input_views: multiple views for branch input
- orthostochastic_project: newton-schulz alternative to sinkhorn
- AttentionPoolReduceStream: learned weighted reduction

### design decisions summary
1. sinkhorn: matches reference, no change needed
2. static mode (default): matches reference -8.0 init for stability
3. architecture: callback pattern (branch_fn) is cleaner for our use case
4. stream embed: orthogonal init is an improvement over reference's zeros
5. for RL routing: will add input-dependent gate rather than dynamic H_res

## training findings

### static mHC results (validated)

depth=20, static mode, 5000 steps:
- stream_similarity: 0.24 (good differentiation)
- sinkhorn errors: 0 (perfect convergence)
- val/bpb: 1.086 (good model quality)
- gate_value: 0.01 (stays at init - this is expected)
- H_res stays near identity, differentiation happens via H_pre/H_post

### dynamic mHC findings (experimental, unstable)

dynamic per-token H_res has fundamental tradeoffs at depth > 12:
- low tau (0.05): sinkhorn struggles, but streams differentiate
- high tau (0.2): sinkhorn converges, but streams collapse
- no configuration found that achieves both

### key metrics to watch

- `mhc/stream_similarity`: should be < 0.3 for good differentiation
- `mhc/sinkhorn_row_err_raw`: should be 0 or very small
- `mhc/gate_value`: for static, stays at 0.01 (expected)
- `mhc/H_pre_norm_mean`, `mhc/H_post_norm_mean`: should increase (learning)

### torch.compile considerations

- sinkhorn must be excluded from torch.compile (`@torch.compiler.disable()`)
- muon optimizer has its own `@torch.compile` on newton-schulz
- use `skip_compile=True` to avoid issues

### runpod setup notes

- data stored in `/root/.cache/nanochat/base_data/` (not persisted across sessions)
- symlink to `/workspace/` for persistence: `ln -s /workspace/nanochat/base_data /root/.cache/nanochat/base_data`
- install `python3.10-dev` for torch.compile support: `apt install python3.10-dev`
