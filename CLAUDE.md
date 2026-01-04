# mHC-gsm8k

multi-head communication (mHC) controller for GSM8K math problems. trains GPT-2 models with mHC-enhanced residual streams, then uses RL to learn when to route through different streams.

## project structure

```
mHC-gsm8k/
├── nanochat/              # forked GPT-2 training framework with mHC
│   ├── nanochat/          # core library
│   │   ├── gpt.py         # GPT model with mHC integration
│   │   ├── mhc.py         # mHC layer implementation (Sinkhorn-Knopp)
│   │   ├── muon.py        # Muon optimizer
│   │   └── ...
│   ├── scripts/           # training scripts
│   │   └── base_train.py  # main training entry point
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
- `mhc_num_streams`: number of residual streams (default 4)
- `mhc_sinkhorn_iters`: iterations for Sinkhorn-Knopp (default 50)
- `mhc_sinkhorn_tau`: temperature for softmax (default 0.1)
- `gate`: learnable parameter controlling mixing strength

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
./sanity_mhc.sh 300  # 300 steps, depth 12
```

### full training
```bash
cd nanochat
python -m scripts.base_train \
    --mhc_enabled=True \
    --mhc_num_streams=4 \
    --depth=12 \
    --num_iterations=5000
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

- nanochat mHC training: working
- routing wrapper: implemented, untested with RL
- gymnasium env: implemented
- controller policy: not yet implemented
