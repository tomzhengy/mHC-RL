# PRD-2: Dynamic mHC Integration with nanochat GPT-2

## Overview

Train GPT-2 models with Dynamic Manifold-Constrained Hyper-Connections (mHC) using nanochat, then tune routing with an RL controller on GSM8K.

**Hypothesis**: Dynamic input-dependent routing (mHC) improves reasoning, and RL can discover better routing policies than learned defaults.

---

## Phase 1: Baseline Training ✅ IN PROGRESS

### 1.1 Goal
Train a vanilla nanochat GPT-2 model as the control group.

### 1.2 Configuration
```python
# baseline config
depth = 12              # 12 layers (GPT-2 small scale)
model_dim = 768         # 12 * 64
num_heads = 6           # head_dim = 128
max_seq_len = 1024
total_batch_size = 524288
num_iterations = 100000  # ~50B tokens

# no mHC
mhc_enabled = False
```

### 1.3 Status
**Currently training on Lambda cluster (8x H100)**

### 1.4 Deliverables
- [ ] Trained baseline checkpoint
- [ ] Val BPB curve
- [ ] GSM8K accuracy (0-shot, 8-shot)
- [ ] HellaSwag accuracy
- [ ] Inference speed benchmark

---

## Phase 2: Dynamic mHC Training (4-5 days) ✅ IMPLEMENTED

### 2.1 Goal
Train with dynamic mHC where H_res, H_pre, H_post are generated **per-token** from input for maximum expressiveness.

### 2.1.1 Implementation Status (Completed)

**Files created:**
- `nanochat/nanochat/mhc.py` - Core mHC module with `DynamicMHC` class
- `nanochat/scripts/test_mhc_integration.py` - Integration tests

**Files modified:**
- `nanochat/nanochat/gpt.py` - Added mHC config params, Block integration, stream expand/reduce
- `nanochat/scripts/base_train.py` - Added mHC CLI options

**Key implementation notes:**
1. **Stream embeddings**: Added learnable `stream_embed` parameter to GPT class to differentiate initially-identical streams
2. **Gate control**: The `gate` parameter interpolates H_res between identity (g=0) and learned mixing (g=1)
3. **Initialization**: H_res starts as identity (via -8 off-diagonal logits), allowing stable training startup
4. **Gradient flow**: Verified gradients flow through mHC params when streams become different during training
5. **Parameter overhead**: ~9.4% increase over baseline (mostly from projection weights)

### 2.2 Architecture Overview

**Widened state**: `[B, T, n*C]` where n=4 streams

**Per sub-block (attention and MLP)**:
```python
x_norm = RMSNorm(x)                              # [B, T, n*C]

# generate per-token matrices via linear projections
H_res_tilde = proj_res(x_norm).view(B, T, n, n)  # [B, T, n, n]
H_pre_tilde = proj_pre(x_norm)                   # [B, T, n]
H_post_tilde = proj_post(x_norm)                 # [B, T, n]

# apply constraints (as per mHC paper)
H_res = sinkhorn(H_res_tilde, iters=20)          # doubly stochastic
H_pre = sigmoid(H_pre_tilde)                     # non-negative [0,1]
H_post = 2 * sigmoid(H_post_tilde)               # non-negative [0,2]

# apply mHC around branch (attention or MLP)
x_pre = apply_H_pre(x, H_pre)                    # mix streams → [B, T, C]
y = branch(x_pre)                                # attention or MLP
y_post = apply_H_post(y, H_post)                 # expand back → [B, T, n, C]
output = apply_H_res(x, H_res) + y_post          # residual + branch
```

### 2.3 New Config Parameters

```python
@dataclass
class GPTConfig:
    # ... existing params ...
    
    # mHC parameters
    mhc_enabled: bool = False
    mhc_num_streams: int = 4          # n = expansion rate
    mhc_sinkhorn_iters: int = 20      # iterations for doubly-stochastic
    mhc_sinkhorn_tau: float = 0.05    # temperature (lower = sharper routing)
```

### 2.4 New File: `nanochat/nanochat/mhc.py`

```python
"""
Dynamic Manifold-Constrained Hyper-Connections (mHC) module.
Per-token matrix generation for maximum expressiveness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 20, tau: float = 0.05) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm to project matrix onto doubly-stochastic manifold.
    
    Args:
        logits: Raw logits [..., n, n]
        num_iters: Number of Sinkhorn iterations
        tau: Temperature (lower = sharper)
    
    Returns:
        Doubly stochastic matrix [..., n, n]
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    
    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class DynamicMHC(nn.Module):
    """
    Dynamic mHC: H_res, H_pre, H_post are generated per-token from input.
    
    Architecture:
    - Input: [B, T, n*C] (flattened streams)
    - Generates [B, T, n, n] H_res, [B, T, n] H_pre/H_post per token
    - Output: [B, T, n*C]
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
        widened_dim = dim * n
        
        # RMSNorm for input normalization
        self.norm = nn.RMSNorm(widened_dim)
        
        # projections to generate matrices per-token
        self.proj_H_res = nn.Linear(widened_dim, n * n, bias=False)
        self.proj_H_pre = nn.Linear(widened_dim, n, bias=False)
        self.proj_H_post = nn.Linear(widened_dim, n, bias=False)
        
        # initialize projections small (start near identity behavior)
        nn.init.normal_(self.proj_H_res.weight, std=0.01)
        nn.init.normal_(self.proj_H_pre.weight, std=0.01)
        nn.init.normal_(self.proj_H_post.weight, std=0.01)
        
        # learnable base matrices (static component added to dynamic)
        # H_res base: identity-like (strong diagonal)
        init_h_res = torch.full((n, n), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_base = nn.Parameter(init_h_res)
        
        # H_pre base: select main stream (stream 0)
        init_h_pre = torch.full((n,), -8.0)
        init_h_pre[0] = 0.0
        self.H_pre_base = nn.Parameter(init_h_pre)
        
        # H_post base: distribute equally
        self.H_post_base = nn.Parameter(torch.zeros(n))
        
        # controllable gate for RL tuning
        # g=1 means full mHC, g=0 means identity (no mixing)
        # initialized to 1.0 (full mHC by default)
        self.gate = nn.Parameter(torch.tensor([1.0]))
    
    def get_matrices(self, x: torch.Tensor):
        """
        Generate per-token matrices from input.
        
        Args:
            x: Input [B, T, n*C] (flattened streams)
        
        Returns:
            H_res [B, T, n, n], H_pre [B, T, n], H_post [B, T, n]
        """
        B, T, _ = x.shape
        n = self.num_streams
        
        # normalize input
        x_norm = self.norm(x)
        
        # generate dynamic adjustments
        H_res_delta = self.proj_H_res(x_norm).view(B, T, n, n)
        H_pre_delta = self.proj_H_pre(x_norm)
        H_post_delta = self.proj_H_post(x_norm)
        
        # combine base + dynamic
        H_res_logits = self.H_res_base + H_res_delta
        H_pre_logits = self.H_pre_base + H_pre_delta
        H_post_logits = self.H_post_base + H_post_delta
        
        # apply constraints
        H_res = sinkhorn_log(H_res_logits, self.sinkhorn_iters, self.sinkhorn_tau)
        H_pre = torch.sigmoid(H_pre_logits)        # [0, 1]
        H_post = 2.0 * torch.sigmoid(H_post_logits) # [0, 2]
        
        # apply gate: interpolate between identity and computed H_res
        # H_res_gated = (1 - g) * I + g * H_res
        g = torch.sigmoid(self.gate)
        
        # during training, add noise to gate for robustness (for RL tuning later)
        if self.training:
            gate_noise = torch.rand(1, device=x.device) * 0.4 + 0.8  # [0.8, 1.2]
            g = g * gate_noise
        
        I = torch.eye(n, device=H_res.device, dtype=H_res.dtype)
        H_res = (1 - g) * I + g * H_res
        
        return H_res, H_pre, H_post
    
    def forward(self, x: torch.Tensor, branch_fn) -> torch.Tensor:
        """
        Apply dynamic mHC around a branch function.
        
        Args:
            x: Input tensor [B, T, n*C] (flattened streams)
            branch_fn: Function to apply (attention or MLP)
        
        Returns:
            Output tensor [B, T, n*C]
        """
        B, T, nC = x.shape
        n = self.num_streams
        C = nC // n
        
        # get per-token matrices
        H_res, H_pre, H_post = self.get_matrices(x)  # [B,T,n,n], [B,T,n], [B,T,n]
        
        # unflatten to [B, T, n, C]
        x_unflat = x.view(B, T, n, C)
        
        # width connection: weighted sum across streams for branch input
        # x_pre[b,t,c] = sum_i H_pre[b,t,i] * x[b,t,i,c]
        x_pre = einsum(x_unflat, H_pre, 'b t n c, b t n -> b t c')  # [B, T, C]
        
        # apply branch (attention or MLP)
        y = branch_fn(x_pre)  # [B, T, C]
        
        # depth connection
        # x_mixed[b,t,j,c] = sum_i H_res[b,t,i,j] * x[b,t,i,c]
        x_mixed = einsum(x_unflat, H_res, 'b t n c, b t n m -> b t m c')  # [B, T, n, C]
        
        # y_expanded[b,t,j,c] = H_post[b,t,j] * y[b,t,c]
        y_expanded = einsum(y, H_post, 'b t c, b t n -> b t n c')  # [B, T, n, C]
        
        # combine and flatten back
        output = x_mixed + y_expanded  # [B, T, n, C]
        return output.view(B, T, nC)  # [B, T, n*C]
    
    def set_gate(self, value: float):
        """Set gate value (for RL control). Value in [0, 1]."""
        with torch.no_grad():
            # inverse sigmoid to get logit
            value = max(1e-6, min(1 - 1e-6, value))  # clamp to avoid inf
            logit = torch.log(torch.tensor(value / (1 - value)))
            self.gate.fill_(logit)
```

### 2.5 Modified `Block` class in `gpt.py`

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        
        # mHC wrappers (if enabled)
        self.mhc_enabled = getattr(config, 'mhc_enabled', False)
        if self.mhc_enabled:
            from nanochat.mhc import DynamicMHC
            mhc_kwargs = dict(
                dim=config.n_embd,
                num_streams=config.mhc_num_streams,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                sinkhorn_tau=config.mhc_sinkhorn_tau,
                layer_idx=layer_idx,
            )
            self.mhc_attn = DynamicMHC(**mhc_kwargs)
            self.mhc_mlp = DynamicMHC(**mhc_kwargs)

    def forward(self, x, cos_sin, kv_cache):
        if not self.mhc_enabled:
            # standard residual connections
            x = x + self.attn(norm(x), cos_sin, kv_cache)
            x = x + self.mlp(norm(x))
        else:
            # mHC connections
            def attn_branch(z):
                return self.attn(norm(z), cos_sin, kv_cache)
            def mlp_branch(z):
                return self.mlp(norm(z))
            
            x = self.mhc_attn(x, attn_branch)
            x = self.mhc_mlp(x, mlp_branch)
        return x
```

### 2.6 Modified `GPT` class

```python
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        
        # ... existing embedding code ...
        
        # stream expansion/reduction (if mHC enabled)
        self.mhc_enabled = getattr(config, 'mhc_enabled', False)
        if self.mhc_enabled:
            self.mhc_num_streams = config.mhc_num_streams
    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        
        # ... rotary embeddings code ...
        
        # embedding
        x = self.transformer.wte(idx)
        x = norm(x)
        
        # expand to streams (if mHC): replicate across n streams
        if self.mhc_enabled:
            n = self.mhc_num_streams
            x = x.unsqueeze(-2).expand(-1, -1, n, -1).reshape(B, T, -1)  # [B, T, n*C]
        
        # transformer blocks
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        
        # reduce streams (if mHC): sum across streams
        if self.mhc_enabled:
            n = self.mhc_num_streams
            C = x.shape[-1] // n
            x = x.view(B, T, n, C).sum(dim=-2)  # [B, T, C]
        
        x = norm(x)
        
        # ... lm_head and loss code ...
    
    def set_mhc_gate(self, value: float):
        """Set gate value across all mHC modules (for RL control)."""
        if not self.mhc_enabled:
            return
        for block in self.transformer.h:
            block.mhc_attn.set_gate(value)
            block.mhc_mlp.set_gate(value)
    
    def set_mhc_gates_per_layer(self, values: list[float]):
        """Set per-layer gate values. values[i] applies to layer i."""
        if not self.mhc_enabled:
            return
        for i, block in enumerate(self.transformer.h):
            block.mhc_attn.set_gate(values[i])
            block.mhc_mlp.set_gate(values[i])
```

### 2.7 Training Configuration

```python
# dynamic mHC config (add to base_train.py)
mhc_enabled = True
mhc_num_streams = 4
mhc_sinkhorn_iters = 20
mhc_sinkhorn_tau = 0.05
```

### 2.8 Deliverables
- [ ] `mhc.py` implementation
- [ ] Modified `gpt.py` with mHC integration
- [ ] Trained dynamic mHC checkpoint
- [ ] Comparison: dynamic mHC vs baseline
  - Val BPB
  - GSM8K accuracy
  - HellaSwag accuracy
  - Inference speed (~10-15% slower expected)

---

## Phase 3: RL Controller Integration (1 week)

### 3.1 Goal
Train an RL agent to control the gate parameter(s) to maximize GSM8K accuracy.

### 3.2 What the Controller Tunes

| Option | Action Space | Complexity |
|--------|--------------|------------|
| **Global gate** | 1 scalar ∈ [0, 1] | Simplest |
| **Per-layer gates** | L scalars (12 for GPT-2 small) | Medium |
| **Per-layer + per-sublayer** | 2L scalars (24) | Higher |

**Recommendation**: Start with global gate, upgrade to per-layer if promising.

### 3.3 Environment Design

```python
class GSM8KMHCEnv(gymnasium.Env):
    """
    Environment for tuning mHC gate on GSM8K problems.
    
    Episode: One GSM8K problem
    Action: Gate value(s)
    Reward: +1 correct, 0 wrong
    """
    
    def __init__(self, model, tokenizer, dataset, max_new_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        
        # action space: global gate ∈ [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # observation: problem embedding
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.problem_idx = self.np_random.integers(0, len(self.dataset))
        problem = self.dataset[self.problem_idx]
        self.question = problem["question"]
        self.expected = problem["final_answer"]
        
        # get embedding of question as observation
        tokens = self.tokenizer(self.question)
        with torch.no_grad():
            emb = self.model.get_embedding(tokens)
        obs = emb.mean(dim=1).squeeze().cpu().numpy()
        return obs, {}
    
    def step(self, action):
        gate = float(action[0])
        
        # set gate in all mHC modules
        self.model.set_mhc_gate(gate)
        
        # generate answer
        response = self.model.generate(self.question, max_tokens=self.max_new_tokens)
        predicted = extract_answer(response)
        
        # compute reward
        correct = answers_match(predicted, self.expected)
        reward = 1.0 if correct else 0.0
        
        obs = np.zeros(768, dtype=np.float32)  # dummy obs for terminal
        return obs, reward, True, False, {"correct": correct, "gate": gate}
```

### 3.4 Training Protocol

```python
# freeze base model, only train gate parameters
for param in model.parameters():
    param.requires_grad = False

# unfreeze only gate parameters
for block in model.transformer.h:
    block.mhc_attn.gate.requires_grad = True
    block.mhc_mlp.gate.requires_grad = True

# PPO training
ppo = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.1,  # high for exploration
    verbose=1,
)
ppo.learn(total_timesteps=50000)
```

### 3.5 Deliverables
- [ ] `GSM8KMHCEnv` implementation
- [ ] Trained RL policy
- [ ] Comparison: RL-tuned vs learned default gates
- [ ] Analysis: what gate values does RL find?
- [ ] Ablation: per-layer vs global gate

---

## Phase 4: Analysis & Writing (3-4 days)

### 4.1 Experiments to Run

| Experiment | Baseline | Dynamic mHC | RL-Tuned |
|------------|----------|-------------|----------|
| Val BPB | ✓ | ✓ | ✓ |
| GSM8K 0-shot | ✓ | ✓ | ✓ |
| GSM8K 8-shot | ✓ | ✓ | ✓ |
| HellaSwag | ✓ | ✓ | ✓ |
| ARC-Easy | ✓ | ✓ | ✓ |
| Inference latency | ✓ | ✓ | ✓ |
| Peak memory | ✓ | ✓ | ✓ |

### 4.2 Visualizations
- Training loss curves (baseline vs mHC)
- H_res heatmaps across layers
- Gate value distribution from RL
- Per-layer gate vs accuracy scatter

### 4.3 Deliverables
- [ ] All benchmark results in tables
- [ ] Visualizations
- [ ] Analysis of learned routing patterns
- [ ] Draft paper/report

---

## File Structure

```
mHC-gsm8k/
├── nanochat/
│   └── nanochat/
│       ├── gpt.py           # modified with mHC integration
│       ├── mhc.py           # NEW: dynamic mHC module
│       └── ...
├── configs/
│   ├── baseline.py          # baseline training config
│   └── dynamic_mhc.py       # dynamic mHC config
├── envs/
│   ├── gsm8k_env.py         # existing
│   └── mhc_env.py           # NEW: mHC-specific env for RL
├── scripts/
│   ├── train_baseline.sh    # baseline training
│   ├── train_mhc.sh         # mHC training
│   └── train_rl.py          # RL controller training
├── analysis/
│   ├── visualize_hres.py    # H_res visualization
│   └── compare_results.py   # benchmark comparison
└── results/
    ├── baseline/
    ├── dynamic_mhc/
    └── rl_tuned/
```

---

## Compute Budget (8x H100)

| Phase | Duration | GPU Hours | Notes |
|-------|----------|-----------|-------|
| Phase 1: Baseline | 2 days | 384 | ✅ In progress |
| Phase 2: Dynamic mHC | 4 days | 768 | Next |
| Phase 3: RL | 3 days | 576 | Many short rollouts |
| Phase 4: Analysis | 2 days | 192 | Eval runs |
| **Total** | **11 days** | **~1920** | |

---

## Success Criteria

### Minimum Viable
- [ ] Dynamic mHC trains stably
- [ ] mHC matches or beats baseline on Val BPB
- [ ] RL finds non-trivial gate values

### Target
- [ ] mHC improves GSM8K by ≥2% over baseline
- [ ] RL improves GSM8K by ≥1% over learned defaults
- [ ] Clear visualization of routing patterns

### Stretch
- [ ] Per-layer RL gates outperform global gate
- [ ] Interesting layer-wise routing patterns emerge
- [ ] Publishable results

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| mHC training unstable | Reduce tau, increase Sinkhorn iters, check gradients |
| No improvement over baseline | Tune n, tau, or try different init |
| RL doesn't converge | Increase entropy, try different action spaces |
| OOM with dynamic mHC | Reduce batch size, gradient checkpointing |
| Sinkhorn too slow | Reduce iters at inference |

---

## References

1. **mHC Paper**: arXiv:2512.24880 - Manifold-Constrained Hyper-Connections
2. **HC Paper**: arXiv:2409.19606 - Hyper-Connections  
3. **tokenbender/mHC**: GitHub implementation reference
4. **nanochat**: Karpathy's training framework

---

## Changelog

- **v1.1** (2026-01-02): Removed static mHC phase, streamlined to dynamic-only
- **v1.0** (2026-01-02): Initial PRD based on research discussion
