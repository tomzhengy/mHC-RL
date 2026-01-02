# Nia Sources Tracking

## Research Papers

| Title | arXiv ID | Source ID | Status | Notes |
|-------|----------|-----------|--------|-------|
| mHC: Manifold-Constrained Hyper-Connections | 2512.24880 | 820ff393-cc17-41e7-9e22-b2df85e7dd92 | ✅ Indexed | Core paper for this project - DeepSeek's multi-stream residual architecture |

## Repositories

_None indexed yet for this project_

## Documentation

_None indexed yet for this project_

---

## Quick Reference

### mHC Paper Key Concepts

- **Authors**: Zhenda Xie, Yixuan Wei, Huanqi Cao + 16 more (DeepSeek-AI)
- **Core idea**: Project residual connection matrices onto a constrained manifold to restore identity mapping property while allowing multi-stream mixing
- **Key equations**:
  - `x_{l+1} = H^res_l · x_l + H^post_l · F(H^pre_l · x_l, W_l)`
  - Manifold projection: `P_M_res(H^res_l)`
- **Key hyperparameters**:
  - Expansion rate λ = 4
  - Gating factor init = 0.01
  - Sinkhorn-Knopp ε_max = 20
- **Benchmarks used**: GSM8K, BBH, DROP, MMLU, etc.

