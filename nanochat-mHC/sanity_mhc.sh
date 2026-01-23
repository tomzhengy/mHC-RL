#!/bin/bash
# Sanity run: Single-GPU mHC training with small batch size
# Purpose: Replicate Jan 10 settings that achieved 0.087 stream_sim
# Logs: row/col errors (raw + used) via wandb

set -e

# Configuration
STEPS=${1:-5000}  # default 5000 steps, can override via CLI: ./sanity_mhc.sh 500
DEPTH=${2:-12}   # smaller model for quick sanity check
WANDB_RUN=${WANDB_RUN:-"mhc-sanity-$(date +%Y%m%d-%H%M%S)"}

echo "======================================"
echo "mHC Sanity Run (Small Batch)"
echo "======================================"
echo "Steps: $STEPS"
echo "Depth: $DEPTH"
echo "device_batch_size: 4"
echo "total_batch_size: 32768"
echo "Gate noise: off"
echo "torch.compile: off"
echo "WandB run: $WANDB_RUN"
echo ""

# Ensure we're in the nanochat directory
cd "$(dirname "$0")"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

echo "Starting training..."
echo ""

# export SKIP_COMPILE for muon optimizer (reads env var)
export SKIP_COMPILE=True

python -m scripts.base_train \
    --depth=$DEPTH \
    --num_iterations=$STEPS \
    --skip_compile=True \
    --mhc_enabled=True \
    --mhc_num_streams=4 \
    --mhc_sinkhorn_iters=20 \
    --mhc_sinkhorn_tau=0.05 \
    --mhc_gate_noise=False \
    --device_batch_size=4 \
    --total_batch_size=32768 \
    --eval_every=500 \
    --core_metric_every=-1 \
    --sample_every=100 \
    --save_every=5000 \
    --run=$WANDB_RUN

echo ""
echo "======================================"
echo "Sanity run complete!"
echo "======================================"
echo ""
echo "Check wandb for mHC metrics:"
echo "  - mhc/sinkhorn_row_err_raw   (base matrix only)"
echo "  - mhc/sinkhorn_col_err_raw   (base matrix only)"
echo "  - mhc/sinkhorn_row_err_used  (actual forward pass H_res)"
echo "  - mhc/sinkhorn_col_err_used  (actual forward pass H_res)"
echo "  - mhc/H_res_diag_mean        (diagonal dominance)"
echo "  - mhc/gate_value"
echo ""
echo "Expected: both raw and used errors should be < 1e-6"
echo "If used >> raw, check gate interpolation or dynamic deltas."
