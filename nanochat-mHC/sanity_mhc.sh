#!/bin/bash
# Sanity run: Multi-GPU mHC training
# Purpose: Verify mHC integration
# Logs: row/col errors (raw + used) via wandb

set -e

# Configuration
STEPS=${1:-5000}  # default 5000 steps, can override via CLI: ./sanity_mhc.sh 500
DEPTH=${2:-12}   # smaller model for quick sanity check
NGPUS=${NGPUS:-4}  # number of GPUs
WANDB_RUN=${WANDB_RUN:-"mhc-sanity-$(date +%Y%m%d-%H%M%S)"}

echo "======================================"
echo "mHC Sanity Run"
echo "======================================"
echo "Steps: $STEPS"
echo "Depth: $DEPTH"
echo "GPUs: $NGPUS"
echo "Seed: ${SEED:-42}"
echo "Gate noise: on"
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

torchrun --nproc_per_node=$NGPUS -m scripts.base_train -- \
    --depth=$DEPTH \
    --num_iterations=$STEPS \
    --seed=${SEED:-42} \
    --skip_compile=${SKIP_COMPILE:-True} \
    --mhc_enabled=True \
    --mhc_num_streams=4 \
    --mhc_sinkhorn_iters=50 \
    --mhc_sinkhorn_tau=0.1 \
    --mhc_gate_noise=True \
    --device_batch_size=16 \
    --total_batch_size=131072 \
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
