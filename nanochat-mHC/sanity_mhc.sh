#!/bin/bash
# Sanity run: mHC training with configurable batch size
# Usage:
#   ./sanity_mhc.sh [steps] [depth]           # small batch (single GPU)
#   LARGE_BATCH=1 ./sanity_mhc.sh [steps] [depth]  # large batch (multi-GPU)
#   NGPUS=4 LARGE_BATCH=1 ./sanity_mhc.sh     # specify GPU count

set -e

# Configuration
STEPS=${1:-5000}
DEPTH=${2:-20}
WANDB_RUN=${WANDB_RUN:-"mhc-sanity-$(date +%Y%m%d-%H%M%S)"}

# Batch size config
if [ "${LARGE_BATCH:-0}" = "1" ]; then
    DEVICE_BATCH_SIZE=16
    TOTAL_BATCH_SIZE=131072
    NGPUS=${NGPUS:-8}
    BATCH_MODE="Large Batch (multi-GPU)"
    USE_TORCHRUN=1
else
    DEVICE_BATCH_SIZE=4
    TOTAL_BATCH_SIZE=32768
    NGPUS=1
    BATCH_MODE="Small Batch (single-GPU)"
    USE_TORCHRUN=0
fi

echo "======================================"
echo "mHC Sanity Run - $BATCH_MODE"
echo "======================================"
echo "Steps: $STEPS"
echo "Depth: $DEPTH"
echo "GPUs: $NGPUS"
echo "device_batch_size: $DEVICE_BATCH_SIZE"
echo "total_batch_size: $TOTAL_BATCH_SIZE"
echo "Mode: static (reference implementation)"
echo "Hyperparams: H_res=-8.0 (reference), tau=0.05, iters=20"
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

# Common args
TRAIN_ARGS="
    --depth=$DEPTH
    --num_iterations=$STEPS
    --skip_compile=True
    --mhc_enabled=True
    --mhc_static=True
    --mhc_num_streams=4
    --mhc_sinkhorn_iters=20
    --mhc_sinkhorn_tau=0.05
    --device_batch_size=$DEVICE_BATCH_SIZE
    --total_batch_size=$TOTAL_BATCH_SIZE
    --eval_every=500
    --core_metric_every=-1
    --sample_every=100
    --save_every=5000
    --run=$WANDB_RUN
"

if [ "$USE_TORCHRUN" = "1" ]; then
    torchrun --standalone --nproc_per_node=$NGPUS -m scripts.base_train -- $TRAIN_ARGS
else
    python -m scripts.base_train $TRAIN_ARGS
fi

echo ""
echo "======================================"
echo "Sanity run complete!"
echo "======================================"
echo ""
echo "Check wandb for mHC metrics:"
echo "  - mhc/stream_similarity"
echo "  - mhc/gate_value"
echo "  - mhc/sinkhorn_row_err_raw"
echo "  - mhc/H_res_diag_mean"
echo ""
