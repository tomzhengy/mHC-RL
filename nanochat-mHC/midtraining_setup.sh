#!/bin/bash
set -e

echo "=== nanochat-mHC Midtraining Setup ==="

# install screen for background jobs, python headers for torch.compile
apt update && apt install -y screen python3.10-dev

# set base dir for nanochat artifacts (use /workspace for persistence on cloud GPUs)
export NANOCHAT_BASE_DIR="/workspace/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# install uv
echo "Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

# create venv and sync dependencies
echo "Syncing dependencies..."
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# install rust for rustbpe tokenizer
echo "Installing Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# build rustbpe tokenizer
echo "Building rustbpe tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

source .venv/bin/activate

# huggingface login (token passed as env var HF_TOKEN)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    hf auth login --token $HF_TOKEN
fi

# download pre-trained tokenizer
echo "Downloading tokenizer from tomzhengy/nanochat-tokenizer..."
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
mkdir -p $TOKENIZER_DIR
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tomzhengy/nanochat-tokenizer', local_dir='$TOKENIZER_DIR')
"
mkdir -p ~/.cache/nanochat/tokenizer
cp -r $TOKENIZER_DIR/* ~/.cache/nanochat/tokenizer/ 2>/dev/null || true

# download mHC checkpoint from huggingface
echo "Downloading mHC checkpoint from tomzhengy/nanochat-mhc-d20-static..."
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
mkdir -p $CKPT_DIR
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tomzhengy/nanochat-mhc-d20-static', local_dir='$CKPT_DIR')
"
# rename to checkpoint manager format (model_{step:06d}.pt)
if [ -f "$CKPT_DIR/model.pt" ] && [ ! -f "$CKPT_DIR/model_000001.pt" ]; then
    mv "$CKPT_DIR/model.pt" "$CKPT_DIR/model_000001.pt"
    mv "$CKPT_DIR/meta.json" "$CKPT_DIR/meta_000001.json"
    echo "Renamed checkpoint files to model_000001.pt / meta_000001.json"
fi

# symlink base_checkpoints to ~/.cache/nanochat for compatibility with default paths
echo "Creating checkpoint symlinks..."
mkdir -p ~/.cache/nanochat
rm -rf ~/.cache/nanochat/base_checkpoints 2>/dev/null || true
ln -s $NANOCHAT_BASE_DIR/base_checkpoints ~/.cache/nanochat/base_checkpoints

echo ""
echo "=== Midtraining Setup Complete ==="
echo ""
echo "Run midtraining:"
echo "  source .venv/bin/activate"
echo "  export TORCH_COMPILE_DISABLE=1"
echo "  python -m scripts.mid_train --model_tag=d20 --run=mid-d20-mhc"
