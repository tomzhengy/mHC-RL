#!/bin/bash
set -e

echo "=== nanochat-mHC GPU Setup ==="

# set base dir for nanochat artifacts (use /workspace for more disk space on cloud GPUs)
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

# activate venv for remaining commands
source .venv/bin/activate

# huggingface login (token passed as env var HF_TOKEN)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    hf auth login --token $HF_TOKEN
fi

# download pre-trained tokenizer from HuggingFace
# use python snapshot_download instead of hf cli to avoid symlink issues
echo "Downloading tokenizer from tomzhengy/nanochat-tokenizer..."
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
mkdir -p $TOKENIZER_DIR
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tomzhengy/nanochat-tokenizer', local_dir='$TOKENIZER_DIR')
"
# also copy to ~/.cache/nanochat for compatibility with default paths
mkdir -p ~/.cache/nanochat/tokenizer
cp -r $TOKENIZER_DIR/* ~/.cache/nanochat/tokenizer/ 2>/dev/null || true

# download 240 data shards (~24GB, enough for speedrun training)
echo "Downloading 240 data shards (~24GB)..."
python -m nanochat.dataset -n 240

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now start training:"
echo "  source .venv/bin/activate"
echo "  ./sanity_mhc.sh 300"
