#!/bin/bash
set -e

echo "=== nanochat-mHC GPU Setup ==="

# set base dir for nanochat artifacts
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# install uv
echo "Installing uv..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
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

# huggingface login (token passed as env var HF_TOKEN)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
fi

# download pre-trained tokenizer from HuggingFace
echo "Downloading tokenizer from tomzhengy/nanochat-tokenizer..."
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
mkdir -p $TOKENIZER_DIR
huggingface-cli download tomzhengy/nanochat-tokenizer --local-dir $TOKENIZER_DIR

# download 240 data shards (~24GB, enough for speedrun training)
echo "Downloading 240 data shards (~24GB)..."
python -m nanochat.dataset -n 240 &
DOWNLOAD_PID=$!

echo ""
echo "=== Setup Complete ==="
echo "Data download running in background (PID: $DOWNLOAD_PID)"
echo ""
echo "To train now (with partial data):"
echo "  source .venv/bin/activate"
echo "  ./sanity_mhc.sh 300"
echo ""
echo "To check download progress:"
echo "  ls -la $NANOCHAT_BASE_DIR/base_data/ | wc -l"
