#!/bin/bash
set -e

echo "=== nanochat-mHC Eval Setup ==="

# install screen for background jobs, python headers for torch.compile
apt update && apt install -y screen python3.10-dev

# set base dir for nanochat artifacts (use /workspace for persistence on cloud GPUs)
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat}"
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

# pre-download CORE eval bundle (~162MB)
echo "Pre-downloading CORE eval bundle (~162MB)..."
python -c "
from nanochat.common import get_base_dir, download_file_with_lock
import os, shutil, zipfile, tempfile

EVAL_BUNDLE_URL = 'https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip'

def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, 'eval_bundle')
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, 'eval_bundle')
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print(f'Placed eval_bundle directory at {eval_bundle_dir}')

download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)
print('CORE eval bundle ready.')
"

# optionally download checkpoint from HuggingFace
# env vars: HF_CKPT_REPO (required), CKPT_SOURCE (base|mid|sft|rl, default base), CKPT_MODEL_TAG
if [ -n "$HF_CKPT_REPO" ]; then
    CKPT_SOURCE="${CKPT_SOURCE:-base}"
    case "$CKPT_SOURCE" in
        base) CKPT_SUBDIR="base_checkpoints" ;;
        mid)  CKPT_SUBDIR="mid_checkpoints" ;;
        sft)  CKPT_SUBDIR="chatsft_checkpoints" ;;
        rl)   CKPT_SUBDIR="chatrl_checkpoints" ;;
        *)    echo "ERROR: CKPT_SOURCE must be one of: base, mid, sft, rl"; exit 1 ;;
    esac

    # default model tag from repo name if not set
    CKPT_MODEL_TAG="${CKPT_MODEL_TAG:-$(basename $HF_CKPT_REPO)}"
    CKPT_DIR="$NANOCHAT_BASE_DIR/$CKPT_SUBDIR/$CKPT_MODEL_TAG"
    mkdir -p $CKPT_DIR

    echo "Downloading checkpoint from $HF_CKPT_REPO to $CKPT_DIR..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_CKPT_REPO', local_dir='$CKPT_DIR')
"

    # rename to checkpoint manager format if needed (model.pt -> model_000001.pt)
    if [ -f "$CKPT_DIR/model.pt" ] && [ ! -f "$CKPT_DIR/model_000001.pt" ]; then
        mv "$CKPT_DIR/model.pt" "$CKPT_DIR/model_000001.pt"
        mv "$CKPT_DIR/meta.json" "$CKPT_DIR/meta_000001.json"
        echo "Renamed checkpoint files to model_000001.pt / meta_000001.json"
    fi

    # symlink to ~/.cache/nanochat for compatibility with default paths
    echo "Creating checkpoint symlinks..."
    mkdir -p ~/.cache/nanochat
    rm -rf ~/.cache/nanochat/$CKPT_SUBDIR 2>/dev/null || true
    ln -s $NANOCHAT_BASE_DIR/$CKPT_SUBDIR ~/.cache/nanochat/$CKPT_SUBDIR
fi

echo ""
echo "=== Eval Setup Complete ==="
echo ""
echo "Run evaluation:"
echo "  source .venv/bin/activate"
echo "  python -m scripts.eval_compare --models base:d20 --evals core --dashboard"
