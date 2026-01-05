"""
Model loader for frozen mHC model.

Loads pretrained nanochat GPT model with mHC (DynamicMHC) and freezes all
parameters for RL controller training.
"""

import sys
from pathlib import Path

import torch

# add nanochat to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanochat-mHC"))
from nanochat.checkpoint_manager import build_model


def load_frozen_mhc_model(checkpoint_dir: str, step: int, device: str = "cuda"):
    """
    Load pretrained mHC model and freeze all parameters.

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g. nanochat-mHC/base_checkpoints/d12)
        step: Training step to load (e.g. 5000)
        device: Device to load model on

    Returns:
        model: Frozen GPT model with mHC
        tokenizer: nanochat tokenizer
        meta: Training metadata dict
    """
    # resolve device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    # build model from checkpoint
    model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval")

    # freeze all parameters
    model.requires_grad_(False)
    model.eval()

    return model, tokenizer, meta
