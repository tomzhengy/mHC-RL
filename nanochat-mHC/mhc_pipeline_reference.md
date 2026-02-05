# mHC Pipeline Reference

Commands to continue training after pretraining is complete.

## Prerequisites

Your mHC pretrained model should be at:

```
~/.cache/nanochat/base_checkpoints/d20/
```

## 1. Download Identity Conversations

```bash
curl -L -o $HOME/.cache/nanochat/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

## 2. Midtraining

Teaches conversation special tokens, tool use, multiple choice.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=8 --run=mhc-d20-mid
```

Saves to: `~/.cache/nanochat/mid_checkpoints/d20/`

### Eval after midtraining

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

## 3. Supervised Finetuning (SFT)

Domain adaptation on ARC, GSM8K, SmolTalk, etc.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device_batch_size=4 --run=mhc-d20-sft
```

Saves to: `~/.cache/nanochat/chatsft_checkpoints/d20/`

### Final eval (includes GSM8K, ARC, etc.)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

## 4. RL (Optional)

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=mhc-d20-rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

## Notes

- Use `device_batch_size=8` for midtraining (matches pretraining, avoids OOM with mHC)
- Use `device_batch_size=4` for SFT (variable length sequences need more memory)
- Model config (mHC settings) is loaded from `meta.json` automatically
- All scripts auto-detect the `d20` model tag from checkpoint directory

## HuggingFace Upload

Model uploaded to: https://huggingface.co/tomzhengy/nanochat-mhc-d20-static

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='/root/.cache/nanochat/base_checkpoints/d20/model_021400.pt',
    path_in_repo='model.pt',
    repo_id='tomzhengy/nanochat-mhc-d20-static'
)
```
