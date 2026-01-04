"""
Pre-flight checks for mHC training.
Run this BEFORE starting a long training job to catch issues early.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py mhc_enabled=True
"""

import os
import sys
import torch
import time
from contextlib import nullcontext

# add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# configuration (can override via CLI)
mhc_enabled = True
mhc_num_streams = 4
mhc_sinkhorn_iters = 20
mhc_sinkhorn_tau = 0.05
depth = 8  # small model for quick testing
max_seq_len = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# simple CLI overrides (--key=value or key=value)
from ast import literal_eval
for arg in sys.argv[1:]:
    if '=' in arg:
        key, val = arg.lstrip('-').split('=', 1)
        if key in globals():
            try:
                globals()[key] = literal_eval(val)
            except:
                globals()[key] = val
            print(f"Override: {key} = {globals()[key]}")


def print_check(name, passed, details=""):
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}" + (f" - {details}" if details else ""))
    return passed


def main():
    print("=" * 60)
    print("mHC Pre-flight Checks")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"mHC enabled: {mhc_enabled}")
    print(f"Streams: {mhc_num_streams}")
    print(f"Model depth: {depth}")
    print()
    
    all_passed = True
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    
    # ---------------------------------------------------------------------
    # 1. model creation
    # ---------------------------------------------------------------------
    print("1. Model Creation")
    print("-" * 40)
    
    try:
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        n_layer = depth
        n_embd = depth * 64
        n_head = max(1, (n_embd + 127) // 128)
        
        config = GPTConfig(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=n_embd,
            sequence_len=max_seq_len,
            mhc_enabled=mhc_enabled,
            mhc_num_streams=mhc_num_streams,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_sinkhorn_tau=mhc_sinkhorn_tau,
        )
        
        model = GPT(config).to(device)
        model.train()
        
        all_passed &= print_check("Model created successfully", True)
        
        # count parameters
        total_params = sum(p.numel() for p in model.parameters())
        mhc_params = sum(p.numel() for n, p in model.named_parameters() if 'mhc' in n or 'stream_embed' in n)
        
        print(f"   Total parameters: {total_params:,}")
        if mhc_enabled:
            print(f"   mHC parameters: {mhc_params:,} ({100*mhc_params/total_params:.1f}%)")
            all_passed &= print_check("mHC parameters exist", mhc_params > 0)
        
    except Exception as e:
        all_passed &= print_check("Model created successfully", False, str(e))
        return False
    
    # ---------------------------------------------------------------------
    # 2. forward pass
    # ---------------------------------------------------------------------
    print("\n2. Forward Pass")
    print("-" * 40)
    
    try:
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # nanochat GPT returns loss when targets provided, logits when not
        with autocast_ctx:
            loss = model(input_ids, targets=input_ids)
            logits = model(input_ids)  # no targets = returns logits
        
        all_passed &= print_check("Forward pass works", True)
        all_passed &= print_check("Logits shape correct", 
            logits.shape == (batch_size, seq_len, vocab_size),
            f"got {logits.shape}")
        all_passed &= print_check("Loss is finite", 
            torch.isfinite(loss).item(),
            f"loss = {loss.item():.4f}")
        all_passed &= print_check("No NaN in logits", 
            not torch.isnan(logits).any().item())
        
    except Exception as e:
        all_passed &= print_check("Forward pass works", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # ---------------------------------------------------------------------
    # 3. backward pass & gradients
    # ---------------------------------------------------------------------
    print("\n3. Backward Pass & Gradients")
    print("-" * 40)
    
    try:
        model.zero_grad()
        loss.backward()
        
        all_passed &= print_check("Backward pass works", True)
        
        # check gradients exist
        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total = sum(1 for p in model.parameters())
        all_passed &= print_check("Gradients computed", 
            has_grads == total,
            f"{has_grads}/{total} params have grads")
        
        # check for NaN gradients
        nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        all_passed &= print_check("No NaN gradients", nan_grads == 0, f"{nan_grads} params have NaN grads")
        
        # check mHC-specific gradients
        if mhc_enabled:
            mhc_grad_info = []
            for name, p in model.named_parameters():
                if 'mhc' in name or 'stream_embed' in name:
                    if p.grad is not None:
                        grad_norm = p.grad.norm().item()
                        mhc_grad_info.append((name.split('.')[-1], grad_norm))
            
            has_nonzero = any(g > 0 for _, g in mhc_grad_info)
            all_passed &= print_check("mHC gradients are non-zero", has_nonzero)
            
            print("   mHC gradient norms:")
            for name, grad in mhc_grad_info[:6]:  # show first 6
                print(f"      {name}: {grad:.2e}")
        
    except Exception as e:
        all_passed &= print_check("Backward pass works", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # ---------------------------------------------------------------------
    # 4. training loop (5 steps)
    # ---------------------------------------------------------------------
    print("\n4. Mini Training Loop (5 steps)")
    print("-" * 40)
    
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        losses = []
        for step in range(5):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            optimizer.zero_grad()
            with autocast_ctx:
                loss = model(input_ids, targets=input_ids)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"   Step {step}: loss = {loss.item():.4f}")
        
        # check loss decreased (or at least didn't explode)
        all_passed &= print_check("Loss didn't explode", 
            losses[-1] < losses[0] * 2,
            f"start={losses[0]:.4f}, end={losses[-1]:.4f}")
        
        all_passed &= print_check("All losses finite", 
            all(l < 1e6 for l in losses))
        
    except Exception as e:
        all_passed &= print_check("Training loop works", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    
    # ---------------------------------------------------------------------
    # 5. stream divergence check (mHC only)
    # ---------------------------------------------------------------------
    if mhc_enabled:
        print("\n5. Stream Divergence Check")
        print("-" * 40)
        
        try:
            se = model.stream_embed.data
            se_norm = se / (se.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = se_norm @ se_norm.T
            n = se.shape[0]
            off_diag = sim_matrix[~torch.eye(n, dtype=bool, device=device)]
            avg_sim = off_diag.mean().item()
            
            all_passed &= print_check("Stream embeddings initialized differently", 
                avg_sim < 0.99,
                f"avg similarity = {avg_sim:.4f}")
            
            print(f"   Stream embed norms: {[f'{se[i].norm().item():.4f}' for i in range(n)]}")
            
        except Exception as e:
            all_passed &= print_check("Stream divergence check", False, str(e))
    
    # ---------------------------------------------------------------------
    # 6. memory check
    # ---------------------------------------------------------------------
    print("\n6. Memory Usage")
    print("-" * 40)
    
    if device == "cuda":
        mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"   Peak allocated: {mem_alloc:.2f} GB")
        print(f"   Peak reserved: {mem_reserved:.2f} GB")
        all_passed &= print_check("Memory usage reasonable", mem_alloc < 40, f"{mem_alloc:.1f} GB")
    else:
        print("   (skipped - not on CUDA)")
    
    # ---------------------------------------------------------------------
    # 7. generation test
    # ---------------------------------------------------------------------
    print("\n7. Generation Test")
    print("-" * 40)
    
    try:
        model.eval()
        prompt = "The meaning of life is"
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=device)
        
        with torch.no_grad():
            with autocast_ctx:
                for _ in range(20):
                    logits = model(input_ids)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
        
        output = tokenizer.decode(input_ids[0].tolist())
        all_passed &= print_check("Generation works", len(output) > len(prompt))
        print(f"   Generated: {output[:80]}...")
        
    except Exception as e:
        all_passed &= print_check("Generation works", False, str(e))
    
    # ---------------------------------------------------------------------
    # 8. determinism check
    # ---------------------------------------------------------------------
    print("\n8. Determinism Check")
    print("-" * 40)
    
    try:
        model.eval()
        torch.manual_seed(42)
        input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
        
        with torch.no_grad():
            with autocast_ctx:
                logits1 = model(input_ids)
                logits2 = model(input_ids)
        
        diff = (logits1 - logits2).abs().max().item()
        all_passed &= print_check("Forward pass is deterministic", diff < 1e-5, f"max diff = {diff:.2e}")
        
    except Exception as e:
        all_passed &= print_check("Determinism check", False, str(e))
    
    # ---------------------------------------------------------------------
    # summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_passed:
        print("\033[92m✓ ALL CHECKS PASSED - Ready to train!\033[0m")
    else:
        print("\033[91m✗ SOME CHECKS FAILED - Review before training\033[0m")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

