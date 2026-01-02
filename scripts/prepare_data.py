#!/usr/bin/env python3
"""
Download and prepare GSM8K dataset for RL training.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --output_dir data/
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def extract_final_answer(answer_text: str) -> str | None:
    """
    Extract the final numeric answer from GSM8K answer text.
    
    GSM8K answers end with "#### <number>" format.
    Returns the number as a string (to preserve formatting like negative signs).
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        # Remove commas from numbers like "1,234"
        return match.group(1).replace(",", "")
    return None


def prepare_gsm8k(output_dir: Path, include_svamp: bool = False) -> dict:
    """
    Download and prepare GSM8K (and optionally SVAMP for OOD eval).
    
    Returns stats about the prepared data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    # ─────────────────────────────────────────────────────────────
    # GSM8K - Main training/eval dataset
    # ─────────────────────────────────────────────────────────────
    print("📥 Downloading GSM8K from Hugging Face...")
    gsm8k = load_dataset("openai/gsm8k", "main")
    
    for split in ["train", "test"]:
        data = gsm8k[split]
        processed = []
        
        for item in tqdm(data, desc=f"Processing GSM8K {split}"):
            question = item["question"]
            answer_text = item["answer"]
            final_answer = extract_final_answer(answer_text)
            
            if final_answer is None:
                print(f"⚠️  Could not extract answer from: {answer_text[:100]}...")
                continue
            
            processed.append({
                "question": question,
                "answer_text": answer_text,  # Full chain-of-thought
                "final_answer": final_answer,  # Just the number for reward
                "source": "gsm8k",
            })
        
        # Save as JSONL for streaming during training
        output_path = output_dir / f"gsm8k_{split}.jsonl"
        with open(output_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        
        stats[f"gsm8k_{split}"] = len(processed)
        print(f"✅ Saved {len(processed)} examples to {output_path}")
    
    # ─────────────────────────────────────────────────────────────
    # SVAMP - OOD evaluation dataset (optional)
    # ─────────────────────────────────────────────────────────────
    if include_svamp:
        print("\n📥 Downloading SVAMP for OOD evaluation...")
        try:
            svamp = load_dataset("ChilleD/SVAMP", split="test")
            processed = []
            
            for item in tqdm(svamp, desc="Processing SVAMP"):
                # SVAMP has different field names
                question = item.get("question") or item.get("Body") + " " + item.get("Question", "")
                final_answer = str(item.get("Answer", ""))
                
                processed.append({
                    "question": question,
                    "answer_text": "",  # SVAMP doesn't have CoT
                    "final_answer": final_answer,
                    "source": "svamp",
                })
            
            output_path = output_dir / "svamp_test.jsonl"
            with open(output_path, "w") as f:
                for item in processed:
                    f.write(json.dumps(item) + "\n")
            
            stats["svamp_test"] = len(processed)
            print(f"✅ Saved {len(processed)} examples to {output_path}")
            
        except Exception as e:
            print(f"⚠️  Could not download SVAMP: {e}")
            print("   You can add it later for OOD evaluation.")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset for RL training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--include_svamp",
        action="store_true",
        help="Also download SVAMP for OOD evaluation",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("GSM8K Data Preparation")
    print("=" * 60)
    
    stats = prepare_gsm8k(output_dir, include_svamp=args.include_svamp)
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    for name, count in stats.items():
        print(f"  {name}: {count:,} examples")
    
    print(f"\n📁 Data saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Check the data: head -5 data/gsm8k_train.jsonl")
    print("  2. Start building the RL environment in envs/gsm8k_env.py")


if __name__ == "__main__":
    main()

