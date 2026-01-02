"""
Test script for GSM8K RL environment.

Run with: uv run python scripts/test_env.py --device cuda
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing import MultiStreamDecoder
from envs import GSM8KEnv


def test_env_basic(model_name: str, device: str = "cuda"):
    """Test basic environment functionality."""
    print(f"\n{'='*60}")
    print("Test 1: Environment Creation")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model: {model_name}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()
    
    # Wrap model
    print("Wrapping with MultiStreamDecoder...")
    wrapped = MultiStreamDecoder(
        base_model,
        n_streams=4,
        mixing_mode="row_stochastic",
    )
    wrapped.freeze_base()
    wrapped.eval()
    if device == "cuda":
        wrapped.mixing = wrapped.mixing.cuda()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create environment
    print("Creating GSM8KEnv...")
    env = GSM8KEnv(
        model=wrapped,
        tokenizer=tokenizer,
        data_path="data/gsm8k_train.jsonl",
        max_new_tokens=150,
        device=device,
    )
    
    print(f"✅ Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Dataset size: {len(env.data)}")
    
    return env


def test_env_reset(env: GSM8KEnv):
    """Test environment reset."""
    print(f"\n{'='*60}")
    print("Test 2: Environment Reset")
    print(f"{'='*60}")
    
    obs, info = env.reset(seed=42)
    
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation value: {obs}")
    print(f"   Question: {info['question'][:100]}...")
    print(f"   Expected answer: {info['expected_answer']}")
    print(f"   Prompt length: {info['prompt_length']} tokens")
    
    print(f"✅ Reset works correctly")
    return obs, info


def test_env_step(env: GSM8KEnv):
    """Test environment step with different actions."""
    print(f"\n{'='*60}")
    print("Test 3: Environment Step (All Gate Values)")
    print(f"{'='*60}")
    
    results = []
    
    for action in range(5):
        # Reset to same problem
        env.reset(seed=42)
        
        # Step with action
        obs, reward, terminated, truncated, info = env.step(action)
        
        results.append({
            "action": action,
            "g": info["gate_value"],
            "reward": reward,
            "predicted": info["predicted_answer"],
            "correct": info["correct"],
        })
        
        print(f"   Action {action} (g={info['gate_value']}):")
        print(f"     Predicted: {info['predicted_answer']}")
        print(f"     Expected: {info['expected_answer']}")
        print(f"     Correct: {info['correct']}")
        print(f"     Reward: {reward}")
    
    print(f"✅ Step works for all gate values")
    return results


def test_env_multiple_episodes(env: GSM8KEnv, n_episodes: int = 10):
    """Test running multiple episodes."""
    print(f"\n{'='*60}")
    print(f"Test 4: Multiple Episodes (n={n_episodes})")
    print(f"{'='*60}")
    
    env.reset_stats()
    
    for i in range(n_episodes):
        obs, info = env.reset()
        
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Episode {i+1}: g={info['gate_value']}, "
              f"correct={info['correct']}, "
              f"predicted={info['predicted_answer']}, "
              f"expected={info['expected_answer']}")
    
    stats = env.get_stats()
    print(f"\n   Cumulative stats:")
    print(f"     Episodes: {stats['episodes']}")
    print(f"     Correct: {stats['correct']}")
    print(f"     Accuracy: {stats['accuracy']:.1%}")
    
    print(f"✅ Multiple episodes work correctly")
    return stats


def test_env_render(env: GSM8KEnv):
    """Test environment rendering."""
    print(f"\n{'='*60}")
    print("Test 5: Environment Render")
    print(f"{'='*60}")
    
    env.reset(seed=123)
    env.render()
    
    print(f"✅ Render works correctly")


def main():
    parser = argparse.ArgumentParser(description="Test GSM8K RL environment")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model to test with",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for multi-episode test",
    )
    args = parser.parse_args()
    
    print(f"Testing GSM8K environment with {args.model} on {args.device}")
    
    # Run tests
    env = test_env_basic(args.model, args.device)
    test_env_reset(env)
    test_env_step(env)
    test_env_multiple_episodes(env, args.episodes)
    test_env_render(env)
    
    print(f"\n{'='*60}")
    print("All environment tests completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

