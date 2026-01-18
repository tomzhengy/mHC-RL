"""
Evaluation script for mHC gate controller.

Includes:
- Gate sanity check: verify gate actually affects model output
- Fixed gate baselines: evaluate each gate value independently
- Random baseline: random gate selection
- Learned policy evaluation: evaluate trained policy
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "nanochat-mHC"))

from controller.model_loader import load_frozen_mhc_model
from controller.policy import GatePolicy
from envs.gsm8k_env import GSM8KEnv


def verify_gate_affects_output(model, tokenizer, prompt: str = "What is 2 + 2?"):
    """
    Verify that changing the gate value actually affects model output.

    Sets g=0.0 and g=1.0 on the same prompt and asserts that logits differ.
    This catches bugs where the gate is accidentally reset or ignored.

    Args:
        model: nanochat GPT model with mHC
        tokenizer: nanochat tokenizer
        prompt: Test prompt to use

    Raises:
        AssertionError: If gate has no effect on logits
    """
    print(f"running gate sanity check on prompt: '{prompt}'")

    # tokenize prompt
    bos = tokenizer.get_bos_token_id()
    tokens = tokenizer.encode(prompt, prepend=bos)
    input_ids = torch.tensor([tokens], device=model.get_device())

    # forward with g=0.0
    model.set_mhc_gate(0.0)
    with torch.no_grad():
        logits_g0 = model(input_ids)

    # forward with g=1.0
    model.set_mhc_gate(1.0)
    with torch.no_grad():
        logits_g1 = model(input_ids)

    # compute difference
    diff = (logits_g0 - logits_g1).abs().mean().item()

    if diff < 1e-6:
        raise AssertionError(
            f"gate has no effect! logit diff = {diff:.2e}\n"
            "check that set_mhc_gate() is being called correctly"
        )

    print(f"gate sanity check passed: logit diff = {diff:.4f}")
    return diff


def run_fixed_gate(env: GSM8KEnv, gate_value: float, n_episodes: int) -> dict:
    """
    Run evaluation with a fixed gate value.

    Args:
        env: GSM8K environment
        gate_value: Fixed gate value to use (0.0, 0.25, 0.5, 0.75, 1.0)
        n_episodes: Number of episodes to run

    Returns:
        Dict with accuracy and episode details
    """
    # find action index for this gate value
    action = env.GATE_VALUES.index(gate_value)

    correct = 0
    total = 0
    results = []

    for i in range(n_episodes):
        obs, info = env.reset()
        obs, reward, terminated, truncated, step_info = env.step(action)

        correct += int(step_info["correct"])
        total += 1
        results.append({
            "question": step_info["question"][:100],
            "expected": step_info["expected_answer"],
            "predicted": step_info["predicted_answer"],
            "correct": step_info["correct"],
        })

        if (i + 1) % 10 == 0:
            print(f"  g={gate_value}: {i+1}/{n_episodes} episodes, accuracy={correct/total:.1%}")

    return {
        "gate_value": gate_value,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "results": results,
    }


def run_random_gate(env: GSM8KEnv, n_episodes: int, seed: int = 42) -> dict:
    """
    Run evaluation with random gate selection.

    Args:
        env: GSM8K environment
        n_episodes: Number of episodes to run
        seed: Random seed

    Returns:
        Dict with accuracy and episode details
    """
    rng = np.random.default_rng(seed)

    correct = 0
    total = 0
    action_counts = defaultdict(int)

    for i in range(n_episodes):
        obs, info = env.reset()
        action = rng.integers(0, len(env.GATE_VALUES))
        action_counts[action] += 1
        obs, reward, terminated, truncated, step_info = env.step(action)

        correct += int(step_info["correct"])
        total += 1

        if (i + 1) % 10 == 0:
            print(f"  random: {i+1}/{n_episodes} episodes, accuracy={correct/total:.1%}")

    return {
        "gate_value": "random",
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "action_counts": dict(action_counts),
    }


def run_policy(
    env: GSM8KEnv,
    policy: GatePolicy,
    n_episodes: int,
    deterministic: bool = True,
) -> dict:
    """
    Run evaluation with a learned policy.

    Args:
        env: GSM8K environment
        policy: Trained GatePolicy network
        n_episodes: Number of episodes to run
        deterministic: If True, use argmax actions; else sample

    Returns:
        Dict with accuracy and per-gate breakdown
    """
    policy.eval()
    device = next(policy.parameters()).device

    correct = 0
    total = 0
    action_counts = defaultdict(int)
    per_gate_correct = defaultdict(int)
    per_gate_total = defaultdict(int)

    for i in range(n_episodes):
        obs, info = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = policy.get_action(obs_tensor, deterministic=deterministic)
        action = action.item()
        action_counts[action] += 1

        obs, reward, terminated, truncated, step_info = env.step(action)

        correct += int(step_info["correct"])
        total += 1
        per_gate_correct[action] += int(step_info["correct"])
        per_gate_total[action] += 1

        if (i + 1) % 10 == 0:
            print(f"  policy: {i+1}/{n_episodes} episodes, accuracy={correct/total:.1%}")

    # compute per-gate accuracy
    per_gate_accuracy = {}
    for action in range(len(env.GATE_VALUES)):
        if per_gate_total[action] > 0:
            per_gate_accuracy[action] = per_gate_correct[action] / per_gate_total[action]
        else:
            per_gate_accuracy[action] = None

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "action_counts": dict(action_counts),
        "per_gate_accuracy": per_gate_accuracy,
    }


def run_baselines(
    env: GSM8KEnv,
    n_episodes: int,
    gate_values: list = None,
) -> dict:
    """
    Run all baselines: fixed gates + random.

    Args:
        env: GSM8K environment
        n_episodes: Number of episodes per baseline
        gate_values: List of gate values to test (default: all 5)

    Returns:
        Dict with results for each baseline
    """
    if gate_values is None:
        gate_values = env.GATE_VALUES

    results = {}

    # fixed gate baselines
    for g in gate_values:
        print(f"\nevaluating fixed gate g={g}...")
        env.reset_stats()
        result = run_fixed_gate(env, g, n_episodes)
        results[f"g={g}"] = result
        print(f"  g={g}: accuracy={result['accuracy']:.1%}")

    # random baseline
    print("\nevaluating random gate...")
    env.reset_stats()
    result = run_random_gate(env, n_episodes)
    results["random"] = result
    print(f"  random: accuracy={result['accuracy']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="evaluate mhc gate controller")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="path to model checkpoint directory")
    parser.add_argument("--step", type=int, default=None,
                        help="checkpoint step to load (default: latest)")
    parser.add_argument("--data_path", type=str, default="data/gsm8k_train.jsonl",
                        help="path to gsm8k data file")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="number of episodes per evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="device to use")
    parser.add_argument("--sanity_check", action="store_true",
                        help="run gate sanity check only")
    parser.add_argument("--run_baselines", action="store_true",
                        help="run baseline evaluations")
    parser.add_argument("--policy_path", type=str, default=None,
                        help="path to trained policy checkpoint")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "both"],
                        help="which data split(s) to evaluate on")
    args = parser.parse_args()

    # find latest step if not specified
    if args.step is None:
        from nanochat.checkpoint_manager import find_last_step
        args.step = find_last_step(args.checkpoint_dir)
        print(f"using latest checkpoint step: {args.step}")

    # load model
    print(f"loading model from {args.checkpoint_dir} step {args.step}...")
    model, tokenizer, meta = load_frozen_mhc_model(
        args.checkpoint_dir, args.step, args.device
    )
    print(f"model loaded: {meta.get('model_config', {})}")

    # sanity check
    if args.sanity_check:
        verify_gate_affects_output(model, tokenizer)
        return

    # determine which splits to run
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]

    all_results = {}
    for split in splits:
        # adjust data path for split
        if split == "train":
            data_path = args.data_path
        else:
            # assume test file follows same naming pattern
            data_path = args.data_path.replace("train", "test")

        print(f"\n{'='*60}")
        print(f"evaluating on {split} split: {data_path}")
        print(f"{'='*60}")

        # create environment
        try:
            env = GSM8KEnv(
                model=model,
                tokenizer=tokenizer,
                data_path=data_path,
                device=args.device,
            )
        except FileNotFoundError as e:
            print(f"warning: {e}")
            print(f"skipping {split} split")
            continue

        # run baselines
        if args.run_baselines:
            results = run_baselines(env, args.n_episodes)
            all_results[split] = results

            # print summary
            print(f"\n{split} split summary:")
            print("-" * 40)
            for name, result in results.items():
                print(f"  {name}: {result['accuracy']:.1%}")

        # run policy evaluation
        if args.policy_path:
            print(f"\nloading policy from {args.policy_path}...")
            policy = GatePolicy()
            policy.load_state_dict(torch.load(args.policy_path, map_location=args.device))
            policy = policy.to(args.device)

            print(f"\nevaluating learned policy on {split}...")
            result = run_policy(env, policy, args.n_episodes)
            all_results[f"{split}_policy"] = result

            print(f"\npolicy results on {split}:")
            print(f"  accuracy: {result['accuracy']:.1%}")
            print(f"  action distribution: {result['action_counts']}")
            print(f"  per-gate accuracy: {result['per_gate_accuracy']}")

    # final summary across splits
    if len(all_results) > 0:
        print("\n" + "=" * 60)
        print("final summary")
        print("=" * 60)
        for key, results in all_results.items():
            if isinstance(results, dict) and "accuracy" in results:
                print(f"  {key}: {results['accuracy']:.1%}")
            elif isinstance(results, dict):
                for name, result in results.items():
                    print(f"  {key}/{name}: {result['accuracy']:.1%}")


if __name__ == "__main__":
    main()
