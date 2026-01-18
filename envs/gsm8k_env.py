"""
GSM8K Gymnasium Environment for RL-controlled routing.

Episode structure:
- reset(): Sample a GSM8K problem, return observation
- step(action): Set routing gate g, generate answer, return reward

Action space: Discrete(5) -> g in {0.0, 0.25, 0.5, 0.75, 1.0}
Observation space: Box with 2 features (prompt_length, question_length)
Reward: +1 for correct answer, 0 otherwise
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# add nanochat to path
sys.path.insert(0, str(Path(__file__).parent.parent / "nanochat-mHC"))
from nanochat.gpt import GPT
from nanochat.engine import Engine


class GSM8KEnv(gym.Env):
    """
    Gymnasium environment for GSM8K math problems with RL-controlled routing.

    Each episode is ONE math problem:
    1. reset() loads a problem
    2. step(action) generates an answer with routing gate g
    3. Reward based on exact match

    This is a one-step environment (done=True after step).
    """

    metadata = {"render_modes": ["human"]}

    # Gate values corresponding to discrete actions
    GATE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(
        self,
        model: GPT,
        tokenizer,
        data_path: str = "data/gsm8k_train.jsonl",
        max_new_tokens: int = 256,
        prompt_template: str = "Question: {question}\nAnswer: Let me solve this step by step.",
        seed: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: nanochat GPT model with mHC (DynamicMHC)
            tokenizer: nanochat tokenizer (RustBPETokenizer)
            data_path: Path to GSM8K JSONL file
            max_new_tokens: Maximum tokens to generate per episode
            prompt_template: Template for formatting questions
            seed: Random seed for reproducibility
            device: Device for model inference
        """
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.engine = Engine(model, tokenizer)
        self.max_new_tokens = max_new_tokens
        self.prompt_template = prompt_template
        self.device = device

        # sequence_len from model config for feature normalization
        self.sequence_len = model.config.sequence_len

        # Load dataset
        self.data = self._load_data(data_path)
        if len(self.data) == 0:
            raise ValueError(f"No data loaded from {data_path}")

        # Action space: 5 discrete gate values
        self.action_space = spaces.Discrete(len(self.GATE_VALUES))

        # Observation space: 2 features
        # - prompt_length: normalized by model.config.sequence_len
        # - question_length: character count clamped to 1000
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Episode state
        self._current_problem: Optional[dict] = None
        self._current_prompt: Optional[str] = None
        self._current_tokens: Optional[list] = None
        self._rng = np.random.default_rng(seed)

        # Statistics tracking
        self.episode_count = 0
        self.correct_count = 0
    
    def _load_data(self, data_path: str) -> list[dict]:
        """Load GSM8K data from JSONL file."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Run `python scripts/prepare_data.py` first."
            )
        
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """
        Extract numeric answer from generated text.
        
        Looks for common patterns:
        - "#### 42" (GSM8K format)
        - "= 42"
        - "answer is 42"
        - Last number in text
        """
        # Try GSM8K format first
        match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
        if match:
            return match.group(1).replace(",", "")
        
        # Try common answer patterns
        patterns = [
            r"(?:answer|result|total|solution)\s*(?:is|=|:)\s*\$?(-?\d+(?:,\d+)*(?:\.\d+)?)",
            r"=\s*\$?(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$",
            r"\$(-?\d+(?:,\d+)*(?:\.\d+)?)\$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "")
        
        # Fall back to last number in text
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "")
        
        return None
    
    def _normalize_number(self, s: Optional[str]) -> Optional[float]:
        """
        Convert string to canonical numeric form.
        
        Handles: "42", "42.0", " 42 ", "42,000", etc.
        Returns None if not parseable.
        """
        if s is None:
            return None
        
        # Clean up: strip whitespace, remove commas
        s = s.strip().replace(",", "")
        
        try:
            # Try parsing as float
            return float(s)
        except ValueError:
            return None
    
    def _answers_match(self, predicted: Optional[str], expected: Optional[str]) -> bool:
        """
        Compare answers with numeric normalization.
        
        - "42" == "42.0" == " 42 " → True
        - GSM8K answers are integers, so we compare as int when possible
        - Falls back to string comparison if not numeric
        """
        if predicted is None or expected is None:
            return False
        
        pred_num = self._normalize_number(predicted)
        exp_num = self._normalize_number(expected)
        
        # If both parse as numbers, compare numerically
        if pred_num is not None and exp_num is not None:
            # GSM8K uses integers - check if they're equal as integers
            # (handles 42 vs 42.0)
            if pred_num == int(pred_num) and exp_num == int(exp_num):
                return int(pred_num) == int(exp_num)
            # Otherwise compare as floats with small tolerance
            return abs(pred_num - exp_num) < 1e-6
        
        # Fall back to string comparison (stripped)
        return predicted.strip() == expected.strip()
    
    def _get_expected_answer(self, problem: dict) -> Optional[str]:
        """
        Get expected answer from problem dict with fallback logic.
        
        Handles different dataset formats:
        - Our prepared format: {"final_answer": "42"}
        - Raw HF GSM8K: {"answer": "... #### 42"}
        - Raw HF with "Answer" key
        """
        # Try our prepared format first
        if "final_answer" in problem:
            return problem["final_answer"]
        
        # Fallback: extract from "answer" field (raw HF format)
        if "answer" in problem:
            return self._extract_answer(problem["answer"])
        
        # Fallback: try "Answer" key (some datasets use capital)
        if "Answer" in problem:
            ans = problem["Answer"]
            # Could be a number directly or a string
            if isinstance(ans, (int, float)):
                return str(int(ans) if ans == int(ans) else ans)
            return self._extract_answer(str(ans))
        
        return None
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment with a new GSM8K problem.

        Returns:
            observation: Array with 2 features (prompt_length, question_length)
            info: Dict with problem details
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random problem
        idx = self._rng.integers(0, len(self.data))
        self._current_problem = self.data[idx]

        # Format prompt
        self._current_prompt = self.prompt_template.format(
            question=self._current_problem["question"]
        )

        # Tokenize using nanochat tokenizer (returns list of ints)
        bos = self.tokenizer.get_bos_token_id()
        self._current_tokens = self.tokenizer.encode(self._current_prompt, prepend=bos)

        # Compute observation: 2 features
        prompt_length = len(self._current_tokens)
        question_length = len(self._current_problem["question"])
        obs = np.array([
            prompt_length / float(self.sequence_len),
            min(question_length, 1000) / 1000.0,
        ], dtype=np.float32)

        info = {
            "question": self._current_problem["question"],
            "expected_answer": self._get_expected_answer(self._current_problem),
            "prompt_length": prompt_length,
            "question_length": question_length,
        }

        return obs, info
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one episode step (generate answer with given routing gate).

        Args:
            action: Discrete action (0-4) mapped to gate value

        Returns:
            observation: Same as reset (episode ends)
            reward: +1 for correct, 0 otherwise
            terminated: Always True (one-step episodes)
            truncated: Always False
            info: Generation details
        """
        if self._current_problem is None:
            raise RuntimeError("Must call reset() before step()")

        # Map action to gate value
        g = self.GATE_VALUES[action]

        # Set mHC gate before generation
        self.model.set_mhc_gate(g)

        # Generate answer using nanochat Engine
        prompt_len = len(self._current_tokens)
        results, masks = self.engine.generate_batch(
            self._current_tokens,
            num_samples=1,
            max_tokens=self.max_new_tokens,
            temperature=0.0,  # greedy decoding
        )

        # results[0] contains prompt + generated tokens
        # slice off the prompt to get just the response
        response_ids = results[0][prompt_len:]
        response = self.tokenizer.decode(response_ids)

        # Check if generation was truncated (hit max_new_tokens without EOS)
        num_generated = len(response_ids)
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        hit_eos = assistant_end in response_ids or bos in response_ids
        response_truncated = (num_generated >= self.max_new_tokens) and not hit_eos

        # Extract predicted answer
        predicted = self._extract_answer(response)
        expected = self._get_expected_answer(self._current_problem)

        # Compute reward (numeric match)
        correct = self._answers_match(predicted, expected)
        reward = 1.0 if correct else 0.0

        # Update statistics
        self.episode_count += 1
        if correct:
            self.correct_count += 1

        # Build info
        question_length = len(self._current_problem["question"])
        info = {
            "question": self._current_problem["question"],
            "expected_answer": expected,
            "predicted_answer": predicted,
            "correct": correct,
            "gate_value": g,
            "generated_text": response[:500],  # truncate for logging
            "cumulative_accuracy": self.correct_count / self.episode_count,
            "response_truncated": response_truncated,
            "num_tokens_generated": num_generated,
        }

        # Observation doesn't change (episode ends)
        obs = np.array([
            prompt_len / float(self.sequence_len),
            min(question_length, 1000) / 1000.0,
        ], dtype=np.float32)

        # Episode always terminates after one step
        terminated = True
        truncated = False

        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render current problem (human-readable)."""
        if self._current_problem is None:
            print("No problem loaded. Call reset() first.")
            return
        
        print(f"\n{'='*60}")
        print(f"Question: {self._current_problem['question']}")
        print(f"Expected: {self._current_problem['final_answer']}")
        print(f"{'='*60}")
    
    def get_stats(self) -> dict[str, float]:
        """Get cumulative statistics."""
        return {
            "episodes": self.episode_count,
            "correct": self.correct_count,
            "accuracy": self.correct_count / max(1, self.episode_count),
        }
    
    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self.episode_count = 0
        self.correct_count = 0


class GSM8KVecEnv:
    """
    Vectorized wrapper for running multiple GSM8K environments in parallel.

    Useful for batched REINFORCE rollouts.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer,
        num_envs: int = 4,
        **env_kwargs,
    ):
        """
        Args:
            model: Shared model across environments
            tokenizer: Shared tokenizer
            num_envs: Number of parallel environments
            **env_kwargs: Arguments passed to GSM8KEnv
        """
        self.num_envs = num_envs
        self.envs = [
            GSM8KEnv(model, tokenizer, seed=i, **env_kwargs)
            for i in range(num_envs)
        ]
        
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
    
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, list[dict]]:
        """Reset all environments."""
        obs_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        
        return np.stack(obs_list), info_list
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments with CleanRL-style terminal info."""
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            # Auto-reset on termination, but preserve terminal state for logging
            if terminated or truncated:
                # Store terminal observation/info BEFORE reset
                # This lets CleanRL attribute reward to correct episode
                info["terminal_observation"] = obs.copy()
                info["terminal_info"] = {
                    k: v for k, v in info.items() 
                    if k not in ("terminal_observation", "terminal_info")
                }
                
                # Now reset for next episode
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list,
        )
    
    def get_stats(self) -> dict[str, float]:
        """Get aggregated statistics across all environments."""
        total_episodes = sum(env.episode_count for env in self.envs)
        total_correct = sum(env.correct_count for env in self.envs)
        return {
            "episodes": total_episodes,
            "correct": total_correct,
            "accuracy": total_correct / max(1, total_episodes),
        }

