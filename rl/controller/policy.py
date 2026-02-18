"""
Gate policy network for REINFORCE.

Simple MLP policy for contextual bandit (no value head needed for REINFORCE).
Uses average reward as baseline, not learned value function.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class GatePolicy(nn.Module):
    """
    Simple MLP policy for selecting mHC gate values.

    Architecture: 2-layer MLP with ReLU activations.
    No value head - REINFORCE uses average reward as baseline.

    Default config:
        obs_dim=2 (prompt_length, question_length)
        hidden_dim=64
        n_actions=5 (gate values: 0.0, 0.25, 0.5, 0.75, 1.0)
    """

    # gate values corresponding to action indices
    GATE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

    def __init__(self, obs_dim: int = 2, hidden_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.n_actions = n_actions
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action logits.

        Args:
            obs: Observation tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            Logits tensor of shape (batch, n_actions) or (n_actions,)
        """
        return self.network(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and return log probability and entropy.

        Args:
            obs: Observation tensor
            deterministic: If True, return argmax action instead of sampling

        Returns:
            action: Sampled or argmax action
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of given actions under current policy.

        Args:
            obs: Observation tensor
            action: Action tensor

        Returns:
            Log probability of the actions
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)

    def action_to_gate(self, action: int) -> float:
        """Convert action index to gate value."""
        return self.GATE_VALUES[action]
