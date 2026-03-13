"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_size = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, action_dim * chunk_size))
        self.model = nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.sample_actions(state)
        return mse_loss(pred_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        actions = self.model.forward(state)
        return actions.reshape(-1, self.chunk_size, self.action_dim)
        


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # v_θ takes (state, noisy_action_chunk_flat, τ) as input
        input_dim = state_dim + action_dim * chunk_size + 1
        layers: list[nn.Module] = []
        in_size = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        # Output is the predicted velocity, same shape as a flattened action chunk
        layers.append(nn.Linear(in_size, action_dim * chunk_size))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        # A_0 ~ N(0, I): sample noise with same shape as action_chunk
        a_0 = torch.randn_like(action_chunk)  # (batch, chunk_size, action_dim)
        # τ ~ U(0, 1)
        tau = torch.rand(batch_size, 1, 1, device=state.device)  # (batch, 1, 1)
        # Interpolation: A_τ = τ * A + (1 - τ) * A_0
        a_tau = tau * action_chunk + (1 - tau) * a_0  # (batch, chunk_size, action_dim)
        # Target velocity: A - A_0
        target = action_chunk - a_0  # (batch, chunk_size, action_dim)
        # Flatten noisy action chunk for network input
        a_tau_flat = a_tau.reshape(batch_size, -1)  # (batch, chunk_size * action_dim)
        tau_flat = tau.reshape(batch_size, 1)  # (batch, 1)
        # Network input: concatenate state, flattened noisy action, and τ
        net_input = torch.cat([state, a_tau_flat, tau_flat], dim=-1)
        # Predicted velocity
        v_pred = self.model(net_input)  # (batch, chunk_size * action_dim)
        v_pred = v_pred.reshape(batch_size, self.chunk_size, self.action_dim)
        return mse_loss(v_pred, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        # Start from pure noise: A_0 ~ N(0, I)
        a_tau = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=state.device
        )
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = i / num_steps
            tau_tensor = torch.full(
                (batch_size, 1), tau, device=state.device
            )
            a_tau_flat = a_tau.reshape(batch_size, -1)
            net_input = torch.cat([state, a_tau_flat, tau_tensor], dim=-1)
            v_pred = self.model(net_input)
            v_pred = v_pred.reshape(batch_size, self.chunk_size, self.action_dim)
            # Euler step: A_{τ + 1/n} = A_τ + (1/n) * v_θ(o, A_τ, τ)
            a_tau = a_tau + dt * v_pred
        return a_tau  # (batch, chunk_size, action_dim)



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
