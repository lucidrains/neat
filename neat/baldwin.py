"""Baldwinian NEAT: fixed predictor + gradient-updated memory nodes.

    delta_hat = W @ m          # W frozen
    m <- m - lr * dL/dm        # only m is a leaf
    controller sees [x, m]
"""

from __future__ import annotations

import torch

from neat.neat import NEAT


def exists(v):
    return v is not None


class PredictiveCodingMemory:
    """A handful of gradient-updated memory nodes and a fixed linear predictor."""

    def __init__(
        self,
        dim: int,
        pop_size: int,
        num_memory: int = 4,
        *,
        lr: float = 0.05,
        ttt: bool = True,
        seed: int | None = None
    ):
        assert dim > 0 and pop_size > 0 and num_memory > 0

        self.dim = dim
        self.pop_size = pop_size
        self.num_memory = num_memory
        self.lr = lr
        self.ttt = ttt

        generator = torch.Generator()
        if exists(seed):
            generator.manual_seed(seed)

        self.weights = torch.randn(dim, num_memory, generator = generator) / num_memory ** 0.5
        self.memory = torch.zeros(pop_size, num_memory)

    @property
    def state_dim(self) -> int:
        return self.dim + self.num_memory

    def reset(self):
        self.memory.zero_()

    def set_ttt(self, enabled: bool):
        self.ttt = bool(enabled)

    def observe(self, obs, indices = None):
        was_numpy = not torch.is_tensor(obs)
        obs = torch.as_tensor(obs, dtype = torch.float32)

        single = obs.ndim == 1
        if single:
            obs = obs[None]

        memory = self.memory[indices] if exists(indices) else self.memory
        out = torch.cat((obs, memory), dim = -1)

        if single:
            out = out[0]

        return out.cpu().numpy() if was_numpy else out

    def update(self, obs, next_obs, dones = None) -> float | None:
        if not self.ttt or self.lr <= 0.0:
            return None

        obs = torch.as_tensor(obs, dtype = torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype = torch.float32)

        if obs.ndim == 1:
            obs, next_obs = obs[None], next_obs[None]

        valid = ~torch.as_tensor(dones).bool() if exists(dones) else torch.ones(len(obs), dtype = torch.bool)
        if not valid.any():
            return None

        delta = (next_obs - obs).detach()
        memory = self.memory.detach().clone().requires_grad_(True)

        error = delta - memory @ self.weights.t()
        loss = error[valid].square().sum(dim = -1).mean()
        loss.backward()

        with torch.no_grad():
            grad = memory.grad if exists(memory.grad) else torch.zeros_like(memory)
            memory = memory.detach() - self.lr * grad.clamp(-1.0, 1.0)
            memory[~valid] = 0.0
            self.memory = memory.clamp(-5.0, 5.0)

        return loss.item()


class BaldwinNEAT(NEAT):
    """NEAT with `[x, m]` as controller input.

    `ttt=False` freezes `m` at zero: same input shape, no lifetime adaptation.
    """

    def __init__(
        self,
        *dims,
        pop_size: int,
        num_memory: int = 4,
        memory_lr: float = 0.05,
        ttt: bool = True,
        carry_across_episodes: bool = False,
        memory_seed: int | None = None,
        **kwargs
    ):
        assert len(dims) >= 2

        self.memory = PredictiveCodingMemory(
            dims[0],
            pop_size,
            num_memory = num_memory,
            lr = memory_lr,
            ttt = ttt,
            seed = memory_seed
        )
        self.carry_across_episodes = carry_across_episodes

        super().__init__(self.memory.state_dim, *dims[1:], pop_size = pop_size, **kwargs)

    def reset_recurrent_state(self):
        super().reset_recurrent_state()
        self.memory.reset()

    def reset_episode(self):
        super().reset_recurrent_state()
        if not self.carry_across_episodes:
            self.memory.reset()

    def set_ttt(self, enabled: bool):
        self.memory.set_ttt(enabled)
        return self

    def observe_transition(self, obs, next_obs, dones = None):
        return self.memory.update(obs, next_obs, dones = dones)

    def forward(self, state, *args, **kwargs):
        state = self.memory.observe(state)
        return super().forward(state, *args, **kwargs)

    def single_forward(self, index: int, state, *args, **kwargs):
        state = self.memory.observe(state, indices = [index])
        return super().single_forward(index, state, *args, **kwargs)
