"""
PettingZoo → wmatrpo env wrapper.

PettingZoo is the standard MARL benchmark suite (the multi-agent analogue of
Gymnasium). Wrapping it lets us run W-MATRPO and IPPO on the same environments
used in MADDPG, MAPPO, IPPO, COMA, and related papers — which is what a
candidacy committee will expect for any "real" benchmark.

Currently this wrapper is a SCAFFOLD. The wmatrpo algorithms in this package
were built for stateless, single-step cooperative tasks (the differential game
and El Farol). PettingZoo environments are multi-step and state-dependent. To
actually train on them, we need three additional capabilities:

  1. State-dependent policies (μ and σ as functions of observation).
  2. Episode-level trajectory collection with discounted returns / GAE.
  3. Optionally recurrent policies for partial observability.

This file provides the scaffold and the env wrapper. The Simple-Spread runner
(`wmatrpo.scripts.simple_spread`) raises NotImplementedError on the missing
pieces with a clear pointer to what to implement.

Suggested install:
    pip install pettingzoo[mpe]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PettingZooConfig:
    env_name: str = "simple_spread_v3"
    n_agents: int = 3
    max_cycles: int = 25
    continuous_actions: bool = True
    state_dim: int = -1                # filled in at construction
    action_dim_per_agent: int = -1     # filled in at construction


class PettingZooEnv:
    """
    Thin wrapper around a PettingZoo ParallelEnv that exposes the same surface
    as our DifferentialGameEnv / ElFarolEnv: `reward`, `initial_observation`,
    `clamp_actions`. Plus a `reset`/`step` pair for episode rollout (which the
    stateless envs don't need).

    Caveat: this exposes a *single-step* interface only via `reward(actions)`;
    the proper episodic interface is `reset()` / `step(actions)`. Algorithms
    that want to use PettingZoo properly should switch to the episodic path.
    """

    def __init__(self, cfg: PettingZooConfig):
        try:
            import pettingzoo.mpe as mpe
        except ImportError as e:
            raise ImportError(
                "PettingZoo is required for this wrapper. Install with:\n"
                "    pip install pettingzoo[mpe]\n"
                f"(import error: {e})"
            )

        # ---- build the underlying parallel env ----
        env_factory_name = cfg.env_name
        if not env_factory_name.startswith("simple_"):
            raise NotImplementedError(
                f"Only MPE 'simple_*' envs scaffolded so far; got {env_factory_name}."
            )
        env_module = __import__(f"pettingzoo.mpe.{env_factory_name}",
                                fromlist=["parallel_env"])
        self._env = env_module.parallel_env(
            N=cfg.n_agents,
            max_cycles=cfg.max_cycles,
            continuous_actions=cfg.continuous_actions,
        )
        self._obs, _info = self._env.reset(seed=0)

        # discover dimensions from the env
        agent_ids = self._env.possible_agents
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        sample_agent = agent_ids[0]
        obs_space = self._env.observation_space(sample_agent)
        act_space = self._env.action_space(sample_agent)

        cfg.state_dim = int(np.prod(obs_space.shape))
        cfg.action_dim_per_agent = (
            int(np.prod(act_space.shape)) if cfg.continuous_actions
            else int(act_space.n)
        )
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.action_low = float(act_space.low.min()) if cfg.continuous_actions else 0.0
        self.action_high = float(act_space.high.max()) if cfg.continuous_actions else 1.0

        self._done = False

    # ----- episodic interface (the proper one) -----
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset env. Returns initial obs tensor of shape (n_agents, obs_dim)."""
        obs, _info = self._env.reset(seed=seed)
        self._obs = obs
        self._done = False
        return self._dict_to_tensor(obs)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        """
        actions: (n_agents, action_dim_per_agent) tensor.
        returns: next_obs (n_agents, obs_dim), rewards (n_agents,), done, info.
        """
        actions_dict = {
            aid: actions[i].detach().cpu().numpy().astype(np.float32)
            for i, aid in enumerate(self.agent_ids)
        }
        next_obs, rewards, terms, truncs, info = self._env.step(actions_dict)
        rewards_tensor = torch.tensor(
            [rewards.get(aid, 0.0) for aid in self.agent_ids], dtype=torch.float32
        )
        done = all(terms.values()) or all(truncs.values())
        self._obs = next_obs
        return self._dict_to_tensor(next_obs), rewards_tensor, done, info

    # ----- single-step compatibility shim (placeholder) -----
    def reward(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "PettingZoo envs are episodic; use reset()/step() rather than reward().\n"
            "See wmatrpo/scripts/simple_spread.py and the roadmap doc."
        )

    def initial_observation(self, batch_size: int = 1) -> torch.Tensor:
        raise NotImplementedError(
            "PettingZoo obs is per-agent and per-episode; call reset() instead."
        )

    def clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.clamp(self.action_low, self.action_high)

    # ----- helpers -----
    def _dict_to_tensor(self, obs_dict) -> torch.Tensor:
        """Convert {agent_id: np_array} → tensor (n_agents, obs_dim)."""
        arrs = [np.asarray(obs_dict[aid], dtype=np.float32).flatten()
                for aid in self.agent_ids]
        return torch.tensor(np.stack(arrs, axis=0))

    def close(self):
        self._env.close()


def make_simple_spread(n_agents: int = 3, max_cycles: int = 25) -> PettingZooEnv:
    """Convenience constructor: MPE simple_spread, N agents, 25 timesteps."""
    return PettingZooEnv(PettingZooConfig(
        env_name="simple_spread_v3",
        n_agents=n_agents,
        max_cycles=max_cycles,
        continuous_actions=True,
    ))
