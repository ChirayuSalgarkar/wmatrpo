"""W-MATRPO — Faithful reference implementation."""
from wmatrpo.env import DifferentialGameEnv, DifferentialGameConfig
from wmatrpo.envs import ElFarolEnv, ElFarolConfig
from wmatrpo.policy import GaussianPolicy
from wmatrpo.critic import CentralizedCritic, DecentralizedCritic, build_critic
from wmatrpo.dual_solver import DualSolver
from wmatrpo.caatr import CAATR
from wmatrpo.algorithm import WMATRPO, WMATRPOConfig
from wmatrpo.ippo import IPPO, IPPOConfig
from wmatrpo.mappo import MAPPO, MAPPOConfig
from wmatrpo.happo import HAPPO, HAPPOConfig
from wmatrpo.trainer import Trainer

__all__ = [
    "DifferentialGameEnv",
    "DifferentialGameConfig",
    "ElFarolEnv",
    "ElFarolConfig",
    "GaussianPolicy",
    "CentralizedCritic",
    "DecentralizedCritic",
    "build_critic",
    "DualSolver",
    "CAATR",
    "WMATRPO",
    "WMATRPOConfig",
    "IPPO",
    "IPPOConfig",
    "Trainer",
]
