"""Utility helpers: seeding, config loading."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
import random
from typing import Type, TypeVar

import numpy as np
import torch
import yaml


T = TypeVar("T")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dataclass_from_dict(cls: Type[T], d: dict) -> T:
    """
    Instantiate a dataclass from a flat dict, ignoring keys not in the dataclass.
    Lets configs include extra context without crashing.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in allowed}
    return cls(**filtered)


def dump_dataclass(obj) -> dict:
    return asdict(obj)
