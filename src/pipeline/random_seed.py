"""Shared seed setting for notebooks and model configuration."""

from __future__ import annotations

import random

import numpy as np


DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, *, include_torch: bool = False) -> int:
    """Seed Python and NumPy, optionally PyTorch, then return the seed.

    Pass the returned value as `random_state` to scikit-learn objects because
    setting global RNGs alone does not configure every estimator or splitter.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    random.seed(seed)
    np.random.seed(seed)
    if include_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed
