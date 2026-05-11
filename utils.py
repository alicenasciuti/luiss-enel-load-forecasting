"""
utils.py
========

Utility functions used across the whole project.

Contents
--------
- set_global_seed(seed): set the random seed for `random`, `numpy` and (if
  available) `torch`/`tensorflow` to ensure reproducibility of experiments.

Role in the project
-------------------
This module does not perform any analysis on its own: it only provides
helpers that are imported by every other module (data_loader, preprocessing,
eda, modelling, evaluation) to keep behaviour consistent.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
