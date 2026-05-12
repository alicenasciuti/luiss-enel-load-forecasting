"""
File Title : Project Utilities
File Name  : utils.py

Description:
Contains shared utility functions and project-wide helper logic
used to support consistent behaviour across multiple modules.
This file is expected to implement reusable utilities related
to experiment reproducibility, environment configuration,
random seed management, framework initialization support,
system-level helper functions, and other generic services
required throughout the project lifecycle.

Role in Project:
Provides the foundational support layer of the project
architecture by supplying reusable helper utilities that are
consumed by data loading, preprocessing, exploratory analysis,
modelling, and evaluation components. This module helps ensure
consistency, reproducibility, and centralized management of
common operations across the entire project pipeline.
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
