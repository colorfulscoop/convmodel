import torch
import numpy as np
import random


def set_reproducibility(seed: int = None, deterministic: bool = False):
    """
    Refer to the document for details
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)
