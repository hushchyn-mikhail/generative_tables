import os
import random
import time
from dataclasses import dataclass
import numpy as np

def improve_reproducibility(seed: int, deterministic: bool = False) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed

@dataclass
class Timer:
    _t0: float = 0.0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        self._t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self._t0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.seconds = time.time() - self._t0
