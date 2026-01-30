import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = True


def seed_everything(cfg: SeedConfig) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)  # avoids hard errors for some ops
