from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class EmbedConfig:
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"
    fp16: bool = False


@torch.no_grad()
def compute_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool = False,
    desc: str = "embed",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (features, targets) for a dataset loader."""
    model.eval()
    feats = []
    ys = []
    use_amp = fp16 and device.type == "cuda"
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            f = model.forward_features(x)
        feats.append(f.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(ys, axis=0)
    return X, Y
