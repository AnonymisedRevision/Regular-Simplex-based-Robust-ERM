from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class SkinPaths:
    train_dir: Path
    test_dir: Path


def resolve_skin_paths(data_root: str | Path) -> SkinPaths:
    root = Path(data_root)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}' to exist. "
            "Your data_root should point to the SKIN/ folder."
        )
    return SkinPaths(train_dir=train_dir, test_dir=test_dir)


def build_imagefolder_datasets(
    data_root: str | Path,
    train_transform,
    eval_transform,
) -> Tuple[Dataset, Dataset]:
    """Build train and test datasets.

    Uses torchvision.datasets.ImageFolder when torchvision is available; otherwise falls back to
    a minimal pure-Python FolderDataset.
    """
    paths = resolve_skin_paths(data_root)
    try:
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(paths.train_dir.as_posix(), transform=train_transform)
        test_ds = ImageFolder(paths.test_dir.as_posix(), transform=eval_transform)
        return train_ds, test_ds
    except Exception:
        from .folder_dataset import FolderDataset
        train_ds = FolderDataset(paths.train_dir.as_posix(), transform=train_transform)
        test_ds = FolderDataset(paths.test_dir.as_posix(), transform=eval_transform)
        return train_ds, test_ds


def stratified_split_indices(
    targets: list[int] | np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) with approximate stratification."""
    targets = np.asarray(targets).astype(int)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    for c in np.unique(targets):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        n_val = int(round(len(idx_c) * val_ratio))
        val_idx.append(idx_c[:n_val])
        train_idx.append(idx_c[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def subset_from_indices(ds: Dataset, indices: np.ndarray) -> Subset:
    return Subset(ds, indices.tolist())
