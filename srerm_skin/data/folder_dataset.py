from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderDataset(Dataset):
    """Minimal ImageFolder-like dataset to avoid hard dependency on torchvision."""

    def __init__(self, root: str | Path, transform: Callable | None = None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.transform = transform
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class subfolders found in {self.root}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []
        for c in self.classes:
            cdir = self.root / c
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p.as_posix(), self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

        # match torchvision ImageFolder attribute name
        self.targets = [y for _, y in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y
