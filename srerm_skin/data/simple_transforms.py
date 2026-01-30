from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Resize shortest side to size, keep aspect
        w, h = img.size
        if min(w, h) == self.size:
            return img
        if w < h:
            new_w = self.size
            new_h = int(round(h * (self.size / w)))
        else:
            new_h = self.size
            new_w = int(round(w * (self.size / h)))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)


class CenterCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th, tw = self.size, self.size
        i = max(0, int(round((h - th) / 2.0)))
        j = max(0, int(round((w - tw) / 2.0)))
        return img.crop((j, i, j + tw, i + th))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class ColorJitter:
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.brightness > 0:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(b)
        if self.contrast > 0:
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(c)
        if self.saturation > 0:
            s = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(s)
        return img


class RandomResizedCrop:
    def __init__(self, size: int, scale=(0.8, 1.0), ratio=(3/4, 4/3)):
        self.size = int(size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect)))
            h = int(round(math.sqrt(target_area / aspect)))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                img = img.crop((j, i, j + w, i + h))
                return img.resize((self.size, self.size), resample=Image.BILINEAR)
        # fallback to center crop after resize
        img = Resize(self.size)(img)
        return CenterCrop(self.size)(img)


class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
