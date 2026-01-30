from __future__ import annotations

from dataclasses import dataclass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformConfig:
    image_size: int = 224
    train_aug: bool = True


def build_transforms(cfg: TransformConfig):
    """Return (train_transform, eval_transform).

    Preference order:
    1) torchvision transforms (if torchvision imports successfully)
    2) pure-Python PIL transforms (fallback)
    """
    try:
        import torchvision
        from torchvision import transforms as T

        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        eval_tf = weights.transforms()

        if not cfg.train_aug:
            return eval_tf, eval_tf

        normalize = eval_tf.transforms[-1]  # Normalize(mean,std)

        train_tf = T.Compose([
            T.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            normalize,
        ])
        return train_tf, eval_tf

    except Exception:
        # Fallback: minimal PIL-based transforms
        from .simple_transforms import (
            Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
        )

        eval_tf = Compose([
            Resize(256),
            CenterCrop(cfg.image_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        if not cfg.train_aug:
            return eval_tf, eval_tf

        train_tf = Compose([
            RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
            RandomHorizontalFlip(0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_tf, eval_tf
