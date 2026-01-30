from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MobileNetConfig:
    num_classes: int = 2
    pretrained: bool = True
    dropout: float | None = None  # optionally override classifier dropout
    backbone: str | None = None   # optional override (timm model name)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small used as both encoder and predictor.

    The implementation prefers `torchvision.models.mobilenet_v3_small` when torchvision is available
    (closest to the typical research pipeline). If torchvision is unavailable or broken, it falls back
    to `timm.create_model('mobilenetv3_small_100', ...)`.

    - `forward_features(x)` returns pooled penultimate features.
    - `forward(x)` returns logits.
    """

    def __init__(self, cfg: MobileNetConfig):
        super().__init__()
        self.cfg = cfg
        self.impl = None
        self._use_timm = False

        # Try torchvision first
        try:
            import torchvision
            import torch.nn as nn

            weights = None
            if cfg.pretrained:
                weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            m = torchvision.models.mobilenet_v3_small(weights=weights)

            # Optionally replace dropout
            if cfg.dropout is not None:
                for i, layer in enumerate(m.classifier):
                    if isinstance(layer, nn.Dropout):
                        m.classifier[i] = nn.Dropout(p=float(cfg.dropout))
                        break

            # Replace final layer
            last = m.classifier[-1]
            if not isinstance(last, nn.Linear):
                raise RuntimeError("Unexpected MobileNetV3 classifier layout.")
            m.classifier[-1] = nn.Linear(last.in_features, cfg.num_classes)

            self.impl = m
            self._use_timm = False
            return
        except Exception:
            pass

        # Fallback to timm
        try:
            import timm
        except Exception as e:
            raise RuntimeError(
                "Neither torchvision nor timm could be used to build MobileNetV3. "
                "Install a compatible torchvision, or install timm."
            ) from e

        name = cfg.backbone or "mobilenetv3_small_100"
        self.impl = timm.create_model(name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
        self._use_timm = True

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_timm:
            x = self.impl.features(x)
            x = self.impl.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        # timm path
        f = self.impl.forward_features(x)
        if f.ndim == 4:
            f = f.mean(dim=(2, 3))
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)
