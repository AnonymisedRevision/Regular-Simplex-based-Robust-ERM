from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.skin_dataset import build_imagefolder_datasets
from ..data.transforms import TransformConfig, build_transforms
from ..models.mobilenetv3 import MobileNetConfig, MobileNetV3Small
from ..training.srerm import eval_on_loader


def parse_args():
    p = argparse.ArgumentParser("Evaluate a saved checkpoint on SKIN/test")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt with 'model_state'.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_false", dest="fp16")
    return p.parse_args()


def main():
    args = parse_args()
    _, eval_tf = build_transforms(TransformConfig(train_aug=False))
    _, test_ds = build_imagefolder_datasets(args.data_root, eval_tf, eval_tf)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    model = MobileNetV3Small(MobileNetConfig(num_classes=2, pretrained=True))
    model.load_state_dict(state, strict=True)
    model.to(device)

    metrics = eval_on_loader(model, loader, device=device, fp16=args.fp16)
    print(metrics)


if __name__ == "__main__":
    main()
