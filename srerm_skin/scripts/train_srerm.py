from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from ..data.skin_dataset import build_imagefolder_datasets, stratified_split_indices
from ..data.transforms import TransformConfig, build_transforms
from ..models.mobilenetv3 import MobileNetConfig, MobileNetV3Small
from ..training.baseline import BaselineTrainConfig, train_baseline, compute_anchor_losses
from ..training.embedding import compute_embeddings
from ..training.dimred import DimRedConfig, fit_pca
from ..geometry.simplex import (
    regular_simplex_directions,
    preventive_displacement,
    verify_regular_simplex,
)
from ..sampling.subset_builder import SubsetBuildConfig, build_simplex_subsets
from ..training.srerm import SRERMTrainConfig, train_srerm, eval_on_loader
from ..utils.seed import SeedConfig, seed_everything
from ..utils.io import ensure_dir, save_json
from ..utils.logging import setup_logger


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SR-ERM training on SKIN (MobileNetV3-Small)")

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # Reproducibility
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--no_deterministic", action="store_false", dest="deterministic")

    # Transforms
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--no_train_aug", action="store_true")

    # Dim reduction / simplex
    p.add_argument("--latent_dim", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0)
    p.add_argument("--tau_R", type=float, default=-1.0)
    p.add_argument("--vertex_k", type=int, default=800)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--refine_swaps", type=int, default=0)
    p.add_argument("--radius", type=float, default=-1.0)

    # Loss / imbalance
    p.add_argument("--balanced_loss", action="store_true")

    # Baseline
    p.add_argument("--baseline_epochs", type=int, default=1)
    p.add_argument("--baseline_lr", type=float, default=3e-4)
    p.add_argument("--baseline_wd", type=float, default=1e-4)

    # SR-ERM
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps_per_epoch", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--tolerance_tau", type=float, default=0.0)
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--init_from", choices=["baseline", "imagenet"], default="baseline")

    # Runtime
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_false", dest="fp16")
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)

    seed_everything(SeedConfig(seed=args.seed, deterministic=args.deterministic))
    logger.info(f"seed={args.seed} deterministic={args.deterministic}")

    writer = SummaryWriter((out_dir / "tensorboard").as_posix())

    # ------------------------------------------------------------------
    # Handle zip dataset
    # ------------------------------------------------------------------
    if args.data_root.lower().endswith(".zip"):
        import zipfile

        data_extract = out_dir / "data_extracted"
        if not (data_extract / "train").exists() and not (data_extract / "SKIN" / "train").exists():
            logger.info(f"Extracting {args.data_root} -> {data_extract}")
            with zipfile.ZipFile(args.data_root, "r") as zf:
                zf.extractall(data_extract)

        if (data_extract / "SKIN" / "train").exists():
            args.data_root = (data_extract / "SKIN").as_posix()
        else:
            args.data_root = data_extract.as_posix()

        logger.info(f"Using data_root={args.data_root}")

    # ------------------------------------------------------------------
    # Datasets / transforms
    # ------------------------------------------------------------------
    train_tf, eval_tf = build_transforms(
        TransformConfig(image_size=args.image_size, train_aug=not args.no_train_aug)
    )

    train_ds_aug, test_ds_eval = build_imagefolder_datasets(args.data_root, train_tf, eval_tf)
    train_ds_eval, _ = build_imagefolder_datasets(args.data_root, eval_tf, eval_tf)

    if len(train_ds_aug.classes) != 2:
        raise ValueError("Expected binary classification")

    logger.info(
        f"classes={train_ds_aug.classes} "
        f"n_train={len(train_ds_aug)} n_test={len(test_ds_eval)}"
    )

    # ------------------------------------------------------------------
    # S_-1 split
    # ------------------------------------------------------------------
    train_idx, val_idx = stratified_split_indices(
        train_ds_eval.targets, val_ratio=args.val_ratio, seed=args.seed
    )
    logger.info(f"S_-1 split: |train|={len(train_idx)} |val|={len(val_idx)}")

    # ------------------------------------------------------------------
    # Optional class-balanced loss
    # ------------------------------------------------------------------
    class_weights = None
    if args.balanced_loss:
        y_tr = np.array([train_ds_eval.targets[i] for i in train_idx.tolist()])
        counts = np.bincount(y_tr, minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        class_weights = (inv / inv.sum() * 2.0).tolist()
        logger.info(f"class_weights={class_weights}")

    # ------------------------------------------------------------------
    # Baseline training
    # ------------------------------------------------------------------
    baseline_tr_loader = DataLoader(
        Subset(train_ds_aug, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    baseline_val_loader = DataLoader(
        Subset(train_ds_eval, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    baseline_model = MobileNetV3Small(
        MobileNetConfig(num_classes=2, pretrained=True)
    )

    baseline_cfg = BaselineTrainConfig(
        epochs=args.baseline_epochs,
        lr=args.baseline_lr,
        weight_decay=args.baseline_wd,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        fp16=args.fp16,
        class_weights=class_weights,
    )

    logger.info("Training baseline ERM on S_-1 ...")
    baseline_stats = train_baseline(
        baseline_model,
        baseline_tr_loader,
        baseline_val_loader,
        baseline_cfg,
        out_dir,
        logger,
    )
    save_json(baseline_stats, out_dir / "baseline_stats.json")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    baseline_model.to(device)

    # ------------------------------------------------------------------
    # Anchors (Algorithm 3)
    # ------------------------------------------------------------------
    b_tr_orig, b_val_orig = compute_anchor_losses(
        baseline_model,
        baseline_tr_loader,
        baseline_val_loader,
        device,
        args.fp16,
        class_weights=class_weights,
    )

    # ------------------------------------------------------------------
    # Embeddings + PCA (Algorithm 1)
    # ------------------------------------------------------------------
    embed_loader = DataLoader(
        train_ds_eval,
        batch_size=max(64, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
    )

    X, Y = compute_embeddings(
        baseline_model, embed_loader, device=device, fp16=args.fp16
    )

    pca, X_hat = fit_pca(X, DimRedConfig(latent_dim=args.latent_dim))
    d_hat = X_hat.shape[1]
    p_hat = X_hat.mean(axis=0)

    radial = np.linalg.norm(X_hat - p_hat[None, :], axis=1)
    tau_R = radial.mean() if args.tau_R <= 0 else args.tau_R
    v = preventive_displacement(tau_R, d_hat, args.epsilon)

    logger.info(f"d_hat={d_hat} tau_R={tau_R:.4f} v={v:.4f}")

    # ------------------------------------------------------------------
    # Simplex subsets (Algorithm 2)
    # ------------------------------------------------------------------
    U = regular_simplex_directions(d_hat)
    verify_regular_simplex(U)

    Pk = p_hat[None, :] + v * U

    subset_cfg = SubsetBuildConfig(
        vertex_k=args.vertex_k,
        val_ratio=args.val_ratio,
        seed=args.seed,
        refine_swaps=args.refine_swaps,
        radius=(args.radius if args.radius > 0 else None),
    )

    train_by_k, val_by_k, subset_stats = build_simplex_subsets(
        X_hat=X_hat, p_hat=p_hat, vertices=Pk, cfg=subset_cfg
    )

    train_by_k[-1] = train_idx.astype(int)
    val_by_k[-1] = val_idx.astype(int)

    save_json(subset_stats, out_dir / "subset_stats.json")

    save_json(
        {
            "train_indices_by_k": {str(k): v.tolist() for k, v in train_by_k.items()},
            "val_indices_by_k": {str(k): v.tolist() for k, v in val_by_k.items()},
        },
        out_dir / "subsets.json",
    )

    # ------------------------------------------------------------------
    # SR-ERM (Algorithm 3)
    # ------------------------------------------------------------------
    baseline_state = {
        k: v.detach().cpu().clone()
        for k, v in baseline_model.state_dict().items()
    }

    def model_factory():
        m = MobileNetV3Small(MobileNetConfig(num_classes=2, pretrained=True))
        if args.init_from == "baseline":
            m.load_state_dict(baseline_state)
        return m

    srerm_cfg = SRERMTrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.wd,
        num_workers=args.num_workers,
        device=args.device,
        fp16=args.fp16,
        tolerance_tau=args.tolerance_tau,
        patience=args.patience,
        num_runs=args.num_runs,
        base_seed=args.seed,
        class_weights=class_weights,
    )

    logger.info("Training SR-ERM ...")
    srerm_result = train_srerm(
        model_factory,
        train_by_k,
        val_by_k,
        train_ds_aug,
        train_ds_eval,
        b_tr_orig,
        b_val_orig,
        srerm_cfg,
        out_dir,
        logger,
    )

    save_json(
        {k: v for k, v in srerm_result.items() if k != "selected_model_state"},
        out_dir / "srerm_result.json",
    )

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    test_loader = DataLoader(
        test_ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    baseline_metrics = eval_on_loader(
        baseline_model, test_loader, device=device, fp16=args.fp16
    )

    srerm_model = model_factory()
    srerm_model.load_state_dict(srerm_result["selected_model_state"])
    srerm_model.to(device)

    srerm_metrics = eval_on_loader(
        srerm_model, test_loader, device=device, fp16=args.fp16
    )

    save_json(
        {
            "baseline_test": baseline_metrics,
            "srerm_test": srerm_metrics,
        },
        out_dir / "metrics.json",
    )

    writer.add_scalar("test/baseline_accuracy", baseline_metrics["accuracy"], 0)
    writer.add_scalar("test/srerm_accuracy", srerm_metrics["accuracy"], 0)
    writer.close()

    logger.info(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
