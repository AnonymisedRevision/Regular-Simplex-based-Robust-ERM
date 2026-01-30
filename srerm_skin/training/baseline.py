from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint
from ..utils.metrics import compute_binary_metrics_from_logits


@dataclass(frozen=True)
class BaselineTrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    fp16: bool = True
    class_weights: list[float] | None = None  # optional inverse-frequency weights
    log_every: int = 50


def _avg_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, fp16: bool) -> float:
    model.eval()
    total = 0.0
    n = 0
    use_amp = fp16 and device.type == "cuda"
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
    return float(total / max(1, n))


@torch.no_grad()
def eval_logits_targets(model: nn.Module, loader: DataLoader, device: torch.device, fp16: bool) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all = []
    y_all = []
    use_amp = fp16 and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: BaselineTrainConfig,
    out_dir: str | Path,
    logger,
) -> Dict[str, float]:
    """Standard ERM training on S_{-1}^{tr} with best-val checkpointing."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    w = None
    if cfg.class_weights is not None:
        w = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

    best_val = float("inf")
    best_path = Path(out_dir) / "checkpoints" / "baseline_best.pt"

    global_step = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"baseline ep {ep}/{cfg.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.fp16 and device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % cfg.log_every == 0:
                pbar.set_postfix(loss=float(loss.item()))

        val_loss = _avg_loss(model, val_loader, loss_fn, device, cfg.fp16)
        tr_loss = _avg_loss(model, train_loader, loss_fn, device, cfg.fp16)
        logger.info(f"[baseline] epoch={ep} train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, {
                "model_state": model.state_dict(),
                "epoch": ep,
                "val_loss": val_loss,
                "train_loss": tr_loss,
                "class_weights": cfg.class_weights,
            })

    # Load best for final metrics
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    val_logits, val_y = eval_logits_targets(model, val_loader, device, cfg.fp16)
    metrics = compute_binary_metrics_from_logits(val_logits, val_y)

    return {
        "baseline_best_val_loss": float(ckpt["val_loss"]),
        "baseline_val_accuracy": metrics.accuracy,
        "baseline_val_f1": metrics.f1,
        "baseline_val_auc": metrics.auc if metrics.auc is not None else -1.0,
    }


def compute_anchor_losses(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    fp16: bool,
    class_weights: list[float] | None = None,
) -> tuple[float, float]:
    """Compute b_orig^{tr} and b_orig^{val} (Algorithm 3, Step 2)."""
    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    b_tr = _avg_loss(model, train_loader, loss_fn, device, fp16)
    b_val = _avg_loss(model, val_loader, loss_fn, device, fp16)
    return b_tr, b_val
