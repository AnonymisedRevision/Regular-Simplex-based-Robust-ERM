from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint
from ..utils.metrics import compute_binary_metrics_from_logits


@dataclass(frozen=True)
class SRERMTrainConfig:
    epochs: int = 5
    steps_per_epoch: int = 300
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    fp16: bool = True
    tolerance_tau: float = 0.0   # Condition 1 tolerance (τ ≈ 0)
    patience: int = 200          # patience in steps (Algorithm 3, step 16)
    num_runs: int = 10            # M in Algorithm 3
    base_seed: int = 42
    class_weights: Optional[Sequence[float]] = None



def _make_loader(ds, indices: np.ndarray, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
    subset = Subset(ds, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def _infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _avg_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, fp16: bool) -> float:
    model.eval()
    total = 0.0
    n = 0
    use_amp = fp16 and device.type == "cuda"
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
def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device, fp16: bool) -> Dict[str, float]:
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
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0)
    m = compute_binary_metrics_from_logits(logits, y)
    return {
        "accuracy": m.accuracy,
        "f1": m.f1,
        "auc": (m.auc if m.auc is not None else -1.0),
        "confusion_matrix": m.confusion_matrix,
    }


def train_srerm(
    model_factory,
    # domain indices include k=-1 and k=0..d_hat
    train_indices_by_k: Dict[int, np.ndarray],
    val_indices_by_k: Dict[int, np.ndarray],
    dataset_train,
    dataset_val,
    b_tr_orig: float,
    b_val_orig: float,
    cfg: SRERMTrainConfig,
    out_dir: str | Path,
    logger,
) -> Dict[str, object]:
    """Train SR-ERM with hard-max updates and dual monitors (Appendix C, Algorithm 3)."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    w = None
    if cfg.class_weights is not None:
        w = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    K_list = sorted(train_indices_by_k.keys())  # includes -1
    if -1 not in K_list:
        raise ValueError("Expected domain -1 (unshifted source) to be present.")

    run_records: list[dict] = []

    for run in range(cfg.num_runs):
        seed = cfg.base_seed + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_factory()
        model.to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

        # Build per-domain infinite loaders
        train_loaders = {
            k: _make_loader(dataset_train, train_indices_by_k[k], cfg.batch_size, cfg.num_workers, shuffle=True, drop_last=True)
            for k in K_list
        }
        train_iters = {k: _infinite_batches(train_loaders[k]) for k in K_list}

        # Validation loaders for full-avg evaluation
        val_loaders = {
            k: _make_loader(dataset_val, val_indices_by_k[k], cfg.batch_size, cfg.num_workers, shuffle=False, drop_last=False)
            for k in K_list
        }

        best_gamma = float("inf")
        best_state = None
        patience_ctr = 0

        total_steps = cfg.epochs * cfg.steps_per_epoch
        pbar = tqdm(range(1, total_steps + 1), desc=f"srerm run {run+1}/{cfg.num_runs}")

        for step in pbar:
            # 1) sample one minibatch from each domain and compute its loss (no-grad pass)
            losses = {}
            batches = {}
            use_amp = cfg.fp16 and device.type == "cuda"

            with torch.no_grad():
                model.train()  # keep train-mode for dropout/bn parity
                for k in K_list:
                    x, y = next(train_iters[k])
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(x)
                        loss = loss_fn(logits, y)
                    losses[k] = float(loss.item())
                    batches[k] = (x, y)

            # 2) robust step: pick worst domain and update on it (Algorithm 3, step 10)
            k_star = max(losses, key=lambda kk: losses[kk])
            x_star, y_star = batches[k_star]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_star = model(x_star)
                loss_star = loss_fn(logits_star, y_star)
            scaler.scale(loss_star).backward()
            scaler.step(opt)
            scaler.update()

            # 3) Monitor Γ̂_train (Algorithm 3, step 11)
            gamma_train = max(losses.values()) - float(b_tr_orig)

            # best checkpoint tracking (Algorithm 3, steps 12-15)
            if gamma_train <= best_gamma:
                best_gamma = gamma_train
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if step % 25 == 0:
                pbar.set_postfix(gamma_train=float(gamma_train), worst_k=int(k_star), worst_loss=float(losses[k_star]))

            # Early stop (Algorithm 3, step 16)
            if gamma_train <= cfg.tolerance_tau or patience_ctr >= cfg.patience:
                break

        if best_state is None:
            best_state = model.state_dict()

        # Freeze θ̄ = θ_best
        model.load_state_dict(best_state)

        # Validation monitor Γ̂_val (Algorithm 3, steps 20-22)
        val_losses = {k: _avg_loss(model, val_loaders[k], loss_fn, device, cfg.fp16) for k in K_list}
        gamma_val = max(val_losses.values()) - float(b_val_orig)

        logger.info(
            f"[srerm] run={run+1}/{cfg.num_runs} best_gamma_train={best_gamma:.4f} "
            f"gamma_val={gamma_val:.4f} val_losses={{{' '.join([str(k)+':'+format(v,'.3f') for k,v in val_losses.items()])}}}"
        )

        # Save run checkpoint
        run_path = ckpt_dir / f"srerm_run{run+1}_best.pt"
        save_checkpoint(run_path, {
            "model_state": best_state,
            "best_gamma_train": best_gamma,
            "gamma_val": gamma_val,
            "val_losses": val_losses,
            "cfg": cfg.__dict__,
        })

        run_records.append({
            "run": run + 1,
            "seed": seed,
            "best_gamma_train": float(best_gamma),
            "gamma_val": float(gamma_val),
            "checkpoint": str(run_path),
            "val_losses": {str(k): float(v) for k, v in val_losses.items()},
        })

    # Select model with minimal Γ̂_val (Algorithm 3, step 23)
    best_rec = min(run_records, key=lambda r: r["gamma_val"])
    best_ckpt = torch.load(best_rec["checkpoint"], map_location="cpu")
    selected_state = best_ckpt["model_state"]

    # Save a canonical selected checkpoint
    selected_path = ckpt_dir / "srerm_selected.pt"
    save_checkpoint(selected_path, {
        "model_state": selected_state,
        "selected_from_run": best_rec["run"],
        "gamma_val": best_rec["gamma_val"],
        "best_gamma_train": best_rec["best_gamma_train"],
        "run_records": run_records,
        "cfg": cfg.__dict__,
    })

    # Load selected state into a fresh model instance for caller convenience
    selected_model = model_factory()
    selected_model.load_state_dict(selected_state)

    return {
        "selected_checkpoint": str(selected_path),
        "selected_from_run": best_rec["run"],
        "gamma_val": float(best_rec["gamma_val"]),
        "best_gamma_train": float(best_rec["best_gamma_train"]),
        "run_records": run_records,
        "selected_model_state": selected_state,  # caller may choose to save/load differently
        "K_list": [int(k) for k in K_list],
    }
