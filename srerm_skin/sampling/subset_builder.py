from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SubsetBuildConfig:
    vertex_k: int = 256                 # per-vertex KNN budget (Algorithm 2, step 9)
    val_ratio: float = 0.2              # Algorithm 2, step 16
    seed: int = 42
    refine_swaps: int = 0               # optional greedy refinement (see paper discussion)
    refine_pool_multiplier: int = 4     # candidates = refine_pool_multiplier * vertex_k
    radius: float | None = None         # optional fixed-radius neighborhood instead of KNN


def _split_indices(indices: np.ndarray, val_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = indices.copy()
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx


def build_simplex_subsets(
    X_hat: np.ndarray,
    p_hat: np.ndarray,
    vertices: np.ndarray,
    cfg: SubsetBuildConfig,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, float]]:
    """Construct simplex subsets S_k by KNN (or radius) sampling in latent space.

    Inputs:
    - X_hat: (N, d̂) latent representations of the full source S
    - p_hat: (d̂,) latent centroid
    - vertices: (d̂+1, d̂) unit simplex directions u_k (centered regular simplex)
    - cfg: subset build config

    Returns:
    - train_indices_by_k: dict k -> np.ndarray of indices (into X_hat) for S_k^{tr}
    - val_indices_by_k: dict k -> np.ndarray of indices for S_k^{val}
    - stats: centroid-mismatch and distance diagnostics

    Paper alignment: Appendix C, Algorithm 2. The security threshold τ_R is expected to be computed
    outside (mean radial spread), and the displacement v applied to form p̂_k = p̂ + v u_k.
    """
    if X_hat.ndim != 2:
        raise ValueError("X_hat must be 2D")
    N, d_hat = X_hat.shape
    if p_hat.shape != (d_hat,):
        raise ValueError("p_hat must have shape (d_hat,)")
    if vertices.shape != (d_hat + 1, d_hat):
        raise ValueError("vertices must have shape (d_hat+1, d_hat)")

    rng = np.random.default_rng(cfg.seed)
    train_by_k: dict[int, np.ndarray] = {}
    val_by_k: dict[int, np.ndarray] = {}

    centroid_errors = []
    mean_dists = []

    for k in range(d_hat + 1):
        p_k = vertices[k]  # NOTE: caller should pass already-displaced vertices if desired
        # p_k here is a point in R^{d̂}
        diffs = X_hat - p_k[None, :]
        dists = np.linalg.norm(diffs, axis=1)

        if cfg.radius is not None:
            chosen = np.where(dists <= float(cfg.radius))[0]
            if len(chosen) < cfg.vertex_k:
                # fall back to KNN if too sparse
                chosen = np.argsort(dists)[: cfg.vertex_k]
            else:
                # If too many within radius, downsample deterministically by distance order
                chosen = chosen[np.argsort(dists[chosen])][: cfg.vertex_k]
        else:
            chosen = np.argsort(dists)[: cfg.vertex_k]

        # Optional centroid-matching refinement by swaps
        if cfg.refine_swaps > 0:
            pool_size = int(cfg.refine_pool_multiplier * cfg.vertex_k)
            pool = np.argsort(dists)[:pool_size]
            chosen = _refine_by_swaps(X_hat, pool, chosen, target=p_k, swaps=cfg.refine_swaps, rng=rng)

        # Split into train/val for dual monitoring
        tr_idx, va_idx = _split_indices(chosen.astype(int), cfg.val_ratio, rng)
        train_by_k[k] = tr_idx
        val_by_k[k] = va_idx

        # diagnostics
        centroid = X_hat[chosen].mean(axis=0)
        centroid_errors.append(float(np.linalg.norm(centroid - p_k)))
        mean_dists.append(float(dists[chosen].mean()))

    stats = {
        "mean_centroid_error": float(np.mean(centroid_errors)),
        "max_centroid_error": float(np.max(centroid_errors)),
        "mean_knn_distance": float(np.mean(mean_dists)),
        "max_knn_distance": float(np.max(mean_dists)),
    }
    return train_by_k, val_by_k, stats


def _refine_by_swaps(
    X_hat: np.ndarray,
    pool: np.ndarray,
    chosen: np.ndarray,
    target: np.ndarray,
    swaps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy refinement: attempt swaps between chosen and pool to reduce ||centroid - target||."""
    chosen = chosen.copy().astype(int)
    pool = pool.astype(int)
    chosen_set = set(chosen.tolist())

    # candidates not in chosen
    pool_ex = np.array([i for i in pool.tolist() if i not in chosen_set], dtype=int)
    if len(pool_ex) == 0:
        return chosen

    centroid = X_hat[chosen].mean(axis=0)
    best_err = float(np.linalg.norm(centroid - target))

    for _ in range(int(swaps)):
        out_idx = int(rng.choice(chosen))
        in_idx = int(rng.choice(pool_ex))

        # compute updated centroid efficiently
        new_centroid = centroid + (X_hat[in_idx] - X_hat[out_idx]) / len(chosen)
        new_err = float(np.linalg.norm(new_centroid - target))

        if new_err < best_err:
            # accept
            pos = int(np.where(chosen == out_idx)[0][0])
            chosen[pos] = in_idx
            centroid = new_centroid
            best_err = new_err

            chosen_set.remove(out_idx)
            chosen_set.add(in_idx)

            # update pool_ex lazily: keep it simple
            pool_ex = np.array([i for i in pool.tolist() if i not in chosen_set], dtype=int)
            if len(pool_ex) == 0:
                break

    return chosen
