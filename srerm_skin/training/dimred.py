from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DimRedConfig:
    latent_dim: int = 10
    dmax: int = 2048  # threshold in Algorithm 1
    whiten: bool = False


def fit_pca(X: np.ndarray, cfg: DimRedConfig):
    """Fit PCA and return (pca, X_hat)."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X.shape
    if d <= cfg.latent_dim:
        # No reduction needed
        return None, X.astype(np.float32)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=int(cfg.latent_dim), whiten=bool(cfg.whiten), svd_solver="randomized", random_state=0)
    X_hat = pca.fit_transform(X).astype(np.float32)
    return pca, X_hat


def transform_pca(pca, X: np.ndarray) -> np.ndarray:
    if pca is None:
        return X.astype(np.float32)
    return pca.transform(X).astype(np.float32)
