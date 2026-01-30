from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def regular_simplex_directions(d: int) -> np.ndarray:
    """Return (d+1, d) array of unit vectors forming a centered regular d-simplex.

    Properties:
    - u_i · u_j = 1 if i=j, and -1/d otherwise
    - sum_k u_k = 0

    This matches Algorithm 2, Step 1 (Appendix C). 
    """
    if d < 1:
        raise ValueError("d must be >= 1")

    # Build vertices in R^{d+1} lying in the subspace orthogonal to 1.
    n = d + 1
    I = np.eye(n)
    ones = np.ones((n, n)) / n
    V = I - ones  # rows sum to 0
    # Scale so that row norms are 1 in the (d)-dim subspace.
    U_full = np.sqrt(n / d) * V  # rows have unit norm, pairwise dot -1/d, but in R^{n}

    # Orthonormal basis for the subspace orthogonal to 1:
    # Take the first d standard basis vectors, center them, then QR to get an orthonormal basis.
    A = I[:, :d] - (1 / n) * np.ones((n, d))
    Q, _ = np.linalg.qr(A)  # Q is (n, d) with orthonormal columns spanning the subspace

    U = U_full @ Q  # (n, d), now represented in R^d

    # Numerical cleanup: enforce zero-sum exactly up to tolerance by recentering
    U = U - U.mean(axis=0, keepdims=True)

    # Normalize to unit (protect against numerical drift)
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    U = U / norms

    return U


def preventive_displacement(tau_R: float, d_hat: int, epsilon: float) -> float:
    """Preventive displacement policy v = (2 - ε) τ_R / d̂ (Eq. 41)."""
    if d_hat <= 0:
        raise ValueError("d_hat must be >= 1")
    if not (0.0 < epsilon <= 1.0):
        raise ValueError("epsilon must be in (0,1]")
    return (2.0 - float(epsilon)) * float(tau_R) / float(d_hat)


def verify_regular_simplex(U: np.ndarray, atol: float = 1e-3) -> None:
    """Raise if U does not satisfy the regular simplex Gram structure within tolerance."""
    n, d = U.shape
    if n != d + 1:
        raise ValueError("U must have shape (d+1, d)")
    G = U @ U.T
    off = -1.0 / d
    for i in range(n):
        if abs(G[i, i] - 1.0) > atol:
            raise ValueError(f"Diag check failed at {i}: {G[i,i]}")
        for j in range(n):
            if i == j:
                continue
            if abs(G[i, j] - off) > atol:
                raise ValueError(f"Off-diag check failed at ({i},{j}): {G[i,j]} vs {off}")
    s = U.sum(axis=0)
    if np.linalg.norm(s) > 1e-2:
        raise ValueError(f"Zero-sum check failed: ||sum u_k|| = {np.linalg.norm(s)}")
