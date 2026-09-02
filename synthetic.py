"""Point-cloud pairs whose answer you chose, so you can check the method before trusting it.

Real data comes later. Everything in this lab should first be proved on a pair where you know
the rotation because you generated it.
"""

from __future__ import annotations

import numpy as np


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation, via the QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def make_pair(
    n: int = 500,
    *,
    outlier_fraction: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (P, Q, R_true, t_true) with Q = (R_true @ P.T).T + t_true, then corrupted.

    `outlier_fraction` of the correspondences in Q are replaced with uniform noise — they are
    still *matched* to a row of P, they are just matched wrongly. That is what a real feature
    matcher hands you.
    """
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    rotation = random_rotation(rng)
    translation = rng.normal(size=3)

    q = p @ rotation.T + translation
    if noise:
        q = q + rng.normal(scale=noise, size=q.shape)

    outliers = int(round(outlier_fraction * n))
    if outliers:
        idx = rng.choice(n, size=outliers, replace=False)
        q[idx] = rng.uniform(-3, 3, size=(outliers, 3))

    return p, q, rotation, translation
