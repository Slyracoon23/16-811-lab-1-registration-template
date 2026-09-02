"""The technique. This is the only file in the repository you have to write.

What ships here runs and is wrong: it matches the centroids and gives up on the rotation. That is
deliberate — the harness is proved before you touch it, so your first edit moves a real number.

What you are implementing (Gallier & Quaintance, ch. 20; Horn 1987):

    H = sum_i (p_i - p_bar)(q_i - q_bar)^T          the cross-covariance of the centred clouds
    H = U S V^T                                     its SVD
    R = V diag(1, 1, det(V U^T)) U^T                the nearest *rotation*, not merely orthogonal
    t = q_bar - R p_bar

The determinant term is not optional. Without it you get the nearest orthogonal matrix, which on
noisy or degenerate data is a reflection roughly half the time — a mirror-image scan that scores
plausibly and is wrong.
"""

from __future__ import annotations

import numpy as np

ASSUMPTIONS: list[str] = [
    "P[i] corresponds to Q[i]. The matching is somebody else's problem.",
    "The transform is rigid: rotation and translation, no scale.",
]


def register(p: np.ndarray, q: np.ndarray, *, prune: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) carrying `p` onto `q`.

    `prune` selects the inlier stage: "none" uses every correspondence, "consistency" keeps the
    largest mutually-consistent set first (step 4 of the lab).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # PLACEHOLDER — runs, and is wrong. Replace with the four lines above.
    rotation = np.eye(3)
    translation = q.mean(axis=0) - rotation @ p.mean(axis=0)
    return rotation, translation
