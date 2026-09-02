"""The ruler's own tests. These pass on a fresh clone — the evaluator ships finished."""

from __future__ import annotations

import numpy as np

from evaluate import rotation_error_deg, translation_error
from synthetic import make_pair, random_rotation


def test_perfect_scores_zero():
    _, _, rotation, translation = make_pair(seed=0)
    assert rotation_error_deg(rotation, rotation) < 1e-9
    assert translation_error(translation, translation) < 1e-12


def test_chance_matches_the_theoretical_mean():
    """Not 90 degrees.

    The geodesic angle between two uniform rotations has density (1 - cos t)/pi on [0, pi], so its
    mean is pi/2 + 2/pi = 2.207 rad = 126.5 deg. A chance baseline that reads 90 means the sampler
    is not uniform, and every "we beat chance" claim built on it is worth less than it looks.
    """
    rng = np.random.default_rng(0)
    _, _, rotation, _ = make_pair(seed=0)
    errs = [rotation_error_deg(random_rotation(rng), rotation) for _ in range(2000)]
    expected = np.degrees(np.pi / 2 + 2 / np.pi)
    assert abs(float(np.mean(errs)) - expected) < 5.0
