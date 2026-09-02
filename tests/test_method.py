"""Your to-do list, as tests.

These fail against the placeholder in `method.py` and pass when it is Horn's solution. Run them
with `make check`. The order is the order to fix them in.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import rotation_error_deg, translation_error
from method import register
from synthetic import make_pair, random_rotation


def test_recovers_a_clean_transform_exactly():
    """Step 1. No noise, no outliers — this is the case Horn solves in closed form."""
    p, q, rotation, translation = make_pair(n=200, seed=0)
    estimated_r, estimated_t = register(p, q)
    assert rotation_error_deg(estimated_r, rotation) < 1e-4  # the ruler's floor; see evaluate.py
    assert translation_error(estimated_t, translation) < 1e-9


def test_returns_a_rotation_not_merely_an_orthogonal_matrix():
    """The determinant term. det(R) must be +1; a reflection scores -1 and looks plausible."""
    p, q, _, _ = make_pair(n=200, seed=3)
    rotation, _ = register(p, q)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8), "not orthogonal"
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-8), "reflection, not rotation"


def test_survives_a_degenerate_planar_cloud():
    """Coplanar points make H rank-deficient. This is where a missing det() term shows up."""
    rng = np.random.default_rng(7)
    p = np.column_stack([rng.normal(size=(300, 2)), np.zeros(300)])
    rotation = random_rotation(rng)
    q = p @ rotation.T
    estimated_r, _ = register(p, q)
    assert np.linalg.det(estimated_r) == pytest.approx(1.0, abs=1e-6)
    assert rotation_error_deg(estimated_r, rotation) < 1e-4


def test_is_unaffected_by_where_the_clouds_sit():
    """Centring. Translate both clouds far from the origin; the rotation must not change."""
    p, q, rotation, _ = make_pair(n=200, seed=1)
    near, _ = register(p, q)
    far, _ = register(p + 1000.0, q + 1000.0)
    assert rotation_error_deg(near, rotation) < 1e-4
    assert rotation_error_deg(far, rotation) < 1e-4, "the translation is leaking into H"


def test_tolerates_small_noise():
    """Least squares should degrade gracefully — this is what it is good at."""
    p, q, rotation, _ = make_pair(n=500, noise=0.01, seed=2)
    estimated_r, _ = register(p, q)
    assert rotation_error_deg(estimated_r, rotation) < 1.0


@pytest.mark.xfail(reason="Step 4: needs the consistency graph. Remove this mark when it passes.")
def test_survives_ninety_percent_outliers():
    """Step 4. Plain least squares cannot do this — that is the point of the lab."""
    p, q, rotation, _ = make_pair(n=500, outlier_fraction=0.9, noise=0.005, seed=4)
    estimated_r, _ = register(p, q, prune="consistency")
    assert rotation_error_deg(estimated_r, rotation) < 5.0
