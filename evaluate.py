"""The ruler. Complete, and it does not import `method`.

An evaluator that has seen the answer cannot be trusted with it, so this file knows nothing about
how the transform was produced. Build confidence in the ruler first: `python evaluate.py --self-test`
scores a deliberately useless answer and a perfect one, and both must land where you predict.
"""

from __future__ import annotations

import argparse

import numpy as np

from synthetic import make_pair, random_rotation


def rotation_error_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees.

    A conditioning caveat worth knowing, because it sets the floor on every number this lab
    reports: arccos is ill-conditioned near 1, where arccos(1 - e) ~ sqrt(2e). So a rotation
    accurate to 1e-15 reads as ~1e-6 degrees here, and no tolerance tighter than about 1e-4 deg
    is meaningful in float64. That is the ruler's limit, not your method's — chapter 8, showing
    up in the first thing you build.
    """
    cos = (np.trace(estimated.T @ truth) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def translation_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(estimated) - np.asarray(truth)))


def self_test() -> None:
    _, _, rotation, translation = make_pair(seed=0)
    rng = np.random.default_rng(1)

    chance = np.mean([rotation_error_deg(random_rotation(rng), rotation) for _ in range(500)])
    print(f"chance  rotation error: {chance:6.2f} deg   (expect ~126.5, not 90 — see below)")
    print(f"perfect rotation error: {rotation_error_deg(rotation, rotation):6.2f} deg   (expect 0)")
    print(f"perfect translation   : {translation_error(translation, translation):6.3f}     (expect 0)")
    print("\nIf either 'perfect' line is not ~0, the ruler is wrong and every later number is fiction.")
    print(
        "Chance is ~126.5 deg, not 90: the angle between two uniform rotations has density\n"
        "(1 - cos t)/pi on [0, pi], whose mean is pi/2 + 2/pi = 2.207 rad. Guessing 90 here is\n"
        "the first place this lab will mislead you if you do not check it."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        self_test()
