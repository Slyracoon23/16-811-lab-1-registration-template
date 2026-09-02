"""A floor and a ceiling, so a number from `method.py` means something on day one."""

from __future__ import annotations

import numpy as np

from synthetic import random_rotation


def chance(p: np.ndarray, q: np.ndarray, *, seed: int = 0):
    """A random rotation. Anything not clearly beating this is not working."""
    rng = np.random.default_rng(seed)
    rotation = random_rotation(rng)
    return rotation, q.mean(axis=0) - rotation @ p.mean(axis=0)


def oracle(rotation: np.ndarray, translation: np.ndarray):
    """The answer. The ceiling — your clean-data result should reach it to ~1e-12."""
    return rotation, translation


def open3d_ransac(p: np.ndarray, q: np.ndarray):
    """Open3D's RANSAC registration, if Open3D is installed. Optional by design.

    The reference must never block the lab: if the import fails you lose one comparison column,
    not the ability to work.
    """
    try:
        import open3d as o3d
    except ImportError:
        return None

    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(p)
    target.points = o3d.utility.Vector3dVector(q)
    corres = o3d.utility.Vector2iVector(np.tile(np.arange(len(p))[:, None], (1, 2)))
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target, corres, 0.05,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [], o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
    )
    return result.transformation[:3, :3], result.transformation[:3, 3]
