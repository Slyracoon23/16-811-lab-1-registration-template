"""Two partial views of one object, the way a depth camera gives them to you.

The scaffold used to register two clouds of Gaussian noise, which is not registration — it is a
matrix identity with a story attached. Real scans differ in three ways that all matter here, and
this file reproduces each:

- **Partial overlap.** A camera sees the side facing it. Two views of the same object share only
  the part both can see, so a correspondence set is never complete.
- **Structure.** Points lie on surfaces, not in a ball. Surfaces are locally planar, which makes
  the cross-covariance far worse conditioned than random points do — chapter 8, showing up in the
  data rather than in a theorem.
- **Wrong matches.** Feature matching on real scans gets most correspondences wrong, and they are
  wrong in a structured way: they land on other parts of the same object.

There is no renderer here and no GL context, so this runs anywhere: visibility is decided by
whether a surface normal faces the camera, which is what a depth camera's answer amounts to.
"""

from __future__ import annotations

import numpy as np


def _box(rng, n, size=(0.6, 0.4, 0.3)):
    """Points on the six faces of a box, with their outward normals."""
    sx, sy, sz = size
    faces = [
        (np.array([1.0, 0, 0]), lambda u, v: np.stack([np.full_like(u, sx), u * sy, v * sz], 1)),
        (np.array([-1.0, 0, 0]), lambda u, v: np.stack([np.full_like(u, -sx), u * sy, v * sz], 1)),
        (np.array([0, 1.0, 0]), lambda u, v: np.stack([u * sx, np.full_like(u, sy), v * sz], 1)),
        (np.array([0, -1.0, 0]), lambda u, v: np.stack([u * sx, np.full_like(u, -sy), v * sz], 1)),
        (np.array([0, 0, 1.0]), lambda u, v: np.stack([u * sx, v * sy, np.full_like(u, sz)], 1)),
        (np.array([0, 0, -1.0]), lambda u, v: np.stack([u * sx, v * sy, np.full_like(u, -sz)], 1)),
    ]
    points, normals = [], []
    for normal, place in faces:
        u, v = rng.uniform(-1, 1, n // 6), rng.uniform(-1, 1, n // 6)
        points.append(place(u, v))
        normals.append(np.tile(normal, (n // 6, 1)))
    return np.vstack(points), np.vstack(normals)


def _cylinder(rng, n, radius=0.22, height=0.7, centre=(0.0, 0.0, 0.45)):
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-height / 2, height / 2, n)
    normals = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], 1)
    points = normals * radius + np.array([0.0, 0.0, 0.0])
    points[:, 2] = z
    return points + np.asarray(centre), normals


def object_surface(points: int = 4000, seed: int = 0):
    """A box with a cylinder standing on it. Returns (points, normals)."""
    rng = np.random.default_rng(seed)
    box_points, box_normals = _box(rng, points * 2 // 3)
    cyl_points, cyl_normals = _cylinder(rng, points // 3)
    return np.vstack([box_points, cyl_points]), np.vstack([box_normals, cyl_normals])


def visible_from(points, normals, camera):
    """Indices of points whose surface faces `camera`. A depth camera's answer, without a renderer."""
    to_camera = np.asarray(camera) - points
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    return np.flatnonzero(np.sum(to_camera * normals, axis=1) > 0.15)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def view_pair(*, outlier_fraction=0.0, noise=0.002, seed=0, points=4000):
    """Return (source, target, R_true, t_true, inlier_mask).

    Two cameras look at the object from different sides. The correspondences are the points both
    can see; `outlier_fraction` of them are then re-pointed at other parts of the object, which is
    how feature matching fails in practice — not at random in space.
    """
    rng = np.random.default_rng(seed)
    surface, normals = object_surface(points, seed=seed)

    first = np.array([2.5, 0.4, 0.8])
    second = np.array([0.7, 2.4, 1.0])
    seen_a = set(visible_from(surface, normals, first).tolist())
    seen_b = set(visible_from(surface, normals, second).tolist())
    shared = np.array(sorted(seen_a & seen_b))
    if len(shared) < 50:
        raise RuntimeError("the two views barely overlap; move the cameras")

    rotation, translation = random_rotation(rng), rng.normal(scale=0.3, size=3)
    source = surface[shared] + rng.normal(scale=noise, size=(len(shared), 3))
    target = source @ rotation.T + translation + rng.normal(scale=noise, size=(len(shared), 3))

    inliers = np.ones(len(shared), dtype=bool)
    wrong = int(round(outlier_fraction * len(shared)))
    if wrong:
        idx = rng.choice(len(shared), size=wrong, replace=False)
        elsewhere = rng.choice(len(surface), size=wrong)
        target[idx] = surface[elsewhere] @ rotation.T + translation
        inliers[idx] = False
    return source, target, rotation, translation, inliers
