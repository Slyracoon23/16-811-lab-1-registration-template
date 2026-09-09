"""Rerun: see the registration instead of reading its error.

A rotation error in degrees tells you that something is wrong. It never tells you *what* — whether
the estimate is a reflection, whether your inlier set kept the wrong half of the object, whether
the two views barely overlapped in the first place. All three look like "4.3 degrees" in a table
and are obvious in three seconds on screen.

Rerun is what robotics teams reach for here. It logs from Python, it needs no server, and it writes
a `.rrd` file you can open later or hand to somebody else — which is why the failure that took an
afternoon in a terminal becomes a screenshot in a pull request.

    python3 view.py --outliers 0.6 --save run.rrd     # then: rerun run.rrd

Two stages on a "stage" timeline: the raw correspondences, then the same pair with your estimate
applied. Scrub between them and a bad rotation is unmistakable.
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr

SOURCE = (90, 140, 255)
TARGET = (255, 170, 60)
INLIER = (90, 220, 140)
OUTLIER = (220, 70, 90)


def log_pair(source, target, *, inliers=None, kept=None, transform=None, stride=4):
    """Log the two views, the correspondences, and the alignment your estimate produces."""
    rr.log("world/source", rr.Points3D(source, colors=SOURCE, radii=0.004))
    rr.log("world/target", rr.Points3D(target, colors=TARGET, radii=0.004))

    # Correspondences as line segments. Every fourth by default: all of them is a solid wall.
    pairs = np.stack([source[::stride], target[::stride]], axis=1)
    if inliers is not None:
        colors = np.where(inliers[::stride, None], np.array(INLIER), np.array(OUTLIER))
    else:
        colors = np.tile(np.array(INLIER), (len(pairs), 1))
    rr.log("world/correspondences", rr.LineStrips3D(pairs, colors=colors, radii=0.0008))

    if kept is not None:
        rr.log("world/kept", rr.Points3D(source[kept], colors=INLIER, radii=0.007))

    if transform is not None:
        rotation, translation = transform
        rr.log("world/aligned", rr.Points3D(source @ np.asarray(rotation).T + translation, colors=INLIER, radii=0.004))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outliers", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="registration.rrd")
    args = parser.parse_args()

    import scene
    from method import register

    source, target, rotation, translation, inliers = scene.view_pair(outlier_fraction=args.outliers, seed=args.seed)

    rr.init("16-811-lab-1-registration")
    rr.save(args.save)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.set_time("stage", sequence=0)
    log_pair(source, target, inliers=inliers)

    rr.set_time("stage", sequence=1)
    estimated_r, estimated_t = register(source, target)
    log_pair(source, target, inliers=inliers, transform=(estimated_r, estimated_t))

    truth = np.degrees(np.arccos(np.clip((np.trace(np.asarray(estimated_r).T @ rotation) - 1) / 2, -1, 1)))
    print(f"{len(source)} correspondences, {int(args.outliers * 100)}% wrong")
    print(f"rotation error: {truth:.2f} deg")
    print(f"wrote {args.save} — open it with:  rerun {args.save}")


if __name__ == "__main__":
    main()
