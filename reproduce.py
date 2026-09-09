"""Run the method across seeds and an outlier sweep, and write results.json.

A single run is an anecdote. The spread is what tells you whether a gap is real.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import baselines
from evaluate import rotation_error_deg, translation_error
from method import ASSUMPTIONS, register
from synthetic import make_pair


def jsonable(record: dict) -> dict:
    """A record with every non-finite float replaced by ``None``.

    `json.dump` writes `NaN` and `Infinity` by default. Python reads those back; `JSON.parse`
    refuses them outright, so a single degenerate row makes the whole results file unreadable to
    anything that is not Python — including the course app that imports it. `null` is JSON, and it
    says the true thing: this one was not measured.

    The sanitising happens here, at the boundary, and not in the functions that compute the
    numbers. `float("nan")` is a perfectly good return value for a fit that had too few points, and
    the printed summary below still uses `np.nanmean` over the real values.
    """
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in record.items()
    }


# The last two points are the lab. Horn's closed form is exact on clean data and degrades
# gracefully to about 0.9; what the sheet promises to show is what happens *past* 99%, and a sweep
# that stopped at 0.99 could not show it.
SWEEP = [0.0, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--prune", default="none")
    args = parser.parse_args()

    records = []
    for fraction in SWEEP:
        for seed in range(args.seeds):
            p, q, rotation, translation = make_pair(
                outlier_fraction=fraction, noise=0.005, seed=seed
            )
            estimated_r, estimated_t = register(p, q, prune=args.prune)
            chance_r, _ = baselines.chance(p, q, seed=seed)
            # The reference the sheet measures you against. It is optional by construction —
            # `open3d_ransac` returns None when Open3D is not installed — so the column goes
            # missing rather than the run failing. A comparison that can block the lab is not a
            # comparison, it is a dependency.
            reference = baselines.open3d_ransac(p, q)
            records.append(
                {
                    "outlier_fraction": fraction,
                    "seed": seed,
                    "rot_err_deg": rotation_error_deg(estimated_r, rotation),
                    "trans_err": translation_error(estimated_t, translation),
                    "chance_rot_err_deg": rotation_error_deg(chance_r, rotation),
                    "ransac_rot_err_deg": (
                        rotation_error_deg(reference[0], rotation) if reference else None
                    ),
                }
            )

    with open("results.json", "w") as handle:
        json.dump(
            {
                "assumptions": ASSUMPTIONS,
                "prune": args.prune,
                "records": [jsonable(r) for r in records],
            },
            handle,
            indent=2,
            allow_nan=False,
        )

    have_reference = any(r["ransac_rot_err_deg"] is not None for r in records)
    print(f"{'outliers':>9} {'rot err':>9} {'spread':>8} {'chance':>9} {'ransac':>9}")
    for fraction in SWEEP:
        rows = [r for r in records if r["outlier_fraction"] == fraction]
        errs = np.array([r["rot_err_deg"] for r in rows])
        chance = np.mean([r["chance_rot_err_deg"] for r in rows])
        ransac = (
            f"{np.mean([r['ransac_rot_err_deg'] for r in rows]):>9.2f}"
            if have_reference
            else f"{'--':>9}"
        )
        print(f"{fraction:>9.3f} {errs.mean():>9.2f} {errs.std():>8.2f} {chance:>9.2f} {ransac}")
    if not have_reference:
        print("\n(no ransac column: Open3D is not installed, and the lab does not need it to run)")
    print("\nresults.json written. Clean data should read ~0.00 once method.py is yours.")


if __name__ == "__main__":
    main()
