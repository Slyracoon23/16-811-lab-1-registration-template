"""Run the method across seeds and an outlier sweep, and write results.json.

A single run is an anecdote. The spread is what tells you whether a gap is real.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import baselines
from evaluate import rotation_error_deg, translation_error
from method import ASSUMPTIONS, register
from synthetic import make_pair

SWEEP = [0.0, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]


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
            records.append(
                {
                    "outlier_fraction": fraction,
                    "seed": seed,
                    "rot_err_deg": rotation_error_deg(estimated_r, rotation),
                    "trans_err": translation_error(estimated_t, translation),
                    "chance_rot_err_deg": rotation_error_deg(chance_r, rotation),
                }
            )

    with open("results.json", "w") as handle:
        json.dump({"assumptions": ASSUMPTIONS, "prune": args.prune, "records": records}, handle, indent=2)

    print(f"{'outliers':>9} {'rot err':>9} {'spread':>8} {'chance':>9}")
    for fraction in SWEEP:
        rows = [r for r in records if r["outlier_fraction"] == fraction]
        errs = np.array([r["rot_err_deg"] for r in rows])
        chance = np.mean([r["chance_rot_err_deg"] for r in rows])
        print(f"{fraction:>9.2f} {errs.mean():>9.2f} {errs.std():>8.2f} {chance:>9.2f}")
    print("\nresults.json written. Clean data should read ~0.00 once method.py is yours.")


if __name__ == "__main__":
    main()
