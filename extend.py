"""The half that is yours: what to try once it reproduces.

Switches, not branches — a reader of your write-up should be able to turn each idea off from the
command line and watch the number move. And it has to be the *same* number `reproduce.py` reports,
or you have changed the subject rather than improved the result.
"""

from __future__ import annotations

import argparse
import json

VARIANTS = ["none", "spectral", "conditioning", "weighted"]


def run(variant: str, seed: int) -> float:
    """Run one variant on one seed and return the rotation error in degrees."""
    raise NotImplementedError(f"Your idea: {variant}. See 'Past the paper' on the lab sheet.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    chosen = VARIANTS if args.all else [args.variant or "none"]
    records = [
        {"variant": variant, "seed": seed, "rot_err_deg": run(variant, seed)}
        for variant in chosen
        for seed in range(args.seeds)
    ]
    with open("extensions.json", "w") as handle:
        json.dump({"records": records}, handle, indent=2)
    print(f"{len(records)} records into extensions.json")


if __name__ == "__main__":
    main()
