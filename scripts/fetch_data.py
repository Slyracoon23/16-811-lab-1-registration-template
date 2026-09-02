"""Real scans, when you are ready for them.

Deliberately fetches nothing by default. Everything in this lab should first be proved on
`synthetic.py`, where you chose the answer — so the first run needs no download at all, and a
slow or broken mirror can never be the reason you did not start.

3DMatch: https://3dmatch.cs.princeton.edu/  (RGB-D fragments with ground-truth transforms)
KITTI odometry: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Both want a few GB and an account in KITTI's case. Download by hand, drop the fragments in
data/3dmatch/, and point `--pair` at two of them.
"""

from __future__ import annotations

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), help="two .ply fragments to load")
    args = parser.parse_args()
    if not args.pair:
        print(__doc__)
        sys.exit(0)
    print(f"Loading {args.pair[0]} and {args.pair[1]} — implement this when you get to step 3.")
