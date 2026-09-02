# 16-811 · Lab 1 · Register two point clouds when most of the matches are wrong

Write Horn's closed-form registration out of the polar decomposition, find the outlier fraction
that destroys it, and rebuild it into a pipeline that survives past 99%.

**Technique:** Horn, *Closed-Form Solution of Absolute Orientation Using Unit Quaternions*,
J. Opt. Soc. Am. A 4(4), 1987.
**Checked against:** [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) (Yang, Shi & Carlone, T-RO 2021).
**From the book:** Gallier & Quaintance ch. 20 (polar form), 15 (quaternions), 18 (graphs), 6 (determinant).

## Start

Open in the dev container — everything is installed — then:

```bash
make check        # 8 tests. 4 fail. Those 4 are the job.
make reproduce    # runs now, with a deliberately wrong method. Beat that number.
```

Nothing here raises `NotImplementedError`. `method.py` ships something that runs and is wrong:
it matches the centroids and gives up on the rotation. The harness is proved before you touch it,
so your first edit moves a real number.

## What you write

**`method.py`, one function.** Everything else is scaffolding.

```
H = Σ (pᵢ - p̄)(qᵢ - q̄)ᵀ          the cross-covariance of the centred clouds
H = U Σ Vᵀ                        its SVD
R = V diag(1, 1, det(V Uᵀ)) Uᵀ    the nearest rotation, not merely orthogonal
t = q̄ - R p̄
```

The determinant term is not optional. Without it you get the nearest *orthogonal* matrix, which on
noisy or degenerate data is a reflection roughly half the time — a mirror-image scan that scores
plausibly and is wrong. One of the tests exists only to catch that.

## The files

| | |
|---|---|
| `method.py` | **Yours.** The technique. ~20 lines when finished |
| `evaluate.py` | The ruler. Complete, and it never imports `method` |
| `synthetic.py` | Pairs whose answer you chose. Prove everything here before real data |
| `baselines.py` | A floor (random) and a ceiling (the truth), so a number means something |
| `reproduce.py` | Seed loop and outlier sweep → `results.json` |
| `extend.py` | Your own ideas, as switches |
| `tests/` | The to-do list |

## Two things the lab will teach you whether you want it or not

**Chance is not 90°.** The geodesic angle between two uniform random rotations has density
`(1 − cos θ)/π`, so its mean is `π/2 + 2/π ≈ 126.5°`. A chance baseline reading 90 means your
sampler is not uniform, and every "we beat chance" claim resting on it is worth less than it looks.

**The ruler has a floor.** `arccos` is ill-conditioned near 1 — `arccos(1 − ε) ≈ √(2ε)` — so a
rotation accurate to 1e-15 reads as ~1e-6 degrees. No tolerance tighter than about 1e-4 deg is
meaningful in float64. That is chapter 8 showing up in the first thing you build.

## TEASER++ (optional)

The reference is deliberately not in the image. Build it when you reach step 5:

```bash
git clone --depth 1 https://github.com/MIT-SPARK/TEASER-plusplus.git
cmake -S TEASER-plusplus -B build -DTEASERPP_PYTHON_XX=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target install
```

If it fails, keep going — `make reproduce` runs without it and skips one comparison column. The
hardest dependency must never be the reason you stop.
