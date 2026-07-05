# Curve Parameter Recovery — θ, M, X

Recovers the unknown parameters of the parametric curve

```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

from 1,500 unordered, unlabeled `(x, y)` samples with hidden `t ∈ (6, 60)`.

## Result

| Parameter | Recovered value | Allowed range |
|---|---|---|
| θ | **30.0000°** (π/6 rad) | 0° < θ < 50° |
| M | **0.030000** | −0.05 < M < 0.05 |
| X | **55.0000** | 0 < X < 100 |

Mean L1 distance per point between the fitted curve and the data:
**≈ 2.7 × 10⁻⁶** — consistent with the input CSV being rounded to 6
decimal places, i.e. this is the data's own precision floor, not a
limitation of the fit.

**Desmos / submission string:**
```
\left(t*\cos(0.523599)-e^{0.030000\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)+55.000000,42+t*\sin(0.523599)+e^{0.030000\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)\right)
```

## Approach, in brief

The model is a rotation of the 1-D curve `t ↦ (t, s(t))`, where
`s(t) = e^(Mt)·sin(0.3t)`. Rotation is exactly invertible, so for any
candidate `(θ, X)` we can recover exact `t` and `s(t)` per point in closed
form — collapsing the fitting problem from 1,503 unknowns down to a
2-parameter search over `(θ, X)`, with `M` and every per-point `t` falling
out for free. A global search (differential evolution) finds `(θ, X)`,
then all unknowns are jointly polished with an analytic-Jacobian
nonlinear least squares.

**Full process narrative + mathematical derivation:**
- [docs/PROCESS.md](docs/PROCESS.md) — plain-language, step-by-step account of how this was approached
- [docs/MATH_APPROACH.md](docs/MATH_APPROACH.md) — the equations and derivation

## Project files

```
.
├── README.md          this file — overview, result, how to run
├── docs/
│   ├── PROCESS.md        plain-language step-by-step process
│   └── MATH_APPROACH.md   the mathematical derivation
├── solve.py             the pipeline: model, Stage A, Stage B, CLI
├── test_solve.py         sanity checks
├── requirements.txt
└── data/xy_data.csv
```

`solve.py` is a single file, organized top-to-bottom in the same order as
`APPROACH.md`'s reasoning: curve model → Stage A (reduced search) →
Stage B (joint refine) → CLI.

## How to run

```bash
pip install -r requirements.txt
python solve.py --data data/xy_data.csv
```

## How to run tests

```bash
pytest -v
# or, without pytest installed:
python test_solve.py
```

Tests check: (1) `inverse_rotate` is an exact inverse of `curve` on
synthetic data with known parameters, (2) the full pipeline recovers a
known synthetic `(θ, M, X)` to high precision, and (3) the pipeline
converges to the expected values on the actual assignment data.
