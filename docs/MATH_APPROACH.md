# Mathematical Approach

Equations and derivation behind the solution. For the plain-language version of the process, see [PROCESS.md](PROCESS.md).

## 1. Problem

Given `N = 1500` unordered, unlabeled `(x, y)` samples from

```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

for a hidden parameter `t ∈ (6, 60)`, recover:

| Parameter | Range |
|---|---|
| θ | 0° < θ < 50° |
| M | −0.05 < M < 0.05 |
| X | 0 < X < 100 |

No row is labeled with its `t`, and rows are unordered. Since `t > 0`
throughout the domain, `|t| = t` everywhere below — the absolute value
never actually matters for this problem, it's just notation.

## 2. Key insight: the model is a rotation

Define the "wiggle" function:

```
s(t) = e^(M·t) · sin(0.3t)
```

Rewrite the model using this:

```
x − X = t·cosθ − s(t)·sinθ
y − 42 = t·sinθ + s(t)·cosθ
```

The right-hand side is exactly the 2×2 rotation matrix `R(θ)` applied to
the point `(t, s(t))`, followed by a translation to `(X, 42)`:

```
[x − X]   [cosθ  −sinθ] [ t   ]
[y − 42] = [sinθ   cosθ] [s(t)]
```

So every data point is a rotated-and-shifted copy of the 1-D curve
`t ↦ (t, s(t))`. Only three numbers — `θ` (rotation), `M` (shape of the
wiggle), `X` (translation) — determine the whole picture. `t` is a private
value attached to each individual point, not a global unknown we need to
search over jointly with the others.

## 3. Eliminating M in closed form

A rotation matrix is exactly invertible (its inverse is `R(−θ)`, i.e. just
transpose it). So for **any** candidate `(θ, X)` — not just the correct
one — we can invert the equation above and recover:

```
t̂ =  (x − X)·cosθ + (y − 42)·sinθ
ẑ = −(x − X)·sinθ + (y − 42)·cosθ
```

This is exact, not an approximation, not a lookup — it's a linear system
being inverted algebraically.

If `(θ, X)` happen to be correct, then `ẑᵢ` must equal `s(t̂ᵢ)` for **one
single shared value of M**, simultaneously for every point in the
dataset. So fitting `M` is just a 1-D least-squares problem:

```
M* = argmin_M  Σᵢ ( e^(M·t̂ᵢ)·sin(0.3·t̂ᵢ) − ẑᵢ )²
```

**This is the core simplification.** Without it, the problem has
`3 + N = 1503` unknowns (`θ, M, X`, plus `t₁, …, t_N`). With it, `M` and
every `t̂ᵢ` fall out for free as a byproduct of the closed-form inversion,
and the real search space shrinks to just **2 unknowns**: `θ` and `X`.

## 4. Stage A — global search over (θ, X)

Scoring a candidate `(θ, X)` means: invert to get `(t̂, ẑ)`, fit the best
`M`, and use the resulting sum-of-squares error as the score. This 2-D
error surface has local minima — different phase alignments of the
`sin(0.3t)` wiggle can locally resemble a good fit over part of the data —
so a purely local optimizer isn't safe from an arbitrary start.

A global optimizer, **Differential Evolution** (Storn & Price, 1997), is
used over the bounded box `θ ∈ (0°, 50°)`, `X ∈ (0, 100)`. Because this is
only 2-dimensional, a global search here is fast and reliable — unlike
searching all 1503 unknowns globally, which would be impractical.

## 5. Stage B — joint refinement with an analytic Jacobian

Stage A gives an approximate `(θ, M, X)` and, via the same inversion, an
initial `t̂ᵢ` for every point. All `1503` unknowns are then refined
**jointly** using `scipy.optimize.least_squares` (Trust-Region-Reflective
algorithm), with a hand-derived analytic Jacobian rather than a
finite-differenced one.

The Jacobian is **block-sparse**: each `tᵢ` only affects its own residual
pair `(xᵢ, yᵢ)` — changing `t₅` has zero effect on the residual for point
`10`. The partial derivatives (with `s = s(t)`):

```
∂x/∂θ = −t·sinθ − s·cosθ         ∂y/∂θ = t·cosθ − s·sinθ
∂x/∂M = −sinθ · (t·s)            ∂y/∂M =  cosθ · (t·s)
∂x/∂X = 1                        ∂y/∂X = 0
∂x/∂t =  cosθ − sinθ·s'(t)       ∂y/∂t =  sinθ + cosθ·s'(t)

where s'(t) = M·s(t) + 0.3·e^(M·t)·cos(0.3t)
```

Because the Jacobian is sparse and exact, this stage converges in well
under a second even at `N = 1500`.

## 6. Result and why it's trustworthy

The pipeline converges to:

```
θ = 30.0000°  (π/6 rad)
M = 0.030000
X = 55.0000
```

with mean L1 residual per point ≈ 2.7 × 10⁻⁶. Three independent checks
support this being the true answer, not just "a good fit":

1. **Clean numbers.** The fit lands on round values well inside its own
   residual noise, not on arbitrary decimals like 29.87° or 0.0287.
2. **Stable re-solve.** Warm-starting `least_squares` from the converged
   point reproduces an identical result — evidence of genuine
   convergence rather than a solver stopping early.
3. **Round-trip check.** Plugging the exact values `(π/6, 0.03, 55)`
   directly back into the model reproduces the CSV to within ~2×10⁻⁵
   absolute error — consistent with the input file being rounded to 6
   decimal digits. In other words, the residual we see is the data's own
   rounding, not error in the algorithm.

## References

- Storn, R., & Price, K. (1997). Differential evolution — a simple and
  efficient heuristic for global optimization over continuous spaces.
  *Journal of Global Optimization*, 11(4), 341–359.
- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for
  scientific computing in Python. *Nature Methods*, 17, 261–272.
