# How I Approached This — Step by Step

This is the plain-language version of "what I actually did," in order.
For the equations behind each step, see [MATH_APPROACH.md](MATH_APPROACH.md).

## 1. First, figure out what's really unknown

- The equations name 3 unknowns: `θ`, `M`, `X`.
- But the CSV has 1500 rows, and **none of them say what `t` is** for that row.
- So there isn't really a 3-unknown problem — there's a **1503-unknown**
  problem: `θ`, `M`, `X`, plus one hidden `t` per data point.
- Realizing this early matters. It rules out just throwing the data at a
  generic curve-fitting function and hoping — that would need a starting
  guess for 1500 unknowns, which isn't realistic.

## 2. Look for a shortcut in the equations themselves

- Both `x(t)` and `y(t)` are built from the same two pieces: `t` itself,
  and a wiggly term, mixed together using `cosθ` and `sinθ`.
- That specific mixing pattern is exactly what a **rotation** looks like.
- This is the one idea the whole solution hangs on: if this is a rotation,
  then it can be **undone**. And undoing a rotation is just algebra — no
  guessing required.

## 3. Use that shortcut to remove unknowns, not add optimizer time

- If I *knew* the correct `θ` and `X`, I could "un-rotate" every data point
  and get back its exact `t` value and its exact wiggle value — for free,
  with a formula, not a search.
- That also means I don't need to search for `M` separately. Once I have
  the un-rotated wiggle values, fitting `M` is a tiny, one-number problem.
- Net effect: instead of searching for 1503 unknowns, I only need to
  search for **2**: `θ` and `X`. Everything else falls out automatically.

## 4. Pick the right kind of search for those 2 unknowns

- I tested a plain local optimizer first (start somewhere, walk downhill).
- It got stuck in the wrong place from several different starting points.
- Why: the wiggly part of the curve repeats itself, so nearby-but-wrong
  guesses can *look* almost as good as the right one.
- Fix: use a **global** search method instead of a local one, since it
  doesn't rely on one lucky starting guess. This is fine to do here
  because we're only searching 2 numbers, not 1503.

## 5. Once close, polish to full precision

- The global search gets you *close* but not exact.
- So the last step re-fits everything — all 1503 numbers together this
  time — starting from the close answer, using a solver built for exactly
  this kind of large-but-structured problem.
- This step also uses the exact formula for how each unknown affects the
  output (rather than the solver guessing it numerically), which makes it
  both faster and more precise.

