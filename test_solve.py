"""
test_solve.py
=============
A few sanity checks on solve.py. Run with:  pytest -v
(or: python test_solve.py  -- it will run without pytest too)
"""
import numpy as np
import pandas as pd

from solve import curve, inverse_rotate, wiggle, fit_all


def test_inverse_rotate_is_exact_inverse():
    """inverse_rotate should exactly undo curve()'s rotation for any (theta, X)."""
    rng = np.random.default_rng(0)
    theta, M, X = np.deg2rad(23.4), 0.017, 41.2
    t = rng.uniform(6, 60, size=200)

    x, y = curve(t, theta, M, X)
    t_hat, z_hat = inverse_rotate(x, y, theta, X)

    assert np.allclose(t_hat, t, atol=1e-10)
    assert np.allclose(z_hat, wiggle(t, M), atol=1e-10)


def test_fit_recovers_known_synthetic_params():
    """On noise-free synthetic data, the pipeline should recover the exact
    (theta, M, X) used to generate it."""
    rng = np.random.default_rng(42)
    theta_true, M_true, X_true = np.deg2rad(30.0), 0.03, 55.0
    t_true = rng.uniform(6, 60, size=400)
    x, y = curve(t_true, theta_true, M_true, X_true)

    theta_hat, M_hat, X_hat, t_hat, l1 = fit_all(x, y, seed=1)

    assert abs(theta_hat - theta_true) < 1e-6
    assert abs(M_hat - M_true) < 1e-6
    assert abs(X_hat - X_true) < 1e-4
    assert l1.mean() < 1e-6


def test_fit_on_provided_dataset():
    """Regression check against the actual assignment data."""
    df = pd.read_csv("data/xy_data.csv")
    x = df["x"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)

    theta, M, X, t, l1 = fit_all(x, y, seed=1)

    assert abs(np.degrees(theta) - 30.0) < 1e-2
    assert abs(M - 0.03) < 1e-3
    assert abs(X - 55.0) < 1e-2
    assert (t > 6).all() and (t < 60).all()
    assert l1.mean() < 1e-4


if __name__ == "__main__":
    # Allows running "python test_solve.py" directly, without pytest.
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
    print("\nALL TESTS PASSED")
