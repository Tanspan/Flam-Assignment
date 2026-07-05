from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, differential_evolution, least_squares
from scipy.sparse import lil_matrix

# Problem constants (fixed by the assignment)

Y_OFFSET = 42.0
OMEGA = 0.3
THETA_MIN, THETA_MAX = 0.0, np.deg2rad(50.0)
M_MIN, M_MAX = -0.05, 0.05
X_MIN, X_MAX = 0.0, 100.0
T_MIN, T_MAX = 6.0, 60.0


# 1. Curve model

def wiggle(t: np.ndarray, M: float) -> np.ndarray:
    """s(t) = e^(M*t) * sin(omega*t).  t > 0 on this domain, so |t| = t."""
    return np.exp(M * t) * np.sin(OMEGA * t)


def curve(t: np.ndarray, theta: float, M: float, X: float):
    """Evaluate the model at parameter value(s) t. Returns (x, y)."""
    s = wiggle(t, M)
    x = t * np.cos(theta) - s * np.sin(theta) + X
    y = Y_OFFSET + t * np.sin(theta) + s * np.cos(theta)
    return x, y


def inverse_rotate(x: np.ndarray, y: np.ndarray, theta: float, X: float):
    """
    Exact closed-form inverse of the rotation, for ANY candidate (theta, X):

        t_hat =  (x - X)*cos(theta) + (y - 42)*sin(theta)
        z_hat = -(x - X)*sin(theta) + (y - 42)*cos(theta)

    If (theta, X) are correct, z_hat should equal wiggle(t_hat, M) for a
    single shared M. This is the key identity the whole fit is built on.
    """
    ct, st = np.cos(theta), np.sin(theta)
    t_hat = (x - X) * ct + (y - Y_OFFSET) * st
    z_hat = -(x - X) * st + (y - Y_OFFSET) * ct
    return t_hat, z_hat

# 2. Stage A - reduced 2-parameter search over (theta, X), with M
#    eliminated in closed form (profiled out via a 1-D fit per candidate)

def fit_M_given_tz(t: np.ndarray, z: np.ndarray):
    """Best-fit scalar M (1-D bounded least squares) for given (t, z)."""
    sin_wt = np.sin(OMEGA * t)

    def sse(M):
        return np.sum((np.exp(M * t) * sin_wt - z) ** 2)

    res = minimize_scalar(sse, bounds=(M_MIN, M_MAX), method="bounded",
                           options={"xatol": 1e-13})
    return res.x, res.fun


def reduced_objective(params, x, y):
    theta, X = params
    t_hat, z_hat = inverse_rotate(x, y, theta, X)
    _, sse = fit_M_given_tz(t_hat, z_hat)
    return sse


def fit_reduced(x: np.ndarray, y: np.ndarray, seed: int = 1):
    """Global search over (theta, X) only; M and per-point t fall out free."""
    de = differential_evolution(
        reduced_objective, [(THETA_MIN, THETA_MAX), (X_MIN, X_MAX)],
        args=(x, y), seed=seed, tol=1e-12, maxiter=100, popsize=20, polish=True,
    )
    theta, X = de.x
    t_hat, z_hat = inverse_rotate(x, y, theta, X)
    M, sse = fit_M_given_tz(t_hat, z_hat)
    t_hat = np.clip(t_hat, T_MIN, T_MAX)
    return theta, M, X, t_hat, sse


# 3. Stage B - joint least squares over ALL unknowns, analytic Jacobian

def residuals(params, x, y):
    theta, M, X = params[0], params[1], params[2]
    t = params[3:]
    cx, cy = curve(t, theta, M, X)
    return np.concatenate([cx - x, cy - y])


def jacobian(params, x, y):
    theta, M, X = params[0], params[1], params[2]
    t = params[3:]
    n = len(t)

    s = wiggle(t, M)
    ds_dM = t * s
    ds_dt = M * s + OMEGA * np.exp(M * t) * np.cos(OMEGA * t)
    sinT, cosT = np.sin(theta), np.cos(theta)

    dcx_dtheta = -t * sinT - s * cosT
    dcx_dM = -sinT * ds_dM
    dcx_dX = np.ones(n)
    dcx_dt = cosT - sinT * ds_dt

    dcy_dtheta = t * cosT - s * sinT
    dcy_dM = cosT * ds_dM
    dcy_dX = np.zeros(n)
    dcy_dt = sinT + cosT * ds_dt

    J = lil_matrix((2 * n, 3 + n))
    J[:n, 0] = dcx_dtheta.reshape(-1, 1)
    J[:n, 1] = dcx_dM.reshape(-1, 1)
    J[:n, 2] = dcx_dX.reshape(-1, 1)
    J[n:, 0] = dcy_dtheta.reshape(-1, 1)
    J[n:, 1] = dcy_dM.reshape(-1, 1)
    J[n:, 2] = dcy_dX.reshape(-1, 1)

    rows_x, rows_y, cols_t = np.arange(n), np.arange(n, 2 * n), np.arange(3, 3 + n)
    J[rows_x, cols_t] = dcx_dt
    J[rows_y, cols_t] = dcy_dt
    return J.tocsr()


def joint_refine(x, y, theta0, M0, X0, t0):
    n = len(t0)
    p0 = np.concatenate([[theta0, M0, X0], t0])
    lb = np.concatenate([[THETA_MIN, M_MIN, X_MIN], np.full(n, T_MIN)])
    ub = np.concatenate([[THETA_MAX, M_MAX, X_MAX], np.full(n, T_MAX)])
    return least_squares(
        residuals, p0, args=(x, y), jac=jacobian,
        tr_solver="lsmr", bounds=(lb, ub), method="trf",
        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000,
    )


# 4. End-to-end pipeline

def fit_all(x: np.ndarray, y: np.ndarray, seed: int = 1):
    """Run Stage A then Stage B. Returns (theta, M, X, t_array, l1_per_point)."""
    theta0, M0, X0, t0, _ = fit_reduced(x, y, seed=seed)
    res = joint_refine(x, y, theta0, M0, X0, t0)
    n = len(x)
    theta, M, X = res.x[0], res.x[1], res.x[2]
    t = res.x[3:]
    r = res.fun
    l1_per_point = np.abs(r[:n]) + np.abs(r[n:])
    return theta, M, X, t, l1_per_point


def to_desmos_latex(theta: float, M: float, X: float) -> str:
    return (
        r"\left(t*\cos(%.6f)-e^{%.6f\left|t\right|}\cdot\sin(0.3t)\sin(%.6f)+%.6f,"
        r"42+t*\sin(%.6f)+e^{%.6f\left|t\right|}\cdot\sin(0.3t)\cos(%.6f)\right)"
    ) % (theta, M, theta, X, theta, M, theta)


# 5. CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/xy_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    x = df["x"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)
    print(f"Loaded {len(x)} points from {args.data}")

    theta, M, X, t, l1_per_point = fit_all(x, y)

    print("\n=== Recovered parameters ===")
    print(f"theta = {theta:.8f} rad = {np.degrees(theta):.6f} deg")
    print(f"M     = {M:.8f}")
    print(f"X     = {X:.8f}")

    print("\n=== Fit quality ===")
    print(f"mean L1 distance / point: {l1_per_point.mean():.6e}")
    print(f"max  L1 distance / point: {l1_per_point.max():.6e}")

    print("\n=== Desmos / LaTeX submission string ===")
    print(to_desmos_latex(theta, M, X))


if __name__ == "__main__":
    main()
