"""
generate_reference_values.py
=============================
Generates reference_values.json: fixed input/output pairs computed by
the validated Python implementation (sa_appa_experiments/gp.py,
methods.py, benchmark.py), for validate_against_python.m to check the
MATLAB/Octave translation against.

Run from anywhere; it locates sa_appa_experiments/ (needs gp.py,
methods.py, benchmark.py on the path) relative to its own location,
assuming the standard sibling-directory layout of this package:

    python3 sa_appa_matlab/generate_reference_values.py

Writes sa_appa_matlab/reference_values.json (overwriting any existing
copy -- the one shipped in this package was produced by this exact
script; re-running it should reproduce it byte-for-byte given the same
NumPy version, since every random draw here uses a fixed seed).

Design note on the APPA-step reference: methods.py's _appa_step() draws
random numbers from an internal RNG stream during the update, in a
specific per-particle sequence. NumPy's and MATLAB/Octave's random number
generators are different algorithms, so "the same seed" does not produce
the same draws across languages -- comparing two RNG-driven runs directly
would not test the translation, only the (expected, uninteresting) fact
that two different RNGs disagree. Instead, this script draws all the
random numbers ONCE in Python, in the exact sequence _appa_step() would
consume them, and ships them as explicit inputs alongside the resulting
output. Both appa_step_reference() below and MATLAB's
appa_step_deterministic.m take those same numbers as explicit arguments
rather than drawing them internally, so the comparison is of the UPDATE
FORMULA only, which is the thing a translation can actually get right or
wrong.
"""

import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "sa_appa_experiments"))

from gp import GPSurrogate
from methods import _ei, _pof
from benchmark import SUITE

OUT_PATH = os.path.join(_THIS_DIR, "reference_values.json")


def gp_reference():
    X = np.array([[0.1, 0.2], [0.4, 0.1], [0.6, 0.7],
                  [0.9, 0.3], [0.3, 0.8], [0.55, 0.45]])
    y = np.array([1.2, 0.5, 2.3, 1.8, 0.9, 1.5])
    gp = GPSurrogate()
    gp.fit(X, y)
    Xstar = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]])
    mu, sigma = gp.predict(Xstar)
    return dict(X=X.tolist(), y=y.tolist(), Xstar=Xstar.tolist(),
                lengthscale=gp.lengthscale, mu=mu.tolist(), sigma=sigma.tolist())


def ei_pof_reference():
    mu_ei = np.array([0.5, 1.2, -0.3, 2.0])
    sigma_ei = np.array([0.2, 0.05, 0.4, 0.01])
    f_best = 0.8
    ei_vals = _ei(mu_ei, sigma_ei, f_best)
    mu_g = [np.array([0.1, -0.2, 0.05, 0.3])]
    sigma_g = [np.array([0.15, 0.1, 0.2, 0.05])]
    pof_vals = _pof(mu_g, sigma_g)
    ei_ref = dict(mu=mu_ei.tolist(), sigma=sigma_ei.tolist(), f_best=f_best, ei=ei_vals.tolist())
    pof_ref = dict(mu_g=mu_g[0].tolist(), sigma_g=sigma_g[0].tolist(), pof=pof_vals.tolist())
    return ei_ref, pof_ref


def benchmark_reference():
    tp1, tp2 = SUITE["TP1"], SUITE["TP2"]
    out = []
    for x, y in [([1.0, -0.5], [0.3]), ([-1.2, 0.8], [1.5]), ([0.0, 0.0], [0.0])]:
        x, y = np.array(x), np.array(y)
        out.append(dict(problem="TP1", x=x.tolist(), y=y.tolist(),
                         F=float(tp1.F(x, y)), f=float(tp1.f(x, y)), g=tp1.g(x, y).tolist()))
    for x, y in [([3.0, 2.0], [1.0, 2.0, 0.5]), ([7.0, 1.0], [0.5, 0.5, 0.5]),
                 ([0.0, 0.0], [0.0, 0.0, 0.0])]:
        x, y = np.array(x), np.array(y)
        out.append(dict(problem="TP2", x=x.tolist(), y=y.tolist(),
                         F=float(tp2.F(x, y)), f=float(tp2.f(x, y)), g=tp2.g(x, y).tolist()))
    return out


def appa_step_reference():
    """See module docstring: random draws are generated once, in the
    exact sequence methods.py::_appa_step() would consume them, and
    shipped explicitly rather than redrawn on the MATLAB side."""

    def appa_step_deterministic(X, best_idx, worst_idx, lam, U):
        N, d = X.shape
        Xnew = X.copy()
        p_f = lam["p_f"]
        for i in range(N):
            if i == worst_idx:
                continue
            if U["role"][i] <= p_f:
                r1 = U["r1"][i]
                diff = X[best_idx] - X[i]
                n = np.linalg.norm(diff) + 1e-12
                r2 = U["r2"][i]
                Xnew[i] = X[i] + lam["explore"] * r1 * diff / n + lam["local"] * (r2 * 2 - 1)
            else:
                r4 = U["r4"][i]
                diff = X[i] - X[worst_idx]
                n = np.linalg.norm(diff) + 1e-12
                Xnew[i] = X[i] + lam["evade"] * r4 * diff / n
        r5 = U["r5"]
        diff = X[best_idx] - X[worst_idx]
        Xnew[worst_idx] = X[worst_idx] + lam["chase"] * r5 * diff + lam["global_"] * (U["r2global"] * 2 - 1)
        return Xnew

    rng = np.random.default_rng(777)
    N, d = 6, 2
    X = rng.uniform(0, 1, size=(N, d))
    best_idx, worst_idx = 2, 5
    lam = dict(explore=0.6, evade=0.4, chase=0.5, local=0.05, global_=0.1, p_f=0.6)
    U = dict(
        role=rng.uniform(size=N),
        r1=rng.uniform(size=N),
        r2=[rng.uniform(size=d) for _ in range(N)],
        r4=rng.uniform(size=N),
        r5=float(rng.uniform()),
        r2global=rng.uniform(size=d),
    )
    Xnew = appa_step_deterministic(X, best_idx, worst_idx, lam, U)

    U_json = dict(role=U["role"].tolist(), r1=U["r1"].tolist(),
                  r2=[r.tolist() for r in U["r2"]], r4=U["r4"].tolist(),
                  r5=U["r5"], r2global=U["r2global"].tolist())
    return dict(X=X.tolist(), best_idx=best_idx + 1, worst_idx=worst_idx + 1,  # 1-indexed for MATLAB
                lam=lam, U=U_json, Xnew=Xnew.tolist())


def main():
    np.random.seed(12345)
    ei_ref, pof_ref = ei_pof_reference()
    out = dict(
        gp=gp_reference(),
        ei=ei_ref,
        pof=pof_ref,
        benchmark=benchmark_reference(),
        appa_step=appa_step_reference(),
    )
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
