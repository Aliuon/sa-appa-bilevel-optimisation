"""
phase7_data.py
==============
Data for the Phase 7 surrogate-fidelity figures (Part A). Read-only with respect to the locked
results: every SA-APPA (M1) run used here is a re-run under its locked seed, checked against the
locked trajectory before any GP snapshot is used.

Ground truth (never charged to any method):
  * TP3 and TP8: the LL problem is a strictly convex QP with simple bounds, solved exactly by
    bounded-variable least squares (scipy.optimize.lsq_linear, method 'bvls') and checked against
    the closed-form projection.
  * all other instances: a reference LL solve, SciPy differential evolution with population
    30 x n_ll, up to 300 generations, tol 1e-10, L-BFGS-B polish, best of two seeds. This is far
    more effort than the study's LL solver but is still a heuristic, and is described as such.

Run selection rule (fixed before any figure was drawn): for each instance, sort the 30 locked
M1 runs by final best feasible UL objective (runs without a feasible point last), ties by run id,
and take the 15th (the lower median).

Outputs: results/phase7/*.npz and results/phase7/fidelity.json
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, differential_evolution, minimize
from scipy.stats import qmc

from benchmark import SUITE
from methods import run_ablation_method, METHOD_SPECS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "phase7")
os.makedirs(OUT, exist_ok=True)
PEN = 1e6
TWO_SIGMA = 2.0
HELDOUT_SEED = 20261007
TP_LOWDIM = [f"TP{k}" for k in range(1, 9)]
SMD_SMALL = [f"SMD{k}_p3q3" for k in range(1, 13)]


# ---------------------------------------------------------------------------- ground truth LL
def ystar_exact(iid, x):
    """Exact LL response for the two instances whose LL is a bound-constrained strictly convex QP."""
    x = np.asarray(x, float)
    if iid == "TP3":   # min ||x - y||^2, y in [0,10]^2
        lo, hi, b = np.zeros(2), np.full(2, 10.0), x
    elif iid == "TP8":  # min ||y - (x - 20)||^2, y_i in [-10, min(20, (x_i - 10)/2)]
        lo = np.full(2, -10.0); hi = np.minimum(20.0, (x - 10.0) / 2.0); b = x - 20.0
    else:
        raise ValueError(iid)
    r = lsq_linear(np.eye(2), b, bounds=(lo, hi), method="bvls")
    closed = np.clip(b, lo, hi)
    assert np.max(np.abs(r.x - closed)) < 1e-9, (iid, x, r.x, closed)
    return r.x


def ystar_reference(inst, x):
    def obj(y):
        g = inst.g(x, y)
        v = np.sum(np.maximum(g, 0) ** 2) if g.size else 0.0
        return inst.f(x, y) + PEN * v
    best = None
    for s in (0, 1):
        r = differential_evolution(obj, list(map(tuple, inst.y_bounds)), seed=s, popsize=30,
                                   maxiter=300, tol=1e-10, polish=True, init="latinhypercube")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def ystar(iid, x):
    if iid in ("TP3", "TP8"):
        return ystar_exact(iid, x)
    return ystar_reference(SUITE[iid], x)


def truth(iid, x):
    inst = SUITE[iid]
    y = ystar(iid, x)
    g = inst.g(x, y)
    llf = bool(np.all(g <= 1e-6)) if g.size else True
    return dict(y=y, theta=float(inst.F(x, y)), phi=np.atleast_1d(inst.G(x, y)).astype(float),
                fstar=float(inst.f(x, y)), ll_feasible=llf)


# ---------------------------------------------------------------------------- run selection
def median_run(iid, method="M1"):
    d = pd.read_csv(os.path.join(RES, "summary.csv"))
    m = d[(d.method_id == method) & (d.instance_id == iid)].copy()
    m["key"] = np.where(m.feasible_found.astype(bool), m.best, np.inf)
    m = m.sort_values(["key", "run_id"])
    return int(m.iloc[14].run_id), float(m.iloc[14].best)


def locked_traj(method, iid, run):
    tag = f'"method_id": "{method}", "instance_id": "{iid}", "run_id": {run},'
    with open(os.path.join(RES, "trajectories.jsonl")) as fh:
        for line in fh:
            if tag in line:
                return [r["best_feasible_obj"] for r in json.loads(line)["rows"]]
    raise KeyError(tag)


def rerun_with_snapshots(iid, run, method="M1"):
    """Re-run under the locked seed, capturing SA-APPA's own GP objects after the initial design
    and at the end of the run. Returns the snapshots and the reproduction check."""
    snaps = {}

    def hook(stage, X, F, G, theta_gp, phi_gps):
        snaps[stage] = dict(X=X, F=F, G=np.array(G, dtype=float).reshape(len(F), -1),
                            theta_gp=theta_gp, phi_gps=phi_gps)
    sp = METHOD_SPECS[method]
    traj, _ = run_ablation_method(SUITE[iid], method, sp["engine"], sp["constraint_mode"],
                                  sp["adaptation"], sp["init"], seed=run, hook=hook)
    new = [r["best_feasible_obj"] for r in traj.rows]
    old = locked_traj(method, iid, run)

    def rel(a, b):
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return 0.0 if (a is None or not np.isfinite(a)) == (b is None or not np.isfinite(b)) else np.inf
        return abs(a - b) / max(1.0, abs(b))
    dev = max(rel(a, b) for a, b in zip(new, old)) if len(new) == len(old) else np.inf
    return snaps, dict(instance=iid, method=method, run=run, n_rows=len(new),
                       max_rel_dev=float(dev), reproduces=bool(dev <= 1e-6))


# ---------------------------------------------------------------------------- metrics
def gp_metrics(gp, Xh, th):
    mu, sd = gp.predict(Xh)
    err = mu - th
    cov = np.mean(np.abs(err) <= TWO_SIGMA * sd) if len(th) else np.nan
    return dict(n=int(len(th)), rmse=float(np.sqrt(np.mean(err ** 2))), max_abs=float(np.max(np.abs(err))),
                coverage_2sd=float(cov), mean_sd=float(np.mean(sd)),
                theta_range=float(np.ptp(th)) if len(th) else np.nan)


def heldout(iid, n):
    inst = SUITE[iid]
    U = qmc.LatinHypercube(d=inst.n_ul, seed=HELDOUT_SEED).random(n)
    X = qmc.scale(U, inst.x_bounds[:, 0], inst.x_bounds[:, 1])
    T = [truth(iid, x) for x in X]
    keep = np.array([t["ll_feasible"] for t in T])
    return X, np.array([t["theta"] for t in T]), np.array([t["phi"] for t in T]), keep
