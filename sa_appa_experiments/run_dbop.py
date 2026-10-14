"""
run_dbop.py: DBOP case study runs (Section 5.12).

- reference: best-known optimum by constrained differential evolution (5 seeds);
- the 14 configurations x 30 runs x 60 nested solves, same seeds as the main study;
- the source paper's adaptive genetic algorithm (AGA; 100 x 100) x 30 runs as an
  evaluation-rich reference, translated from the authors' IMCE.m (penalty formulation of the source).
Output: results/dbop/{reference.json, summary_dbop.csv, trajectories_dbop.jsonl, aga_dbop.csv}.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import dbop_case as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "dbop")
METHODS = [f"M{i}" for i in range(1, 15)]
RUNS = 30


def J_G(x):
    y = C.follower(x)
    return C.F(x, y), C.G(x, y)


def reference():
    from scipy.optimize import differential_evolution, NonlinearConstraint
    best = None
    rows = []
    for s in range(5):
        r = differential_evolution(lambda x: J_G(x)[0], list(map(tuple, C.X_BOUNDS)),
                                   constraints=NonlinearConstraint(lambda x: J_G(x)[1].max(), -np.inf, 0),
                                   seed=s, maxiter=2000, popsize=30, tol=1e-12, polish=True)
        J, G = J_G(r.x)
        rows.append(dict(seed=s, J=J, maxG=float(G.max()), nfev=int(r.nfev), x=r.x.tolist()))
        if G.max() <= 1e-6 and (best is None or J < best["J"]):
            best = rows[-1]
    return dict(best=best, runs=rows)


# ---------------- AGA (translation of IMCE.m run_aga) ----------------
def aga(seed, NP=100, kmax=100, pc0=0.9, pm0=0.2, lam=0.1, dpm=0.05, s0=0.1, om=0.05, W=1e9):
    rng = np.random.default_rng(seed)
    lb, ub = C.X_BOUNDS[:, 0], C.X_BOUNDS[:, 1]
    nev = [0]
    best_feas = [np.inf]
    curve = []

    def fit(x):
        J, G = J_G(x)
        nev[0] += 1
        pen = np.sum(np.maximum(G[:-1], 0)) + (1e6 if G[-1] >= 0 else 0.0)
        if np.all(G <= 1e-6) and G[-1] < 0 and J < best_feas[0]:
            best_feas[0] = J
        if nev[0] in (60, 100, 500, 1000, 2500, 5000, 10100):
            curve.append((nev[0], best_feas[0]))
        return J + W * pen

    pop = lb + rng.random((NP, 7)) * (ub - lb)
    f = np.array([fit(x) for x in pop])
    fb, xb = f.min(), pop[f.argmin()].copy()
    T0 = max(f.std(), 1e-6); Tmin = T0 * 1e-3; kmid = kmax / 2
    for k in range(kmax):
        pc = pc0 / (1 + np.exp(lam * (k - kmid))); pm = pm0 * np.exp(-dpm * k); sig = s0 * np.exp(-om * k)
        temp = T0 * (Tmin / T0) ** (k / (kmax - 1))
        sel = np.empty((NP - 1, 7))
        for j in range(NP - 1):
            i1, i2 = rng.integers(NP, size=2)
            b, w = (i1, i2) if f[i1] <= f[i2] else (i2, i1)
            z = np.clip(-(f[w] - f[b]) / max(temp, 1e-12), -700, 700)
            sel[j] = pop[b] if rng.random() < 1 / (1 + np.exp(z)) else pop[w]
        off = sel.copy(); idx = rng.permutation(NP - 1)
        for a, b in zip(idx[0::2], idx[1::2]):
            if rng.random() < pc:
                u = rng.random()
                off[a] = np.clip(u * sel[a] + (1 - u) * sel[b], lb, ub)
                off[b] = np.clip((1 - u) * sel[a] + u * sel[b], lb, ub)
        for i in range(NP - 1):
            if rng.random() < pm:
                off[i] = np.clip(off[i] + rng.standard_normal(7) * sig * (ub - lb), lb, ub)
        pop = np.vstack([xb, off]); f = np.array([fit(x) for x in pop])
        if f.min() < fb:
            fb, xb = f.min(), pop[f.argmin()].copy()
    J, G = J_G(xb)
    return dict(seed=seed, best_fitness=float(fb), J=float(J), feasible=bool(np.all(G <= 1e-6) and G[-1] < 0),
                best_feasible=float(best_feas[0]), evals=nev[0], curve=curve, x=xb.tolist())


def one(args):
    m, r = args
    C.patch_exact_ll()
    import methods, methods_m14
    inst = C.instance()
    t0 = time.time()
    traj, _ = methods_m14.run_m14(inst, seed=r) if m == "M14" else methods.run_method(inst, m, r)
    return dict(method_id=m, run_id=r, best=traj.best, feasible_found=traj.found_feasible,
                cpu_time_s=time.time() - t0,
                curve=[(row["eval_count"], row["best_feasible_obj"]) for row in traj.rows])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "ref"):
        ref = reference()
        json.dump(ref, open(os.path.join(OUT, "reference.json"), "w"), indent=1)
        print("reference J* =", ref["best"]["J"])
    if what in ("all", "runs"):
        jobs = [(m, r) for m in METHODS for r in range(RUNS)]
        with Pool(min(8, os.cpu_count())) as pool:
            res = pool.map(one, jobs, chunksize=4)
        pd.DataFrame([{k: v for k, v in d.items() if k != "curve"} for d in res]).to_csv(
            os.path.join(OUT, "summary_dbop.csv"), index=False)
        with open(os.path.join(OUT, "trajectories_dbop.jsonl"), "w") as fh:
            for d in res:
                fh.write(json.dumps(dict(method_id=d["method_id"], run_id=d["run_id"], curve=d["curve"])) + "\n")
        print("runs done:", len(res))
    if what in ("all", "aga"):
        with Pool(min(8, os.cpu_count())) as pool:
            res = pool.map(aga, range(RUNS))
        pd.DataFrame([{k: v for k, v in d.items() if k not in ("curve", "x")} for d in res]).to_csv(
            os.path.join(OUT, "aga_dbop.csv"), index=False)
        json.dump(res, open(os.path.join(OUT, "aga_dbop.json"), "w"))
        print("AGA done")
