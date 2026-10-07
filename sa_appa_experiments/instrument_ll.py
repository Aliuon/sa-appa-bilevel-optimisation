"""
instrument_ll.py
================
Instrumented re-run for the evaluation-count figures (Phase 7, Part B1).

Re-runs runs 0-4 (seed = run_id, the locked seed convention) of all 14 configurations on all
44 instances with a counter wrapped around the lower-level objective f(x, y). The wrapper
returns exactly the value of f, so the run is unchanged; this is verified by comparing every
re-run with the locked results (summary.csv / summary_m14.csv and the trajectory files).

Counted quantities per run:
  n_ul      number of full nested LL solves (= UL evaluations), from the trajectory length
  n_ll_f    number of calls to the LL objective f(x, y), over all nested solves

Writes results/ll_counts_subset.csv (resumable) and nothing else; locked files are only read.
"""
import argparse, copy, csv, json, os, time, multiprocessing as mp
import numpy as np

from benchmark import SUITE
from methods import run_method, METHOD_SPECS
from methods_m14 import run_m14

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "ll_counts_subset.csv")
FIELDS = ["method_id", "instance_id", "run_id", "n_ul", "n_ll_f", "best", "feasible_found",
          "locked_best", "best_match", "traj_match", "traj_max_rel_dev", "cpu_time_s"]
METHODS = list(METHOD_SPECS.keys()) + ["M14"]


class _Counter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, x, y):
        self.n += 1
        return self.f(x, y)


def _load_locked():
    import pandas as pd
    d = pd.concat([pd.read_csv(os.path.join(RES, "summary.csv")),
                   pd.read_csv(os.path.join(RES, "summary_m14.csv"))])
    return {(r.method_id, r.instance_id, int(r.run_id)): (r.best, bool(r.feasible_found))
            for r in d.itertuples()}


def _load_locked_traj(keys):
    want = set(keys); out = {}
    for fn in ("trajectories.jsonl", "trajectories_m14.jsonl"):
        with open(os.path.join(RES, fn)) as fh:
            for line in fh:
                j = json.loads(line)
                k = (j["method_id"], j["instance_id"], int(j["run_id"]))
                if k in want:
                    out[k] = [round(r["best_feasible_obj"], 9) if r["best_feasible_obj"] is not None
                              and np.isfinite(r["best_feasible_obj"]) else None for r in j["rows"]]
    return out


TOL = 1e-6   # relative tolerance for "reproduces"; the exact deviation is recorded as well


def _rel(a, b):
    if a is None or b is None:
        return 0.0 if (a is None and b is None) else float("inf")
    return abs(a - b) / max(1.0, abs(b))


def _same(a, b, tol=TOL):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)):
        return (a is None or (isinstance(a, float) and np.isnan(a))) and \
               (b is None or (isinstance(b, float) and np.isnan(b)))
    return abs(a - b) <= tol * max(1.0, abs(b))


def _worker(args):
    mid, iid, r, locked, ltraj = args
    inst = copy.copy(SUITE[iid])
    cnt = _Counter(inst.f)
    inst.f = cnt
    if mid == "M14":
        traj, cpu = run_m14(inst, seed=r)
    else:
        traj, cpu = run_method(inst, mid, seed=r)
    rows = [round(x["best_feasible_obj"], 9) if x["best_feasible_obj"] is not None
            and np.isfinite(x["best_feasible_obj"]) else None for x in traj.rows]
    if ltraj is not None and len(rows) == len(ltraj):
        dev = max((_rel(a, b) for a, b in zip(rows, ltraj)), default=0.0)
    else:
        dev = float("inf")
    tm = dev <= TOL
    lb = locked[0]
    return dict(method_id=mid, instance_id=iid, run_id=r, n_ul=len(traj.rows), n_ll_f=cnt.n,
                best=traj.best, feasible_found=traj.found_feasible, locked_best=lb,
                best_match=_same(traj.best if traj.found_feasible else np.nan,
                                 lb if locked[1] else np.nan),
                traj_match=tm, traj_max_rel_dev=dev, cpu_time_s=cpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    order = ([i for i in SUITE if i.startswith("TP")] + [i for i in SUITE if i.endswith("p3q3")] +
             [i for i in SUITE if i.endswith("p30q30")] + [i for i in SUITE if i.endswith("p10q10")])
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {(r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(fh)}
    keys = [(m, i, r) for i in order for m in METHODS for r in range(a.runs) if (m, i, r) not in done]
    locked = _load_locked(); ltraj = _load_locked_traj(keys)
    work = [(m, i, r, locked[(m, i, r)], ltraj.get((m, i, r))) for (m, i, r) in keys]
    print(f"{len(work)} runs to do ({len(done)} done)", flush=True)
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    t0 = time.time(); n = 0; bad = 0
    with mp.Pool(a.workers) as pool:
        for row in pool.imap_unordered(_worker, work, chunksize=1):
            w.writerow(row); fh.flush(); n += 1
            bad += (not row["best_match"]) or (not row["traj_match"])
            if n % 50 == 0 or n == len(work):
                print(f"{n}/{len(work)} done, {bad} mismatches, {(time.time()-t0)/60:.1f} min", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
