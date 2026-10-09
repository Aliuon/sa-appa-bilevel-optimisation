"""
run_m14.py
==========
Runs M14K (M14 with top-K LL training sets, sensitivity check; matched-budget BL-SAEA re-implementation, methods_m14.py) on the
same 44 reconciled instances x 30 seeds as M1-M13, with the same seed
convention (seed = run_id) as run_all.py.

Writes to SEPARATE files so the locked M1-M13 outputs are never touched:
    results/summary_m14k.csv
    results/trajectories_m14k.jsonl
merge_m14.py then writes results/summary_all14.csv for the analysis.
Resumable: completed (instance, run) pairs are skipped on restart.
"""

import argparse, csv, json, multiprocessing as mp, os, time
import numpy as np

from benchmark import SUITE
from methods_m14 import run_m14

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SUMMARY = os.path.join(RES, "summary_m14k.csv")
TRAJ = os.path.join(RES, "trajectories_m14k.jsonl")
FIELDS = ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"]


def _worker(args):
    iid, r = args
    t0 = time.time()
    try:
        traj, cpu = run_m14(SUITE[iid], seed=r, method_id="M14", ll_topk=True)
        return dict(method_id="M14K", instance_id=iid, run_id=r, best=traj.best,
                    feasible_found=traj.found_feasible, cpu_time_s=cpu,
                    rows=traj.rows, error=None)
    except Exception as e:  # recorded, never hidden
        return dict(method_id="M14K", instance_id=iid, run_id=r, best=np.nan,
                    feasible_found=False, cpu_time_s=time.time() - t0,
                    rows=[], error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    a = ap.parse_args()
    done = set()
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as fh:
            for row in csv.DictReader(fh):
                done.add((row["instance_id"], int(row["run_id"])))
    def _k_lt_budget(i):
        inst = SUITE[i]
        return (inst.x_bounds.shape[0] + 1) * (inst.y_bounds.shape[0] + 2) // 2 < 60
    work = [(i, r) for i in SUITE if _k_lt_budget(i) for r in range(a.runs) if (i, r) not in done]
    print(f"M14: {len(work)} runs to do ({len(done)} already done)", flush=True)
    new = not os.path.exists(SUMMARY)
    sf = open(SUMMARY, "a", newline="")
    w = csv.DictWriter(sf, fieldnames=FIELDS)
    if new:
        w.writeheader()
    tf = open(TRAJ, "a")
    t0, n = time.time(), 0
    with mp.Pool(a.workers) as pool:
        for res in pool.imap_unordered(_worker, work, chunksize=1):
            w.writerow({k: res[k] for k in FIELDS}); sf.flush()
            tf.write(json.dumps(dict(method_id="M14", instance_id=res["instance_id"],
                                     run_id=res["run_id"], rows=res["rows"],
                                     error=res["error"])) + "\n"); tf.flush()
            n += 1
            if res["error"]:
                print("ERROR", res["instance_id"], res["run_id"], res["error"], flush=True)
            if n % 50 == 0 or n == len(work):
                el = time.time() - t0
                print(f"{n}/{len(work)} done, {el/60:.1f} min elapsed, "
                      f"ETA {(len(work)-n)*el/n/60:.1f} min", flush=True)
    print("M14 sweep finished", flush=True)


if __name__ == "__main__":
    main()
