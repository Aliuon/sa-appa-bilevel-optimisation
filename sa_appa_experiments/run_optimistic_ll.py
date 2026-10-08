"""
run_optimistic_ll.py
====================
SMD6 sensitivity run with the optimistic tie-break of Assumption A1 (optimistic_ll.py).

Re-runs all 14 configurations on SMD6 at (3,3), (10,10) and (30,30), 30 runs each, with the
locked seed convention (seed = run_id), the same budget (60 nested solves) and the same settings;
the only change is that every nested LL solve is followed by the optimistic refinement.

Restricted to SMD6 on purpose: it is the only instance in the suite whose UL objective differs
across tied LL optima (on SMD11 F is constant over the tie, Appendix A). On instances where the
numerical LL solution is not exactly optimal, the refinement would move within a sub-optimal
level set of f and could report values better than the true optimum, so it is not a safe
default for the whole suite.

Writes results/optll/summary_optll.csv and trajectories_optll.jsonl (resumable). Locked files
are only read.
"""
import argparse, csv, json, multiprocessing as mp, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "optll")
SUMMARY = os.path.join(OUT, "summary_optll.csv")
TRAJ = os.path.join(OUT, "trajectories_optll.jsonl")
FIELDS = ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s",
          "refine_calls", "refine_accepted", "refine_F_evals", "refine_f_evals"]
INSTANCES = ["SMD6_p3q3", "SMD6_p10q10", "SMD6_p30q30"]


def _init():
    import optimistic_ll
    optimistic_ll.patch()


def _worker(args):
    mid, iid, r = args
    import optimistic_ll
    from benchmark import SUITE
    import methods, methods_m14
    for k in optimistic_ll.COUNTS:
        optimistic_ll.COUNTS[k] = 0
    t0 = time.time()
    try:
        if mid == "M14":
            traj, cpu = methods_m14.run_m14(SUITE[iid], seed=r)
        else:
            traj, cpu = methods.run_method(SUITE[iid], mid, seed=r)
        err, rows, best, ff = None, traj.rows, traj.best, traj.found_feasible
    except Exception as e:
        err, rows, best, ff, cpu = f"{type(e).__name__}: {e}", [], np.nan, False, time.time() - t0
    c = dict(optimistic_ll.COUNTS)
    return dict(method_id=mid, instance_id=iid, run_id=r, best=best, feasible_found=ff, cpu_time_s=cpu,
                refine_calls=c["calls"], refine_accepted=c["accepted"], refine_F_evals=c["F_evals"],
                refine_f_evals=c["f_evals"], rows=rows, error=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from methods import METHOD_SPECS
    methods_all = list(METHOD_SPECS.keys()) + ["M14"]
    done = set()
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as fh:
            done = {(r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(fh)}
    work = [(m, i, r) for i in INSTANCES for m in methods_all for r in range(a.runs) if (m, i, r) not in done]
    print(f"{len(work)} runs to do ({len(done)} done)", flush=True)
    new = not os.path.exists(SUMMARY)
    sf = open(SUMMARY, "a", newline=""); w = csv.DictWriter(sf, fieldnames=FIELDS)
    if new:
        w.writeheader()
    tf = open(TRAJ, "a")
    t0, n = time.time(), 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for res in pool.imap_unordered(_worker, work, chunksize=1):
            w.writerow({k: res[k] for k in FIELDS}); sf.flush()
            tf.write(json.dumps(dict(method_id=res["method_id"], instance_id=res["instance_id"],
                                     run_id=res["run_id"], rows=res["rows"], error=res["error"]),
                                default=float) + "\n"); tf.flush()
            n += 1
            if res["error"]:
                print("ERROR", res["method_id"], res["instance_id"], res["run_id"], res["error"], flush=True)
            if n % 25 == 0 or n == len(work):
                print(f"{n}/{len(work)} done, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
