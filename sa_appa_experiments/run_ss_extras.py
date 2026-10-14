"""
run_ss_extras.py
================
Side experiments repeated with strong Stackelberg selection (strong_stackelberg.py) on the
instances where it changes results (run_strong_stackelberg.AFFECTED), so that every analysis in
the paper uses one lower-level protocol:

  s5   : Scenario 5 tuning diagnostic (M1 with the tuned adaptive rule), 30 runs per instance
  m14k : M14K (BL-SAEA with top-K LL training sets) where K < budget, 30 runs per instance
  b1   : instrumented re-run for the evaluation counts (runs 0-4, all 14 configurations); each
         run is checked against the strong Stackelberg main run with the same seed

Outputs go to results/strong_stackelberg/ (resumable). Nothing else is written.
"""
import argparse, copy, csv, json, multiprocessing as mp, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "strong_stackelberg")


def _init():
    import strong_stackelberg
    strong_stackelberg.patch()


def _s5(args):
    iid, seed = args
    from benchmark import SUITE
    from methods import run_ablation_method
    from rerun_scenario5_tuned import BEST_CONFIG
    t0 = time.time()
    traj, cpu = run_ablation_method(SUITE[iid], "M1", "appa", "cei", "adaptive", "lhs", seed=seed, **BEST_CONFIG)
    last = traj.rows[-1]
    return dict(method_id="M1_tuned", instance_id=iid, run_id=seed, best=last["best_feasible_obj"],
                feasible_found=last["is_feasible"], cpu_time_s=time.time() - t0), \
        dict(method_id="M1_tuned", instance_id=iid, run_id=seed,
             rows=[dict(eval_count=r["eval_count"], iteration=r["iteration"],
                        best_feasible_obj=r["best_feasible_obj"], is_feasible=r["is_feasible"]) for r in traj.rows])


def _m14k(args):
    iid, r = args
    from benchmark import SUITE
    from methods_m14 import run_m14
    traj, cpu = run_m14(SUITE[iid], seed=r, method_id="M14", ll_topk=True)
    return dict(method_id="M14K", instance_id=iid, run_id=r, best=traj.best,
                feasible_found=traj.found_feasible, cpu_time_s=cpu), \
        dict(method_id="M14", instance_id=iid, run_id=r, rows=traj.rows, error=None)


class _Counter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, x, y):
        self.n += 1
        return self.f(x, y)


def _b1(args):
    mid, iid, r, ref = args
    from benchmark import SUITE
    from methods import run_method
    from methods_m14 import run_m14
    inst = copy.copy(SUITE[iid]); cnt = _Counter(inst.f); inst.f = cnt
    traj, cpu = run_m14(inst, seed=r) if mid == "M14" else run_method(inst, mid, seed=r)
    best = traj.best if traj.found_feasible else np.nan
    match = (np.isnan(best) and np.isnan(ref)) or (abs(best - ref) <= 1e-6 * max(1.0, abs(ref)))
    return dict(method_id=mid, instance_id=iid, run_id=r, n_ul=len(traj.rows), n_ll_f=cnt.n,
                best=traj.best, feasible_found=traj.found_feasible, ss_best=ref, best_match=bool(match),
                cpu_time_s=cpu), None


def run(name, fn, work, fields, traj_name=None, workers=2):
    path = os.path.join(OUT, f"{name}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {tuple(str(row[k]) for k in ("method_id", "instance_id", "run_id")) for row in csv.DictReader(fh)}
    print(f"[{name}] {len(work)} jobs, {len(done)} done", flush=True)
    new = not os.path.exists(path)
    fh = open(path, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    tf = open(os.path.join(OUT, traj_name), "a") if traj_name else None
    t0, n = time.time(), 0
    with mp.Pool(workers, initializer=_init) as pool:
        for row, tr in pool.imap_unordered(fn, work, chunksize=1):
            w.writerow({k: row[k] for k in fields}); fh.flush()
            if tf and tr:
                tf.write(json.dumps(tr, default=float) + "\n"); tf.flush()
            n += 1
            if n % 25 == 0:
                print(f"[{name}] {n}/{len(work)} {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[{name}] finished", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="s5,m14k,b1"); a = ap.parse_args()
    from run_strong_stackelberg import AFFECTED
    from benchmark import SUITE
    from methods import METHOD_SPECS
    todo = a.only.split(",")
    if "s5" in todo:
        p = os.path.join(OUT, "scenario5_tuned_ss.csv")
        done = set()
        if os.path.exists(p):
            done = {(r["instance_id"], int(r["run_id"])) for r in csv.DictReader(open(p))}
        work = [(i, s) for i in AFFECTED for s in range(30) if (i, s) not in done]
        run("scenario5_tuned_ss", _s5, work, ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"],
            "scenario5_tuned_trajectories_ss.jsonl", a.workers)
    if "m14k" in todo:
        def k_ok(i):
            inst = SUITE[i]
            return (inst.x_bounds.shape[0] + 1) * (inst.y_bounds.shape[0] + 2) // 2 < 60
        p = os.path.join(OUT, "summary_m14k_ss.csv")
        done = set()
        if os.path.exists(p):
            done = {(r["instance_id"], int(r["run_id"])) for r in csv.DictReader(open(p))}
        work = [(i, r) for i in AFFECTED if k_ok(i) for r in range(30) if (i, r) not in done]
        run("summary_m14k_ss", _m14k, work, ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"],
            "trajectories_m14k_ss.jsonl", a.workers)
    if "b1" in todo:
        ref = {}
        for row in csv.DictReader(open(os.path.join(OUT, "summary_ss.csv"))):
            if row["mode"] == "full":
                b = float(row["best"]) if row["feasible_found"] == "True" else np.nan
                ref[(row["method_id"], row["instance_id"], int(row["run_id"]))] = b
        p = os.path.join(OUT, "ll_counts_ss.csv")
        done = set()
        if os.path.exists(p):
            done = {(r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(open(p))}
        ms = list(METHOD_SPECS.keys()) + ["M14"]
        work = [(m, i, r, ref[(m, i, r)]) for i in AFFECTED for m in ms for r in range(5) if (m, i, r) not in done]
        run("ll_counts_ss", _b1, work, ["method_id", "instance_id", "run_id", "n_ul", "n_ll_f", "best", "feasible_found",
                                        "ss_best", "best_match", "cpu_time_s"], None, a.workers)


if __name__ == "__main__":
    main()
