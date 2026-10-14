"""
run_strong_stackelberg.py
=========================
Main-protocol runs with strong Stackelberg (optimistic) selection among tied LL optima
(strong_stackelberg.py) applied to every nested LL solve.

The diagnostic (ss_diagnostic.py, 20 UL points per instance) found ties that change the
leader's value or UL feasibility on 8 of the 44 instances: TP2, TP4, SMD6 and SMD11 at all
three scales. On the other 36 the selection never fired there. This is checked, not assumed:

  scan : the other 36 instances x 14 configurations x run 0, with the selection active. Every
         run must report zero accepted selections and a best value identical to the locked one;
         any instance where a selection fires is moved to the full re-run.
  full : the affected instances x 14 configurations x 30 runs (seed = run_id, 60 nested solves,
         all settings as locked).
The scan found replacements on SMD12 (9 of 42 runs), so SMD12 at all three scales was added to
AFFECTED and re-run in full; its scan rows are kept for the record but not used.

Writes results/strong_stackelberg/summary_ss.csv and trajectories_ss.jsonl (resumable).
Locked files are only read.
"""
import argparse, csv, json, multiprocessing as mp, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "strong_stackelberg")
SUMMARY = os.path.join(OUT, "summary_ss.csv")
TRAJ = os.path.join(OUT, "trajectories_ss.jsonl")
FIELDS = ["mode", "method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s",
          "ss_calls", "ss_gated", "ss_accepted", "ss_accepted_feas", "ss_ball_rejected",
          "ss_F_evals", "ss_f_evals", "ss_g_evals", "ss_G_evals"]
AFFECTED = ["TP2", "TP4", "SMD6_p3q3", "SMD6_p10q10", "SMD6_p30q30",
            "SMD11_p3q3", "SMD11_p10q10", "SMD11_p30q30",
            # added after the scan: the selection replaced the response in 9 of the 42 SMD12 scan
            # runs (SMD12 has tied LL optima but its LL solve usually fails the accuracy check)
            "SMD12_p3q3", "SMD12_p10q10", "SMD12_p30q30"]


def _init():
    import strong_stackelberg
    strong_stackelberg.patch()


def _worker(args):
    mode, mid, iid, r = args
    import strong_stackelberg as ss
    from benchmark import SUITE
    import methods, methods_m14
    ss.reset_counts()
    t0 = time.time()
    try:
        if mid == "M14":
            traj, cpu = methods_m14.run_m14(SUITE[iid], seed=r)
        else:
            traj, cpu = methods.run_method(SUITE[iid], mid, seed=r)
        err, rows, best, ff = None, traj.rows, traj.best, traj.found_feasible
    except Exception as e:
        err, rows, best, ff, cpu = f"{type(e).__name__}: {e}", [], np.nan, False, time.time() - t0
    c = dict(ss.COUNTS)
    out = dict(mode=mode, method_id=mid, instance_id=iid, run_id=r, best=best, feasible_found=ff,
               cpu_time_s=cpu, rows=rows, error=err)
    for k in ["calls", "gated", "accepted", "accepted_feas", "ball_rejected",
              "F_evals", "f_evals", "g_evals", "G_evals"]:
        out["ss_" + k] = c[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--extra", default="", help="comma list of instances added to the full re-run")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from methods import METHOD_SPECS
    from benchmark import SUITE
    methods_all = list(METHOD_SPECS.keys()) + ["M14"]
    affected = AFFECTED + [i for i in a.extra.split(",") if i]
    others = [i for i in SUITE if i not in affected and not i.startswith("SMD12")]
    done = set()
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as fh:
            done = {(r["mode"], r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(fh)}
    work = [("full", m, i, r) for i in affected for m in methods_all for r in range(a.runs)]
    work += [("scan", m, i, 0) for i in others for m in methods_all]
    work = [w for w in work if w not in done]
    print(f"{len(work)} runs to do ({len(done)} done)", flush=True)
    new = not os.path.exists(SUMMARY)
    sf = open(SUMMARY, "a", newline=""); w = csv.DictWriter(sf, fieldnames=FIELDS)
    if new:
        w.writeheader()
    tf = open(TRAJ, "a")
    t0, n = time.time(), 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for res in pool.imap(_worker, work, chunksize=1):
            w.writerow({k: res[k] for k in FIELDS}); sf.flush()
            tf.write(json.dumps(dict(mode=res["mode"], method_id=res["method_id"], instance_id=res["instance_id"],
                                     run_id=res["run_id"], rows=res["rows"], error=res["error"]),
                                default=float) + "\n"); tf.flush()
            n += 1
            if res["error"]:
                print("ERROR", res["method_id"], res["instance_id"], res["run_id"], res["error"], flush=True)
            if n % 25 == 0 or n == len(work):
                print(f"{n}/{len(work)} done, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
