"""
rerun_scenario5_tuned.py
=========================
Task 29, final step: ONE re-run of Scenario 5 (M1 vs M9) on the FULL
44-instance suite, 30 seeds each, using the best config found by the
held-out diagnostic search (diagnostic_scenario5.py): rand7, avg_rank=1.625
vs baseline's 1.750 (lower is better; M9's existing/unchanged results are
NOT re-run -- only M1-tuned is computed here).

Per the explicit instruction this script runs EXACTLY ONCE with this one
config -- no further search, no re-running to try for a better outcome.
"""

import time
import multiprocessing as mp
import pandas as pd

from benchmark import SUITE
from methods import run_ablation_method

BEST_CONFIG = dict(pf_eta=0.055, beta=0.379, gamma=0.527, delta=0.401, pf_min=0.1, pf_max=0.9)
N_RUNS = 30
OUT_PATH = "results/scenario5_tuned_summary.csv"
TRAJ_OUT_PATH = "results/scenario5_tuned_trajectories.jsonl"


def _run_one(args):
    inst_id, seed = args
    inst = SUITE[inst_id]
    t0 = time.time()
    traj, cpu = run_ablation_method(
        inst, "M1", "appa", "cei", "adaptive", "lhs", seed=seed, **BEST_CONFIG
    )
    last = traj.rows[-1]
    return dict(method_id="M1_tuned", instance_id=inst_id, run_id=seed,
                best=last["best_feasible_obj"], feasible_found=last["is_feasible"],
                cpu_time_s=time.time() - t0), dict(
        method_id="M1_tuned", instance_id=inst_id, run_id=seed,
        rows=[dict(eval_count=r["eval_count"], iteration=r["iteration"],
                    best_feasible_obj=r["best_feasible_obj"], is_feasible=r["is_feasible"])
              for r in traj.rows])


def main():
    jobs = [(inst_id, seed) for inst_id in SUITE.keys() for seed in range(N_RUNS)]
    print(f"Scenario 5 tuned re-run: {len(jobs)} runs (44 instances x 30 seeds), config={BEST_CONFIG}")

    t0 = time.time()
    with mp.Pool(processes=2) as pool:
        results = pool.map(_run_one, jobs)
    print(f"Done in {time.time()-t0:.1f}s")

    summary_rows = [r[0] for r in results]
    traj_rows = [r[1] for r in results]

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

    import json
    with open(TRAJ_OUT_PATH, "w") as f:
        for t in traj_rows:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {TRAJ_OUT_PATH}")


if __name__ == "__main__":
    main()
