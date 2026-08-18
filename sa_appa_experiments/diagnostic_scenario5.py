"""
diagnostic_scenario5.py
========================
Task 29 (post-sign-off supplementary analysis): before drawing any
conclusion from Scenario 5 (M1 adaptive vs M9 fixed-schedule, the LOCKED
result is T2=29.16, p=2.7e-06, M9 better -- see results/summary.csv /
task6 notes, NOT re-run or reframed here), check whether M1's own adaptive
mechanism was ever tuned. It was not: beta/gamma/delta were hardcoded
(0.2/0.3/0.3, methods.py) and eta/adaptive-p_f (Eq. pf) was not
implemented at all in the code that produced the locked results (p_f was
constant 0.6 regardless of adaptation flag). See methods.py's new pf_eta
kwarg (Task 29) for the additive, opt-in fix -- inactive on every call
that produced the locked results.

This script:
  1. Runs a small random search over (eta, beta, gamma, delta) for M1's
     APPA engine, evaluated ONLY on a held-out subset of 8 instances (not
     used to draw the final Scenario 5 conclusion), 5 seeds per config.
  2. Scores each config by its mean 2-way Friedman-style rank against M9's
     EXISTING (already-locked, unchanged) mean-of-30 performance on the
     same 8 held-out instances.
  3. Picks the best config. The actual full-suite re-run of Scenario 5
     with that config (30 seeds x 44 instances) is a separate script
     (rerun_scenario5_tuned.py) run only after this search completes.
"""

import json
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

from benchmark import SUITE
from methods import run_ablation_method

HELDOUT_INSTANCES = ["TP3", "TP7", "SMD1_p3q3", "SMD1_p30q30", "SMD5_p10q10",
                      "SMD7_p30q30", "SMD9_p3q3", "SMD4_p10q10"]
N_SEEDS_DIAG = 5
N_RANDOM_CONFIGS = 10

OUT_PATH = "results/diag_scenario5_search.csv"


def sample_configs(n, rng):
    configs = [dict(label="baseline_default", pf_eta=None, beta=0.2, gamma=0.3, delta=0.3)]
    for i in range(n):
        configs.append(dict(
            label=f"rand{i}",
            pf_eta=float(rng.uniform(0.02, 0.4)),
            beta=float(rng.uniform(0.05, 0.6)),
            gamma=float(rng.uniform(0.1, 0.6)),
            delta=float(rng.uniform(0.1, 0.6)),
        ))
    return configs


def _run_one(args):
    label, cfg, inst_id, seed = args
    inst = SUITE[inst_id]
    t0 = time.time()
    traj, cpu = run_ablation_method(
        inst, "M1", "appa", "cei", "adaptive", "lhs", seed=seed,
        pf_eta=cfg["pf_eta"], pf_min=0.1, pf_max=0.9,
        beta=cfg["beta"], gamma=cfg["gamma"], delta=cfg["delta"],
    )
    last = traj.rows[-1]
    return dict(label=label, instance_id=inst_id, seed=seed,
                best=last["best_feasible_obj"], feasible=last["is_feasible"],
                cpu_time_s=time.time() - t0)


def main():
    rng = np.random.default_rng(12345)
    configs = sample_configs(N_RANDOM_CONFIGS, rng)
    cfg_by_label = {c["label"]: c for c in configs}

    jobs = []
    for c in configs:
        for inst_id in HELDOUT_INSTANCES:
            for seed in range(N_SEEDS_DIAG):
                jobs.append((c["label"], c, inst_id, seed))

    print(f"Diagnostic search: {len(configs)} configs x {len(HELDOUT_INSTANCES)} instances "
          f"x {N_SEEDS_DIAG} seeds = {len(jobs)} runs")

    t0 = time.time()
    with mp.Pool(processes=2) as pool:
        results = pool.map(_run_one, jobs)
    print(f"Done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

    # Load M9's existing (locked, unchanged) per-instance mean on the same
    # held-out instances from results/summary.csv.
    full = pd.read_csv("results/summary.csv")
    m9 = full[(full.method_id == "M9") & (full.instance_id.isin(HELDOUT_INSTANCES))]
    m9_mean = {}
    for inst_id in HELDOUT_INSTANCES:
        sub = m9[m9.instance_id == inst_id]
        feas = sub[sub.feasible_found]
        m9_mean[inst_id] = feas.best.mean() if len(feas) > 0 else np.nan

    print("\nM9 (fixed-schedule, existing/unchanged) mean objective on held-out instances:")
    for k, v in m9_mean.items():
        print(f"  {k}: {v}")

    # Score each config: for each instance, mean-of-5-seeds M1-tuned
    # objective vs M9's mean-of-30; 2-way rank (1=better,2=worse), average
    # across the 8 instances. Lower average rank = better.
    scores = []
    for label, cfg in cfg_by_label.items():
        sub = df[df.label == label]
        ranks = []
        detail = {}
        for inst_id in HELDOUT_INSTANCES:
            isub = sub[sub.instance_id == inst_id]
            ifeas = isub[isub.feasible == True]
            m1_mean = ifeas.best.mean() if len(ifeas) > 0 else np.nan
            m9_val = m9_mean[inst_id]
            if np.isnan(m1_mean) and np.isnan(m9_val):
                r_m1 = 1.5
            elif np.isnan(m1_mean):
                r_m1 = 2.0
            elif np.isnan(m9_val):
                r_m1 = 1.0
            else:
                r_m1 = 1.0 if m1_mean < m9_val else (1.5 if m1_mean == m9_val else 2.0)
            ranks.append(r_m1)
            detail[inst_id] = dict(m1_mean=m1_mean, m9_mean=m9_val, m1_rank=r_m1)
        avg_rank = float(np.mean(ranks))
        n_feas_total = int(sub.feasible.sum())
        scores.append(dict(label=label, cfg=cfg, avg_rank_vs_m9=avg_rank,
                            n_feasible_of=len(sub), n_feasible=n_feas_total, detail=detail))

    scores.sort(key=lambda s: (s["avg_rank_vs_m9"], -s["n_feasible"]))
    print("\n" + "=" * 78)
    print("RANKED CONFIGS (lower avg_rank_vs_m9 = better; baseline_default = current M1)")
    print("=" * 78)
    for s in scores:
        c = s["cfg"]
        eta_str = "None" if c["pf_eta"] is None else f"{c['pf_eta']:.3f}"
        print(f"{s['label']:16s} avg_rank={s['avg_rank_vs_m9']:.3f}  "
              f"n_feas={s['n_feasible']}/{s['n_feasible_of']}  "
              f"eta={eta_str} beta={c['beta']:.3f} gamma={c['gamma']:.3f} delta={c['delta']:.3f}")

    with open("results/diag_scenario5_ranked.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)
    print("\nWrote results/diag_scenario5_ranked.json")
    return scores


if __name__ == "__main__":
    main()
