"""
analysis.py
===========
Reads the real experiment results (results/summary.csv, produced by
run_all.py) and, for each of the 7 scenarios (scenarios.py):
  1. builds the Best/Mean/Std/CPU-Time table (Appendix B), one row per
     reconciled instance, one column-group per method in the scenario;
  2. ranks methods within each instance on the mean-of-30-runs feasible
     objective (methods with zero feasible runs on an instance ranked
     worst, tied -- same rule as Task 4/5's fig5_cd_diagram.py);
  3. runs the Friedman test (Conover's F-approximation) on that scenario's
     rank matrix, via the validated friedman_test.py;
  4. computes the Conover post-hoc paired critical difference (rank-sum
     units) AND the Nemenyi critical distance (mean-rank units, for the
     CD diagram) -- kept explicitly distinct per Task 5's standing rule.

Usage: python3 analysis.py            # all 7 scenarios
       python3 analysis.py --num 3    # just scenario 3
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scenarios import SCENARIOS, SCENARIO_BY_NUM
from benchmark import SUITE
from friedman_test import friedman_f_statistic, f_critical, conover_posthoc_cd, nemenyi_cd, pairwise_table

RESULTS_CSV = "results/summary.csv"


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    return df


def per_cell_stats(df, method_id, instance_id):
    sub = df[(df.method_id == method_id) & (df.instance_id == instance_id)]
    if sub.empty:
        return dict(n_runs=0, n_feasible=0, best=np.nan, mean=np.nan, std=np.nan, cpu_mean=np.nan)
    feas = sub[sub.feasible_found]
    n_feas = len(feas)
    if n_feas > 0:
        best = feas.best.min()
        mean = feas.best.mean()
        std = feas.best.std(ddof=1) if n_feas > 1 else 0.0
    else:
        best = mean = std = np.nan
    return dict(n_runs=len(sub), n_feasible=n_feas, best=best, mean=mean, std=std,
                cpu_mean=sub.cpu_time_s.mean())


def scenario_table(df, scenario):
    methods = scenario["methods"]
    rows = []
    for iid in SUITE.keys():
        row = dict(instance_id=iid)
        for m in methods:
            st = per_cell_stats(df, m, iid)
            row[f"{m}_best"] = st["best"]
            row[f"{m}_mean"] = st["mean"]
            row[f"{m}_std"] = st["std"]
            row[f"{m}_cpu"] = st["cpu_mean"]
            row[f"{m}_nfeas"] = st["n_feasible"]
        rows.append(row)
    return pd.DataFrame(rows)


def rank_matrix_for_scenario(table_df, methods):
    """One row per instance, one column per method, rank 1=best. Methods
    with zero feasible runs on an instance are ranked worst (tied)."""
    n = len(table_df)
    k = len(methods)
    ranks = np.zeros((n, k))
    for i, row in table_df.iterrows():
        means = np.array([row[f"{m}_mean"] for m in methods])
        nfeas = np.array([row[f"{m}_nfeas"] for m in methods])
        BIG = (np.nanmax(means) + 1e9) if np.any(~np.isnan(means)) else 1e9
        score = np.where(nfeas > 0, np.nan_to_num(means, nan=BIG), BIG)
        ranks[i, :] = rankdata(score, method="average")
    return ranks


def run_scenario(df, scenario_num, alpha=(0.05, 0.01)):
    scenario = SCENARIO_BY_NUM[scenario_num]
    methods = scenario["methods"]
    table_df = scenario_table(df, scenario)
    rank_matrix = rank_matrix_for_scenario(table_df, methods)
    n, k = rank_matrix.shape

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05)
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01)
    cd_conover_05, t_crit_05 = conover_posthoc_cd(stat, 0.05)
    cd_conover_01, t_crit_01 = conover_posthoc_cd(stat, 0.01)
    cd_nemenyi_05 = nemenyi_cd(k, n, 0.05)
    mean_ranks = rank_matrix.mean(axis=0)

    return dict(
        scenario=scenario, table=table_df, rank_matrix=rank_matrix,
        stat=stat, f_crit_05=f_crit_05, f_crit_01=f_crit_01,
        cd_conover_05=cd_conover_05, cd_conover_01=cd_conover_01,
        cd_nemenyi_05=cd_nemenyi_05, mean_ranks=dict(zip(methods, mean_ranks)),
    )


def print_scenario_report(result):
    s = result["scenario"]
    stat = result["stat"]
    print("=" * 78)
    print(f"Scenario {s['num']} ({s['phase']}): {s['name']}  --  {', '.join(s['methods'])}")
    print(f"Isolates: {s['isolates']}")
    print(f"n={stat['n']} instances, k={stat['k']} methods")
    print("-" * 78)
    print(f"A2={stat['A2']:.3f}  B2={stat['B2']:.3f}  T2={stat['T2']:.4f}  "
          f"F({stat['df1']},{stat['df2']})  p={stat['p_value']:.4g}")
    decision05 = "REJECT H0" if stat["T2"] > result["f_crit_05"] else "fail to reject"
    decision01 = "REJECT H0" if stat["T2"] > result["f_crit_01"] else "fail to reject"
    print(f"F_crit(0.05)={result['f_crit_05']:.3f} -> {decision05}")
    print(f"F_crit(0.01)={result['f_crit_01']:.3f} -> {decision01}")
    print(f"Mean ranks: " + ", ".join(f"{m}={r:.3f}" for m, r in result["mean_ranks"].items()))
    print(f"Conover post-hoc CD (rank-sum units): alpha=0.05 -> {result['cd_conover_05']:.3f}, "
          f"alpha=0.01 -> {result['cd_conover_01']:.3f}")
    print(f"Nemenyi CD (mean-rank units, for CD diagram): alpha=0.05 -> {result['cd_nemenyi_05']:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=None)
    args = parser.parse_args()
    df = load_results()
    nums = [args.num] if args.num else [s["num"] for s in SCENARIOS]
    for n in nums:
        result = run_scenario(df, n)
        print_scenario_report(result)
