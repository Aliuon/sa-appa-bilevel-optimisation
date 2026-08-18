"""
analysis_extra.py
==================
Supplementary Friedman-test analyses requested after the 7 scenario results
were signed off as final (do NOT touch/re-run/reframe those -- this module
only reads results/summary.csv, never writes to it, and never re-derives
Scenario 1-7's objective-value results).

Two additional analyses, both using the same validated friedman_f_statistic
/ conover_posthoc_cd / nemenyi_cd machinery as analysis.py:

1. Feasibility-rate Friedman test (one per scenario, 7 total): ranks methods
   within each instance by N-feasible-of-30 (higher = better = rank 1)
   instead of mean feasible objective. Reported as its own labelled set of
   7 tests, explicitly separate from the objective-value tests.

2. Constraint-density subset analysis: the same 7 scenario groupings, same
   procedure, restricted to the 8 instances belonging to SMD10/SMD11/SMD12
   (the three benchmark families with the most active/scaling constraints:
   SMD10@(3,3),(10,10) [30,30 excluded, Appendix C], SMD11@(3,3),(10,10),
   (30,30), SMD12@(3,3),(10,10),(30,30)). This adds 7 more tests; combined
   with the 7 full-suite tests that is 14 tests, and a Holm-Bonferroni
   correction is applied across all 14 (both raw and corrected p reported).

Usage: python3 analysis_extra.py
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scenarios import SCENARIOS, SCENARIO_BY_NUM
from benchmark import SUITE
from friedman_test import friedman_f_statistic, f_critical, conover_posthoc_cd, nemenyi_cd
from analysis import load_results, per_cell_stats, scenario_table, run_scenario

CONSTRAINT_DENSE_FAMILIES = ("SMD10", "SMD11", "SMD12")
CONSTRAINT_DENSE_INSTANCES = [iid for iid in SUITE.keys()
                              if iid.split("_p")[0] in CONSTRAINT_DENSE_FAMILIES]


# ---------------------------------------------------------------------------
# 1. Feasibility-rate Friedman test
# ---------------------------------------------------------------------------

def feas_rank_matrix_for_scenario(table_df, methods):
    """One row per instance, one column per method. Ranking variable is
    N-feasible-of-30 (higher is better -> rank 1). Ties (e.g. two methods
    both 0/30 or both 30/30 on an instance) get average ranks, same
    convention as the objective-value rank matrix."""
    n = len(table_df)
    k = len(methods)
    ranks = np.zeros((n, k))
    for i, row in table_df.iterrows():
        nfeas = np.array([row[f"{m}_nfeas"] for m in methods], dtype=float)
        # higher nfeas = better = should get rank 1 -> rank the negative
        ranks[i, :] = rankdata(-nfeas, method="average")
    return ranks


def run_scenario_feasibility(df, scenario_num, instance_ids=None, alpha=(0.05, 0.01)):
    scenario = SCENARIO_BY_NUM[scenario_num]
    methods = scenario["methods"]
    table_df = scenario_table(df, scenario)
    if instance_ids is not None:
        table_df = table_df[table_df.instance_id.isin(instance_ids)].reset_index(drop=True)
    rank_matrix = feas_rank_matrix_for_scenario(table_df, methods)
    n, k = rank_matrix.shape

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05) if not np.isnan(stat["T2"]) else np.nan
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01) if not np.isnan(stat["T2"]) else np.nan
    mean_ranks = rank_matrix.mean(axis=0)
    mean_nfeas = {m: table_df[f"{m}_nfeas"].mean() for m in methods}

    return dict(scenario=scenario, table=table_df, rank_matrix=rank_matrix, stat=stat,
                f_crit_05=f_crit_05, f_crit_01=f_crit_01,
                mean_ranks=dict(zip(methods, mean_ranks)), mean_nfeas=mean_nfeas,
                n_instances=n)


# ---------------------------------------------------------------------------
# 2. Constraint-density subset (objective-value ranking, same as analysis.py
#    but restricted to CONSTRAINT_DENSE_INSTANCES)
# ---------------------------------------------------------------------------

def run_scenario_subset(df, scenario_num, instance_ids, alpha=(0.05, 0.01)):
    scenario = SCENARIO_BY_NUM[scenario_num]
    methods = scenario["methods"]
    table_df = scenario_table(df, scenario)
    table_df = table_df[table_df.instance_id.isin(instance_ids)].reset_index(drop=True)
    from analysis import rank_matrix_for_scenario
    rank_matrix = rank_matrix_for_scenario(table_df, methods)
    n, k = rank_matrix.shape

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05) if not np.isnan(stat["T2"]) else np.nan
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01) if not np.isnan(stat["T2"]) else np.nan
    mean_ranks = rank_matrix.mean(axis=0)

    return dict(scenario=scenario, table=table_df, rank_matrix=rank_matrix, stat=stat,
                f_crit_05=f_crit_05, f_crit_01=f_crit_01,
                mean_ranks=dict(zip(methods, mean_ranks)), n_instances=n)


# ---------------------------------------------------------------------------
# Holm-Bonferroni across the 14 tests (7 full-suite objective-value tests
# already computed by analysis.run_scenario, + 7 constraint-density subset
# tests from run_scenario_subset above)
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals):
    """pvals: list of (label, p). Returns list of (label, p_raw, p_holm,
    reject_at_0.05) sorted by ascending raw p, with the standard
    step-down Holm procedure (monotone-adjusted)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    adj = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        label, p = pvals[idx]
        holm_p = (m - rank) * p
        running_max = max(running_max, holm_p)
        adj[idx] = min(running_max, 1.0)
    out = []
    for i, (label, p) in enumerate(pvals):
        out.append((label, p, adj[i], adj[i] < 0.05))
    return out


def main():
    df = load_results()

    print("\n" + "#" * 78)
    print("# 1. FEASIBILITY-RATE FRIEDMAN TESTS (N-feasible-of-30 as ranking variable)")
    print("#" * 78)
    feas_results = {}
    for s in SCENARIOS:
        r = run_scenario_feasibility(df, s["num"])
        feas_results[s["num"]] = r
        stat = r["stat"]
        print("=" * 78)
        print(f"Scenario {s['num']} feasibility-rate test -- {', '.join(s['methods'])}")
        print(f"n={stat['n']}, k={stat['k']}, T2={stat['T2']:.4f}, p={stat['p_value']:.4g}")
        dec05 = "REJECT H0" if stat["T2"] > r["f_crit_05"] else "fail to reject"
        dec01 = "REJECT H0" if stat["T2"] > r["f_crit_01"] else "fail to reject"
        print(f"  alpha=0.05: F_crit={r['f_crit_05']:.3f} -> {dec05}")
        print(f"  alpha=0.01: F_crit={r['f_crit_01']:.3f} -> {dec01}")
        print(f"  Mean feasibility ranks: " + ", ".join(f"{m}={v:.3f}" for m, v in r["mean_ranks"].items()))
        print(f"  Mean N-feasible/30: " + ", ".join(f"{m}={v:.2f}" for m, v in r["mean_nfeas"].items()))

    print("\n" + "#" * 78)
    print("# 2. CONSTRAINT-DENSITY SUBSET (SMD10/SMD11/SMD12, n=8 instances)")
    print("#" * 78)
    print(f"Subset instances ({len(CONSTRAINT_DENSE_INSTANCES)}): {CONSTRAINT_DENSE_INSTANCES}")
    subset_results = {}
    for s in SCENARIOS:
        r = run_scenario_subset(df, s["num"], CONSTRAINT_DENSE_INSTANCES)
        subset_results[s["num"]] = r
        stat = r["stat"]
        print("=" * 78)
        print(f"Scenario {s['num']} subset test -- {', '.join(s['methods'])}")
        print(f"n={stat['n']}, k={stat['k']}, T2={stat['T2']:.4f}, p={stat['p_value']:.4g}")
        dec05 = "REJECT H0" if stat["T2"] > r["f_crit_05"] else "fail to reject"
        print(f"  alpha=0.05: F_crit={r['f_crit_05']:.3f} -> {dec05}")
        print(f"  Mean ranks (objective): " + ", ".join(f"{m}={v:.3f}" for m, v in r["mean_ranks"].items()))

    print("\n" + "#" * 78)
    print("# 3. HOLM-BONFERRONI ACROSS 14 TESTS (7 full-suite + 7 subset)")
    print("#" * 78)
    full_results = {s["num"]: run_scenario(df, s["num"]) for s in SCENARIOS}
    pvals = []
    for s in SCENARIOS:
        pvals.append((f"S{s['num']}_full", full_results[s["num"]]["stat"]["p_value"]))
    for s in SCENARIOS:
        pvals.append((f"S{s['num']}_subset", subset_results[s["num"]]["stat"]["p_value"]))
    holm = holm_bonferroni(pvals)
    holm_sorted_report = sorted(holm, key=lambda x: x[1])
    for label, p, p_holm, reject in holm_sorted_report:
        print(f"  {label:12s}  p_raw={p:.4g}  p_holm={p_holm:.4g}  reject@0.05={reject}")

    return dict(feas_results=feas_results, subset_results=subset_results,
                full_results=full_results, holm=holm)


if __name__ == "__main__":
    main()
