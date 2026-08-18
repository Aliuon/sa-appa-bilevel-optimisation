"""
task5_analysis.py
==================
Runs the Task 5 Friedman-test analysis: Conover's F-approximation omnibus
test plus the paired-t post-hoc critical difference, computed on the mean
(across N_RUNS=5 runs) objective value per (instance, method) cell, over
the reconciled 44-instance benchmark suite -- run TWICE, separately for
Phase I (M1-M9, internal ablation) and Phase II (M1, M10-M13, external
comparison), per the user's Task 5 instruction that Friedman requires the
same set of treatments evaluated on every block.

Uses the SAME ranking rule as figures/fig5_cd_diagram.py (mean-across-runs
feasible objective, ascending; zero-feasible-run methods ranked worst,
tied) so the CD diagram (Figure 5, Nemenyi-based) and this text-report
(Conover-F-based) omnibus/post-hoc analysis are computed from the same
underlying ranks and are directly comparable, while remaining two
distinct statistical procedures (see friedman_test.py's module docstring
and Task 5's write-up for why they are not conflated).

This script currently runs on synthetic placeholder data
(synthetic_data.generate_synthetic_results) -- see SYNTHETIC_DATA_WARNING.
Swap in a real results loader per README.md to get real numbers; nothing
else here needs to change.
"""

import numpy as np

from config import METHOD_IDS, INSTANCES, PHASE_I_METHODS, PHASE_II_METHODS
from synthetic_data import generate_synthetic_results, SYNTHETIC_DATA_WARNING
from figures.fig5_cd_diagram import _mean_ranks
from friedman_test import friedman_f_statistic, f_critical, conover_posthoc_cd, pairwise_table


def run_phase(results_df, method_ids, phase_label):
    mean_rank, ranks_long, n_instances = _mean_ranks(results_df, method_ids)
    k = len(method_ids)

    wide = ranks_long.pivot(index="instance_id", columns="method_id", values="rank")
    wide = wide.reindex(columns=method_ids)
    rank_matrix = wide.values

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05)
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01)

    cd_05, t_crit_05 = conover_posthoc_cd(stat, 0.05)
    cd_01, t_crit_01 = conover_posthoc_cd(stat, 0.01)
    rows, cds = pairwise_table(stat, alpha_list=(0.05, 0.01))

    print("=" * 78)
    print(f"Phase: {phase_label}  (n={stat['n']} instances, k={stat['k']} methods)")
    print(f"Methods: {', '.join(method_ids)}")
    print("-" * 78)
    print(f"A2 = {stat['A2']:.3f}   B2 = {stat['B2']:.3f}   C1 = {stat['C1']:.3f}")
    print(f"SST = {stat['SST']:.3f}   SS_treatments = {stat['SS_treatments']:.3f}"
          f"   SS_error = {stat['SS_error']:.3f}")
    from scipy.stats import f as f_dist
    if stat["p_value"] == 0:
        p_str = f"< {np.exp(f_dist.logsf(stat['T2'], stat['df1'], stat['df2'])):.3g} (underflowed to 0 in linear scale)"
    else:
        p_str = f"{stat['p_value']:.6g}"
    print(f"T2 = {stat['T2']:.3f}   ~ F({stat['df1']}, {stat['df2']})"
          f"   p = {p_str}")
    print(f"F_crit(alpha=0.05) = {f_crit_05:.3f}  ->  "
          f"{'REJECT H0' if stat['T2'] > f_crit_05 else 'fail to reject H0'}")
    print(f"F_crit(alpha=0.01) = {f_crit_01:.3f}  ->  "
          f"{'REJECT H0' if stat['T2'] > f_crit_01 else 'fail to reject H0'}")
    print("-" * 78)
    print("Rank sums R_j:")
    for m, r in zip(method_ids, stat["rank_sums"]):
        print(f"  {m:>6s}: {r:8.2f}   (mean rank {r/stat['n']:.3f})")
    print("-" * 78)
    print(f"Post-hoc paired critical difference (Conover, t-distribution, "
          f"df={stat['df2']}):")
    print(f"  CD(alpha=0.05) = {cd_05:.3f}  (t_crit={t_crit_05:.3f})")
    print(f"  CD(alpha=0.01) = {cd_01:.3f}  (t_crit={t_crit_01:.3f})")
    print("-" * 78)
    print("Pairwise |R_i - R_j| (sorted, largest first; * = sig. at 0.05, "
          "** = sig. at 0.01):")
    for row in rows:
        mi, mj = method_ids[row["i"]], method_ids[row["j"]]
        flag = "**" if row["significant_0.01"] else ("*" if row["significant_0.05"] else "")
        print(f"  {mi:>6s} vs {mj:<6s}: {row['abs_diff']:8.2f}  {flag}")
    print("=" * 78)
    print()

    return stat, rows, cds


def main():
    print(SYNTHETIC_DATA_WARNING)
    print("Task 5 analysis below runs on SYNTHETIC placeholder data -- see "
          "README.md to substitute real results.\n")

    results_df = generate_synthetic_results()

    phase1 = run_phase(results_df, PHASE_I_METHODS, "Phase I (internal ablation, M1-M9)")
    phase2 = run_phase(results_df, PHASE_II_METHODS, "Phase II (external comparison, M1,M10-M13)")

    return phase1, phase2


if __name__ == "__main__":
    main()
