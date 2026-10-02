"""
analysis_resub.py
=================
Single entry point that regenerates every number the resubmitted manuscript
reports from the scenario tests, so that each figure in the text can be
traced to code:

  * the eight scenario Friedman tests (Scenarios 1-7 unchanged; Scenario 8 =
    M1 vs M14, new), with Conover and Nemenyi post-hoc results and the full
    pairwise decision table for both procedures;
  * the eight feasibility-rate tests and the eight constraint-density subset
    tests;
  * the Holm-Bonferroni correction across the combined 16 tests;
  * per-scale mean ranks for Scenarios 3, 5, 6 and 8;
  * results/scenario_tables/scenario_8.csv (Appendix B raw table).

Writes results/resub_numbers.json. Never writes summary.csv.
"""

import json
import itertools
import numpy as np
import pandas as pd

from scenarios import SCENARIOS
from benchmark import SUITE
from analysis import load_results, run_scenario, scenario_table
from analysis_extra import (run_scenario_feasibility, run_scenario_subset,
                            holm_bonferroni, CONSTRAINT_DENSE_INSTANCES)


def scale_of(iid):
    if iid.startswith("TP"):
        return "TP"
    return iid.split("_")[1]


def main():
    df = load_results(include_m14=True)
    assert set(df.method_id.unique()) >= {f"M{i}" for i in range(1, 15)}, "M14 results missing"
    n_m14 = (df.method_id == "M14").sum()
    import os
    if os.environ.get("RESUB_ALLOW_PARTIAL") != "1":
        assert n_m14 == 44 * 30, f"M14 has {n_m14} rows, expected 1320"

    out = {"scenarios": {}, "feas": {}, "subset": {}, "holm": [], "per_scale": {},
           "methods": {}}

    for s in SCENARIOS:
        num = s["num"]
        r = run_scenario(df, num)
        st = r["stat"]
        methods = s["methods"]
        n = st["n"]
        rank_sums = {m: float(v) for m, v in zip(methods, r["rank_matrix"].sum(axis=0))}
        pairs = []
        for a, b in itertools.combinations(methods, 2):
            d_sum = abs(rank_sums[a] - rank_sums[b])
            d_mean = d_sum / n
            pairs.append(dict(a=a, b=b, rank_sum_diff=d_sum, mean_rank_diff=d_mean,
                              conover_sig05=bool(st["p_value"] < 0.05 and d_sum > r["cd_conover_05"]),
                              nemenyi_sig05=bool(d_mean > r["cd_nemenyi_05"])))
        out["scenarios"][num] = dict(
            name=s["name"], methods=methods, n=n, k=st["k"], A2=st["A2"], B2=st["B2"],
            T2=st["T2"], df1=st["df1"], df2=st["df2"], p=st["p_value"],
            f05=r["f_crit_05"], f01=r["f_crit_01"],
            reject05=bool(st["T2"] > r["f_crit_05"]), reject01=bool(st["T2"] > r["f_crit_01"]),
            mean_ranks={m: float(v) for m, v in r["mean_ranks"].items()},
            rank_sums=rank_sums, cd_conover05=r["cd_conover_05"], cd_conover01=r["cd_conover_01"],
            cd_nemenyi05=r["cd_nemenyi_05"], pairs=pairs)
        if num == 8:
            r["table"].to_csv("results/scenario_tables/scenario_8.csv", index=False)
            tab = r["table"]
            wins = ties = losses = 0
            for _, row in tab.iterrows():
                a, b = row["M1_mean"], row["M14_mean"]
                fa, fb = row["M1_nfeas"] > 0, row["M14_nfeas"] > 0
                if not fa and not fb:
                    ties += 1
                elif fa and (not fb or a < b):
                    wins += 1
                elif fb and (not fa or b < a):
                    losses += 1
                else:
                    ties += 1
            out["scenarios"][num]["M1_better_instances"] = wins
            out["scenarios"][num]["M14_better_instances"] = losses
            out["scenarios"][num]["ties_or_no_comparison"] = ties

        # per-scale mean ranks
        tab = r["table"]
        rm = r["rank_matrix"]
        scales = {}
        for sc in ["TP", "p3q3", "p10q10", "p30q30"]:
            idx = [i for i, iid in enumerate(tab.instance_id) if scale_of(iid) == sc]
            scales[sc] = dict(n=len(idx), mean_ranks={m: float(v) for m, v in zip(methods, rm[idx].mean(axis=0))})
        out["per_scale"][num] = scales

        fr = run_scenario_feasibility(df, num)
        out["feas"][num] = dict(T2=fr["stat"]["T2"], p=fr["stat"]["p_value"],
                                reject05=bool(fr["stat"]["T2"] > fr["f_crit_05"]),
                                mean_nfeas={m: float(v) for m, v in fr["mean_nfeas"].items()})
        sr = run_scenario_subset(df, num, CONSTRAINT_DENSE_INSTANCES)
        out["subset"][num] = dict(n=sr["n_instances"], T2=sr["stat"]["T2"], p=sr["stat"]["p_value"],
                                  mean_ranks={m: float(v) for m, v in sr["mean_ranks"].items()})

    pvals = [(f"S{n}_full", out["scenarios"][n]["p"]) for n in out["scenarios"]] + \
            [(f"S{n}_subset", out["subset"][n]["p"]) for n in out["subset"]]
    for label, p, ph, rej in sorted(holm_bonferroni(pvals), key=lambda x: x[1]):
        out["holm"].append(dict(label=label, p_raw=p, p_holm=ph, reject05=bool(rej)))

    for m in sorted(df.method_id.unique(), key=lambda x: int(x[1:])):
        sub = df[df.method_id == m]
        out["methods"][m] = dict(runs=int(len(sub)), feasible_runs=int(sub.feasible_found.sum()),
                                 mean_cpu=float(sub.cpu_time_s.mean()))
    out["total_runs"] = int(len(df))
    out["total_feasible"] = int(df.feasible_found.sum())

    with open("results/resub_numbers.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # console summary
    for n, s in out["scenarios"].items():
        print(f"S{n} {s['name']}: T2={s['T2']:.4f} p={s['p']:.4g} ranks={ {k: round(v,3) for k,v in s['mean_ranks'].items()} }")
    for h in out["holm"]:
        print(h)


if __name__ == "__main__":
    main()
