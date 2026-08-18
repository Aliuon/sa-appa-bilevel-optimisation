"""
export_scenario_summaries.py
=============================
Task 20 (CD diagrams per scenario): dumps, for each of the 7 scenarios,
exactly the numbers already computed and validated by analysis.py's
run_scenario() (mean ranks, Nemenyi CD, Conover F-statistic/p-value) to a
single JSON file that sa_appa_figures/figures/fig5_cd_diagram_scenarios.py
reads to draw the CD diagrams.

Deliberately does NOT recompute ranks/statistics from the raw trajectory
data via a separate code path (the original fig5_cd_diagram.py did this,
against the synthetic-data convention) -- that would risk a second,
independently-computed set of numbers silently drifting from the ones
already reported in Appendix B's tables and Section 5.7/5.8's prose. This
script is a thin export of analysis.py's own run_scenario() output, so the
CD diagrams are guaranteed consistent with the numbers already in the
manuscript.
"""

import json
import numpy as np

from scenarios import SCENARIOS
from analysis import load_results, run_scenario

OUT_PATH = "results/scenario_cd_summaries.json"


def _to_native(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main():
    df = load_results()
    out = []
    for s in SCENARIOS:
        result = run_scenario(df, s["num"])
        stat = result["stat"]
        out.append(dict(
            num=s["num"],
            phase=s["phase"],
            name=s["name"],
            methods=s["methods"],
            isolates=s["isolates"],
            mean_ranks={m: _to_native(r) for m, r in result["mean_ranks"].items()},
            n_instances=_to_native(stat["n"]),
            k_methods=_to_native(stat["k"]),
            T2=_to_native(stat["T2"]),
            p_value=_to_native(stat["p_value"]),
            df1=_to_native(stat["df1"]),
            df2=_to_native(stat["df2"]),
            f_crit_05=_to_native(result["f_crit_05"]),
            f_crit_01=_to_native(result["f_crit_01"]),
            cd_nemenyi_05=_to_native(result["cd_nemenyi_05"]),
            cd_conover_05=_to_native(result["cd_conover_05"]),
        ))
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
