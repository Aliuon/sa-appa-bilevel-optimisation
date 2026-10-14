"""Scenario 8 sensitivity check for inexact LL responses.

On SMD1-SMD9 the UL optimum is F* = 0 and any UL value below it can only arise when the
returned LL response is not the global LL optimum (e.g. a Rastrigin local minimum in SMD4).
This script counts such runs per method and repeats the Scenario 8 Friedman test with the
affected instances removed. It adds the key 's8_llcheck' to results/paper_numbers.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

TOL = 1e-2
d = pd.concat([pd.read_csv("results/summary.csv"), pd.read_csv("results/summary_m14.csv")])
f = d[d.feasible_found.astype(bool)]
smd = f[f.instance_id.str.match(r"SMD[1-9]_")]
below = smd.assign(b=smd.best < -TOL).groupby("method_id").b.agg(["sum", "count"])
pair = f[f.method_id.isin(["M1", "M14"])]
bad = (pair[pair.instance_id.str.match(r"SMD[1-9]_") & (pair.best < -TOL)]
       .groupby(["instance_id", "method_id"]).size().unstack(fill_value=0))
for c in ["M1", "M14"]:
    if c not in bad:
        bad[c] = 0
m = pair.groupby(["instance_id", "method_id"]).best.mean().unstack()
m = m.reindex(sorted(d.instance_id.unique())).drop(index=list(bad.index))
r = m.rank(axis=1, na_option="bottom")
n, k = r.shape
A2 = (r ** 2).values.sum(); B2 = (r.sum() ** 2).sum() / n; C1 = n * k * (k + 1) ** 2 / 4
T2 = (n - 1) * (B2 - C1) / (A2 - B2)
p = float(1 - stats.f.cdf(T2, k - 1, (n - 1) * (k - 1)))
out = {
    "tol": TOL,
    "below_opt_runs": {mid: int(row["sum"]) for mid, row in below.iterrows()},
    "feasible_smd1_9_runs": {mid: int(row["count"]) for mid, row in below.iterrows()},
    "affected": {i: {"M1": int(row["M1"]), "M14": int(row["M14"])} for i, row in bad.iterrows()},
    "n_kept": int(n), "M14_better": int((m.M14 < m.M1).sum()), "M1_better": int((m.M1 < m.M14).sum()),
    "mean_ranks": {c: float(v) for c, v in r.mean().items()}, "T2": float(T2), "p": p,
}
J = "results/paper_numbers.json"
js = json.load(open(J)); js["s8_llcheck"] = out
json.dump(js, open(J, "w"), indent=1)
print(json.dumps(out, indent=1))


def drop_affected(methods):
    g = f[f.method_id.isin(methods)]
    b = g[g.instance_id.str.match(r"SMD[1-9]_") & (g.best < -TOL)].instance_id.unique()
    mm = g.groupby(["instance_id", "method_id"]).best.mean().unstack().reindex(index=sorted(d.instance_id.unique()), columns=methods)
    mm = mm.drop(index=list(b))
    # instances on which no method has a feasible run stay in, all tied (the main ranking rule)
    rr = mm.rank(axis=1, na_option="bottom")
    nn, kk = rr.shape
    a2 = (rr ** 2).values.sum(); b2 = (rr.sum() ** 2).sum() / nn; c1 = nn * kk * (kk + 1) ** 2 / 4
    t2 = (nn - 1) * (b2 - c1) / (a2 - b2)
    return {"dropped": sorted(map(str, b)), "n": int(nn), "mean_ranks": {c: float(v) for c, v in rr.mean().items()},
            "T2": float(t2), "p": float(1 - stats.f.cdf(t2, kk - 1, (nn - 1) * (kk - 1)))}


js = json.load(open(J))
js["llcheck_scenarios"] = {"3": drop_affected(["M3", "M4", "M5", "M6"]),
                           "6": drop_affected(["M1", "M10", "M11"]),
                           "8": drop_affected(["M1", "M14"])}
json.dump(js, open(J, "w"), indent=1)
print(json.dumps(js["llcheck_scenarios"], indent=1))
