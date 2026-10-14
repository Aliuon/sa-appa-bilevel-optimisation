"""Sensitivity check for M14's LL-mapping training set.

M14 (as reported) fits the LL mapping models on the whole archive. Jiang et al. (2024),
Algorithm 4, fit them per UL point on the K = (Du+1)(Dl+2)/2 archive points most similar in
cosine similarity. K >= 60 (the whole budget) on 24 of the 44 instances, where the two rules
coincide; M14K re-runs the other 20 instances with the published rule. This script repeats
Scenario 8 with M14K in place of M14 on those 20 instances and adds 's8_m14k' to
results/paper_numbers.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

d = pd.concat([pd.read_csv("results/summary.csv"), pd.read_csv("results/summary_m14.csv")])
k = pd.read_csv("results/summary_m14k.csv")
assert len(k) == 600 and k.run_id.nunique() == 30, "M14K sweep incomplete"
inst20 = sorted(k.instance_id.unique())
all_inst = sorted(d.instance_id.unique())


def means(df, mid):
    f = df[(df.method_id == mid) & df.feasible_found.astype(bool)]
    return f.groupby("instance_id").best.mean().reindex(all_inst)


m1 = means(d, "M1")
m14 = means(d, "M14")
m14k = means(k, "M14K")
mix = m14.copy()
mix.loc[inst20] = m14k.loc[inst20]


def friedman2(a, b):
    M = pd.DataFrame({"a": a, "b": b})
    r = M.rank(axis=1, na_option="bottom")
    n, kk = r.shape
    A2 = (r ** 2).values.sum(); B2 = (r.sum() ** 2).sum() / n; C1 = n * kk * (kk + 1) ** 2 / 4
    T2 = (n - 1) * (B2 - C1) / (A2 - B2)
    p = float(1 - stats.f.cdf(T2, kk - 1, (n - 1) * (kk - 1)))
    return {"n": int(n), "mean_ranks": {"M1": float(r.a.mean()), "M14K": float(r.b.mean())},
            "T2": float(T2), "p": p,
            "M14K_better": int((M.b < M.a).sum()), "M1_better": int((M.a < M.b).sum())}


full = friedman2(m1, mix)
sub = {"n": len(inst20),
       "M14K_better_than_M14": int((m14k.loc[inst20] < m14.loc[inst20]).sum()),
       "M14_better_than_M14K": int((m14.loc[inst20] < m14k.loc[inst20]).sum()),
       "wilcoxon_p": float(stats.wilcoxon(m14.loc[inst20].dropna(), m14k.loc[inst20].dropna()).pvalue),
       "M1_vs_M14K_on_20": friedman2(m1.loc[inst20], m14k.loc[inst20]),
       "M1_vs_M14_on_20": friedman2(m1.loc[inst20], m14.loc[inst20])}
out = {"instances": inst20, "s8_with_m14k": full, "on_20": sub,
       "feasible_runs_m14k": int(k.feasible_found.astype(bool).sum()),
       "errors": int(k.best.isna().sum() - (~k.feasible_found.astype(bool)).sum())}
J = "results/paper_numbers.json"
js = json.load(open(J)); js["s8_m14k"] = out
json.dump(js, open(J, "w"), indent=1)
print(json.dumps(out, indent=1))
