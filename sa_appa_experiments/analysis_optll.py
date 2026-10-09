"""
analysis_optll.py
=================
Scenario-level sensitivity of the eight Friedman tests to the SMD6 optimistic tie-break.

1. Recomputes the eight scenario tests from the locked results with the main ranking rule and
   checks that they match results/resub_numbers.json (they must, to 1e-9).
2. Replaces the three SMD6 instances by the optimistic re-run (results/optll/summary_optll.csv)
   and recomputes the tests and the Holm-Bonferroni correction over the sixteen omnibus tests
   (the eight feasibility tests are unchanged: SMD6 has no constraints).
3. Reports, per configuration, the mean final UL value on SMD6 under both LL solves.
Adds the key 'smd6_optll' to results/resub_numbers.json.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
J = os.path.join(RES, "resub_numbers.json")


def friedman(d, methods, instances):
    m = d[d.method_id.isin(methods)]
    f = m[m.feasible_found.astype(bool)]
    mm = f.groupby(["instance_id", "method_id"]).best.mean().unstack().reindex(index=instances, columns=methods)
    rr = mm.rank(axis=1, na_option="bottom")
    n, k = rr.shape
    a2 = (rr ** 2).values.sum(); b2 = (rr.sum() ** 2).sum() / n; c1 = n * k * (k + 1) ** 2 / 4
    t2 = (n - 1) * (b2 - c1) / (a2 - b2)
    p = float(1 - stats.f.cdf(t2, k - 1, (n - 1) * (k - 1)))
    return dict(n=int(n), T2=float(t2), p=p, mean_ranks={c: float(v) for c, v in rr.mean().items()}, means=mm)


def holm(ps):
    lab = list(ps); pv = np.array([ps[k] for k in lab]); o = np.argsort(pv); m = len(pv)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * pv[i])); adj[i] = run
    return {k: float(adj[j]) for j, k in enumerate(lab)}


def main():
    js = json.load(open(J))
    d = pd.concat([pd.read_csv(os.path.join(RES, "summary.csv")), pd.read_csv(os.path.join(RES, "summary_m14.csv"))])
    inst = sorted(d.instance_id.unique())
    o = pd.read_csv(os.path.join(RES, "optll", "summary_optll.csv"))
    smd6 = ["SMD6_p3q3", "SMD6_p10q10", "SMD6_p30q30"]
    exp = 14 * 30 * 3
    assert len(o) == exp, f"optimistic run incomplete: {len(o)}/{exp}"
    d2 = pd.concat([d[~d.instance_id.isin(smd6)], o[d.columns]])
    out = {"check": {}, "scenarios": {}}
    for sk, s in js["scenarios"].items():
        a = friedman(d, s["methods"], inst)
        out["check"][sk] = abs(a["T2"] - s["T2"])
        assert out["check"][sk] < 1e-9, (sk, a["T2"], s["T2"])
        b = friedman(d2, s["methods"], inst)
        best_a = min(a["mean_ranks"], key=a["mean_ranks"].get); best_b = min(b["mean_ranks"], key=b["mean_ranks"].get)
        out["scenarios"][sk] = dict(name=s["name"], methods=s["methods"], p_locked=s["p"], p_optll=b["p"],
                                    T2_optll=b["T2"], mean_ranks_locked=s["mean_ranks"],
                                    mean_ranks_optll=b["mean_ranks"], best_locked=best_a, best_optll=best_b,
                                    reject05_locked=bool(s["p"] < 0.05), reject05_optll=bool(b["p"] < 0.05))
    # Holm over sixteen: replace the eight scenario ("_full") p-values
    ps = {h["label"]: h["p_raw"] for h in js["holm"]}
    for sk, s in out["scenarios"].items():
        lab = f"S{sk}_full"
        assert lab in ps, lab
        ps[lab] = s["p_optll"]
    hb = holm(ps)
    out["holm_optll"] = {k: dict(p_raw=ps[k], p_holm=hb[k], reject05=bool(hb[k] < 0.05)) for k in ps}
    out["holm_locked"] = {h["label"]: dict(p_holm=h["p_holm"], reject05=h["reject05"]) for h in js["holm"]}
    out["holm_changes"] = [k for k in ps if out["holm_optll"][k]["reject05"] != out["holm_locked"][k]["reject05"]]
    # per-configuration means on SMD6
    def means(df):
        f = df[df.instance_id.isin(smd6) & df.feasible_found.astype(bool)]
        return f.groupby(["instance_id", "method_id"]).best.mean().unstack()
    ml, mo = means(d), means(o)
    out["smd6_means"] = {i: {m: dict(locked=float(ml.loc[i, m]), optll=float(mo.loc[i, m])) for m in ml.columns}
                         for i in smd6}
    out["smd6_rank_M1"] = {i: dict(locked=int(ml.loc[i].rank()["M1"]), optll=int(mo.loc[i].rank()["M1"])) for i in smd6}
    out["refine"] = dict(calls=int(o.refine_calls.sum()), accepted=int(o.refine_accepted.sum()),
                         F_evals_per_run_median=float(o.refine_F_evals.median()),
                         f_evals_per_run_median=float(o.refine_f_evals.median()))
    js["smd6_optll"] = out
    json.dump(js, open(J, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "holm_optll"}, indent=1, default=float))


if __name__ == "__main__":
    main()
