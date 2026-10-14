"""
analysis_ss_compare.py
======================
Compares the main results under strong Stackelberg selection (canonical result files after
merge_ss.py) with the no-selection protocol (results/no_tie_selection/), and summarises how often
the selection acted. Run after analysis_paper.py has regenerated results/paper_numbers.json.
Adds the key 'ss' to results/paper_numbers.json.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); ARC = os.path.join(RES, "no_tie_selection")
SS = os.path.join(RES, "strong_stackelberg")
from run_strong_stackelberg import AFFECTED


def means(d, insts):
    f = d[d.instance_id.isin(insts) & d.feasible_found.astype(bool)]
    return f.groupby(["instance_id", "method_id"]).best.mean().unstack()


def main():
    J = os.path.join(RES, "paper_numbers.json")
    new, old = json.load(open(J)), json.load(open(os.path.join(ARC, "paper_numbers.json")))
    out = {"scenarios": {}, "holm": {}}
    for k, s in new["scenarios"].items():
        o = old["scenarios"][k]
        bn = min(s["mean_ranks"], key=s["mean_ranks"].get); bo = min(o["mean_ranks"], key=o["mean_ranks"].get)
        out["scenarios"][k] = dict(name=s["name"], p_new=s["p"], p_old=o["p"], reject05_new=s["reject05"],
                                   reject05_old=o["reject05"], best_new=bn, best_old=bo,
                                   mean_ranks_new=s["mean_ranks"], mean_ranks_old=o["mean_ranks"],
                                   max_rank_shift=max(abs(s["mean_ranks"][m] - o["mean_ranks"][m]) for m in s["mean_ranks"]))
    out["feas"] = {k: dict(p_new=new["feas"][k]["p"], p_old=old["feas"][k]["p"],
                           reject_new=new["feas"][k]["reject05"], reject_old=old["feas"][k]["reject05"])
                   for k in new["feas"]}
    hn = {h["label"]: h for h in new["holm"]}; ho = {h["label"]: h for h in old["holm"]}
    for k in hn:
        out["holm"][k] = dict(p_holm_new=hn[k]["p_holm"], p_holm_old=ho[k]["p_holm"],
                              reject_new=hn[k]["reject05"], reject_old=ho[k]["reject05"])
    out["holm_changes"] = [k for k in hn if hn[k]["reject05"] != ho[k]["reject05"]]
    out["scenario_decision_changes"] = [k for k, v in out["scenarios"].items() if v["reject05_new"] != v["reject05_old"]]
    out["best_changes"] = [k for k, v in out["scenarios"].items() if v["best_new"] != v["best_old"]]
    # per-instance means on the affected instances, both protocols
    dn = pd.concat([pd.read_csv(os.path.join(RES, "summary.csv")), pd.read_csv(os.path.join(RES, "summary_m14.csv"))])
    do = pd.concat([pd.read_csv(os.path.join(ARC, "summary.csv")), pd.read_csv(os.path.join(ARC, "summary_m14.csv"))])
    allm = sorted(set(dn.method_id))
    mn = means(dn, AFFECTED).reindex(index=AFFECTED, columns=allm)
    mo = means(do, AFFECTED).reindex(index=AFFECTED, columns=allm)
    out["means"] = {i: {m: dict(new=float(mn.loc[i, m]) if m in mn.columns and not np.isnan(mn.loc[i, m]) else None,
                                old=float(mo.loc[i, m]) if m in mo.columns and not np.isnan(mo.loc[i, m]) else None)
                        for m in sorted(set(mn.columns) | set(mo.columns))} for i in AFFECTED}
    def rk(mm, i):
        r = mm.loc[i].rank(na_option="bottom")
        return int(r["M1"])
    out["rank_M1"] = {i: dict(new=rk(mn, i), old=rk(mo, i)) for i in AFFECTED}
    fn = dn[dn.instance_id.isin(AFFECTED)].groupby("method_id").feasible_found.mean()
    fo = do[do.instance_id.isin(AFFECTED)].groupby("method_id").feasible_found.mean()
    out["feas_rate_affected"] = {m: dict(new=float(fn[m]), old=float(fo[m])) for m in fn.index}
    # how often the selection acted
    s = pd.read_csv(os.path.join(SS, "summary_ss.csv"))
    full, scan_all = s[s["mode"] == "full"], s[s["mode"] == "scan"]
    scan = scan_all[~scan_all.instance_id.isin(AFFECTED)]
    late = scan_all[scan_all.instance_id.isin(AFFECTED)]
    out["scan_affected"] = dict(instances=sorted(late.instance_id.unique()), runs=len(late),
                                runs_with_accept=int((late.ss_accepted > 0).sum()))
    per = {}
    for i, g in full.groupby("instance_id"):
        per[i] = dict(calls=int(g.ss_calls.sum()), gated=int(g.ss_gated.sum()), accepted=int(g.ss_accepted.sum()),
                      accepted_feas=int(g.ss_accepted_feas.sum()), ball_rejected=int(g.ss_ball_rejected.sum()),
                      runs_with_accept=int((g.ss_accepted > 0).sum()), runs=len(g))
    out["selection"] = dict(per_instance=per, calls=int(full.ss_calls.sum()), accepted=int(full.ss_accepted.sum()),
                            gated=int(full.ss_gated.sum()), accepted_feas=int(full.ss_accepted_feas.sum()),
                            scan_runs=len(scan), scan_accepted=int(scan.ss_accepted.sum()),
                            scan_gated_frac=float(scan.ss_gated.sum() / max(1, scan.ss_calls.sum())),
                            extra_evals_median=dict(F=float(full.ss_F_evals.median()), f=float(full.ss_f_evals.median())))
    # scan runs reproduce the canonical results
    canon = {(r.method_id, r.instance_id, int(r.run_id)): (r.best, bool(r.feasible_found)) for r in dn.itertuples()}
    dev = []
    for r in scan.itertuples():
        b, ff = canon[(r.method_id, r.instance_id, int(r.run_id))]
        if ff and bool(r.feasible_found):
            dev.append(abs(r.best - b) / max(1.0, abs(b)))
        elif ff != bool(r.feasible_found):
            dev.append(np.inf)
    dev = np.array(dev)
    out["scan_repro"] = dict(n=int(len(dev)), within_1e6=int((dev <= 1e-6).sum()), max_rel_dev=float(np.max(dev)) if len(dev) else 0.0)
    new["ss"] = out
    json.dump(new, open(J, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("means",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
