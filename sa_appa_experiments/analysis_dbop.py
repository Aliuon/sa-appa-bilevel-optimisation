"""analysis_dbop.py: statistics and figure for the DBOP case study (Section 5.12).
Writes results/dbop/dbop_numbers.json and ../paper_figures_main/fig_dbop.{pdf,png}."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from friedman_test import rank_matrix_from_wide, friedman_f_statistic, conover_posthoc_cd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results", "dbop")
FIG = os.path.join(HERE, "..", "paper_figures_main")
M = [f"M{i}" for i in range(1, 15)]
LAB = {"M1": "SA-APPA", "M2": "SA-APPA-wLHS", "M3": "SA-PPA", "M4": "SA-GA", "M5": "SA-DE", "M6": "SA-PSO",
       "M7": "SA-APPA-Penalty", "M8": "SA-APPA-EI", "M9": "SA-APPA-Fixed", "M10": "SA-DE-Pen",
       "M11": "SA-PSO-RBF", "M12": "Nest-GA (A)", "M13": "Nest-GA (B)", "M14": "BL-SAEA"}

ref = json.load(open(os.path.join(R, "reference.json")))["best"]; Js = ref["J"]
d = pd.read_csv(os.path.join(R, "summary_dbop.csv"))
a = pd.read_csv(os.path.join(R, "aga_dbop.csv")); A = json.load(open(os.path.join(R, "aga_dbop.json")))
d["score"] = np.where(d.feasible_found, d.best, np.inf)
wide = d.pivot(index="run_id", columns="method_id", values="score")[M]
W = wide.values.copy(); W[np.isinf(W)] = 1e12          # infeasible runs rank last (ties averaged)
rk = rank_matrix_from_wide(W); st = friedman_f_statistic(rk)
cd05, _ = conover_posthoc_cd(st, 0.05)
rs = dict(zip(M, st["rank_sums"])); mr = {m: rs[m] / st["n"] for m in M}
rows = {}
for m in M:
    s = d[(d.method_id == m) & d.feasible_found].best
    rows[m] = dict(label=LAB[m], feasible=int(d[d.method_id == m].feasible_found.sum()), median=float(s.median()),
                   q1=float(s.quantile(.25)), q3=float(s.quantile(.75)), best=float(s.min()),
                   gap_median=float((s.median() - Js) / Js * 100), mean_rank=float(mr[m]),
                   conover_vs_M1=(None if m == "M1" else bool(abs(rs[m] - rs["M1"]) > cd05)))
order = sorted(M, key=lambda m: mr[m])
c = {}
for r in A:
    for n, v in r["curve"]:
        c.setdefault(n, []).append(v)
m1 = d[(d.method_id == "M1")].score.values
aga_final = a.best_feasible.values
u = mannwhitneyu(m1, aga_final, alternative="two-sided")
out = dict(J_star=Js, x_star=ref["x"], friedman=dict(T2=st["T2"], p=st["p_value"], n=st["n"], k=st["k"],
           cd05_rank_sum=cd05), configs=rows, order=order,
           aga=dict(feasible=int(a.feasible.sum()), median_final=float(np.median(aga_final)),
                    median_at={str(k): float(np.median(v)) for k, v in sorted(c.items())},
                    gap_median_final=float((np.median(aga_final) - Js) / Js * 100), evals=int(a.evals.iloc[0]),
                    mw_M1_vs_AGA_p=float(u.pvalue)))
json.dump(out, open(os.path.join(R, "dbop_numbers.json"), "w"), indent=1)

# figure: (a) median best feasible value vs nested solves; (b) final values
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = [json.loads(l) for l in open(os.path.join(R, "trajectories_dbop.jsonl"))]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
show = ["M1", "M3", "M9", "M6", "M14", "M12"]
sty = {"M1": ("C0", "-"), "M3": ("C1", "--"), "M9": ("C2", "-."), "M6": ("C3", ":"), "M14": ("C4", "-"), "M12": ("C7", "--")}
for m in show:
    curves = [t["curve"] for t in T if t["method_id"] == m]
    E = np.arange(1, 61); Y = np.full((len(curves), 60), np.nan)
    for i, cv in enumerate(curves):
        for e, v in cv:
            if 1 <= e <= 60 and v == v:
                Y[i, int(e) - 1:] = v
    with np.errstate(all="ignore"):
        med = np.nanmedian(Y, axis=0)
    ax[0].plot(E, med, color=sty[m][0], ls=sty[m][1], lw=1.6, label=f"{m} ({LAB[m]})")
ax[0].axhline(out["aga"]["median_at"]["60"], color="k", lw=0.8, ls=":", label="AGA, 60 evaluations")
ax[0].axhline(out["aga"]["median_final"], color="k", lw=1.0, ls="-.", label=f"AGA, {out['aga']['evals']} evaluations")
ax[0].axhline(Js, color="k", lw=1.2, label="best known $J^*$")
ax[0].set_xlabel("Nested solves"); ax[0].set_ylabel("Median best feasible $J$ (Mt CO$_2$)")
ax[0].set_xlim(25, 60); ax[0].legend(fontsize=7.5, loc="upper right")
data = [d[(d.method_id == m) & d.feasible_found].best.values for m in order] + [aga_final]
ax[1].boxplot(data, tick_labels=order + ["AGA"], showfliers=True)
ax[1].axhline(Js, color="k", lw=1.0); ax[1].set_ylabel("Best feasible $J$ (Mt CO$_2$)")
ax[1].tick_params(axis="x", rotation=60, labelsize=8)
ax[0].set_title("(a) Convergence (median over 30 runs)", fontsize=10); ax[1].set_title("(b) Final values, feasible runs", fontsize=10)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_dbop.{ext}"), dpi=200)
print(json.dumps({k: out[k] for k in ("J_star", "friedman", "order")}, indent=0)[:600])
print(pd.DataFrame(rows).T[["feasible", "median", "gap_median", "mean_rank", "conover_vs_M1"]].loc[order].to_string())
print(out["aga"])
