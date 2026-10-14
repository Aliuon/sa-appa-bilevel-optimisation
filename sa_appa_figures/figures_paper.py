"""
figures_paper.py
================
Figures for the manuscript, built from the real trajectories of
all fourteen configurations (results/trajectories.jsonl for M1-M13, locked;
results/trajectories_m14.jsonl for M14).

Every pooled figure is split into an internal-ablation view (M1-M9) and a
Phase II view (M1, M10-M14). Colour follows the configuration, not its
rank: M1 is slot 1 (blue) in every panel of every figure.

Definitions are unchanged from fig1_boxplot.py, fig2_performance_profile.py
and fig2b_data_profile.py, except that f_best / f_L / normalisation ranges
are now taken over all fourteen configurations. The convergence-rate figure
uses the cumulative number of nested solves as its time axis (rather than
the iteration counter), because M12-M14 count iterations differently from
M1-M11.

Outputs (PDF + EPS, opaque fills only) to ../paper_figures_main/.
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "sa_appa_experiments")
sys.path.insert(0, EXP)
from benchmark import SUITE  # noqa: E402

OUT = os.path.join(HERE, "..", "paper_figures_main")
ABL = [f"M{i}" for i in range(1, 10)]
EXT = ["M1", "M10", "M11", "M12", "M13", "M14"]
LABEL = {"M1": "M1 SA-APPA", "M2": "M2", "M3": "M3", "M4": "M4 GA", "M5": "M5 DE",
         "M6": "M6 PSO", "M7": "M7", "M8": "M8", "M9": "M9 fixed",
         "M10": "M10 SA-DE-Pen", "M11": "M11 SA-PSO-RBF", "M12": "M12 Nest-GA (A)",
         "M13": "M13 Nest-GA (B)", "M14": "M14 BL-SAEA"}

# Reference categorical palette (dataviz skill, light mode), fixed order.
P = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#52514e"
STYLE = {
    "M1": (P[0], "-", 2.2),
    "M2": (P[1], "-", 1.3), "M3": (P[2], "-", 1.3), "M4": (P[3], "-", 1.3),
    "M5": (P[4], "-", 1.3), "M6": (P[5], "-", 1.3), "M7": (P[6], "-", 1.3),
    "M8": (P[7], "-", 1.3), "M9": (GRAY, "--", 1.3),
    "M10": (P[1], "-", 1.3), "M11": (P[2], "-", 1.3), "M12": (P[3], "-", 1.3),
    "M13": (P[3], "--", 1.3), "M14": (P[7], "-", 1.6),
}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False})


def lighten(c, a=0.82):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a)


def load():
    recs = []
    for fn in ["trajectories.jsonl", "trajectories_m14.jsonl"]:
        with open(os.path.join(EXP, "results", fn)) as fh:
            for line in fh:
                o = json.loads(line)
                for r in o["rows"]:
                    recs.append((o["method_id"], o["instance_id"], o["run_id"], r["eval_count"],
                                 r["iteration"], r["best_feasible_obj"] if r["is_feasible"] else np.nan,
                                 r["is_feasible"]))
    df = pd.DataFrame(recs, columns=["method_id", "instance_id", "run_id", "eval_count",
                                     "iteration", "best_feasible_obj", "is_feasible"])
    assert df.method_id.nunique() == 14
    return df


def final_values(df):
    df = df.sort_values(["method_id", "instance_id", "run_id", "eval_count"])
    return df.groupby(["method_id", "instance_id", "run_id"], as_index=False).last()


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".eps"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


def group_of(iid):
    return "TP1–TP9" if iid.startswith("TP") else {"p3q3": "SMD (3,3)", "p10q10": "SMD (10,10)",
                                                     "p30q30": "SMD (30,30)"}[iid.split("_")[1]]


# ---------------------------------------------------------------- Figure 1
def fig1(last, methods, stem):
    last = last.copy()
    last["norm"] = np.nan
    for iid, sub in last.groupby("instance_id"):   # normalise over ALL 14 configurations
        v = sub.loc[sub.is_feasible, "best_feasible_obj"]
        if v.empty:
            continue
        lo, hi = v.min(), v.max()
        last.loc[v.index, "norm"] = (v - lo) / (hi - lo) if hi > lo else 0.0
    groups = ["TP1–TP9", "SMD (3,3)", "SMD (10,10)", "SMD (30,30)"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, g in zip(axes, groups):
        sub = last[last.instance_id.map(group_of) == g]
        data = [sub.loc[sub.method_id == m, "norm"].dropna().values for m in methods]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6,
                        medianprops=dict(color="#0b0b0b", linewidth=1.0),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="#0b0b0b", markersize=3),
                        flierprops=dict(marker="o", markersize=2, markerfacecolor="#8a8984",
                                        markeredgecolor="none"))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(lighten(STYLE[m][0], 0.35))
            patch.set_edgecolor(STYLE[m][0])
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=90)
        ax.set_title(g)
        ax.grid(axis="y")
    axes[0].set_ylabel("Normalised final feasible objective\n(0 = best found on instance)")
    fig.tight_layout()
    save(fig, stem)


# ---------------------------------------------------------------- Figure 2
def perf_profile(ax, last, methods, eps=1e-8, tau_max=2.0):
    fb = last[last.is_feasible].groupby("instance_id").best_feasible_obj.min()
    insts = list(fb.index)
    means = last[last.is_feasible].groupby(["method_id", "instance_id"]).best_feasible_obj.mean()
    taus = np.linspace(0, tau_max, 400)
    for m in methods:
        e = np.array([((means.get((m, i), np.nan) - fb[i]) / (abs(fb[i]) + eps))
                      if (m, i) in means.index else np.inf for i in insts])
        e = np.where(np.isnan(e), np.inf, e)
        rho = [(e <= t).mean() for t in taus]
        c, ls, lw = STYLE[m]
        ax.step(taus, rho, where="post", color=c, linestyle=ls, linewidth=lw, label=LABEL[m])
    ax.set_xlabel(r"$\tau$ (relative error to best found)")
    ax.set_ylabel(r"$\rho_m(\tau)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(fontsize=6.5, loc="upper left")
    return len(insts)


def data_profile(ax, df, methods, tau, n0=25, kmax=25.0):
    last = final_values(df)
    fL = last[last.is_feasible].groupby("instance_id").best_feasible_obj.min().rename("fL")
    at0 = df[(df.eval_count == n0) & df.is_feasible]
    f0 = at0.groupby("instance_id").best_feasible_obj.mean().rename("f0")
    sub = df[df.method_id.isin(methods)].merge(fL, on="instance_id").merge(f0, on="instance_id")
    sub["ok"] = sub.is_feasible & (sub.best_feasible_obj <= sub.fL + tau * (sub.f0 - sub.fL))
    first = sub[sub.ok].groupby(["method_id", "instance_id", "run_id"]).eval_count.min()
    keys = df[df.method_id.isin(methods)][["method_id", "instance_id", "run_id"]].drop_duplicates()
    keys = keys.merge(first.rename("ev").reset_index(), how="left").fillna({"ev": np.inf})
    keys["kappa"] = keys.ev / (keys.instance_id.map(lambda i: SUITE[i].n_ul) + 1)
    grid = np.linspace(0, kmax, 500)
    for m in methods:
        k = keys.loc[keys.method_id == m, "kappa"].values
        c, ls, lw = STYLE[m]
        ax.step(grid, [(k <= g).mean() for g in grid], where="post", color=c, linestyle=ls,
                linewidth=lw, label=LABEL[m])
    ax.set_xlabel(r"$\kappa$ (nested solves / $(n_{\mathrm{UL}}+1)$)")
    ax.set_ylabel(rf"$d_m(\kappa)$, $\tau={tau}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(fontsize=6.5, loc="upper left")


# ---------------------------------------------------------------- Figures 3/4
REP = ["TP2", "SMD4_p3q3", "SMD4_p10q10", "SMD4_p30q30"]


def fig3(df, methods, stem):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, iid in zip(axes.ravel(), REP):
        d = df[df.instance_id == iid]
        omitted = []
        for m in methods:
            s = d[d.method_id == m]
            piv = s.pivot_table(index="eval_count", columns="run_id", values="best_feasible_obj",
                                aggfunc="last").sort_index().ffill()
            if piv.dropna(how="all").empty:
                omitted.append(m)
                continue
            c, ls, lw = STYLE[m]
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), facecolor=lighten(c),
                            edgecolor="none", zorder=1)
            ax.plot(piv.index, piv.mean(axis=1), color=c, linestyle=ls, linewidth=lw,
                    label=LABEL[m], zorder=3)
        ax.set_title(iid.replace("_", " "))
        ax.set_xlabel("Cumulative nested LL solves")
        ax.set_ylabel("Best feasible objective so far")
        ax.grid(True)
        if omitted:
            ax.text(0.02, 0.02, "No feasible run: " + ", ".join(omitted), transform=ax.transAxes,
                    fontsize=6, color=GRAY)
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, stem)


def fit_order(t, err, floor=1e-10):
    t, err = np.asarray(t, float), np.asarray(err, float)
    mask = (t >= 1) & (err > floor) & np.isfinite(err)
    if mask.sum() < 3:
        return np.nan
    x, y = np.log(t[mask]), np.log(err[mask])
    if np.ptp(x) == 0:
        return np.nan
    return -np.polyfit(x, y, 1)[0]


def fig4(df, methods, stem, floor=1e-10):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, iid in zip(axes.ravel(), REP):
        d = df[df.instance_id == iid]
        for m in methods:
            s = d[d.method_id == m]
            curves, orders = [], []
            for r, g in s.groupby("run_id"):
                g = g.sort_values("eval_count")
                v = g.best_feasible_obj.ffill()
                if v.isna().all():
                    continue
                err = (v - v.iloc[-1]).abs()
                err.index = g.eval_count.values
                curves.append(err)
                o = fit_order(g.eval_count.values, err.values, floor)
                if np.isfinite(o):
                    orders.append(o)
            if not curves:
                continue
            mean_err = pd.concat(curves, axis=1).mean(axis=1)
            mk = mean_err > floor
            c, ls, lw = STYLE[m]
            lab = f"{LABEL[m]} (p={np.mean(orders):.2f}$\\pm${np.std(orders):.2f})" if orders else f"{LABEL[m]} (p n/a)"
            if mk.sum() >= 2:
                ax.loglog(mean_err.index[mk], mean_err.values[mk], color=c, linestyle=ls,
                          linewidth=lw, label=lab)
        ax.set_title(iid.replace("_", " "))
        ax.set_xlabel("Cumulative nested LL solves (log)")
        ax.set_ylabel(r"$|f_t-f_{best,run}|$ (log)")
        ax.grid(True, which="both")
        ax.legend(fontsize=5.5)
    fig.tight_layout()
    save(fig, stem)


def main():
    df = load()
    last = final_values(df)
    fig1(last, ABL, "fig1_ablation")
    fig1(last, EXT, "fig1_external")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    n = perf_profile(axes[0], last, ABL)
    perf_profile(axes[1], last, EXT)
    axes[0].set_title(f"Internal ablation (M1–M9), {n} instances")
    axes[1].set_title(f"Phase II (M1, M10–M14), {n} instances")
    fig.tight_layout()
    save(fig, "fig2_profiles_split")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, tau in enumerate([0.1, 0.25]):
        data_profile(axes[row, 0], df, ABL, tau)
        data_profile(axes[row, 1], df, EXT, tau)
    axes[0, 0].set_title("Internal ablation (M1–M9)")
    axes[0, 1].set_title("Phase II (M1, M10–M14)")
    fig.tight_layout()
    save(fig, "fig2b_data_profiles_split")
    fig3(df, EXT, "fig3_convergence_external")
    fig4(df, EXT, "fig4_rate_external")
    fig3(df, ABL, "fig3_convergence_ablation")   # code-package copies
    fig4(df, ABL, "fig4_rate_ablation")
    print("figures written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
