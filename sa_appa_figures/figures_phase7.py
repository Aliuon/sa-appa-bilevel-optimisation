"""
figures_phase7.py
=================
Phase 7 figures, modelled on the visual ideas of Sinha & Shaikh (IEEE TCYB 52(10), 2022,
Figs 4-9) but drawn from this study's own data and notation. No numbers from that paper are
used or compared.

Part A (surrogate fidelity) reads results/phase7/*.npz and fidelity.json written by
sa_appa_experiments/phase7_run.py. SA-APPA's GPs model theta(x) = F(x, y*(x)) and the UL
constraint maps phi_j(x); nothing here is drawn or described as an approximation of the
reaction set Omega(x), the follower response y*(x) or the LL value function f*(x).

Part B (evaluation counts) reads the locked trajectories (all 30 runs) and
results/ll_counts_subset.csv (instrumented re-run of runs 0-4, instrument_ll.py).

Every figure is its own file, written as .pdf and .eps with opaque fills only; a .png preview
and a .caption.txt are written beside it, and every figure is listed in manifest.json.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "sa_appa_experiments")
RES = os.path.join(EXP, "results")
P7 = os.path.join(RES, "phase7")
OUT = os.path.join(HERE, "..", "paper_figures_phase7")
sys.path.insert(0, EXP)
from benchmark import SUITE  # noqa: E402

# ----------------------------------------------------------------------------- style
# Colour and line style are the map used by every manuscript figure (figures_paper.STYLE);
# Phase 7 adds a marker and a bar hatch per configuration so that identity survives greyscale.
P = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, INK, MUTED = "#52514e", "#0b0b0b", "#8a8984"
METHOD_STYLE = {   # colour, line style, marker, hatch
    "M1": (P[0], "-", "o", ""),     "M2": (P[1], "-", "s", "////"), "M3": (P[2], "-", "^", "\\\\\\\\"),
    "M4": (P[3], "-", "D", "xxxx"), "M5": (P[4], "-", "v", "...."), "M6": (P[5], "-", "P", "----"),
    "M7": (P[6], "-", "X", "++++"), "M8": (P[7], "-", "<", "||||"), "M9": (GRAY, "--", ">", "oo"),
    "M10": (P[1], "-", "s", "\\\\||"), "M11": (P[2], "-", "^", "//--"), "M12": (P[3], "-", "D", "x.."),
    "M13": (P[3], "--", "d", "**"), "M14": (P[7], "-", "*", "OO"),
}
LABEL = {"M1": "M1 SA-APPA", "M2": "M2 uniform init.", "M3": "M3 no GA", "M4": "M4 GA engine",
         "M5": "M5 DE engine", "M6": "M6 PSO engine", "M7": "M7 penalised EI", "M8": "M8 plain EI",
         "M9": "M9 fixed schedule", "M10": "M10 SA-DE-Pen", "M11": "M11 SA-PSO-RBF",
         "M12": "M12 Nest-GA (A)", "M13": "M13 Nest-GA (B)", "M14": "M14 BL-SAEA"}
GROUPS = {"ablA": ["M1", "M2", "M3", "M4", "M5", "M6"],
          "ablB": ["M1", "M7", "M8", "M9"],
          "ext": ["M1", "M10", "M11", "M12", "M13", "M14"],
          "all": [f"M{i}" for i in range(1, 15)]}
GROUP_TXT = {"ablA": "internal ablation, design and engine (M1-M6)",
             "ablB": "internal ablation, constraint handling and step-size rule (M1, M7-M9)",
             "ext": "Phase II comparators (M1, M10-M14)",
             "all": "all fourteen configurations"}
SETS = {"tp": [f"TP{k}" for k in range(1, 10)],
        "smd_s": [f"SMD{k}_p3q3" for k in range(1, 13)],
        "smd_m": [f"SMD{k}_p10q10" for k in range(1, 13)],
        "smd_l": [f"SMD{k}_p30q30" for k in range(1, 13)]}
SET_TXT = {"tp": "TP1-TP9", "smd_s": "SMD1-SMD12 at (p, q) = (3, 3)",
           "smd_m": "SMD1-SMD12 at (p, q) = (10, 10)", "smd_l": "SMD1-SMD12 at (p, q) = (30, 30)"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False,
                     "hatch.linewidth": 0.6, "legend.framealpha": 1.0, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": GRAY, "ytick.color": GRAY, "ps.fonttype": 42, "pdf.fonttype": 42})
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#123e73"])
MANIFEST = []


def lighten(c, a=0.6):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a)


def save(fig, stem, caption, meta):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{stem}.png"), dpi=150, bbox_inches="tight")
    open(os.path.join(OUT, f"{stem}.caption.txt"), "w").write(caption + "\n")
    plt.close(fig)
    MANIFEST.append(dict(file=stem, caption=caption, **meta))


def ax3d(figsize=(4.6, 3.8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=26, azim=-62)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(1.0); a.pane.set_facecolor((1, 1, 1, 1)); a.pane.set_edgecolor(to_rgb("#e4e3df") + (1,))
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True)); ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    return fig, ax


def zlabel(ax, txt):
    ax.set_zlabel(txt, rotation=90, labelpad=6)


def J():
    return json.load(open(os.path.join(P7, "fidelity.json")))


# ============================================================================ PART A
def fig_A1():
    d = np.load(os.path.join(P7, "tp3_truth.npz"))
    xs, ts = d["xstar"], float(d["theta_star"])
    for key, zlab, stem, what in [
            ("fstar", "$f^*(\\mathbf{x})$", "A1a_tp3_llvalue_truth",
             "the LL optimal value function $f^*(\\mathbf{x})=f(\\mathbf{x},\\mathbf{y}^*(\\mathbf{x}))$"),
            ("theta", "$\\theta(\\mathbf{x})$", "A1b_tp3_theta_truth",
             "the UL objective on the follower's response, $\\theta(\\mathbf{x})=F(\\mathbf{x},\\mathbf{y}^*(\\mathbf{x}))$")]:
        fig, ax = ax3d()
        Z = d[key]
        ax.plot_wireframe(d["X1"], d["X2"], Z, rstride=2, cstride=2, color=GRAY, linewidth=0.45)
        zs = float(d["fstar"][0, 0] * 0 + (25.0 if key == "fstar" else ts))
        ax.scatter([xs[0]], [xs[1]], [zs], s=46, marker="o", facecolor="white", edgecolor=P[7],
                   linewidth=1.6, depthshade=False, zorder=10, label="bilevel optimum $\\mathbf{x}^*=(15,10)$")
        # UL feasible region (x1 + 2 x2 >= 30, x1 + x2 <= 25) drawn on the floor
        z0 = float(np.min(Z))
        g = np.linspace(0, 15, 200)
        ax.plot(g, (30 - g) / 2, z0, color=P[0], linewidth=1.0, label="UL constraint boundaries")
        ax.plot(g, 25 - g, z0, color=P[0], linewidth=1.0, linestyle="--")
        ax.set_xlim(0, 15); ax.set_ylim(0, 15)
        zlabel(ax, zlab)
        ax.legend(loc="upper left", fontsize=7)
        cap = (f"TP3 (Appendix A), ground truth: {what} on a {d['X1'].shape[0]} x {d['X1'].shape[1]} grid "
               "over the UL box [0, 15]^2. The LL problem min ||x - y||^2 over y in [0, 10]^2 is a strictly "
               "convex quadratic programme, so y*(x) = min(x, 10) is single-valued, as Assumption A2 requires; "
               "it was solved by bounded-variable least squares (SciPy lsq_linear, 'bvls') and checked "
               "against the closed-form projection. The kinks at x_i = 10 are where the LL bound becomes "
               "active. The open circle marks the bilevel optimum x* = (15, 10), y* = (10, 10), "
               "theta = 325, f* = 25; the blue lines on the floor are the UL constraint boundaries "
               "x1 + 2 x2 = 30 (solid) and x1 + x2 = 25 (dashed). Ground truth only; no method was charged.")
        save(fig, stem, cap, dict(part="A1", instance="TP3", run=None, seed=None))


def fig_A2():
    fj = J()["A2"]
    for stage, n_tr in (("init", 25), ("final", 60)):
        g = np.load(os.path.join(P7, f"tp3_gp_{stage}.npz"))
        t = np.load(os.path.join(P7, "tp3_truth.npz"))
        X1, X2, TH, MU, SD = g["X1"], g["X2"], t["theta"], g["mu"], g["sd"]
        Xtr, Ftr = g["Xtr"], g["Ftr"]
        inc = np.array(fj[stage]["incumbent_x"]); incF = fj[stage]["incumbent_F"]
        m = fj[stage]
        stg = ("after the 25-point initial design" if stage == "init"
               else "at the end of the run (60 nested solves)")
        meta = dict(part="A2", instance="TP3", run=fj["run"], seed=fj["run"], stage=stage)
        common = (f"TP3, SA-APPA (M1), run {fj['run']} (seed {fj['run']}), selected by a rule fixed in "
                  "advance: the lower median (15th of 30) of the locked final UL objectives. The GP is the "
                  f"one SA-APPA held {stg}, trained on {n_tr} truly evaluated points with its own settings "
                  f"(squared-exponential kernel, lengthscale {float(g['lengthscale']):.2f} from the median "
                  "heuristic, signal variance 1, outputs mean-centred and not rescaled); no hyperparameter "
                  "was tuned for this figure.")
        # (i) true theta vs GP mean
        fig, ax = ax3d()
        ax.plot_wireframe(X1, X2, TH, rstride=3, cstride=3, color=GRAY, linewidth=0.45, label="true $\\theta(\\mathbf{x})$")
        ax.plot_wireframe(X1, X2, MU, rstride=3, cstride=3, color=P[0], linewidth=0.55, label="GP posterior mean $\\mu_\\theta(\\mathbf{x})$")
        ax.scatter(Xtr[:, 0], Xtr[:, 1], Ftr, s=10, color=INK, depthshade=False, label="training points")
        ax.scatter([15], [10], [325], s=46, facecolor="white", edgecolor=P[7], linewidth=1.6,
                   depthshade=False, label="optimum $\\mathbf{x}^*$")
        ax.scatter([inc[0]], [inc[1]], [incF], s=50, marker="X", color=P[7], depthshade=False,
                   label="SA-APPA incumbent")
        zlabel(ax, "$\\theta$"); ax.legend(loc="upper left", fontsize=6.5)
        cap = (common + f" Grey: true theta(x); blue: GP posterior mean; black: training points; open circle: "
               f"optimum x* = (15, 10), theta = 325; cross: SA-APPA's best feasible evaluated point at this "
               f"stage, x = ({inc[0]:.2f}, {inc[1]:.2f}), theta = {incF:.2f}. On 2,500 held-out points over "
               f"the box the mean has RMSE {m['whole_box']['rmse']:.2f} and maximum absolute error "
               f"{m['whole_box']['max_abs']:.1f} (theta spans {m['whole_box']['theta_range']:.0f}); within "
               f"1.5 of x* the RMSE is {m['near_optimum']['rmse']:.2f}.")
        save(fig, f"A2{'a' if stage == 'init' else 'd'}_tp3_theta_gp_mean_{stage}", cap, meta)
        # (ii) mean +/- 2 sd wireframes
        fig, ax = ax3d()
        ax.plot_wireframe(X1, X2, TH, rstride=3, cstride=3, color=GRAY, linewidth=0.45, label="true $\\theta$")
        ax.plot_wireframe(X1, X2, MU + 2 * SD, rstride=3, cstride=3, color=P[1], linewidth=0.5, label="$\\mu_\\theta+2\\sigma_\\theta$")
        ax.plot_wireframe(X1, X2, MU - 2 * SD, rstride=3, cstride=3, color=P[6], linewidth=0.5, linestyle="--", label="$\\mu_\\theta-2\\sigma_\\theta$")
        zlabel(ax, "$\\theta$"); ax.legend(loc="upper left", fontsize=6.5)
        cap = (common + f" Uncertainty of the GP: the surfaces mu +/- 2 sigma (multiplier 2) and the true "
               f"theta. Because the signal variance is fixed at 1 while theta varies by about "
               f"{m['whole_box']['theta_range']:.0f}, sigma is at most 1 and averages "
               f"{m['whole_box']['mean_sd']:.4f}; the two surfaces are indistinguishable at this scale. The "
               f"empirical coverage of the +/- 2 sigma interval on 2,500 held-out points is "
               f"{100 * m['whole_box']['coverage_2sd']:.1f}% over the box and "
               f"{100 * m['near_optimum']['coverage_2sd']:.1f}% within 1.5 of x*, against a nominal 95%: "
               "the GP's predictive uncertainty is not calibrated on this problem.")
        save(fig, f"A2{'b' if stage == 'init' else 'e'}_tp3_theta_gp_band_{stage}", cap, meta)
        # (iii) error map
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        E = np.abs(MU - TH)
        from matplotlib.ticker import MaxNLocator
        lv = MaxNLocator(10).tick_values(0, np.nanmax(E))
        cs = ax.contourf(X1, X2, E, levels=lv, cmap=SEQ)
        cb = fig.colorbar(cs, ax=ax); cb.set_label("$|\\mu_\\theta(\\mathbf{x})-\\theta(\\mathbf{x})|$")
        g1 = np.linspace(0, 15, 100)
        ax.plot(g1, (30 - g1) / 2, color=INK, linewidth=0.8); ax.plot(g1, 25 - g1, color=INK, linewidth=0.8, linestyle="--")
        ax.axvline(10, color=MUTED, linewidth=0.6, linestyle=":"); ax.axhline(10, color=MUTED, linewidth=0.6, linestyle=":")
        ax.scatter(Xtr[:, 0], Xtr[:, 1], s=9, color=INK, zorder=3)
        ax.scatter([15], [10], s=40, facecolor="white", edgecolor=P[7], linewidth=1.5, zorder=4, clip_on=False)
        ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        imax = np.unravel_index(np.nanargmax(E), E.shape)
        cap = (common + f" Absolute error of the GP mean on a 61 x 61 grid, with training points (dots), the "
               f"optimum (open circle), the UL constraint boundaries (black) and the LL kink lines x_i = 10 "
               f"(dotted). The error is largest at ({X1[imax]:.1f}, {X2[imax]:.1f}), "
               f"{'a corner of the box far from the training points' if (X1[imax] in (0, 15) and X2[imax] in (0, 15)) else 'away from the training points'}; "
               "inside the feasible region it is smaller but does not vanish near the kink x_2 = 10, on which "
               "the optimum lies, because the smooth kernel cannot represent the change of slope there.")
        save(fig, f"A2{'c' if stage == 'init' else 'f'}_tp3_theta_gp_error_{stage}", cap, meta)


def fig_A3():
    fj = J()["A3"]
    names = {"TP3": ["$-(x_1+2x_2-30)$", "$x_1+x_2-25$"], "TP4": ["$x_1^2+2x_2-4$"],
             "TP5": ["$x_1-x_2$", "$x_1^2+x_2^2-100$"], "TP7": ["$x_1^2+x_2^2-1$"],
             "TP8": ["$x_1+x_2+y^*_1(\\mathbf{x})-2y^*_2(\\mathbf{x})-40$"]}
    for iid, info in fj.items():
        d = np.load(os.path.join(P7, f"{iid.lower()}_phi_final.npz"))
        for j in range(info["m_ul"]):
            fig, ax = ax3d()
            ax.plot_wireframe(d["X1"], d["X2"], d[f"true{j}"], rstride=3, cstride=3, color=GRAY, linewidth=0.45,
                              label=f"true $\\varphi_{j+1}(\\mathbf{{x}})$")
            ax.plot_wireframe(d["X1"], d["X2"], d[f"mu{j}"], rstride=3, cstride=3, color=P[0], linewidth=0.55,
                              label=f"GP mean $\\mu_{{\\varphi_{j+1}}}$")
            ax.scatter(d["Xtr"][:, 0], d["Xtr"][:, 1], d["Gtr"][:, j], s=9, color=INK, depthshade=False,
                       label="training points")
            inc = d["incumbent"]
            if np.all(np.isfinite(inc)):
                zi = float(np.interp(0, [0, 1], [0, 0]))
                ax.scatter([inc[0]], [inc[1]], [0.0], s=46, marker="X", color=P[7], depthshade=False,
                           label="SA-APPA incumbent (at $\\varphi=0$ level)")
            zlabel(ax, f"$\\varphi_{j+1}$"); ax.legend(loc="upper left", fontsize=6.5)
            mt = info["metrics"][f"phi{j}"]
            dep = ("depends on y*(x); its true values use the exact LL response (bound-constrained QP)"
                   if iid == "TP8" else "depends on x only")
            cap = (f"{iid}, UL constraint phi_{j+1}(x) = {names[iid][j].replace('$', '')} ({dep}): true map "
                   f"(grey) and the GP SA-APPA held at the end of run {info['run']} (seed {info['run']}; lower-median "
                   f"run of 30, rule fixed in advance), trained on 60 truly evaluated points. On the 61 x 61 grid "
                   f"the GP mean has RMSE {mt['rmse']:.3g}, maximum absolute error {mt['max_abs']:.3g}, "
                   f"+/- 2 sigma coverage {100 * mt['coverage_2sd']:.0f}% and predicts the sign of phi_{j+1} "
                   f"(feasible or not) correctly at {100 * mt['sign_agreement']:.1f}% of grid points."
                   + (" No run of the median rank found a feasible point on TP4, so no incumbent is marked."
                      if not np.all(np.isfinite(d['incumbent'])) else ""))
            save(fig, f"A3_{iid.lower()}_phi{j+1}", cap, dict(part="A3", instance=iid, run=info["run"], seed=info["run"]))


def fig_A4():
    t = np.load(os.path.join(P7, "tp3_truth.npz"))
    for k, key in ((1, "y1"), (2, "y2")):
        fig, ax = ax3d()
        ax.plot_wireframe(t["X1"], t["X2"], t[key], rstride=2, cstride=2, color=GRAY, linewidth=0.45)
        zlabel(ax, f"$y^*_{k}(\\mathbf{{x}})$"); ax.set_xlim(0, 15); ax.set_ylim(0, 15)
        cap = (f"TP3, true follower response y*_{k}(x) = min(x_{k}, 10) over the UL box, from the exact "
               "solution of the strictly convex LL quadratic programme (single-valued, as Assumption A2 "
               "requires). SA-APPA does not model y*(x); no approximation is drawn. A diagnostic GP of y* "
               "was not produced, because the nested solves of the selected run do not store y and adding "
               "it would require new evaluations.")
        save(fig, f"A4_tp3_ystar{k}", cap, dict(part="A4", instance="TP3", run=None, seed=None))


def fig_A5():
    fj = J()["A5"]
    for iid, m in fj.items():
        if "error" in m:
            continue
        d = np.load(os.path.join(P7, f"a5_{iid}.npz"))
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        th, mu, sd = d["th"], d["mu"], d["sd"]
        ax.errorbar(th, mu, yerr=2 * sd, fmt="o", ms=2.6, color=P[0], ecolor=lighten(P[0], 0.45),
                    elinewidth=0.6, capsize=0)
        lo, hi = float(min(th.min(), mu.min())), float(max(th.max(), mu.max()))
        ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.8, linestyle="--")
        ax.set_xlabel("true $\\theta(\\mathbf{x})$"); ax.set_ylabel("GP mean $\\mu_\\theta(\\mathbf{x})$ $\\pm 2\\sigma$")
        ax.set_title(iid.replace("_p3q3", " (3, 3)"), fontsize=8)
        cap = (f"{iid}: GP posterior mean of theta against the true theta on {m['n']} held-out LHS points, with "
               f"+/- 2 sigma bars, for the GP SA-APPA held at the end of run {m['run']} (lower-median run, rule "
               f"fixed in advance). True values from {m['truth']}"
               + (f"; {m['n_dropped_ll_infeasible']} points without an LL-feasible response were dropped" if m['n_dropped_ll_infeasible'] else "")
               + f". RMSE {m['rmse']:.3g}, coverage {100 * m['coverage_2sd']:.0f}% (nominal 95%). The mean "
               f"predictive standard deviation is {m['mean_sd']:.2g}, so most bars are narrower than the markers.")
        save(fig, f"A5_{iid.lower()}", cap, dict(part="A5", instance=iid, run=m["run"], seed=m["run"]))


def fig_A6():
    """SMD6 at (3, 3): effect of the non-unique LL response (phase7_smd6_check.py)."""
    cj = json.load(open(os.path.join(P7, "smd6_check.json")))
    d = np.load(os.path.join(P7, "smd6_check.npz"))
    tr = cj["training"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(d["Ft_opt"], d["Ft"], "o", ms=3, color=P[0])
    lo, hi = float(min(d["Ft_opt"].min(), d["Ft"].min())), float(max(d["Ft_opt"].max(), d["Ft"].max()))
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("optimistic $\\theta(\\mathbf{x})$"); ax.set_ylabel("value returned by the nested LL solve")
    ax.set_title("SMD6 (3, 3), training points", fontsize=8)
    cap = (f"SMD6 at (3, 3): the {tr['n']} UL values that SA-APPA trained its GP on (run {tr['run']}, the "
           "lower-median run) against the optimistic theta at the same points. The LL of SMD6 is optimal on a "
           "whole line of tail variables (t1 = t2), and the UL objective adds the squared tail, so the nested "
           "differential-evolution solve returns theta_opt + 2 t^2 with t effectively arbitrary. The gap is "
           f"non-negative everywhere, with median {tr['gap_median']:.1f} and maximum {tr['gap_max']:.1f}; the "
           "dashed line is equality.")
    save(fig, "A6a_smd6_training_targets", cap, dict(part="A6", instance="SMD6_p3q3", run=tr["run"], seed=tr["run"]))

    a, c = cj["a5_vs_optimistic"], cj["counterfactual_consistent_targets"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(d["th_opt"], d["mu"], "s", ms=2.6, color=lighten(GRAY, 0.35), label="SA-APPA's GP (targets as observed)")
    ax.plot(d["th_opt"], d["mu_cf"], "o", ms=2.6, color=P[0], label="same GP rule, optimistic targets")
    lo = float(min(d["th_opt"].min(), d["mu"].min(), d["mu_cf"].min()))
    hi = float(max(d["th_opt"].max(), d["mu"].max(), d["mu_cf"].max()))
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("optimistic $\\theta(\\mathbf{x})$"); ax.set_ylabel("GP mean $\\mu_\\theta(\\mathbf{x})$")
    ax.set_title("SMD6 (3, 3), held-out points", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True, framealpha=1, edgecolor="#e4e3df")
    cap = (f"SMD6 at (3, 3), {cj['reference_truth']['n']} held-out points: GP mean against the optimistic theta. Grey squares: "
           f"the GP SA-APPA held at the end of run {tr['run']} (RMSE {a['rmse']:.0f}, correlation {a['corr']:.2f}). Blue "
           "circles: a diagnostic GP with the same kernel, hyperparameter rule and 60 training inputs, but trained "
           f"on the optimistic theta at those inputs (RMSE {c['rmse']:.1f}, correlation {c['corr']:.2f}). The "
           "diagnostic GP is not an SA-APPA run; it shows that the poor fit comes from the inconsistent LL "
           "responses rather than from the GP model.")
    save(fig, "A6b_smd6_consistent_targets", cap, dict(part="A6", instance="SMD6_p3q3", run=tr["run"], seed=tr["run"]))


def fig_A6c():
    """SMD6 at (3, 3): SA-APPA's own GP when the nested LL solve applies the optimistic tie-break."""
    fj = json.load(open(os.path.join(P7, "smd6_optll_fidelity.json")))
    old = json.load(open(os.path.join(P7, "smd6_check.json")))["a5_vs_optimistic"]
    d = np.load(os.path.join(P7, "smd6_optll_fidelity.npz"))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(d["th"], d["mu"], "o", ms=2.6, color=P[0])
    lo, hi = float(min(d["th"].min(), d["mu"].min())), float(max(d["th"].max(), d["mu"].max()))
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("optimistic $\\theta(\\mathbf{x})$"); ax.set_ylabel("GP mean $\\mu_\\theta(\\mathbf{x})$")
    ax.set_title("SMD6 (3, 3), optimistic LL solve", fontsize=8)
    cap = (f"SMD6 at (3, 3), {fj['n']} held-out points: GP mean of SA-APPA (M1) against the optimistic theta when "
           "every nested LL solve is followed by the optimistic tie-break (sensitivity run). The GP is the one held "
           f"at the end of run {fj['run']}, the lower-median of the 30 optimistic runs. RMSE {fj['rmse']:.1f} "
           f"({100 * fj['rmse'] / fj['theta_range']:.0f}% of the range), correlation {fj['corr']:.2f}, against RMSE "
           f"{old['rmse']:.0f} and correlation {old['corr']:.2f} with the locked LL solve. The +/- 2 sigma coverage is "
           f"{100 * fj['coverage_2sd']:.1f}%: the tie-break makes the data consistent but does not change the GP's calibration.")
    save(fig, "A6c_smd6_optimistic_ll", cap, dict(part="A6", instance="SMD6_p3q3", run=fj["run"], seed=fj["run"]))


# ============================================================================ PART B
def load_traj():
    out = {}
    for fn in ("trajectories.jsonl", "trajectories_m14.jsonl"):
        with open(os.path.join(RES, fn)) as fh:
            for line in fh:
                j = json.loads(line)
                b = [r["best_feasible_obj"] for r in j["rows"]]
                out[(j["method_id"], j["instance_id"], int(j["run_id"]))] = np.array(
                    [np.inf if v is None or not np.isfinite(v) else v for v in b])
    return out


def reference_values(T):
    """F_ref(i): best final feasible value over all configurations and runs, except that on
    SMD1-SMD9, whose UL optimum is F* = 0, F_ref = 0 (values below zero come from inexact LL
    responses, Section 5.6). Fixed before any evaluations-to-target figure was drawn."""
    ref = {}
    for i in SUITE:
        v = min(tr[-1] for (m, ii, r), tr in T.items() if ii == i)
        if i.startswith("SMD") and int(i.split("_")[0][3:]) <= 9:
            v = 0.0
        ref[i] = v
    return ref


def evals_to_target(T, ref, tol):
    rows = []
    for (m, i, r), tr in T.items():
        hit = np.nonzero(np.abs(tr - ref[i]) <= tol)[0] if np.isfinite(ref[i]) else []
        hit = [k for k in hit if np.isfinite(tr[k])]
        rows.append(dict(method_id=m, instance_id=i, run_id=r, hit=int(hit[0]) + 1 if len(hit) else np.inf))
    d = pd.DataFrame(rows)

    def lower_median(x):
        x = np.sort(x.values)
        return x[14] if len(x) == 30 else np.nan       # 15th of 30: finite iff >= 15 runs reach
    g = d.groupby(["instance_id", "method_id"]).hit
    return pd.DataFrame({"median": g.apply(lower_median), "n_reached": g.apply(lambda x: int(np.isfinite(x).sum())),
                         "n_runs": g.size()}).reset_index()


def legend_handles(methods, stacked=False):
    from matplotlib.patches import Patch
    hs = []
    for m in methods:
        c, ls, mk, h = METHOD_STYLE[m]
        hs.append(Patch(facecolor=lighten(c, 0.15), edgecolor=INK if m == "M1" else c, hatch=h,
                        linewidth=0.5, label=LABEL[m]))
    if stacked:
        hs.append(Patch(facecolor=MUTED, edgecolor=MUTED, label="lower segment: UL evaluations"))
        hs.append(Patch(facecolor="white", edgecolor=GRAY, hatch="////", label="upper segment: LL evaluations"))
    return hs


def bar_chart(table, value, inst, methods, ylabel, logy, annotate_missing, stacked=None):
    """Grouped bars, one group per instance, one bar per configuration (at most seven per group
    except in the all-configurations appendix version)."""
    width = 0.84 / len(methods)
    fig, ax = plt.subplots(figsize=(max(5.6, 0.16 * len(inst) * len(methods) + 1.2), 3.0))
    x0 = np.arange(len(inst))
    reached_any = {i: False for i in inst}
    for k, m in enumerate(methods):
        c, ls, mk, h = METHOD_STYLE[m]
        sub = table[table.method_id == m].set_index("instance_id").reindex(inst)
        xs = x0 - 0.42 + width * (k + 0.5)
        vals = sub[value].values.astype(float)
        ok = np.isfinite(vals)
        for i, o in zip(inst, ok):
            reached_any[i] |= bool(o)
        if stacked:
            lo = sub[stacked[0]].values.astype(float)
            ax.bar(xs[ok], lo[ok], width * 0.9, color=c, edgecolor=c, linewidth=0.4)
            ax.bar(xs[ok], (vals - lo)[ok], width * 0.9, bottom=lo[ok], color=lighten(c, 0.6),
                   edgecolor=INK if m == "M1" else c, hatch=h, linewidth=0.4)
        else:
            ax.bar(xs[ok], vals[ok], width * 0.9, color=lighten(c, 0.15), edgecolor=INK if m == "M1" else c,
                   hatch=h, linewidth=0.5)
        if annotate_missing:
            for xx, i, o in zip(xs, inst, ok):
                if not o:
                    ax.text(xx, 0.8, "target not reached", rotation=90, ha="center", va="bottom",
                            fontsize=5.0, color=GRAY, clip_on=False)
    ax.set_xticks(x0)
    ax.set_xticklabels([i.split("_")[0] for i in inst], fontsize=7)
    ax.set_xlim(-0.55, len(inst) - 0.45)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(handles=legend_handles(methods, bool(stacked)), ncol=4, fontsize=6, loc="lower left",
              bbox_to_anchor=(0, 1.01), handlelength=2.2, handleheight=1.1)
    return fig, ax


def fig_B2(T=None):
    T = T or load_traj()
    ref = reference_values(T)
    tables = {}
    for tag, tol in (("t1", 1e-2), ("t2", 1.0)):
        e = evals_to_target(T, ref, tol)
        e.to_csv(os.path.join(OUT, f"B2_success_counts_{tag}.csv"), index=False)
        tables[tag] = e
    json.dump({k: float(v) for k, v in ref.items()}, open(os.path.join(OUT, "B2_reference_values.json"), "w"), indent=1)
    # ---- B2
    for tag, tol in (("t1", 1e-2), ("t2", 1.0)):
        for sk, inst in SETS.items():
            if tag == "t2" and sk == "tp":
                continue
            inst = [i for i in inst if i in SUITE]
            for gk, methods in GROUPS.items():
                tb = tables[tag]
                fig, ax = bar_chart(tb, "median", inst, methods, "median nested solves to target",
                                    logy=False, annotate_missing=True)
                ax.set_ylim(0, 62)
                place = "appendix" if (sk == "smd_m" or gk == "all" or tag == "t2") else "main or appendix"
                excl = (" SMD10 at (30, 30) is excluded (no feasible point exists within the box; Appendix C)."
                        if sk == "smd_l" else "")
                cap = (f"Evaluations to target, {SET_TXT[sk]}, {GROUP_TXT[gk]}. Bar: median over the 30 runs "
                       f"(not 31) of the number of full nested LL solves after which the best feasible UL objective "
                       f"first lies within {tol:g} (absolute) of the reference value F_ref; F_ref is the best value "
                       "found by any configuration in any run, except on SMD1-SMD9 where F_ref = 0, the known UL "
                       "optimum. Runs that never reach the target are censored; where fewer than 15 of the 30 runs "
                       "reach it the median is undefined and 'target not reached' is written instead of a bar. "
                       "The budget is 60 nested solves. Success counts are in the companion table. Counts are in "
                       "nested solves, the study's cost unit; LL function evaluations per solve are in the "
                       "budget-composition figure." + excl)
                save(fig, f"B2_{tag}_{sk}_{gk}", cap, dict(part="B2", instance_set=sk, group=gk, target=tol,
                                                         runs="0-29 (all, locked)", placement=place))
    return tables


def fig_B1():
    ll = pd.read_csv(os.path.join(RES, "ll_counts_subset.csv"))
    ll["ok"] = ll.best_match & ll.traj_match
    good = ll[ll.ok]
    b1 = good.groupby(["instance_id", "method_id"]).agg(n_ul=("n_ul", "median"), n_ll=("n_ll_f", "median"),
                                                         n_runs=("run_id", "size")).reset_index()
    b1["total"] = b1.n_ul + b1.n_ll
    b1.to_csv(os.path.join(OUT, "B1_budget_composition.csv"), index=False)
    for sk, inst in SETS.items():
        inst = [i for i in inst if i in SUITE]
        if not b1.instance_id.isin(inst).any():
            print(f"B1: no instrumented runs yet for set {sk}; skipped")
            continue
        for gk, methods in GROUPS.items():
            fig, ax = bar_chart(b1, "total", inst, methods, "median evaluations (log scale)", logy=True,
                                annotate_missing=False, stacked=("n_ul",))
            nexcl = int((~ll.ok & ll.instance_id.isin(inst) & ll.method_id.isin(methods)).sum())
            excl = (" SMD10 at (30, 30) is excluded (no feasible point exists within the box; Appendix C)."
                    if sk == "smd_l" else "")
            cap = (f"Budget composition, {SET_TXT[sk]}, {GROUP_TXT[gk]}. Each bar is the median over runs 0-4 "
                   "(5 of the 30 locked runs, re-run under the same seeds with a counter on the LL objective f) "
                   "of the UL evaluations (solid lower segment; 60 full nested solves for every configuration) "
                   "plus the LL objective evaluations made inside those solves (hatched upper segment). Every "
                   "LL solve is a nested heuristic solve: SciPy differential evolution with population "
                   "N_ll = 6 n_ll, at most T_ll = 20 generations and L-BFGS-B polishing (M14 changes only its "
                   "initial population); none is exact. The y axis is logarithmic, so segment heights are not "
                   "proportional to counts; the UL share is under 1% of the total in every case."
                   + (f" {nexcl} re-run(s) that did not reproduce the locked trajectory within 1e-6 (relative) "
                      "were left out of the median." if nexcl else "") + excl)
            place = "appendix" if (sk == "smd_m" or gk == "all") else "main or appendix"
            save(fig, f"B1_{sk}_{gk}", cap, dict(part="B1", instance_set=sk, group=gk, runs="0-4 (instrumented)",
                                                 placement=place))
    return b1


if __name__ == "__main__":
    which = sys.argv[1:] or ["A1", "A2", "A3", "A4", "A5", "A6", "A6c", "B2", "B1"]
    for w in which:
        globals()[f"fig_{w}"]()
    mp_ = os.path.join(OUT, "manifest.json")
    old = json.load(open(mp_)) if os.path.exists(mp_) else []
    keep = {m["file"] for m in MANIFEST}
    json.dump([m for m in old if m["file"] not in keep] + MANIFEST, open(mp_, "w"), indent=1, default=str)
    print(len(MANIFEST), "figures written")
