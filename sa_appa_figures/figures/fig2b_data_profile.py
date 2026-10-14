"""
fig2b_data_profile.py
======================
Supplementary figure requested after Scenario 1-7 sign-off (this repo's
Task 26 / the project's "item 1"): a Moré & Wild-style DATA PROFILE,
budget-indexed, distinct from the Dolan & More PERFORMANCE PROFILE already
built in fig2_performance_profile.py. Do not conflate the two:

  - Performance profile (fig2): for a FIXED (final) budget, fraction of
    instances with relative error <= tau, plotted as a function of tau.
  - Data profile (this file): for a FIXED relative-error tolerance tau,
    fraction of (instance, run) pairs solved so far, plotted as a
    function of the evaluation BUDGET spent -- following

      More, J.J. and Wild, S.M., "Benchmarking Derivative-Free
      Optimization Algorithms," SIAM Journal on Optimization 20(1),
      2009, pp. 172-191.

Definitions used here (all reproduced on the figure/caption)
--------------------------------------------------------------
For instance i and method m, let f_L(i) be the same empirical best value
used by fig2 (min final feasible objective across ALL methods and runs on
instance i in this study -- not a certified global optimum). Let f_0(i) be
the mean, across all methods and runs, of the best feasible objective
recorded at the end of the initial N0=25-point LHS design (i.e. before any
surrogate-guided infill) -- a solver-independent "before optimisation"
reference value for that instance, since methods do not share one common
starting point in this study (each method/instance/run draws its own LHS
design).

A run (method m, instance i, run r) is "solved" at evaluation budget kappa
if its running-best feasible objective at eval_count=kappa satisfies

    best_feasible_obj(kappa) <= f_L(i) + tau * (f_0(i) - f_L(i))

The data profile is

    d_m(kappa) = (1/|P|) * #{ (i,r) : solved by budget kappa }

plotted as a function of kappa normalised by (n_ul(i)+1) simplex
gradients, per More & Wild's convention -- n_ul (the upper-level/leader
search dimension) is used as "the" problem dimension here since that is
the space the LHS design and CEI-guided infill loop actually search;
the lower level is solved by a separate nested exact solve each time, not
evaluation-budgeted the same way as the outer loop.
The budget axis is normalised by n_ul. Normalising by n_ul+n_ll (the total
problem dimension) would move the curves of the 30-dimensional SMD instances
relative to the low-dimensional TP problems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import METHOD_IDS
from figures.common import save_figure, method_style

DEFAULT_TAU = 0.1

CAPTION = (
    "Figure 2b. Data profile (More and Wild, 2009) across all {n} reconciled "
    "instances x {nr} runs = {npairs} (instance,run) pairs, at relative-error "
    "tolerance tau={tau}. f_L(i) is the same empirical best-found value used "
    "by Figure 2 (no certified global optimum is available uniformly across "
    "the suite); f_0(i) is the mean best-feasible objective at the end of "
    "the initial N0=25-point LHS design, averaged over all methods/runs on "
    "instance i. A run is solved at budget kappa (measured in simplex "
    "gradients, kappa_evals/(n_ul(i)+1)) once its running-best feasible "
    "objective first satisfies f <= f_L(i) + tau*(f_0(i)-f_L(i)). Distinct "
    "from, and not to be conflated with, the Dolan-More performance profile "
    "of Figure 2, which fixes the final budget and varies tau instead."
)


def _f_L_and_f_0(df, n0=25):
    """f_L(i): empirical best final feasible objective across all
    methods/runs (matches fig2_performance_profile.py's convention).
    f_0(i): mean best-feasible-obj at eval_count==n0 across all
    methods/runs on instance i."""
    df = df.sort_values(["method_id", "instance_id", "run_id", "eval_count"])
    last = df.groupby(["method_id", "instance_id", "run_id"], as_index=False).last()
    f_L = (last[last["is_feasible"]]
           .groupby("instance_id")["best_feasible_obj"].min().rename("f_L"))

    at_n0 = df[df["eval_count"] == n0]
    f_0 = (at_n0[at_n0["is_feasible"]]
           .groupby("instance_id")["best_feasible_obj"].mean().rename("f_0"))
    return f_L, f_0


def _solved_matrix(df, f_L, f_0, tau, n_ul_lookup, kappa_grid):
    """For each method, instance, run: the normalised budget (simplex
    gradients) at which the run first satisfies the convergence test, or
    +inf if never. Returns a DataFrame method_id,instance_id,run_id,
    kappa_solved."""
    merged = df.merge(f_L, on="instance_id", how="left").merge(f_0, on="instance_id", how="left")
    merged["threshold"] = merged["f_L"] + tau * (merged["f_0"] - merged["f_L"])
    merged["ok"] = merged["is_feasible"] & (merged["best_feasible_obj"] <= merged["threshold"])
    solved_rows = merged[merged["ok"]]
    first_solve = (solved_rows.groupby(["method_id", "instance_id", "run_id"])["eval_count"]
                   .min().rename("eval_solved").reset_index())

    all_keys = df[["method_id", "instance_id", "run_id"]].drop_duplicates()
    out = all_keys.merge(first_solve, on=["method_id", "instance_id", "run_id"], how="left")
    out["eval_solved"] = out["eval_solved"].fillna(np.inf)
    out["n_ul"] = out["instance_id"].map(n_ul_lookup)
    out["kappa_solved"] = out["eval_solved"] / (out["n_ul"] + 1)
    return out


def make_figure2b(results_df, out_dir, suite, method_ids=None, is_synthetic=None,
                   tau=DEFAULT_TAU, kappa_max=25.0, n_kappa=500, n0=25):
    """
    Parameters
    ----------
    suite : dict[str, Instance]  (benchmark.SUITE) -- needed for n_ul per instance.
    """
    method_ids = method_ids or METHOD_IDS
    if is_synthetic is None:
        is_synthetic = bool(results_df.attrs.get("SYNTHETIC", False))

    f_L, f_0 = _f_L_and_f_0(results_df, n0=n0)
    n_ul_lookup = {iid: inst.n_ul for iid, inst in suite.items()}
    solved = _solved_matrix(results_df, f_L, f_0, tau, n_ul_lookup, None)

    kappa_grid = np.linspace(0, kappa_max, n_kappa)
    n_pairs = solved[solved.method_id == method_ids[0]].shape[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in method_ids:
        sub = solved[solved.method_id == m]["kappa_solved"].values
        d = np.array([(sub <= k).mean() for k in kappa_grid])
        ax.step(kappa_grid, d, where="post", label=m, **method_style(m, method_ids))

    ax.set_xlabel(r"$\kappa$ (evaluation budget, simplex gradients $=\mathrm{evals}/(n_{UL}+1)$)")
    ax.set_ylabel(rf"$d_m(\kappa)$ = fraction solved to $\tau={tau}$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(rf"Data profile (Moré and Wild, 2009), $\tau={tau}$")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    n_instances = results_df["instance_id"].nunique()
    n_runs = solved[solved.method_id == method_ids[0]]["run_id"].nunique()
    caption = CAPTION.format(n=n_instances, nr=n_runs, npairs=n_pairs, tau=tau)
    return save_figure(fig, out_dir, f"fig2b_data_profile_tau{tau}", caption, is_synthetic)
