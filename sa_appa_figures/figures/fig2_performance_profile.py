"""
fig2_performance_profile.py
============================
Figure 2: Dolan-More-style performance profile across all 44 reconciled
instances.

Definitions (reproduced on the figure and in the caption):

  For instance i, let f_best(i) = min over ALL methods and ALL runs of the
  final feasible objective value observed on instance i in this
  experiment. This is an EMPIRICAL best-found value, not a certified
  global optimum -- the figure says so explicitly on its face (see
  `NO_CERTIFIED_OPTIMUM_NOTE` below), because for most of this suite
  (all TP problems, and the non-quadratic SMD problems) no certified
  global optimum is available uniformly across the whole 44-instance set.

  For method m on instance i, the relative error is

      e_{m,i} = (mean_over_runs(final feasible objective) - f_best(i))
                / (|f_best(i)| + eps)

  (eps is a small constant avoiding division by zero when f_best(i) = 0,
  which happens on several SMD instances whose stated optimum is F*=0).
  A method that never found a feasible point on instance i is treated as
  e_{m,i} = +inf for that instance (it never counts toward rho_m(tau) at
  any finite tau).

  The performance profile is

      rho_m(tau) = (1/N) * #{ i : e_{m,i} <= tau }

  plotted as a step function of tau for each method m, where N=44 is the
  number of reconciled instances.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import METHOD_IDS
from figures.common import save_figure, method_style

NO_CERTIFIED_OPTIMUM_NOTE = (
    "'Best' = empirical best found across all methods/runs in this study; "
    "no certified global optimum is available uniformly across the suite."
)

CAPTION = (
    "Figure 2. Performance profile (Dolan & More style) across all {n} "
    "reconciled problem instances. For instance i, f_best(i) is the "
    "empirical best final feasible objective found by any method in any "
    "run in this study -- {note} For method m on instance i, the relative "
    "error is e_{{m,i}} = (mean-over-runs final feasible objective - "
    "f_best(i)) / (|f_best(i)| + eps); a method that never found a "
    "feasible point on instance i is assigned e_{{m,i}} = +infinity. The "
    "curve for method m plots rho_m(tau) = fraction of the {n} instances "
    "with e_{{m,i}} <= tau, as a step function of tau."
)


def _instance_best_and_method_means(results_df):
    df = results_df.sort_values(
        ["method_id", "instance_id", "run_id", "eval_count"]
    )
    last = df.groupby(["method_id", "instance_id", "run_id"], as_index=False).last()

    means = (
        last.groupby(["method_id", "instance_id"])
        .apply(lambda g: g.loc[g["is_feasible"], "best_feasible_obj"].mean(),
               include_groups=False)
        .rename("mean_final_obj")
        .reset_index()
    )
    per_instance_best = (
        last[last["is_feasible"]]
        .groupby("instance_id")["best_feasible_obj"]
        .min()
        .rename("f_best")
    )
    return means, per_instance_best


def make_figure2(results_df, out_dir, method_ids=None, is_synthetic=None,
                  tau_max=2.0, n_tau=400, eps=1e-8):
    """
    Parameters mirror make_figure1. `tau_max`/`n_tau` control the
    resolution of the tau grid the step functions are evaluated on.
    """
    method_ids = method_ids or METHOD_IDS
    if is_synthetic is None:
        is_synthetic = bool(results_df.attrs.get("SYNTHETIC", False))

    means, f_best = _instance_best_and_method_means(results_df)
    n_instances = f_best.shape[0]

    merged = means.merge(f_best, on="instance_id", how="left")
    merged["rel_error"] = np.where(
        merged["mean_final_obj"].notna(),
        (merged["mean_final_obj"] - merged["f_best"]) / (merged["f_best"].abs() + eps),
        np.inf,
    )

    tau_grid = np.linspace(0, tau_max, n_tau)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in method_ids:
        errs = merged.loc[merged["method_id"] == m, "rel_error"].values
        if errs.size < n_instances:
            missing = n_instances - errs.size
            errs = np.concatenate([errs, np.full(missing, np.inf)])
        rho = np.array([(errs <= t).mean() for t in tau_grid])
        ax.step(tau_grid, rho, where="post", label=m, **method_style(m, method_ids))

    ax.set_xlabel(r"$\tau$ (relative error threshold)")
    ax.set_ylabel(r"$\rho_m(\tau)$ = fraction of instances with $e_{m,i} \leq \tau$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Performance profile across all reconciled instances")
    ax.text(0.98, 0.03, NO_CERTIFIED_OPTIMUM_NOTE, transform=ax.transAxes,
             ha="right", va="bottom", fontsize=7, style="italic", wrap=True)
    ax.legend(fontsize=7, ncol=2, loc="lower right", bbox_to_anchor=(0.98, 0.12))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caption = CAPTION.format(n=n_instances, note=NO_CERTIFIED_OPTIMUM_NOTE)
    return save_figure(fig, out_dir, "fig2_performance_profile", caption, is_synthetic)
