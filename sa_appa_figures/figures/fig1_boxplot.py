"""
fig1_boxplot.py
===============
Figure 1: solution-vs-method comparison.

Normalisation (stated here AND reproduced verbatim in the figure caption,
per Task 4's requirement):

    For a given problem instance i, let S_i be the set of every
    (method, run) final feasible objective value observed on instance i
    (i.e. pooled across all methods and all N_RUNS runs -- NOT just one
    method). For a value f in S_i:

        norm(f) = (f - min(S_i)) / (max(S_i) - min(S_i))

    so norm = 0 marks the best value observed on instance i by any method
    in any run, and norm = 1 marks the worst. If max(S_i) == min(S_i)
    (every method found the identical value on instance i -- possible on
    trivial instances), norm is defined as 0 for all values on that
    instance and the instance contributes a degenerate (zero-width) box.

    Only FEASIBLE final values are normalised and plotted; a method that
    never found a feasible point on instance i contributes no observation
    to that instance's boxplot (see is_feasible in the results schema).

One panel per benchmark group (config.FIGURE1_GROUPS: TP1-TP9 pooled, then
one panel per SMD scale), each panel a per-method boxplot of the pooled,
normalised values across every instance in that group.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import METHOD_IDS, INSTANCE_BY_ID, FIGURE1_GROUPS
from figures.common import save_figure

CAPTION = (
    "Figure 1. Per-method distribution of min-max normalised final feasible "
    "objective values, one panel per benchmark group (TP1-TP9 pooled; SMD "
    "problems pooled separately within each of the three (p,q) scales). "
    "For instance i, each observed final feasible objective f is normalised "
    "as norm(f) = (f - min(S_i)) / (max(S_i) - min(S_i)), where S_i is the "
    "set of final feasible objective values from every method and every "
    "run on instance i; norm = 0 therefore marks the best value found by "
    "any method on that instance, and norm = 1 the worst. Boxes show the "
    "pooled distribution (across all instances in the panel's group and "
    "all N_RUNS={n_runs} runs) for each method; a method with no feasible "
    "final value on a given instance contributes no observation for that "
    "instance."
)


def _final_feasible_values(df):
    """One row per (method_id, instance_id, run_id): the last feasible
    best_feasible_obj recorded for that run (NaN if never feasible)."""
    df = df.sort_values(["method_id", "instance_id", "run_id", "eval_count"])
    last = df.groupby(["method_id", "instance_id", "run_id"], as_index=False).last()
    return last[["method_id", "instance_id", "run_id", "best_feasible_obj", "is_feasible"]]


def _normalise_per_instance(final_df):
    out = final_df.copy()
    out["norm"] = np.nan
    for inst_id, sub in out.groupby("instance_id"):
        vals = sub.loc[sub["is_feasible"], "best_feasible_obj"]
        if vals.empty:
            continue
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            out.loc[vals.index, "norm"] = (vals - lo) / (hi - lo)
        else:
            out.loc[vals.index, "norm"] = 0.0
    return out


def make_figure1(results_df, out_dir, method_ids=None, is_synthetic=None,
                  n_runs=5):
    """
    Parameters
    ----------
    results_df : DataFrame with the canonical schema (see synthetic_data.py)
    out_dir : str, directory to write the figure + caption into
    method_ids : optional list restricting which methods are plotted
                 (defaults to config.METHOD_IDS)
    is_synthetic : bool; if None, inferred from results_df.attrs["SYNTHETIC"]

    Returns dict of written file paths (see figures.common.save_figure).
    """
    method_ids = method_ids or METHOD_IDS
    if is_synthetic is None:
        is_synthetic = bool(results_df.attrs.get("SYNTHETIC", False))

    final_df = _final_feasible_values(results_df)
    final_df = final_df[final_df["method_id"].isin(method_ids)]
    norm_df = _normalise_per_instance(final_df)

    fig, axes = plt.subplots(1, len(FIGURE1_GROUPS),
                              figsize=(4.2 * len(FIGURE1_GROUPS), 5),
                              sharey=True)
    if len(FIGURE1_GROUPS) == 1:
        axes = [axes]

    for ax, group in zip(axes, FIGURE1_GROUPS):
        group_instances = {iid for iid, inst in INSTANCE_BY_ID.items()
                            if inst.group == group}
        sub = norm_df[norm_df["instance_id"].isin(group_instances)]
        data, labels = [], []
        for m in method_ids:
            vals = sub.loc[sub["method_id"] == m, "norm"].dropna().values
            data.append(vals)
            labels.append(m)
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(group, fontsize=10)
        ax.set_xlabel("Method")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Min-max normalised final feasible objective\n(0 = best found, 1 = worst found, per instance)")

    fig.suptitle("Solution quality by method, pooled by benchmark group", y=1.02)
    fig.tight_layout()

    caption = CAPTION.format(n_runs=n_runs)
    return save_figure(fig, out_dir, "fig1_solution_by_method", caption, is_synthetic)
