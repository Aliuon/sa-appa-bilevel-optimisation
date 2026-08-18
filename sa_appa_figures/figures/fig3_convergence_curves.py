"""
fig3_convergence_curves.py
============================
Figure 3: convergence curves, one figure PER PROBLEM INSTANCE (44 files),
each showing every method's mean best-feasible-objective-so-far against
expensive evaluation count, with a shaded band spanning the min/max across
that (method, instance) cell's N_RUNS runs.

`make_figure3_all()` loops config.INSTANCES and calls `make_figure3_one()`
once per instance, returning a list of the per-instance outputs -- this is
what "produces each figure as a separate output file" means for a
convergence-curve figure, where "every problem" (Task 4, item 3) is 44
separate instances.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from config import METHOD_IDS, INSTANCES, INSTANCE_BY_ID
from figures.common import save_figure, method_style


def _lighten(color, amount=0.78):
    """Blend `color` towards white by `amount` (0=no change, 1=white) at
    full opacity. Used instead of alpha-blended fills for the min/max band
    -- the EPS backend used for the paper's Section 4 figures has no
    alpha-transparency support, so shaded regions must be opaque, visually
    lighter colours rather than alpha overlays (project format rule)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

CAPTION_TEMPLATE = (
    "Figure 3 ({instance_id}). Convergence on {instance_id} "
    "(n_ul={n_ul}, n_ll={n_ll}{scale_txt}): mean best-feasible objective "
    "found so far vs. cumulative expensive evaluations, averaged across "
    "{n_runs} independent runs per method; shaded band spans the "
    "min-max range across those runs at each evaluation count. Methods "
    "with zero feasible runs throughout the budget are omitted from this "
    "panel and noted below the plot instead."
)


def _method_curve(df, method_id, instance_id):
    sub = df[(df["method_id"] == method_id) & (df["instance_id"] == instance_id)]
    if sub.empty:
        return None
    piv = sub.pivot_table(index="eval_count", columns="run_id",
                            values="best_feasible_obj", aggfunc="last")
    piv = piv.sort_index().ffill()
    if piv.dropna(how="all").empty:
        return None
    mean = piv.mean(axis=1, skipna=True)
    lo = piv.min(axis=1, skipna=True)
    hi = piv.max(axis=1, skipna=True)
    return piv.index.values, mean.values, lo.values, hi.values


def make_figure3_one(results_df, instance_id, out_dir, method_ids=None,
                      is_synthetic=None, n_runs=5, _instance_df=None):
    """
    `_instance_df`: internal performance optimisation -- if the caller
    already has results_df pre-filtered to this instance (see
    make_figure3_all), pass it here to skip re-filtering the full
    (potentially large) results table on every instance. Not part of the
    public API; call without it for normal single-instance use.
    """
    method_ids = method_ids or METHOD_IDS
    if is_synthetic is None:
        is_synthetic = bool(results_df.attrs.get("SYNTHETIC", False))
    inst = INSTANCE_BY_ID[instance_id]
    df = _instance_df if _instance_df is not None else results_df

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    omitted = []
    for m in method_ids:
        curve = _method_curve(df, m, instance_id)
        if curve is None:
            omitted.append(m)
            continue
        x, mean, lo, hi = curve
        style = method_style(m, method_ids)
        line, = ax.plot(x, mean, label=m, linewidth=1.4, **style)
        ax.fill_between(x, lo, hi, facecolor=_lighten(style["color"]),
                         edgecolor="none", alpha=1.0, zorder=1)

    ax.set_xlabel("Cumulative expensive evaluations")
    ax.set_ylabel("Best feasible objective found so far")
    ax.set_title(f"Convergence on {instance_id}")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    if omitted:
        ax.text(0.02, 0.02, f"No feasible runs: {', '.join(omitted)}",
                 transform=ax.transAxes, fontsize=6, color="gray")
    fig.tight_layout()

    scale_txt = f", scale {inst.scale}" if inst.scale else ""
    caption = CAPTION_TEMPLATE.format(
        instance_id=instance_id, n_ul=inst.n_ul, n_ll=inst.n_ll,
        scale_txt=scale_txt, n_runs=n_runs,
    )
    return save_figure(fig, out_dir, f"fig3_convergence_{instance_id}",
                        caption, is_synthetic)


def make_figure3_all(results_df, out_dir, method_ids=None, is_synthetic=None,
                      instances=None, n_runs=5):
    """Loop every reconciled instance (or a caller-supplied subset) and
    produce one convergence-curve figure per instance. Returns a list of
    the per-instance save_figure() result dicts."""
    instances = instances or [inst.id for inst in INSTANCES]
    groups = dict(list(results_df.groupby("instance_id")))
    outputs = []
    for inst_id in instances:
        outputs.append(make_figure3_one(
            results_df, inst_id, out_dir, method_ids=method_ids,
            is_synthetic=is_synthetic, n_runs=n_runs,
            _instance_df=groups.get(inst_id),
        ))
    return outputs
