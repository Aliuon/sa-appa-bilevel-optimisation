"""
fig4_convergence_rate.py
==========================
Figure 4: empirical rate-of-convergence plots.

Task 4 asks for "log-log |f_t - f_best,run| vs iteration for every
algorithm on every problem, with the empirical convergence order fitted
by log-log linear regression and reported on the plot."

Design choice, stated explicitly: producing one file per (method, instance)
pair would be 13 x 44 = 572 separate images, which is unwieldy as a set of
paper figures and not how rate-of-convergence plots are normally presented
in this literature (they're normally one panel per problem, all algorithms
overlaid, which is also what makes the *comparison* of orders legible).
`make_figure4_one()` therefore produces ONE figure per instance with every
method's curve and fitted order shown together -- every algorithm IS shown
on every problem, just not as 572 separate files. If you need the fully
exploded per-(method,instance) version (e.g. for a supplementary archive),
`fit_convergence_order()` is exposed standalone so you can loop it
yourself and call matplotlib directly per pair; that loop is three lines
and deliberately not built in here to avoid generating hundreds of near-
duplicate files by default.

f_best,run is each run's OWN final best-feasible value (not the
cross-method instance best used in Figure 2) -- i.e. this measures how
fast a run approaches its own limit, not how close that limit is to the
best found elsewhere.

Fitting procedure: for each run, error_t = |best_feasible_obj_t -
final_value_of_that_run|. The last few iterations where error_t is at or
near machine-zero (by construction, since final_value defines error=0 at
the last point) are excluded from the fit -- specifically all t after the
last t with error_t > `zero_floor`. log(error_t) is regressed on log(t)
(t >= 1) by ordinary least squares; the reported "empirical order" is the
negative of the fitted slope (so that faster decay = larger, positive
reported order, matching the usual convention that error ~ t^{-order}).
Order is fitted per run, then reported as mean +/- std across the
N_RUNS runs for that (method, instance) cell.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import METHOD_IDS, INSTANCES, INSTANCE_BY_ID
from figures.common import save_figure

CAPTION_TEMPLATE = (
    "Figure 4 ({instance_id}). Log-log convergence rate on {instance_id}: "
    "|f_t - f_best,run| vs. iteration t, one curve per method (mean across "
    "{n_runs} runs; f_best,run is each run's own final best-feasible "
    "value, not the cross-method best of Figure 2). The empirical "
    "convergence order p (error ~ t^-p) is fitted per run by ordinary "
    "least squares on log(error_t) vs. log(t), excluding the near-zero "
    "tail, and reported per method as mean +/- std across runs in the "
    "legend."
)


def fit_convergence_order(iterations, errors, zero_floor=1e-10):
    """
    OLS fit of log(error) ~ log(iteration), on the informative (non-zero,
    non-tail) portion of the sequence.

    Returns (order, intercept, n_points_used) where order = -slope, or
    (np.nan, np.nan, 0) if fewer than 3 informative points are available.
    """
    iterations = np.asarray(iterations, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = (iterations >= 1) & (errors > zero_floor) & np.isfinite(errors)
    if mask.sum() < 3:
        return np.nan, np.nan, 0
    x = np.log(iterations[mask])
    y = np.log(errors[mask])
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    # A degenerate fit (fewer than 3 points left after dropping non-finite
    # values, or zero variance in log(iteration) -- e.g. every informative
    # point shares the same iteration count) leaves polyfit's design matrix
    # singular/rank-deficient; numpy then either divides by a zero scale
    # factor or fails to converge in the underlying SVD. Both are real
    # degenerate-data cases (not bugs to silently retry), so they are
    # treated the same as "insufficient data" rather than crashing the
    # whole 44-instance batch.
    if x.size < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, 0
    try:
        slope, intercept = np.polyfit(x, y, deg=1)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, 0
    return -slope, intercept, int(x.size)


def _per_run_error_curves(df, method_id, instance_id):
    sub = df[(df["method_id"] == method_id) & (df["instance_id"] == instance_id)]
    curves = {}
    for run_id, g in sub.groupby("run_id"):
        g = g.sort_values("iteration")
        vals = g["best_feasible_obj"].ffill()
        if vals.isna().all():
            continue
        final_val = vals.iloc[-1]
        err = (vals - final_val).abs().values
        curves[run_id] = (g["iteration"].values, err)
    return curves


def make_figure4_one(results_df, instance_id, out_dir, method_ids=None,
                      is_synthetic=None, n_runs=5, zero_floor=1e-10,
                      _instance_df=None):
    """`_instance_df`: see fig3_convergence_curves.make_figure3_one -- same
    pre-filtering optimisation, same "internal, optional" status."""
    method_ids = method_ids or METHOD_IDS
    if is_synthetic is None:
        is_synthetic = bool(results_df.attrs.get("SYNTHETIC", False))
    df = _instance_df if _instance_df is not None else results_df

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in method_ids:
        curves = _per_run_error_curves(df, m, instance_id)
        if not curves:
            continue
        orders = []
        # representative curve for plotting: mean error across runs at
        # each shared iteration index (runs share iteration grids by
        # construction of the synthetic/real generator)
        all_iters = sorted(set(np.concatenate([it for it, _ in curves.values()])))
        err_matrix = []
        for run_id, (it, err) in curves.items():
            order, _, npts = fit_convergence_order(it, err, zero_floor)
            if npts >= 3:
                orders.append(order)
            err_matrix.append(pd.Series(err, index=it))
        err_df = pd.concat(err_matrix, axis=1)
        mean_err = err_df.mean(axis=1, skipna=True)
        plot_mask = mean_err > zero_floor
        if plot_mask.sum() >= 2:
            ax.loglog(mean_err.index[plot_mask], mean_err.values[plot_mask],
                       label=_legend_label(m, orders))
        else:
            ax.plot([], [], label=f"{m} (insufficient data)")

    ax.set_xlabel("Iteration t (log scale)")
    ax.set_ylabel(r"$|f_t - f_{best,run}|$ (log scale)")
    ax.set_title(f"Empirical convergence rate on {instance_id}")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    caption = CAPTION_TEMPLATE.format(instance_id=instance_id, n_runs=n_runs)
    return save_figure(fig, out_dir, f"fig4_convergence_rate_{instance_id}",
                        caption, is_synthetic)


def _legend_label(method_id, orders):
    if not orders:
        return f"{method_id} (order n/a)"
    orders = np.array(orders)
    return f"{method_id} (p={orders.mean():.2f}±{orders.std():.2f})"


def make_figure4_all(results_df, out_dir, method_ids=None, is_synthetic=None,
                      instances=None, n_runs=5):
    instances = instances or [inst.id for inst in INSTANCES]
    groups = dict(list(results_df.groupby("instance_id")))
    outputs = []
    for inst_id in instances:
        outputs.append(make_figure4_one(
            results_df, inst_id, out_dir, method_ids=method_ids,
            is_synthetic=is_synthetic, n_runs=n_runs,
            _instance_df=groups.get(inst_id),
        ))
    return outputs
