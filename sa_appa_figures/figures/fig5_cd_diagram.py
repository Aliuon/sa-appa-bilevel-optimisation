"""
fig5_cd_diagram.py
=====================
Figure 5: critical-difference (CD) diagram summarising the Friedman /
Nemenyi ranking (Task 5).

Ranking rule per instance (Task 3, Section 5.3's "ranking feasible and
infeasible incumbents" convention, simplified -- see LIMITATION note
below): within each instance, methods are ranked 1 (best) .. k (worst) on
the mean-across-runs final feasible objective; a method with zero feasible
runs on that instance is ranked strictly worse than every method with at
least one feasible run, tied at the bottom with any other all-infeasible
method (average rank on ties). k = number of methods compared.

LIMITATION: Task 3's full ranking rule also breaks ties *among*
all-infeasible methods by total constraint violation. The canonical
results schema used by this package (see synthetic_data.py) does not
currently carry a constraint-violation column, so all-infeasible methods
on a given instance are tied here rather than broken by violation. Add a
`mean_constraint_violation` column and extend `_rank_instance()` below if
you need the full rule.

Statistic: the omnibus Friedman test statistic reported in this figure's
caption is Conover's (1998) F-approximation (via Villegas, based on
Conover 1998) -- NOT the older chi-square approximation; see
`friedman_test.py` at the package root for the full derivation, a
worked-example validation, and Task 5's Phase I / Phase II results.
See claude/task3-section5-rewrite.md for the citation-pool caveat on
this source. The Nemenyi critical difference used for the diagram's
connecting bars below is a DIFFERENT, standard post-hoc procedure
(Demsar, 2006) to the paired-t-based post-hoc critical difference that
`friedman_test.py` reports alongside the omnibus test -- the two are
not interchangeable and Task 5's write-up keeps them distinct. The
Nemenyi critical difference is

    CD = q_alpha * sqrt(k(k+1) / (6N))

where q_alpha is the studentized-range critical value at family size k
divided by sqrt(2) (computed exactly via scipy.stats.studentized_range,
not a hand-typed table, precisely to avoid transcription error in a
number that determines which differences get reported as significant),
and N is the number of instances (44, or fewer if a method_ids subset
excludes some non-comparable methods). Two methods are joined by a
solid bar in the diagram if the absolute difference in their average
rank is < CD (Demsar, 2006 -- NOT currently in the paper's bibliography;
flagging per the standing citation rule, same as the Friedman-test
source).
"""

import numpy as np
import pandas as pd
from scipy.stats import studentized_range, rankdata
import matplotlib.pyplot as plt

from config import METHOD_IDS
from figures.common import save_figure
from friedman_test import friedman_f_statistic

CAPTION_TEMPLATE = (
    "Figure 5. Critical-difference diagram over {n_instances} instances and "
    "{k} methods. Omnibus Friedman test (Conover's F-approximation, NOT "
    "the chi-square approximation): T2={f_stat:.3f}, F({df1},{df2}), "
    "p={p_value_str}. Nemenyi critical difference CD={cd:.3f} at "
    "alpha={alpha} (a distinct post-hoc procedure to the paired-t-based "
    "critical difference reported alongside the omnibus test in Task 5's "
    "write-up -- see friedman_test.py; the two are not interchangeable). "
    "Methods are positioned by mean rank (lower/left = better); a solid "
    "bar connects methods whose mean-rank difference is below the "
    "Nemenyi CD, i.e. not distinguishable at this significance level. "
    "Ranking rule: within each instance, mean-across-runs final feasible "
    "objective, ascending; methods with zero feasible runs on an "
    "instance are ranked worst (tied) on that instance -- see "
    "fig5_cd_diagram.py docstring for the known limitation vs. the full "
    "Task 3 ranking rule (constraint-violation tie-break not modelled "
    "here)."
)


def _rank_instance(sub):
    """sub: rows for one instance, one row per method (mean_obj, is_feasible)."""
    sub = sub.copy()
    BIG = sub["mean_obj"].max() + 1e9 if sub["mean_obj"].notna().any() else 1e9
    score = np.where(sub["is_feasible"], sub["mean_obj"], BIG)
    sub["rank"] = rankdata(score, method="average")
    return sub


def _mean_ranks(results_df, method_ids):
    df = results_df.sort_values(["method_id", "instance_id", "run_id", "eval_count"])
    last = df.groupby(["method_id", "instance_id", "run_id"], as_index=False).last()
    last = last[last["method_id"].isin(method_ids)]

    per_cell = (
        last.groupby(["method_id", "instance_id"])
        .apply(lambda g: pd.Series({
            "mean_obj": g.loc[g["is_feasible"], "best_feasible_obj"].mean(),
            "is_feasible": bool(g["is_feasible"].any()),
        }), include_groups=False)
        .reset_index()
    )

    ranks = per_cell.groupby("instance_id", group_keys=False).apply(
        _rank_instance, include_groups=False
    )
    # groupby.apply with include_groups=False drops instance_id; recover it
    ranks["instance_id"] = per_cell["instance_id"].values
    ranks["method_id"] = per_cell["method_id"].values

    mean_rank = ranks.groupby("method_id")["rank"].mean().reindex(method_ids)
    n_instances = ranks["instance_id"].nunique()
    return mean_rank, ranks, n_instances


def _friedman_stat(ranks_long, method_ids, n_instances):
    """
    Delegates to friedman_test.friedman_f_statistic() (Task 5), which is
    unit-tested against the Friedmantest.pdf worked example. This
    function previously (Task 4) computed the statistic inline and had a
    bug: it used SST (=A2-C1) as the denominator instead of SS_error
    (=A2-B2/n), and its numerator (`ss_ranks`) was n times too large
    (n * SS_treatments rather than SS_treatments) -- together producing a
    T statistic roughly 9x too large on the Friedmantest.pdf worked
    example (236.38 vs. the correct 25.74). Fixed in Task 5 by routing
    through the validated shared implementation instead of duplicating
    the formula here.
    """
    k = len(method_ids)
    # rank_matrix: one row per instance, one column per method_id, in a
    # fixed method order, matching friedman_f_statistic()'s expected shape.
    wide = ranks_long.pivot(index="instance_id", columns="method_id", values="rank")
    wide = wide.reindex(columns=method_ids)
    rank_matrix = wide.values
    stat = friedman_f_statistic(rank_matrix)
    return stat["T2"], stat["p_value"], stat["df1"], stat["df2"]


def make_figure5(results_df, out_dir, method_ids=None, is_synthetic=None,
                  alpha=0.05, phase_label=None, filename_stem="fig5_cd_diagram"):
    """
    `phase_label`: if given (e.g. "Phase I (M1-M9)"), included in the title
    and caption, and used to build a phase-specific filename_stem, so that
    calling this twice -- once per Task 5's Phase I method set and once
    per its Phase II method set -- writes two distinct figures rather than
    one overwriting the other. Friedman's test requires the same set of
    treatments evaluated on every block; mixing Phase I's internal
    ablation variants and Phase II's external baselines into one 13-way
    ranking would still be a valid Friedman comparison in principle (all
    13 methods share the same 44 instances/blocks here), but conflates
    the paper's own two separate research questions, so main.py produces
    the two phase-specific diagrams by default instead of (or as well as)
    one all-13-methods diagram.
    """
    method_ids = method_ids or METHOD_IDS
    if is_synthetic is None:
        is_synthetic = bool(results_df.attrs.get("SYNTHETIC", False))

    mean_rank, ranks_long, n_instances = _mean_ranks(results_df, method_ids)
    k = len(method_ids)
    f_stat, p_value, df1, df2 = _friedman_stat(ranks_long, method_ids, n_instances)
    if p_value == 0:
        from scipy.stats import f as f_dist
        p_value_str = f"<{np.exp(f_dist.logsf(f_stat, df1, df2)):.2g}"
    else:
        p_value_str = f"{p_value:.4g}"

    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_instances))

    order = mean_rank.sort_values().index.tolist()
    ordered_ranks = mean_rank.loc[order]

    fig, ax = plt.subplots(figsize=(9, 2.8 + 0.15 * k))
    y0 = 0.5
    ax.hlines(y0, ordered_ranks.min() - 0.5, ordered_ranks.max() + 0.5, color="black", linewidth=1)
    for i, m in enumerate(order):
        r = ordered_ranks[m]
        ax.plot([r, r], [y0, y0 + 0.15], color="black", linewidth=1)
        ax.text(r, y0 + 0.18 + 0.12 * (i % 2), f"{m}\n({r:.2f})",
                ha="center", va="bottom", fontsize=7)

    # group methods within CD of each other into cliques, draw connecting bars
    bar_y = y0 - 0.15
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ordered_ranks[order[j + 1]] - ordered_ranks[order[i]] < cd:
            j += 1
        if j > i:
            ax.hlines(bar_y, ordered_ranks[order[i]], ordered_ranks[order[j]],
                       color="tab:red", linewidth=3, alpha=0.7)
            bar_y -= 0.08
        i += 1 if j == i else (j - i)

    ax.set_ylim(bar_y - 0.1, y0 + 0.5 + 0.12 * k)
    ax.set_yticks([])
    ax.set_xlabel("Mean rank (lower = better)")
    title = f"Critical-difference diagram (CD={cd:.3f}, alpha={alpha})"
    if phase_label:
        title = f"{phase_label} -- {title}"
    ax.set_title(title)
    fig.tight_layout()

    caption = CAPTION_TEMPLATE.format(
        n_instances=n_instances, k=k, f_stat=f_stat, p_value_str=p_value_str, cd=cd,
        df1=df1, df2=df2, alpha=alpha,
    )
    if phase_label:
        caption = f"{phase_label}. " + caption
    return save_figure(fig, out_dir, filename_stem, caption, is_synthetic)
