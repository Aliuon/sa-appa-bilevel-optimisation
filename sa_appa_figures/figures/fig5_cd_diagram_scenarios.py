"""
fig5_cd_diagram_scenarios.py
==============================
Task 20: seven critical-difference (CD) diagrams, one per experimental
scenario, matching the fig:cd1--fig:cd7 labels already referenced from
Appendix B's per-scenario Friedman summary paragraphs in the manuscript.

Unlike the original fig5_cd_diagram.py (built for the synthetic-data
Phase I/Phase II convention and its own from-scratch rank computation),
this module reads its mean ranks, Nemenyi critical difference, and
Friedman F-statistic directly from
sa_appa_experiments/results/scenario_cd_summaries.json, produced by
export_scenario_summaries.py as a thin export of analysis.py's already-
validated run_scenario() output -- the same numbers already embedded in
Appendix B's tables and Section 5.7/5.8's prose. This guarantees the CD
diagrams cannot silently drift from the numbers already reported
elsewhere in the paper.

The connecting bar that groups methods not significantly different at the
Nemenyi CD is drawn as a fully opaque line (no alpha), per the project's
EPS/no-transparency format rule -- the original fig5_cd_diagram.py used
`alpha=0.7` on this bar, which is fixed here.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from figures.common import save_figure

CAPTION_TEMPLATE = (
    "Figure {fignum} (Scenario {num}, CD{num}). Critical-difference diagram, "
    "Scenario {num} ({name}): {isolates}. Methods {methods_list}, "
    "n={n_instances} instances. Omnibus Friedman test (Conover's "
    "F-approximation): T2={t2:.3f}, F({df1},{df2}), p={p_str}. Nemenyi "
    "critical difference CD={cd:.3f} at alpha=0.05 (Demsar, 2006 -- a "
    "distinct post-hoc procedure to the Conover paired critical difference "
    "reported alongside the omnibus test in Appendix D, not "
    "interchangeable with it; the Nemenyi CD here is not currently traced "
    "to a bibliography entry -- see the standing citation-verification "
    "note). Methods are positioned by mean rank (lower/left = better); a "
    "solid bar connects methods whose mean-rank difference is below the "
    "Nemenyi CD, i.e.\\ not distinguishable at this significance level. "
    "Mean ranks: {mean_rank_str}."
)


def _p_str(p, t2, df1, df2):
    if p == 0:
        from scipy.stats import f as f_dist
        return f"<{np.exp(f_dist.logsf(t2, df1, df2)):.2g}"
    return f"{p:.4g}"


def _draw_cd(ax, order, ordered_ranks, cd, k):
    y0 = 0.5
    ax.hlines(y0, ordered_ranks.min() - 0.5, ordered_ranks.max() + 0.5,
              color="black", linewidth=1)
    for i, m in enumerate(order):
        r = ordered_ranks[m]
        ax.plot([r, r], [y0, y0 + 0.15], color="black", linewidth=1)
        ax.text(r, y0 + 0.18 + 0.12 * (i % 2), f"{m}\n({r:.2f})",
                ha="center", va="bottom", fontsize=8)

    bar_y = y0 - 0.15
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ordered_ranks[order[j + 1]] - ordered_ranks[order[i]] < cd:
            j += 1
        if j > i:
            # Opaque (no alpha) connecting bar -- EPS backend used for
            # this figure has no transparency support (project format
            # rule); the original implementation used alpha=0.7 here.
            ax.hlines(bar_y, ordered_ranks[order[i]], ordered_ranks[order[j]],
                      color="firebrick", linewidth=3, alpha=1.0)
            bar_y -= 0.1
        i += 1 if j == i else (j - i)
    return bar_y


def make_cd_diagram_for_scenario(s, out_dir, fignum, is_synthetic=False):
    """s: one entry of the list loaded from scenario_cd_summaries.json."""
    method_ids = s["methods"]
    mean_rank = s["mean_ranks"]
    k = s["k_methods"]
    order = sorted(method_ids, key=lambda m: mean_rank[m])
    ordered_ranks = {m: mean_rank[m] for m in order}
    cd = s["cd_nemenyi_05"]

    fig, ax = plt.subplots(figsize=(8, 2.6 + 0.2 * k))
    # _draw_cd expects .min()/.max()/label-indexing; a pandas Series over
    # the plain {method: rank} dict gives that without a second code path.
    import pandas as pd
    ranks_series = pd.Series(ordered_ranks)
    bar_y = _draw_cd(ax, order, ranks_series, cd, k)

    ax.set_ylim(bar_y - 0.1, 0.5 + 0.55 + 0.12 * k)
    ax.set_yticks([])
    ax.set_xlabel("Mean rank (lower = better)")
    ax.set_title(f"Scenario {s['num']}: {s['name']}  (CD={cd:.3f}, "
                 f"$\\alpha=0.05$)")
    fig.tight_layout()

    p_str = _p_str(s["p_value"], s["T2"], s["df1"], s["df2"])
    mean_rank_str = ", ".join(f"{m}={mean_rank[m]:.3f}" for m in order)
    caption = CAPTION_TEMPLATE.format(
        fignum=fignum, num=s["num"], name=s["name"], isolates=s["isolates"],
        methods_list=", ".join(method_ids), n_instances=s["n_instances"],
        t2=s["T2"], df1=s["df1"], df2=s["df2"], p_str=p_str, cd=cd,
        mean_rank_str=mean_rank_str,
    )
    return save_figure(fig, out_dir, f"fig_cd_scenario{s['num']}", caption, is_synthetic)


def make_all_cd_diagrams(json_path, out_dir, is_synthetic=False):
    with open(json_path) as f:
        scenarios = json.load(f)
    outputs = []
    for i, s in enumerate(scenarios, start=1):
        outputs.append(make_cd_diagram_for_scenario(s, out_dir, fignum=5 + i,
                                                      is_synthetic=is_synthetic))
    return outputs


if __name__ == "__main__":
    make_all_cd_diagrams("scenario_cd_summaries.json", "output_real")
