"""
common.py
=========
Shared plotting utilities: consistent styling, the mandatory SYNTHETIC
watermark, and a single save_figure() helper that writes both the image
file and a sibling .caption.txt file, so every figure's caption is
available as plain text for direct use in the paper (and is never only
inside this Python session).
"""

import os
import matplotlib.pyplot as plt

SYNTHETIC_WATERMARK = "SYNTHETIC PLACEHOLDER DATA -- NOT REAL RESULTS"

# ---------------------------------------------------------------------------
# Per-method (colour, linestyle, marker) styling, distinct across all 13
# methods. matplotlib's default colour cycle has only 10 colours, which
# silently wraps and gives M1 and M11 IDENTICAL colours in any 13-method
# plot using bare `ax.plot(..., label=m)` -- a real bug caught by
# cross-checking fig2b's plotted data profile against the underlying
# numbers (the visually "best" curve was actually M11, not M1). Every
# multi-method figure should use METHOD_STYLE[m] explicitly rather than
# relying on the default cycle.
# ---------------------------------------------------------------------------
_METHOD_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a",
]
_METHOD_LINESTYLES = ["-", "--", "-.", ":"]
_METHOD_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def method_style(method_id, method_ids_all):
    """Deterministic (color, linestyle, marker) triple for method_id, given
    the full ordered list of method ids in use, so that no two methods in
    the same figure share a colour even beyond 10 methods."""
    idx = method_ids_all.index(method_id) if method_id in method_ids_all else hash(method_id) % 13
    color = _METHOD_COLORS[idx % len(_METHOD_COLORS)]
    linestyle = _METHOD_LINESTYLES[idx // len(_METHOD_COLORS) % len(_METHOD_LINESTYLES)]
    marker = _METHOD_MARKERS[idx % len(_METHOD_MARKERS)]
    return dict(color=color, linestyle=linestyle)


def add_synthetic_watermark(fig, is_synthetic):
    if not is_synthetic:
        return
    fig.text(0.5, 0.5, SYNTHETIC_WATERMARK, fontsize=28, color="red",
              alpha=0.18, ha="center", va="center", rotation=30, zorder=1000)
    fig.text(0.99, 0.01, SYNTHETIC_WATERMARK, fontsize=7, color="red",
              ha="right", va="bottom", alpha=0.85)


def save_figure(fig, out_dir, filename_stem, caption, is_synthetic, dpi=200):
    """
    Save `fig` as PNG (and PDF, for direct LaTeX inclusion) under
    `out_dir`, plus a `<stem>.caption.txt` file containing `caption`.
    Returns the dict of paths written.

    If `is_synthetic`, both the filename and the caption text are
    prefixed so a placeholder figure can never be silently mistaken for
    a real one, even out of context (e.g. if someone greps a folder of
    figures for filenames, or pastes a caption into the paper without
    reading it first).
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = f"SYNTHETIC_{filename_stem}" if is_synthetic else filename_stem
    full_caption = caption
    if is_synthetic:
        full_caption = (
            "[SYNTHETIC PLACEHOLDER -- DO NOT USE IN THE PAPER AS-IS] " + caption
        )
        add_synthetic_watermark(fig, True)

    png_path = os.path.join(out_dir, f"{stem}.png")
    pdf_path = os.path.join(out_dir, f"{stem}.pdf")
    eps_path = os.path.join(out_dir, f"{stem}.eps")
    cap_path = os.path.join(out_dir, f"{stem}.caption.txt")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    try:
        # EPS has no alpha-transparency support; matplotlib's EPS backend
        # requires this: figures using alpha-blended fills/lines must use
        # opaque fills/hatching instead (project format rule) or this
        # raises/ warns per-artist. Kept as its own try/except so a
        # non-EPS-safe figure still produces PNG/PDF while surfacing the
        # problem loudly instead of silently shipping a broken EPS.
        fig.savefig(eps_path, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover
        eps_path = None
        print(f"[WARNING] EPS export failed for {stem}: {exc} "
              f"-- check for alpha-transparent artists (fills/lines) and "
              f"replace with opaque fills or hatching.")
    with open(cap_path, "w") as fh:
        fh.write(full_caption + "\n")

    plt.close(fig)  # avoid "too many open figures" when generating dozens
                     # of per-instance files (Figures 3 and 4)

    return {"png": png_path, "pdf": pdf_path, "eps": eps_path, "caption_file": cap_path,
            "caption": full_caption}


def new_fig(figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
