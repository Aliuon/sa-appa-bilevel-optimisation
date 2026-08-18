from figures.fig1_boxplot import make_figure1
from figures.fig2_performance_profile import make_figure2
from figures.fig3_convergence_curves import make_figure3_one, make_figure3_all
from figures.fig4_convergence_rate import make_figure4_one, make_figure4_all, fit_convergence_order
from figures.fig5_cd_diagram import make_figure5

__all__ = [
    "make_figure1",
    "make_figure2",
    "make_figure3_one", "make_figure3_all",
    "make_figure4_one", "make_figure4_all", "fit_convergence_order",
    "make_figure5",
]
