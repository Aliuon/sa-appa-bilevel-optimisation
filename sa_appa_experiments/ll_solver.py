"""
ll_solver.py
============
The nested, "exact" lower-level solve used by every one of the 13
methods whenever they need a true (x, y*(x)) pair -- the initial LHS
design, every infill point for surrogate-assisted methods, and every
population member for the fully nested baselines (M12, M13).

Implementation choice, documented rather than left implicit: this uses
scipy.optimize.differential_evolution (a real, independently-validated
global optimiser) with constraints handled by a quadratic penalty on
g_l(x,y) > 0, uniformly across all 44 instances and all 13 methods. The
manuscript does not specify what solves the LL problem "exactly"; using
one consistent, off-the-shelf, non-fabricated solver for this shared
subroutine keeps the comparison between M1-M13 fair (they differ only
in their UL search / constraint-handling / infill choices, which is
what Section 5 actually studies), and is stated here explicitly rather
than silently assumed.
"""

import numpy as np
from scipy.optimize import differential_evolution

PENALTY = 1e6


def ll_solve(inst, x, seed=0, popsize=12, maxiter=60, tol=1e-7):
    bounds = list(map(tuple, inst.y_bounds))

    def obj(y):
        val = inst.f(x, y)
        g = inst.g(x, y)
        if g.size:
            viol = np.sum(np.maximum(g, 0.0) ** 2)
        else:
            viol = 0.0
        return val + PENALTY * viol

    res = differential_evolution(
        obj, bounds, seed=seed, popsize=popsize, maxiter=maxiter,
        tol=tol, mutation=(0.4, 1.0), recombination=0.7, polish=True,
        updating='deferred', workers=1, init='latinhypercube',
    )
    y_star = res.x
    g = inst.g(x, y_star)
    feasible_ll = bool(np.all(g <= 1e-6)) if g.size else True
    F_val = inst.F(x, y_star)
    G_val = inst.G(x, y_star)
    return y_star, F_val, G_val, feasible_ll
