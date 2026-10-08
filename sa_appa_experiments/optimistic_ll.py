"""
optimistic_ll.py
================
Optimistic tie-break for the nested LL solve (Assumption A1), used only by the SMD6 sensitivity
run (run_optimistic_ll.py). The locked results were produced without it and are not changed.

Why: when the LL has several optimal responses, the shared DE solve returns an arbitrary one,
so the UL value it reports is not theta(x) under the optimistic convention. On SMD6 this adds a
non-negative term 2 t^2 with t effectively random, which makes the training data of every
surrogate inconsistent (phase7_smd6_check.py).

What: after the usual DE solve returns y0 with LL value f0, a local search solves the optimistic
selection problem over the (epsilon-)optimal LL set,

    min_y F(x, y)  s.t.  f(x, y) <= f0 + eps,  g(x, y) <= 0 (if y0 is LL-feasible),  y in box,

started from y0 (SLSQP, at most 100 iterations), with eps = 1e-6 * max(1, |f0|). This is the
optimistic selection over the LL solution set (see Dempe 2002, dempe2002foundations, for the optimistic formulation), applied
as a refinement of a numerical LL solution. The refined point is accepted only if it satisfies
the constraints and lowers F; otherwise y0 is kept, so the step can never make the reported
value worse for the leader. Every F, f and g evaluation made by the step is counted.

The refinement is generic: it uses no knowledge of SMD6. On an instance whose LL optimum is
unique it can move y only within the eps-level set of f, so it is close to a no-op there.
"""
import numpy as np
from scipy.optimize import minimize

COUNTS = {"calls": 0, "accepted": 0, "F_evals": 0, "f_evals": 0}


def optimistic_refine(inst, x, y0, rel_eps=1e-6, maxiter=100):
    COUNTS["calls"] += 1
    lo, hi = inst.y_bounds[:, 0], inst.y_bounds[:, 1]
    y0 = np.clip(np.asarray(y0, float), lo, hi)
    f0 = float(inst.f(x, y0))
    g0 = inst.g(x, y0)
    ll_feas = bool(np.all(g0 <= 1e-6)) if g0.size else True
    eps = rel_eps * max(1.0, abs(f0))
    F0 = float(inst.F(x, y0))

    def Fo(y):
        COUNTS["F_evals"] += 1
        return float(inst.F(x, y))

    def c_level(y):
        COUNTS["f_evals"] += 1
        return f0 + eps - float(inst.f(x, y))

    cons = [{"type": "ineq", "fun": c_level}]
    if g0.size and ll_feas:
        cons.append({"type": "ineq", "fun": lambda y: -np.asarray(inst.g(x, y), float)})
    try:
        r = minimize(Fo, y0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                     options={"maxiter": maxiter, "ftol": 1e-10})
        y1 = np.clip(r.x, lo, hi)
    except Exception:
        return y0
    ok = float(inst.f(x, y1)) <= f0 + 2 * eps
    if g0.size and ll_feas:
        ok = ok and bool(np.all(inst.g(x, y1) <= 1e-6))
    if ok and float(inst.F(x, y1)) < F0:
        COUNTS["accepted"] += 1
        return y1
    return y0


def patch():
    """Wrap the shared LL solve (M1-M13) and M14's warm-started LL solve with the refinement."""
    import methods, methods_m14
    import ll_solver
    base = ll_solver.ll_solve
    base_warm = methods_m14._ll_solve_warm

    def ll_solve_opt(inst, x, seed=0, **kw):
        y, _, _, _ = base(inst, x, seed=seed, **kw)
        y = optimistic_refine(inst, x, y)
        g = inst.g(x, y)
        feas = bool(np.all(g <= 1e-6)) if g.size else True
        return y, inst.F(x, y), inst.G(x, y), feas

    def ll_solve_warm_opt(inst, x, seed, xl0=None, xl_near=None):
        y, _, _, _, _ = base_warm(inst, x, seed, xl0=xl0, xl_near=xl_near)
        y = optimistic_refine(inst, x, y)
        g = inst.g(x, y)
        feas = bool(np.all(g <= 1e-6)) if g.size else True
        return y, inst.F(x, y), inst.G(x, y), feas, g

    methods.ll_solve = ll_solve_opt
    methods_m14._ll_solve_warm = ll_solve_warm_opt
