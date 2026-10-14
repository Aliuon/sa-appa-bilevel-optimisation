"""
strong_stackelberg.py
=====================
Strong Stackelberg (optimistic) selection among tied lower-level optima, applied to every
nested LL solve on every test problem.

Reference: D'Amato, Daniele, Mallozzi and Petrone (2012), Equilibrium strategies via GA to
Stackelberg games under multiple follower's best reply, International Journal of Intelligent
Systems 27(2), 74-85 (doi 10.1002/int.21514): when the follower has several best replies, the
leader is assumed to receive the one that is best for the leader.

Why a new version (optimistic_ll.py was the SMD6-only sensitivity step). That step searched the
level set {f <= f0 + eps} around the DE answer y0. When y0 is not an accurate LL solution, that
set contains points that are worse for the follower than its true optimum, and the search moved
the leader's value by exploiting the solver error (SMD10 (3,3): F 2.01 -> 0.29 in one test).
This version only breaks ties; it never corrects or exploits LL solver error:

1. Accuracy gate. A local constrained solve of the LL problem from y0 (SLSQP) gives f_loc. If
   f0 - f_loc > ACC_TOL * max(1, |f_loc|), y0 is not an accurate LL solution and no selection is
   made (y0 is returned unchanged, exactly as in the locked protocol).
2. Selection. Over the eps-optimal set S_eps = {y in box : g(x,y) <= 0, f(x,y) <= f_ref + eps},
   with f_ref = min(f0, f_loc) and eps = REL_EPS * max(1, |f_ref|), the leader's preferred reply is
   found lexicographically: first minimise the UL constraint violation sum(max(G,0)^2), then
   minimise F subject to G <= 0 (or, if no G-feasible tie exists, at the least-violating point).
3. Validation. The candidate y1 must satisfy g <= 1e-6 (when y0 does), f(y1) <= f_ref + 2 eps, and
   a local LL solve started at y1 must not lower f by more than ACC_TOL * max(1, |f_ref|), i.e.
   y1 is itself an (approximately) optimal reply and no better LL basin was uncovered.
4. Materiality. y1 must make the reply UL-feasible (or cut the violation by DELTA_REL relative),
   or keep the violation and lower F by more than DELTA_REL * max(1, |F0|).
5. Tie test. The selection is repeated with eps/100. Along a genuine tie, f is constant and the
   leader's gain is unchanged; inside the small ball of eps-optimal points around a unique LL
   optimum the gain shrinks roughly tenfold. y1 is accepted only if at least TIE_RATIO of the
   gain survives, and the eps/100 reply is the one returned. On problems with a unique LL
   optimum the step therefore returns y0 unchanged.

Every F, f, g and G evaluation made by the step is counted in COUNTS.
"""
import numpy as np
from scipy.optimize import minimize

REL_EPS = 1e-8       # width of the eps-optimal LL set, relative to max(1, |f_ref|)
ACC_TOL = 1e-5       # accuracy gate for y0 and validation tolerance for y1
DELTA_REL = 1e-4     # materiality threshold on F, relative to max(1, |F0|)
TIE_RATIO = 0.5      # gain at eps/100 must be at least this share of the gain at eps
FEAS_TOL = 1e-6

COUNTS = {"calls": 0, "gated": 0, "accepted": 0, "accepted_feas": 0, "ball_rejected": 0,
          "F_evals": 0, "f_evals": 0, "g_evals": 0, "G_evals": 0}


def reset_counts():
    for k in COUNTS:
        COUNTS[k] = 0


def _viol(v):
    v = np.asarray(v, float)
    return float(np.sum(np.maximum(v, 0.0) ** 2)) if v.size else 0.0


def _local_ll(inst, x, y_start, lo, hi, use_g):
    """Local constrained LL solve from y_start; returns (y, f) of the best LL-feasible point seen."""
    def fo(y):
        COUNTS["f_evals"] += 1
        return float(inst.f(x, y))
    cons = []
    if use_g:
        def gc(y):
            COUNTS["g_evals"] += 1
            return -np.asarray(inst.g(x, y), float)
        cons.append({"type": "ineq", "fun": gc})
    try:
        r = minimize(fo, y_start, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                     options={"maxiter": 200, "ftol": 1e-14})
        y = np.clip(r.x, lo, hi)
    except Exception:
        return y_start, float(inst.f(x, y_start))
    fy = float(inst.f(x, y))
    if use_g and np.any(np.asarray(inst.g(x, y)) > FEAS_TOL):
        return y_start, float(inst.f(x, y_start))
    return y, fy


def strong_stackelberg_select(inst, x, y0):
    COUNTS["calls"] += 1
    lo, hi = inst.y_bounds[:, 0], inst.y_bounds[:, 1]
    y0 = np.clip(np.asarray(y0, float), lo, hi)
    f0 = float(inst.f(x, y0))
    g0 = np.asarray(inst.g(x, y0), float)
    use_g = bool(g0.size) and bool(np.all(g0 <= FEAS_TOL))
    if g0.size and not use_g:
        return y0                              # LL-infeasible answer: nothing to select among
    F0 = float(inst.F(x, y0))
    G0 = np.asarray(inst.G(x, y0), float)
    v0 = _viol(G0)

    # 1. accuracy gate
    _, f_loc = _local_ll(inst, x, y0, lo, hi, use_g)
    acc = ACC_TOL * max(1.0, abs(f_loc))
    if f0 - f_loc > acc:
        COUNTS["gated"] += 1
        return y0
    f_ref = min(f0, f_loc)
    eps = REL_EPS * max(1.0, abs(f_ref))

    def Fo(y):
        COUNTS["F_evals"] += 1
        return float(inst.F(x, y))

    def Vo(y):
        COUNTS["G_evals"] += 1
        return _viol(inst.G(x, y))

    bnds = list(zip(lo, hi))
    opts = {"maxiter": 100, "ftol": 1e-12}

    def select(eps):
        """Leader's preferred reply in S_eps, lexicographic (UL violation, then F); validated."""
        def level(y):
            COUNTS["f_evals"] += 1
            return f_ref + eps - float(inst.f(x, y))
        base = [{"type": "ineq", "fun": level}]
        if use_g:
            def gc(y):
                COUNTS["g_evals"] += 1
                return -np.asarray(inst.g(x, y), float)
            base.append({"type": "ineq", "fun": gc})
        start = y0
        G_ok = v0 <= FEAS_TOL ** 2
        if not G_ok and G0.size:
            try:
                r = minimize(Vo, y0, method="SLSQP", bounds=bnds, constraints=base, options=opts)
                ya = np.clip(r.x, lo, hi)
                if Vo(ya) < v0:
                    start = ya
                G_ok = Vo(start) <= FEAS_TOL ** 2
            except Exception:
                pass
        cons = list(base)
        if G_ok and G0.size:
            def Gc(y):
                COUNTS["G_evals"] += 1
                return -np.asarray(inst.G(x, y), float)
            cons.append({"type": "ineq", "fun": Gc})
        try:
            r = minimize(Fo, start, method="SLSQP", bounds=bnds, constraints=cons, options=opts)
            y1 = np.clip(r.x, lo, hi)
        except Exception:
            y1 = start
        best = None
        for cand in (y1, start):
            if cand is y0:
                continue
            if float(inst.f(x, cand)) > f_ref + 2 * eps:
                continue
            if use_g and np.any(np.asarray(inst.g(x, cand)) > FEAS_TOL):
                continue
            _, f_chk = _local_ll(inst, x, cand, lo, hi, use_g)
            if f_chk < f_ref - acc:            # a better LL basin exists: y0 was not optimal
                continue
            key = (_viol(inst.G(x, cand)), float(inst.F(x, cand)))
            if best is None or key < best[1]:
                best = (cand, key)
        return best

    # 2-3. selection at the working tolerance
    sel = select(REL_EPS * max(1.0, abs(f_ref)))
    if sel is None:
        return y0
    cand, (v1, F1) = sel
    gain_v = v0 - v1 if v0 > FEAS_TOL ** 2 else 0.0
    gain_F = F0 - F1 if v1 <= max(v0, FEAS_TOL ** 2) else 0.0
    feas_gain = gain_v > FEAS_TOL ** 2 and (v1 <= FEAS_TOL ** 2 or gain_v > DELTA_REL * max(1.0, v0))
    F_gain = gain_F > DELTA_REL * max(1.0, abs(F0))
    if not (feas_gain or F_gain):
        return y0
    # 4. tie test: the leader's gain must not depend on the width of the optimality tolerance.
    # Along a genuine tie (f constant) the gain is the same at eps/100; inside the small
    # eps-ball around a unique optimum it shrinks roughly tenfold.
    sel2 = select(REL_EPS * 1e-2 * max(1.0, abs(f_ref)))
    if sel2 is None:
        return y0
    _, (v2, F2) = sel2
    if feas_gain and (v0 - v2) < TIE_RATIO * gain_v:
        feas_gain = False
    if F_gain and (F0 - F2) < TIE_RATIO * gain_F:
        F_gain = False
    if not (feas_gain or F_gain):
        COUNTS["ball_rejected"] += 1
        return y0
    COUNTS["accepted"] += 1
    if feas_gain:
        COUNTS["accepted_feas"] += 1
    return sel2[0]


def patch():
    """Wrap the shared LL solve (M1-M13) and M14's warm-started LL solve with the selection."""
    import methods, methods_m14
    import ll_solver
    base = ll_solver.ll_solve
    base_warm = methods_m14._ll_solve_warm

    def ll_solve_ss(inst, x, seed=0, **kw):
        y, F, G, feas = base(inst, x, seed=seed, **kw)
        y1 = strong_stackelberg_select(inst, x, y)
        if y1 is y or np.array_equal(y1, y):
            return y, F, G, feas
        g = inst.g(x, y1)
        feas = bool(np.all(g <= FEAS_TOL)) if g.size else True
        return y1, inst.F(x, y1), inst.G(x, y1), feas

    def ll_solve_warm_ss(inst, x, seed, xl0=None, xl_near=None):
        out = base_warm(inst, x, seed, xl0=xl0, xl_near=xl_near)
        y = out[0]
        y1 = strong_stackelberg_select(inst, x, y)
        if y1 is y or np.array_equal(y1, y):
            return out
        g = inst.g(x, y1)
        feas = bool(np.all(g <= FEAS_TOL)) if g.size else True
        return y1, inst.F(x, y1), inst.G(x, y1), feas, g

    methods.ll_solve = ll_solve_ss
    methods_m14._ll_solve_warm = ll_solve_warm_ss
    return base, base_warm
