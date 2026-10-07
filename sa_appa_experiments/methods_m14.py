"""
methods_m14.py
==============
M14: a matched-budget re-implementation of BL-SAEA,

    H. Jiang, K. Qiu, Y. Tian, X. Zhang, Y. Jin, "Efficient surrogate
    modeling method for evolutionary algorithm to solve bilevel
    optimization problems", IEEE Transactions on Cybernetics 54(7)
    (2024) 4335-4347, doi:10.1109/TCYB.2023.3309598.

Source used for the algorithmic details: the authors' group's own
reference implementation in PlatEMO (BIMK/PlatEMO, file
Algorithms/Multi-objective optimization/BL-SAEA/BLSAEA.m and llSearch.m),
read directly. The published paper itself was not accessible from this
environment (paywalled), so where the paper and the PlatEMO code could
differ, the PlatEMO code is what is followed, except where noted below.

This module is ADDITIVE. It does not import-modify methods.py, gp.py or
ll_solver.py, so the locked M1-M13 results in results/summary.csv are
untouched (verified by the bit-for-bit re-run check in the Task 14 notes).

What is kept from BL-SAEA (PlatEMO BLSAEA.m):
  * UL search operator: binary tournament selection, simulated binary
    crossover (p_c = 1, eta_c = 20) and polynomial mutation
    (p_m = 1/D, eta_m = 20), producing 2 x N_ul offspring per generation.
  * UL surrogate: a Kriging (GP) model of the UL objective F trained on the
    archive of truly evaluated UL points. Offspring are pre-screened by
    non-dominated sorting on (predicted mean, predicted variance), both
    minimised, and only the first front receives a full nested LL solve.
  * LL surrogates: one Kriging model per LL variable, mapping x_u -> x_l*
    over the archive, used to predict a starting point x_l0 for the LL
    search at each screened offspring.
  * LL warm start (llSearch.m, flag=1): the LL initial population is half
    Latin-hypercube points and half seeded around the predictions: the LL
    optimum of the archive point closest in cosine similarity (itself and
    one Gaussian perturbation), Gaussian perturbations of x_l0 (sigma =
    0.2 x LL range, about half of the coordinates perturbed), and x_l0.
  * Fitness: feasibility penalty (objective if UL and LL feasible, else
    total violation + 1e10); (mu+lambda) truncation to N_ul.
  * Quadratic-approximation local search around the current best UL point
    each generation, followed by one full nested LL solve at its result,
    replacing the worst population member.

Deliberate, documented deviations (needed to run under this study's
matched budget and shared LL solver; none of them favours M14 or M1):
  1. Budget: 60 full nested LL solves per run, as for every other method
     (25 LHS points shared with M1's initial design + 35 further solves).
     The screened first front is truncated, by predicted mean, to the
     remaining budget.
  2. LL solver: the same scipy differential_evolution call used by every
     other method (popsize=6, maxiter=20, polish=True), with only its
     initial population replaced by the BL-SAEA warm start. PlatEMO's
     llSearch runs its own GA for up to 2000 generations; that is not
     comparable with the other 13 methods' LL effort and is not used.
  3. Local search: PlatEMO minimises a quadratic approximation of the UL
     objective with SQP while calling the true UL objective at
     approximated LL responses. Those true calls are not counted as
     expensive in BL-SAEA but would be uncounted extra information here,
     so this re-implementation minimises the quadratic approximations of
     the UL objective and UL constraints only (scipy SLSQP inside the UL
     box), then spends one counted nested LL solve at the result.
  4. LL mapping models: PlatEMO's code fits the per-LL-variable models on
     (archive x_u, archive F), which appears to be an indexing slip, since
     the paper describes the models as mapping x_u to the LL optimum. This
     re-implementation fits them on (archive x_u, archive x_l*), as the
     paper describes. [NEEDS AUTHOR CHECK: confirm against the paper.]
  5. Kriging: the study's own GPSurrogate (squared-exponential kernel) is
     used, but with its lengthscale chosen by maximum marginal likelihood
     over a log-spaced grid at every refit, since DACE Kriging (used by
     BL-SAEA) estimates its correlation parameter by maximum likelihood.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from methods import (N0, T_MAX, N_UL_POP, Trajectory, _stable_seed, _lhs,
                     _feasible, _clip)
from gp import GPSurrogate
from ll_solver import PENALTY

BUDGET = N0 + T_MAX            # 60 full nested LL solves, as for M1-M13
PRO_C, DIS_C, DIS_M = 1.0, 20.0, 20.0
LL_POPSIZE, LL_MAXITER = 6, 20  # identical to every other method's LL call


# ---------------------------------------------------------------------------
# Kriging with ML lengthscale
# ---------------------------------------------------------------------------

def _fit_gp_ml(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    y_std = float(np.std(y))
    if not np.isfinite(y_std) or y_std < 1e-12:
        y_std = 1.0
    ys = (y - np.mean(y)) / y_std
    d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
    iu = np.triu_indices_from(d, k=1)
    med = np.median(d[iu]) if len(iu[0]) else 1.0
    med = max(med, 1e-3)
    best, best_ll = None, -np.inf
    for ls in med * np.logspace(-1.5, 1.0, 12):
        gp = GPSurrogate(lengthscale=ls)
        try:
            gp.fit(X, ys)
        except np.linalg.LinAlgError:
            continue
        yc = ys - gp.y_mean
        ll = -0.5 * yc @ gp.alpha - np.sum(np.log(np.diag(gp.L)))
        if ll > best_ll:
            best, best_ll = gp, ll
    if best is None:
        best = GPSurrogate(lengthscale=med).fit(X, ys)
    return best, float(np.mean(y)), y_std


def _predict_gp_ml(model, Xs):
    gp, m, s = model
    mu, sd = gp.predict(Xs)
    return m + s * mu, (s * sd) ** 2


# ---------------------------------------------------------------------------
# UL operators (PlatEMO BLSAEAGA)
# ---------------------------------------------------------------------------

def _tournament(fitness, n, rng):
    a = rng.integers(len(fitness), size=n)
    b = rng.integers(len(fitness), size=n)
    return np.where(fitness[a] <= fitness[b], a, b)


def _sbx_pm(parents, lo, hi, rng):
    half = len(parents) // 2
    p1, p2 = parents[:half], parents[half:2 * half]
    N, D = p1.shape
    mu = rng.uniform(size=(N, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (DIS_C + 1)),
                    (2 - 2 * mu) ** (-1 / (DIS_C + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, size=(N, D))
    beta[rng.uniform(size=(N, D)) < 0.5] = 1.0
    beta[np.repeat(rng.uniform(size=(N, 1)) > PRO_C, D, axis=1)] = 1.0
    off = np.vstack([(p1 + p2) / 2 + beta * (p1 - p2) / 2,
                     (p1 + p2) / 2 - beta * (p1 - p2) / 2])
    Lo = np.broadcast_to(lo, off.shape)
    Up = np.broadcast_to(hi, off.shape)
    off = np.minimum(np.maximum(off, Lo), Up)
    span = np.maximum(Up - Lo, 1e-12)
    site = rng.uniform(size=off.shape) < 1.0 / D
    mu = rng.uniform(size=off.shape)
    t = site & (mu <= 0.5)
    off[t] = off[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) *
             (1 - (off[t] - Lo[t]) / span[t]) ** (DIS_M + 1)) ** (1 / (DIS_M + 1)) - 1)
    t = site & (mu > 0.5)
    off[t] = off[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) *
             (1 - (Up[t] - off[t]) / span[t]) ** (DIS_M + 1)) ** (1 / (DIS_M + 1)))
    return np.minimum(np.maximum(off, Lo), Up)


def _first_front(obj):
    n = len(obj)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        if dom.any():
            keep[i] = False
    return np.flatnonzero(keep)


# ---------------------------------------------------------------------------
# LL solve with BL-SAEA warm start (same DE settings as ll_solver.ll_solve)
# ---------------------------------------------------------------------------

def _ll_solve_warm(inst, x, seed, xl0=None, xl_near=None):
    yb = inst.y_bounds
    lo, hi = yb[:, 0], yb[:, 1]
    n_ll = len(lo)
    S = max(5, LL_POPSIZE * n_ll)   # scipy's population size for popsize=6

    def obj(y):
        val = inst.f(x, y)
        g = inst.g(x, y)
        viol = np.sum(np.maximum(g, 0.0) ** 2) if g.size else 0.0
        return val + PENALTY * viol

    init = 'latinhypercube'
    if xl0 is not None:
        rng = np.random.default_rng(seed)
        n_lhs = S - S // 2
        pop = qmc.scale(qmc.LatinHypercube(d=n_ll, seed=seed).random(n_lhs), lo, hi)
        sigma = 0.2 * (hi - lo)
        seeded = []
        if xl_near is not None:
            seeded.append(np.clip(xl_near, lo, hi))
            pert = xl_near.copy()
            m = rng.uniform(size=n_ll) < 0.5
            pert[m] = pert[m] + sigma[m] * rng.standard_normal(m.sum())
            seeded.append(np.clip(pert, lo, hi))
        x0c = np.clip(xl0, lo, hi)
        while len(seeded) < S // 2 - 1:
            pert = x0c.copy()
            m = rng.uniform(size=n_ll) < 0.5
            pert[m] = pert[m] + sigma[m] * rng.standard_normal(m.sum())
            seeded.append(np.clip(pert, lo, hi))
        seeded.append(x0c)
        init = np.vstack([pop, np.array(seeded[:S - n_lhs])])
        # scipy requires strictly-inside-bounds or on-bounds points; clip
        init = np.clip(init, lo, hi)

    res = differential_evolution(
        obj, list(map(tuple, yb)), seed=seed, popsize=LL_POPSIZE,
        maxiter=LL_MAXITER, tol=1e-7, mutation=(0.4, 1.0), recombination=0.7,
        polish=True, updating='deferred', workers=1, init=init)
    y_star = res.x
    g = inst.g(x, y_star)
    ll_feas = bool(np.all(g <= 1e-6)) if g.size else True
    return y_star, inst.F(x, y_star), inst.G(x, y_star), ll_feas, g


def _fitness(F, G, gll):
    v = (np.sum(np.maximum(G, 0)) if G.size else 0.0) + \
        (np.sum(np.maximum(gll, 0)) if gll.size else 0.0)
    return F if v <= 1e-6 else v + 1e10


def _quad_fit(X, y):
    """Least-squares quadratic model (constant + linear + diagonal +
    cross terms when enough points; otherwise diagonal only), mirroring
    PlatEMO's QuadApprox in intent."""
    n, d = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(d)] + [X[:, i] ** 2 for i in range(d)]
    full = n >= 1 + 2 * d + d * (d - 1) // 2
    if full:
        cols += [X[:, i] * X[:, j] for i in range(d) for j in range(i + 1, d)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    def f(x):
        x = np.asarray(x)
        terms = [1.0] + list(x) + list(x ** 2)
        if full:
            terms += [x[i] * x[j] for i in range(d) for j in range(i + 1, d)]
        return float(np.dot(coef, terms))
    return f


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_m14(inst, seed=0, method_id="M14", ll_topk=False):
    """ll_topk=True trains the LL mapping models, per UL point, on the K archive points most
    similar in cosine similarity, K=(Du+1)(Dl+2)/2, as in Jiang et al. (2024), Algorithm 4.
    The default (False) trains on the whole archive; this is the configuration reported as M14."""
    t0 = time.time()
    lo, hi = inst.x_bounds[:, 0], inst.x_bounds[:, 1]
    rng = np.random.default_rng(_stable_seed(seed, method_id, inst.id, "pop") % (2 ** 31 - 1))
    traj = Trajectory()
    n_eval = 0

    # Initial design: identical points and LL seeds to methods._initial_design
    # (so M14 starts from exactly M1's 25 LHS evaluations).
    X = _lhs(inst.x_bounds, N0, seed)
    Yl, Fv, Gv, fit = [], [], [], []
    for i, x in enumerate(X):
        y, F, G, _, gll = _ll_solve_warm(inst, x, _stable_seed(seed, "init", i) % (2 ** 31 - 1))
        n_eval += 1
        traj.record(n_eval, 0, F, _feasible(G))  # same feasibility rule as M1-M13 reporting
        Yl.append(y); Fv.append(F); Gv.append(G); fit.append(_fitness(F, G, gll))
    A_x, A_y, A_F = X.copy(), np.array(Yl), np.array(Fv)
    A_G = list(Gv)
    A_fit = np.array(fit)

    order = np.argsort(A_fit)[:N_UL_POP]
    P_x, P_y, P_fit = A_x[order], A_y[order], A_fit[order]

    gen = 0
    while n_eval < BUDGET:
        gen += 1
        # --- offspring
        mp1 = _tournament(P_fit, len(P_fit), rng)
        mp2 = _tournament(P_fit, len(P_fit), rng)
        off = np.vstack([_sbx_pm(P_x[mp1], lo, hi, rng), _sbx_pm(P_x[mp2], lo, hi, rng)])

        # --- UL Kriging screening
        ul_model = _fit_gp_ml(A_x, A_F)
        mu, var = _predict_gp_ml(ul_model, off)
        front = _first_front(np.column_stack([mu, var]))
        remaining = BUDGET - n_eval - 1          # keep one solve for local search
        if remaining <= 0:
            front = front[:0]
        elif len(front) > remaining:
            front = front[np.argsort(mu[front])[:remaining]]

        # --- LL mapping models x_u -> x_l*
        K_top = (A_x.shape[1] + 1) * (A_y.shape[1] + 2) // 2
        if not ll_topk or K_top >= len(A_x):
            ll_models = [_fit_gp_ml(A_x, A_y[:, k]) for k in range(A_y.shape[1])]

            def predict_xl(xu):
                return np.array([_predict_gp_ml(m, xu[None, :])[0][0] for m in ll_models])
        else:
            def predict_xl(xu):
                nx = np.linalg.norm(A_x, axis=1) * (np.linalg.norm(xu) + 1e-12) + 1e-12
                sel = np.argsort(-(A_x @ xu) / nx, kind="stable")[:K_top]
                ms = [_fit_gp_ml(A_x[sel], A_y[sel, k]) for k in range(A_y.shape[1])]
                return np.array([_predict_gp_ml(m, xu[None, :])[0][0] for m in ms])

        def nearest_xl(xu):
            nx = np.linalg.norm(A_x, axis=1) * (np.linalg.norm(xu) + 1e-12) + 1e-12
            cos = (A_x @ xu) / nx
            return A_y[int(np.argmax(cos))]

        new_x, new_y, new_fit = [], [], []
        for j in front:
            xu = off[j]
            y, F, G, _, gll = _ll_solve_warm(
                inst, xu, _stable_seed(seed, method_id, inst.id, gen, int(j)) % (2 ** 31 - 1),
                xl0=predict_xl(xu), xl_near=nearest_xl(xu))
            n_eval += 1
            traj.record(n_eval, gen, F, _feasible(G))
            fv = _fitness(F, G, gll)
            A_x = np.vstack([A_x, xu]); A_y = np.vstack([A_y, y])
            A_F = np.append(A_F, F); A_G.append(G); A_fit = np.append(A_fit, fv)
            new_x.append(xu); new_y.append(y); new_fit.append(fv)

        if new_x:
            P_x = np.vstack([P_x, new_x]); P_y = np.vstack([P_y, new_y])
            P_fit = np.append(P_fit, new_fit)
        keep = np.argsort(P_fit)[:N_UL_POP]
        P_x, P_y, P_fit = P_x[keep], P_y[keep], P_fit[keep]

        if n_eval >= BUDGET:
            break

        # --- quadratic-approximation local search around current best
        xb = P_x[0]
        idx = np.argsort(A_fit)[:max(2 * len(xb) + 1, N_UL_POP)]
        Xq = A_x[idx]
        fq = _quad_fit(Xq, A_F[idx])
        cons = []
        if A_G[0].size:
            Garr = np.array([A_G[i] for i in idx])
            for k in range(Garr.shape[1]):
                qk = _quad_fit(Xq, Garr[:, k])
                cons.append({'type': 'ineq', 'fun': (lambda x, q=qk: -q(x))})
        try:
            r = minimize(fq, xb, method='SLSQP', bounds=list(zip(lo, hi)),
                         constraints=cons, options={'maxiter': 100})
            x_ls = _clip(np.atleast_2d(r.x), inst.x_bounds)[0]
        except Exception:
            x_ls = xb.copy()
        y, F, G, _, gll = _ll_solve_warm(
            inst, x_ls, _stable_seed(seed, method_id, inst.id, gen, "ls") % (2 ** 31 - 1),
            xl0=predict_xl(x_ls), xl_near=nearest_xl(x_ls))
        n_eval += 1
        traj.record(n_eval, gen, F, _feasible(G))
        fv = _fitness(F, G, gll)
        A_x = np.vstack([A_x, x_ls]); A_y = np.vstack([A_y, y])
        A_F = np.append(A_F, F); A_G.append(G); A_fit = np.append(A_fit, fv)
        # replace worst population member (PlatEMO behaviour)
        P_x[-1], P_y[-1], P_fit[-1] = x_ls, y, fv
        o = np.argsort(P_fit)
        P_x, P_y, P_fit = P_x[o], P_y[o], P_fit[o]

    return traj, time.time() - t0
