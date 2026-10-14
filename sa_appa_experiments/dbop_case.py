"""
dbop_case.py
============
Application case study: dynamic carbon-tax design for South Africa's power sector.

Model: the dynamic bilevel carbon policy problem (DBOP) of Haroon, Ngnotchouye and Tilahun,
Engineering Letters 34(7) (2026) 2797-2807, with per-technology capacity limits (as in the
authors' IMCE.m), recalibrated as agreed by the authors on 14 October 2026:

  C1  production in GW (capacities 35 / 3.5 / 4.3 GW, eps_max = 42.8 GW); all Table II-III values
      and bounds unchanged, so beta1 is read in $/MWh per GW^2;
  C2  emissions in Mt CO2 per year (GW x t/MWh x 8760 h = 8.76 Mt);
  C3  revenue neutrality as 21 hard upper-level constraints G_t = S_t eps_L - TR_t <= 0, and
      Lemma 1 (beta2 > gamma2) as G_22 = gamma2 - beta2 <= 0;
  C4  per-technology capacity limits eps_i(t) in [0, cap_i];
  C5  the follower's problem is solved in closed form (Theorem 2 of the source, clipped to the
      capacity box); one nested solve is one 21-year simulation. The follower's response is unique,
      so the strong Stackelberg selection never acts and is not applied here.

Leader x = (alpha1, beta1, gamma1, alpha2, beta2, gamma2, beta3).
Follower y = (eps_H(t), eps_M(t), eps_L(t)), t = 0..20 (63 variables); f = minus the total profit
(the per-period problems are separable, so minimising the sum gives each period's optimum).
"""
import numpy as np
from benchmark import Instance

PAR = dict(P=190.0, cH=23.10, cM=137.50, cL=165.00, capH=35.0, capM=3.5, capL=4.3, eps_max=42.8,
           rH=0.95, rM=0.50, rL=0.05, tau_max=40.0, S0=80.0, delta=0.95, T=20)
HOURS = 8.76
ORDER = ["alpha1", "beta1", "gamma1", "alpha2", "beta2", "gamma2", "beta3"]
X_BOUNDS = np.array([[0.15, 0.25], [0.01, 0.05], [0.32, 8.31], [2.0, 5.0], [0.0, 1.2],
                     [0.8, 1.0], [0.06, 0.10]])
T = np.arange(PAR["T"] + 1.0)
NT = len(T)
CAPS = np.array([PAR["capH"], PAR["capM"], PAR["capL"]])


def _p(x):
    p = dict(PAR)
    p.update(dict(zip(ORDER, np.asarray(x, float))))
    return p


def _D(p):
    return p["beta2"] - p["gamma2"] / (1 + np.exp(-T / p["alpha2"]))


def follower(x):
    """Closed-form follower response (Eqs 15-17 of the source), clipped to the capacity box."""
    p = _p(x)
    eH = np.sqrt(np.maximum(p["P"] - p["cH"] - p["gamma1"] * np.exp(p["alpha1"] * T), 0) / (3 * p["beta1"]))
    d = _D(p)
    # Where D(t) <= 0 (outside Lemma 1) the MCE profit is increasing in eps_M, so the optimum is the
    # capacity; IMCE.m sets eps_M = 0 there, which is not the follower's optimum.
    eM = np.full(NT, np.inf)
    m = d > 0
    eM[m] = p["eps_max"] * (p["P"] - p["cM"]) / (2 * p["tau_max"] * d[m])
    Sb = p["S0"] * np.exp(-p["beta3"] * T)
    eL = 0.5 * p["eps_max"] * (1 + (p["P"] - p["cL"]) / Sb)
    e = np.clip(np.stack([eH, eM, eL], 1), 0, CAPS)
    return e.T.reshape(-1)          # [eH(0..T), eM(0..T), eL(0..T)]


def _split(y):
    y = np.asarray(y, float).reshape(3, NT)
    return y[0], y[1], y[2]


def _taxes(p, eH, eM, eL):
    tH = p["gamma1"] * np.exp(p["alpha1"] * T) + p["beta1"] * eH ** 2
    tM = p["tau_max"] * _D(p) * eM / p["eps_max"]
    S = p["S0"] * np.exp(-p["beta3"] * T) * (1 - eL / p["eps_max"])
    return tH, tM, S


def F(x, y):
    p = _p(x)
    eH, eM, eL = _split(y)
    CE = HOURS * (p["rH"] * eH + p["rM"] * eM + p["rL"] * eL)
    return float(np.sum(p["delta"] ** T * CE))


def G(x, y):
    p = _p(x)
    eH, eM, eL = _split(y)
    tH, tM, S = _taxes(p, eH, eM, eL)
    TR = eH * tH + eM * tM
    return np.append(S * eL - TR, p["gamma2"] - p["beta2"])


def f(x, y):
    p = _p(x)
    eH, eM, eL = _split(y)
    tH, tM, S = _taxes(p, eH, eM, eL)
    profit = (p["P"] * (eH + eM + eL) - p["cH"] * eH - p["cM"] * eM - p["cL"] * eL
              - eH * tH - eM * tM + eL * S)
    return float(-np.sum(profit))


def g(x, y):
    return np.array([])


def instance():
    inst = Instance(id="DBOP_SA", family="DBOP", n_ul=7, n_ll=3 * NT, x_bounds=X_BOUNDS.copy(),
                    y_bounds=np.repeat(np.c_[np.zeros(3), CAPS], NT, axis=0), F=F, f=f, G=G, g=g,
                    scale="", notes="DBOP South Africa case study, recalibrated (C1-C5)")
    inst.ll_exact = follower
    return inst


def patch_exact_ll():
    """Route the shared nested LL solve (M1-M13) and M14's warm-started solve to the closed form
    when the instance provides one; other instances are unaffected."""
    import methods
    import methods_m14
    base, base_warm = methods.ll_solve, methods_m14._ll_solve_warm

    def ll_solve(inst, x, seed=0, **kw):
        if getattr(inst, "ll_exact", None) is None:
            return base(inst, x, seed=seed, **kw)
        y = inst.ll_exact(x)
        return y, inst.F(x, y), inst.G(x, y), True

    def ll_solve_warm(inst, x, seed, xl0=None, xl_near=None):
        if getattr(inst, "ll_exact", None) is None:
            return base_warm(inst, x, seed, xl0=xl0, xl_near=xl_near)
        y = inst.ll_exact(x)
        return y, inst.F(x, y), inst.G(x, y), True, np.array([])

    methods.ll_solve = ll_solve
    methods_m14._ll_solve_warm = ll_solve_warm


def check_follower(n=200, seed=0):
    """Check the closed form against a numerical maximisation of each period's profit."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        x = X_BOUNDS[:, 0] + rng.random(7) * (X_BOUNDS[:, 1] - X_BOUNDS[:, 0])
        y = follower(x)
        r = minimize(lambda z: f(x, z), y + 0.01 * rng.standard_normal(y.size),
                     bounds=list(map(tuple, instance().y_bounds)), method="L-BFGS-B",
                     options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
        worst = max(worst, f(x, y) - r.fun)
    return worst
