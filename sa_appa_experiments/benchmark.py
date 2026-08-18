"""
benchmark.py
============
Exact, real (non-synthetic) implementations of the 44-instance reconciled
benchmark suite from the locked Appendix A (TP1-TP9, with TP7 reformulated
per Task 2; SMD1-SMD12 at three scales, with SMD10@(30,30) excluded per
Task 2's feasibility finding -- see Appendix C of the manuscript).

Every Instance exposes:
    F(x, y)   -> upper-level objective (scalar)
    f(x, y)   -> lower-level objective (scalar)
    G(x, y)   -> array of upper-level constraint values, feasible iff all <= 0
    g(x, y)   -> array of lower-level constraint values, feasible iff all <= 0
    x_bounds  -> (n_ul, 2) array of [lo, hi] per UL dimension
    y_bounds  -> (n_ll, 2) array of [lo, hi] per LL dimension

x, y are 1D numpy arrays of length n_ul, n_ll respectively. All functions
operate on a single point (batched evaluation is done by the caller via
vectorised numpy operations across a population, not inside these
functions, to keep the per-instance definitions simple and directly
checkable against the paper's formulas).

This module is the single source of truth for problem definitions --
nothing here is a placeholder. Every formula below is transcribed
directly from the manuscript's locked Appendix A (Task 2 reconciliation).
"""

import numpy as np
from dataclasses import dataclass, field

EPS = 1e-9


@dataclass
class Instance:
    id: str
    family: str
    n_ul: int
    n_ll: int
    x_bounds: np.ndarray
    y_bounds: np.ndarray
    F: callable
    f: callable
    G: callable
    g: callable
    scale: str = ""
    notes: str = ""


def _rosenbrock(z):
    z = np.asarray(z, dtype=float)
    return np.sum((z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2)


def _bounds(lo, hi, n):
    return np.array([[lo, hi]] * n, dtype=float)


# ---------------------------------------------------------------------------
# TP1-TP9
# ---------------------------------------------------------------------------

def _tp_instances():
    insts = []

    # TP1
    insts.append(Instance(
        id="TP1", family="TP", n_ul=2, n_ll=1,
        x_bounds=_bounds(-2, 2, 2), y_bounds=_bounds(-2, 2, 1),
        F=lambda x, y: x[0] ** 2 + x[1] ** 2,
        f=lambda x, y: 2 * x[0] + y[0],
        G=lambda x, y: np.array([]),
        g=lambda x, y: np.array([x[0] ** 2 + y[0] ** 2 - 4, x[0] - y[0]]),
    ))

    # TP2
    insts.append(Instance(
        id="TP2", family="TP", n_ul=2, n_ll=3,
        x_bounds=_bounds(0, 10, 2), y_bounds=_bounds(0, 10, 3),
        F=lambda x, y: (1 + x[0] - x[1] + 2 * y[1]) * (8 - x[0] - 2 * y[0] + y[1] + 5 * y[2]),
        f=lambda x, y: 2 * y[0] - y[1] + y[2],
        G=lambda x, y: np.array([]),
        g=lambda x, y: np.array([
            -y[0] + y[1] + y[2] - 1,
            2 * x[0] - y[0] + 2 * y[1] - 0.5 * y[2] - 1,
            -(2 * x[1] + 2 * y[0] - y[1] - 0.5 * y[2] - 1),
        ]),
    ))

    # TP3 (box [0,15]^2 already includes the audited x>=0 lower bound)
    insts.append(Instance(
        id="TP3", family="TP", n_ul=2, n_ll=2,
        x_bounds=_bounds(0, 15, 2), y_bounds=_bounds(0, 10, 2),
        F=lambda x, y: (x[0] - 30) ** 2 + (x[1] - 20) ** 2 - 20 * y[0] + 20 * y[1],
        f=lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2,
        G=lambda x, y: np.array([-(x[0] + 2 * x[1] - 30), x[0] + x[1] - 25]),
        g=lambda x, y: np.array([]),
    ))

    # TP4
    insts.append(Instance(
        id="TP4", family="TP", n_ul=2, n_ll=2,
        x_bounds=_bounds(0, 10, 2), y_bounds=_bounds(0, 10, 2),
        F=lambda x, y: -x[0] ** 2 - 3 * x[1] ** 2 - 4 * y[0] + y[1] ** 2,
        f=lambda x, y: 2 * x[0] ** 2 + y[0] ** 2 - 5 * y[1],
        G=lambda x, y: np.array([x[0] ** 2 + 2 * x[1] - 4]),
        g=lambda x, y: np.array([
            -(x[0] ** 2 - 2 * x[0] + x[1] ** 2 - 2 * y[0] + y[1] + 3),
            4 - (x[1] + 3 * y[0] - 4 * y[1]),
        ]),
    ))

    # TP5
    insts.append(Instance(
        id="TP5", family="TP", n_ul=2, n_ll=2,
        x_bounds=_bounds(0, 10, 2), y_bounds=_bounds(0, 10, 2),
        F=lambda x, y: -((x[0] + y[0]) * (x[1] + y[1])) / (1 + x[0] * y[0] + x[1] * y[1] + EPS),
        f=lambda x, y: -((x[0] + y[0]) * (x[1] + y[1])) / (1 + x[0] * y[0] + x[1] * y[1] + EPS),
        G=lambda x, y: np.array([x[0] - x[1], x[0] ** 2 + x[1] ** 2 - 100]),
        g=lambda x, y: np.array([-y[0], y[0] - x[0], -y[1], y[1] - x[1]]),
        notes="x>=0 enforced via box lower bound 0",
    ))

    # TP6: n_ul=1 (x2), n_ll=2 (x1,x3); f does not depend on x2 by the paper's own formula
    insts.append(Instance(
        id="TP6", family="TP", n_ul=1, n_ll=2,
        x_bounds=_bounds(np.log(0.1), np.log(10), 1),
        y_bounds=_bounds(np.log(0.1), np.log(10), 2),
        F=lambda x, y: np.exp(x[0]),
        f=lambda x, y: np.exp(-y[0]) + 0.5864 * np.exp(0.6 * y[0]),
        G=lambda x, y: np.array([]),
        g=lambda x, y: np.array([
            0.0332 * np.exp(y[1]) + 0.1 * np.exp(y[0]) - 1,
            4 * np.exp(x[0] - y[1]) + 2 * np.exp(-0.71 * x[0] - y[1]) + 0.0332 * np.exp(-1.3 * x[0]) - 1,
        ]),
        notes="y=(x1,x3) in the paper's own variable names",
    ))

    # TP7 (reformulated per Task 2: arg max negated to arg min)
    insts.append(Instance(
        id="TP7", family="TP", n_ul=2, n_ll=2,
        x_bounds=_bounds(0, 1, 2), y_bounds=_bounds(0, 1, 2),
        F=lambda x, y: (x[0] - 1) ** 2 + (x[1] - 1) ** 2 + (y[0] - 1) ** 2,
        f=lambda x, y: -(y[0] - x[0] ** 2) ** 2 - (y[1] - x[1] ** 2) ** 2,
        G=lambda x, y: np.array([x[0] ** 2 + x[1] ** 2 - 1]),
        g=lambda x, y: np.array([]),
        notes="reformulated (Task 2): known theta(x) discontinuity at x1=1/sqrt(2); "
              "excluded from convergence-theorem scope, kept as adversarial test case",
    ))

    # TP8
    insts.append(Instance(
        id="TP8", family="TP", n_ul=2, n_ll=2,
        x_bounds=_bounds(0, 50, 2), y_bounds=_bounds(-10, 20, 2),
        F=lambda x, y: abs(2 * x[0] + 2 * x[1] - 3 * y[0] - 3 * y[1] - 60),
        f=lambda x, y: (y[0] - x[0] + 20) ** 2 + (y[1] - x[1] + 20) ** 2,
        G=lambda x, y: np.array([x[0] + x[1] + y[0] - 2 * y[1] - 40]),
        g=lambda x, y: np.array([10 - x[0] + 2 * y[0], 10 - x[1] + 2 * y[1]]),
    ))

    # TP9
    def _tp9_f(x, y):
        yy = np.where(np.abs(y) < EPS, EPS, y)
        idx = np.arange(1, len(x) + 1)
        return np.exp(1 + (1 / 4000) * np.sum(x ** 2 * y ** 2)
                      - np.prod(np.cos(x / (yy * np.sqrt(idx)))))

    insts.append(Instance(
        id="TP9", family="TP", n_ul=10, n_ll=10,
        x_bounds=_bounds(-5, 5, 10), y_bounds=_bounds(-np.pi, np.pi, 10),
        F=lambda x, y: np.sum(np.abs(x - 1) + np.abs(y)),
        f=_tp9_f,
        G=lambda x, y: np.array([]),
        g=lambda x, y: np.array([]),
    ))

    return insts


# ---------------------------------------------------------------------------
# SMD1-SMD12
# ---------------------------------------------------------------------------

def _split(x, y, p, r, q, s=0):
    xu1, xu2 = x[:p], x[p:p + r]
    xl1, xl2 = y[:q + s], y[q + s:q + s + r]
    return xu1, xu2, xl1, xl2


def _smd_instance(base, p, q, r=2, s=0):
    xu1_b = _bounds(-5, 10, p)
    xu2_b = _bounds(-5, 10, r)
    xl1_b = _bounds(-5, 10, q + s)
    xl2_b = _bounds(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, r)

    if base == "SMD1":
        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum(xl1 ** 2) + np.sum(xu2 ** 2) + np.sum((xu2 - np.tan(xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum(xl1 ** 2) + np.sum((xu2 - np.tan(xl2)) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD2":
        xu2_b = _bounds(-5, 1, r)
        xl2_b = _bounds(1e-6, np.e, r)

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) - np.sum(xl1 ** 2) + np.sum(xu2 ** 2) - np.sum((xu2 - np.log(xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum(xl1 ** 2) + np.sum((xu2 - np.log(xl2)) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD3":
        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum(xl1 ** 2) + np.sum(xu2 ** 2) + np.sum((xu2 ** 2 - np.tan(xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            rastrigin = q + np.sum(xl1 ** 2 - np.cos(2 * np.pi * xl1))
            return np.sum(xu1 ** 2) + rastrigin + np.sum((xu2 ** 2 - np.tan(xl2)) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD4":
        xu2_b = _bounds(-1, 1, r)
        xl2_b = _bounds(0, np.e, r)

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) - np.sum(xl1 ** 2) + np.sum(xu2 ** 2) - np.sum((np.abs(xu2) - np.log(1 + xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            rastrigin = q + np.sum(xl1 ** 2 - np.cos(2 * np.pi * xl1))
            return np.sum(xu1 ** 2) + rastrigin + np.sum((np.abs(xu2) - np.log(1 + xl2)) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD5":
        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) - _rosenbrock(xl1) + np.sum(xu2 ** 2) - np.sum((np.abs(xu2) - xl2 ** 2) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + _rosenbrock(xl1) + np.sum((np.abs(xu2) - xl2 ** 2) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD6":
        xl1_b = _bounds(-5, 10, q + s)

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q, s)
            return (np.sum(xu1 ** 2) - np.sum(xl1[:q] ** 2) + np.sum(xl1[q:q + s] ** 2)
                    + np.sum(xu2 ** 2) - np.sum((xu2 - xl2) ** 2))

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q, s)
            tail = xl1[q:q + s]
            f2 = np.sum(xl1[:q] ** 2) + (np.sum((tail[1:] - tail[:-1]) ** 2) if s > 1 else 0.0)
            return np.sum(xu1 ** 2) + f2 + np.sum((xu2 - xl2) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD7":
        xu2_b = _bounds(-5, 1, r)
        xl2_b = _bounds(1e-6, np.e, r)

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            idx = np.arange(1, p + 1)
            griewank = 1 + (1 / 400) * np.sum(xu1 ** 2) - np.prod(np.cos(xu1 / np.sqrt(idx)))
            return griewank - np.sum(xl1 ** 2) + np.sum(xu2 ** 2) - np.sum((xu2 - np.log(xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 3) + np.sum(xl1 ** 2) + np.sum((xu2 - np.log(xl2)) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD8":
        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            ackley = (20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(xu1 ** 2)))
                      - np.exp(np.mean(np.cos(2 * np.pi * xu1))))
            return ackley - _rosenbrock(xl1) + np.sum(xu2 ** 2) - np.sum((xu2 - xl2 ** 3) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(np.abs(xu1)) + _rosenbrock(xl1) + np.sum((xu2 - xl2 ** 3) ** 2)
        G = lambda x, y: np.array([])
        g = lambda x, y: np.array([])

    elif base == "SMD9":
        xu2_b = _bounds(-5, 1, r)
        xl2_b = _bounds(-1 + 1e-6, -1 + np.e, r)
        a = b = 1.0

        def _ann(u1, u2, a, b):
            s_ = (np.sum(u1 ** 2) + np.sum(u2 ** 2)) / a
            return s_ - np.floor(s_ + 1 / (2 * b))

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) - np.sum(xl1 ** 2) + np.sum(xu2 ** 2) - np.sum((xu2 - np.log(1 + xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum(xl1 ** 2) + np.sum((xu2 - np.log(1 + xl2)) ** 2)

        def G(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.array([-_ann(xu1, xu2, a, b)])

        def g(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.array([-_ann(xl1, xl2, a, b)])

    elif base == "SMD10":
        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum((xu1 - 2) ** 2) + np.sum(xl1 ** 2) + np.sum((xu2 - 2) ** 2) - np.sum((xu2 - np.tan(xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum((xl1 - 2) ** 2) + np.sum((xu2 - np.tan(xl2)) ** 2)

        def G(x, y):
            # vectorised, identical formula to the per-j loop version:
            # -(xu1_j - (sum(xu1^3)-xu1_j^3) - sum(xu2^3)) for j=1..p, then
            # -(xu2_j - (sum(xu2^3)-xu2_j^3) - sum(xu1^3)) for j=1..r
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            su1, su2 = np.sum(xu1 ** 3), np.sum(xu2 ** 3)
            out1 = -(xu1 - (su1 - xu1 ** 3) - su2)
            out2 = -(xu2 - (su2 - xu2 ** 3) - su1)
            return np.concatenate([out1, out2])

        def g(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            sl1 = np.sum(xl1 ** 3)
            return -(xl1 - (sl1 - xl1 ** 3))

    elif base == "SMD11":
        xu2_b = _bounds(-1, 1, r)
        xl2_b = _bounds(1 / np.e, np.e, r)

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) - np.sum(xl1 ** 2) + np.sum(xu2 ** 2) - np.sum((xu2 - np.log(xl2)) ** 2)

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum(xl1 ** 2) + np.sum((xu2 - np.log(xl2)) ** 2)

        def G(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            arg = np.maximum(1.0 / r + np.log(xl2), 0.0)  # domain safety guard (implementation note)
            return np.sqrt(arg) - xu2

        def g(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.array([1.0 - np.sum((xu2 - np.log(xl2)) ** 2)])

    elif base == "SMD12":
        xu2_b = _bounds(-14.10, 14.10, r)
        xl2_b = _bounds(-1.5 + 1e-6, 1.5 - 1e-6, r)

        def F(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return (np.sum((xu1 - 2) ** 2) + np.sum(xl1 ** 2) + np.sum((xu2 - 2) ** 2)
                    + np.sum(np.tan(np.abs(xl2))) - np.sum((xu2 - np.tan(xl2)) ** 2))

        def f(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            return np.sum(xu1 ** 2) + np.sum((xl1 - 2) ** 2) + np.sum((xu2 - np.tan(xl2)) ** 2)

        def G(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            out0 = np.tan(xl2) - xu2
            su1, su2 = np.sum(xu1 ** 3), np.sum(xu2 ** 3)
            out1 = -(xu1 - (su1 - xu1 ** 3) - su2)
            out2 = -(xu2 - (su2 - xu2 ** 3) - su1)
            return np.concatenate([out0, out1, out2])

        def g(x, y):
            xu1, xu2, xl1, xl2 = _split(x, y, p, r, q)
            sl1 = np.sum(xl1 ** 3)
            out0 = np.array([1.0 - np.sum((xu2 - np.tan(xl2)) ** 2)])
            out1 = -(xl1 - (sl1 - xl1 ** 3))
            return np.concatenate([out0, out1])

    else:
        raise ValueError(base)

    n_ul = p + r
    n_ll = (q + s) + r
    x_bounds = np.vstack([xu1_b, xu2_b])
    y_bounds = np.vstack([xl1_b, xl2_b])
    return n_ul, n_ll, x_bounds, y_bounds, F, f, G, g


def _smd_instances():
    scales = [(3, 3), (10, 10), (30, 30)]
    r = 2
    out = []
    for i in range(1, 13):
        base = f"SMD{i}"
        s = 2 if base == "SMD6" else 0
        for (p, q) in scales:
            if base == "SMD10" and (p, q) == (30, 30):
                continue  # excluded, Task 2 / Appendix C
            n_ul, n_ll, xb, yb, F, f, G, g = _smd_instance(base, p, q, r=r, s=s)
            out.append(Instance(
                id=f"{base}_p{p}q{q}", family="SMD", n_ul=n_ul, n_ll=n_ll,
                x_bounds=xb, y_bounds=yb, F=F, f=f, G=G, g=g,
                scale=f"({p},{q})",
            ))
    return out


def build_suite():
    insts = _tp_instances() + _smd_instances()
    assert len(insts) == 44, f"expected 44 reconciled instances, got {len(insts)}"
    return {inst.id: inst for inst in insts}


SUITE = build_suite()
INSTANCE_IDS = list(SUITE.keys())

if __name__ == "__main__":
    print(f"{len(SUITE)} instances built.")
    rng = np.random.default_rng(0)
    for iid, inst in SUITE.items():
        x = rng.uniform(inst.x_bounds[:, 0], inst.x_bounds[:, 1])
        y = rng.uniform(inst.y_bounds[:, 0], inst.y_bounds[:, 1])
        try:
            Fv = inst.F(x, y)
            fv = inst.f(x, y)
            Gv = inst.G(x, y)
            gv = inst.g(x, y)
            assert np.isfinite(Fv) and np.isfinite(fv)
            assert np.all(np.isfinite(Gv)) and np.all(np.isfinite(gv))
        except Exception as e:
            print(f"FAIL {iid}: {e}")
            raise
    print("All instances evaluate to finite values at a random feasible-box point. OK.")
