#!/usr/bin/env python3
"""
SA_APPA_4nd.py
==============
Single-file Python version of the SA-APPA experiment codebase (resubmission state, October 2026):
"A Surrogate-Assisted Prey-Predator Algorithm for Constrained Bilevel Problems: A Controlled
Component Study under a Matched Budget" (Haroon, Ngnotchouye and Tilahun).

Run it as

    python SA_APPA_4nd.py                       # use the stored results, rebuild statistics and figures
    python SA_APPA_4nd.py --run-sweep           # (re)run all 14 configurations x 44 instances x 30 runs
    python SA_APPA_4nd.py --quick               # smoke test: 3 instances, 3 runs, all 14 configurations
    python SA_APPA_4nd.py --extra-figs          # also draw the Phase 7 figures (MAKE_EXTRA_FIGS)
    python SA_APPA_4nd.py --validate-only       # self-validation report only

WHAT IS IN THIS FILE
  The file is assembled by tools/build_sa_appa_4nd.py from the repository modules. The algorithm,
  benchmark and statistics code (benchmark.py, gp.py, ll_solver.py, methods.py, methods_m14.py,
  scenarios.py, friedman_test.py, analysis.py, analysis_extra.py) is copied verbatim apart from the
  imports, so it is the code that produced the reported results. The drivers, the figure code and
  the Phase 7 code follow, with a few names changed where two modules used the same one.

    configurations   M1-M9 internal ablation, M10-M13 in-house baselines (M12 and M13 are the
                     same nested GA with different seed streams), M14 matched-budget
                     re-implementation of BL-SAEA (Jiang et al., IEEE TCYB 54(7), 2024)
    scenarios        eight scenario-level Friedman tests (Conover's F-approximation), Conover and
                     Nemenyi post-hoc procedures, feasibility-rate and constraint-density tests,
                     Holm-Bonferroni correction across the sixteen omnibus tests
    figures          paper figures 1-4 (split ablation / Phase II views) and the eight CD diagrams
    Phase 7          (MAKE_EXTRA_FIGS) surrogate-fidelity figures (TP3 ground truth, the GP of
                     theta that SA-APPA held, constraint maps, held-out RMSE and +/- 2 sigma
                     coverage), budget composition (UL + LL evaluations, instrumented re-run of
                     runs 0-4) and evaluations to target
    self-validation  re-runs a few locked runs and compares them with the stored results, and
                     recomputes every test statistic from the stored CSV files and compares it
                     with results/resub_numbers.json

RESULTS AND OUTPUT
  Results are read from and written to RESULTS_DIR. By default this is
  ./sa_appa_4nd_output/results; if it is empty and the repository's
  sa_appa_experiments/results/ exists next to this file, the stored summary and trajectory
  files are copied from there first (the repository files themselves are never written).
  Figures go to ./sa_appa_4nd_output/figures and ./sa_appa_4nd_output/figures_phase7.

REPRODUCIBILITY
  Run r uses seed r; every random stream inside a run is derived from (r, configuration,
  instance) by an MD5 hash (_stable_seed). Re-running in another NumPy / SciPy version is not
  guaranteed to reproduce every run: a small floating-point difference early in a run can change
  a later decision and send the run elsewhere. In this environment (NumPy 2.5, SciPy 1.18), 14 of
  1,238 re-run runs (TP1-TP9 and SMD at (3, 3), runs 0-4) differed from the stored trajectories by
  more than 1e-6 (relative), some by a large amount (for example M2 on TP9, run 0: 27.77 against
  36.09 stored). All 14 used the prey-predator engine (M1, M2, M3, M7, M8); M4-M6 and M9-M14
  reproduced in every case. The self-validation report states the tolerance and lists every
  deviation; it never edits the stored results.

REQUIREMENTS
  Python 3.10 or later, numpy, scipy, pandas, matplotlib. A full sweep is about 54 CPU-hours
  (18,480 runs); it runs in parallel over processes and can be resumed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import copy
import csv
import itertools
import json
import multiprocessing as mp
import numpy as np
import os
import os as _os
import pandas as pd
import sys
import time
from dataclasses import dataclass, field
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import stats
from scipy.optimize import differential_evolution, lsq_linear, minimize
from scipy.stats import f as f_dist, norm, qmc, rankdata, studentized_range, t as t_dist

# ================================================================================================
# CONFIGURATION (edit here or use the command-line flags)
# ================================================================================================
QUICK_TEST = False          # 3 instances x 3 runs x 14 configurations
RUN_SWEEP = False           # run the experiment; otherwise use the stored results
MAKE_EXTRA_FIGS = False     # Phase 7 figures (Part A fidelity, B1 budget composition, B2 to target)
SELF_VALIDATE = True        # write the self-validation report
N_WORKERS = max(1, (os.cpu_count() or 1))
VALIDATION_RUNS = [("TP1", 0), ("SMD1_p3q3", 0)]   # locked runs re-run for every configuration

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "sa_appa_4nd_output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
P7_DIR = os.path.join(RESULTS_DIR, "phase7")
P7_FIG_DIR = os.path.join(OUTPUT_DIR, "figures_phase7")
LL_COUNTS_CSV = os.path.join(RESULTS_DIR, "ll_counts_subset.csv")
REPO_RESULTS = os.path.join(HERE, "sa_appa_experiments", "results")
STORED_FILES = ["summary.csv", "trajectories.jsonl", "summary_m14.csv", "trajectories_m14.jsonl"]


def configure_paths(output_dir):
    """Point every module-level path at output_dir (call before anything reads or writes)."""
    global OUTPUT_DIR, RESULTS_DIR, FIG_DIR, P7_DIR, P7_FIG_DIR, LL_COUNTS_CSV
    OUTPUT_DIR = os.path.abspath(output_dir)
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    P7_DIR = os.path.join(RESULTS_DIR, "phase7")
    P7_FIG_DIR = os.path.join(OUTPUT_DIR, "figures_phase7")
    LL_COUNTS_CSV = os.path.join(RESULTS_DIR, "ll_counts_subset.csv")
    for d in (RESULTS_DIR, os.path.join(RESULTS_DIR, "scenario_tables"), FIG_DIR, P7_DIR, P7_FIG_DIR):
        os.makedirs(d, exist_ok=True)


# ================================================================================================
# BENCHMARK SUITE (benchmark.py)
# ================================================================================================

# benchmark.py
# ============
# Exact, real (non-synthetic) implementations of the 44-instance reconciled
# benchmark suite from the locked Appendix A (TP1-TP9, with TP7 reformulated
# per Task 2; SMD1-SMD12 at three scales, with SMD10@(30,30) excluded per
# Task 2's feasibility finding -- see Appendix C of the manuscript).
#
# Every Instance exposes:
#     F(x, y)   -> upper-level objective (scalar)
#     f(x, y)   -> lower-level objective (scalar)
#     G(x, y)   -> array of upper-level constraint values, feasible iff all <= 0
#     g(x, y)   -> array of lower-level constraint values, feasible iff all <= 0
#     x_bounds  -> (n_ul, 2) array of [lo, hi] per UL dimension
#     y_bounds  -> (n_ll, 2) array of [lo, hi] per LL dimension
#
# x, y are 1D numpy arrays of length n_ul, n_ll respectively. All functions
# operate on a single point (batched evaluation is done by the caller via
# vectorised numpy operations across a population, not inside these
# functions, to keep the per-instance definitions simple and directly
# checkable against the paper's formulas).
#
# This module is the single source of truth for problem definitions --
# nothing here is a placeholder. Every formula below is transcribed
# directly from the manuscript's locked Appendix A (Task 2 reconciliation).
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


# ================================================================================================
# SURROGATES (gp.py)
# ================================================================================================

# gp.py
# =====
# Lightweight Gaussian Process regressor, implemented directly with numpy
# Cholesky factorisation (Definition 4 of the manuscript: predictive mean
# Eq.(5), predictive variance Eq.(6)), rather than a heavier third-party
# library, so that repeated retraining (once per generation, per Lemma 3)
# is fast at the population/instance/run counts this project needs.
#
# Kernel: squared-exponential (RBF) with automatic relevance determination
# disabled (single lengthscale per surrogate, set once via the median
# heuristic on the initial design and held fixed thereafter -- this keeps
# retraining to "refactorise the Cholesky of the growing training set" per
# Assumption A4 / Lemma 3's cost model, rather than re-running maximum
# likelihood hyperparameter estimation at every one of the thousands of
# retraining events this project performs; Assumption A4 explicitly
# permits either "fixed or re-estimated by maximum likelihood" -- we use
# the fixed-hyperparameter branch and say so, matching Lemma 3's own
# "assumes hyperparameters are held fixed" cost-model note added in
# Task 1's clarifying addition).
#
# A simple RBF interpolant (RBFSurrogate) is also provided for M11
# (SPSO-RBF, Table 1's only non-GP surrogate method).
JITTER = 1e-8


class GPSurrogate:
    def __init__(self, lengthscale=None, signal_var=1.0, noise_var=1e-6):
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.X = None
        self.y_mean = 0.0
        self.y = None
        self.L = None
        self.alpha = None

    def _kernel(self, A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return self.signal_var * np.exp(-0.5 * d2 / (self.lengthscale ** 2))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.lengthscale is None:
            # median-heuristic lengthscale, set once on first fit and held fixed
            if X.shape[0] > 1:
                d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
                iu = np.triu_indices_from(d, k=1)
                med = np.median(d[iu]) if len(iu[0]) else 1.0
                self.lengthscale = max(med, 1e-3)
            else:
                self.lengthscale = 1.0
        self.X = X
        self.y_mean = float(np.mean(y))
        yc = y - self.y_mean
        K = self._kernel(X, X) + (self.noise_var + JITTER) * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yc))
        self.y = y
        return self

    def predict(self, Xstar, return_std=True):
        Xstar = np.atleast_2d(Xstar)
        Ks = self._kernel(Xstar, self.X)
        mu = self.y_mean + Ks @ self.alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, Ks.T)
        var = self.signal_var - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, np.sqrt(var)


class RBFSurrogate:
    """Deterministic (multiquadric) RBF interpolant for M11 (SPSO-RBF), the
    only non-GP surrogate in Table 1. Has no native predictive variance;
    for the EI-style infill M11 uses per Table 1, we substitute a
    distance-to-nearest-training-point exploration term in place of GP
    variance, a standard proxy in the RBF-assisted global-optimisation
    literature (e.g. Regis & Shoemaker-style algorithms) -- documented
    here explicitly since the original SPSO-RBF paper's exact acquisition
    formula was not obtainable (paywalled, per Task 3's citation note)."""

    def __init__(self, epsilon=None):
        self.epsilon = epsilon
        self.X = None
        self.w = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X = X
        if self.epsilon is None:
            d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
            iu = np.triu_indices_from(d, k=1)
            self.epsilon = max(np.median(d[iu]) if len(iu[0]) else 1.0, 1e-3)
        Phi = np.sqrt(1.0 + (np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)) / self.epsilon ** 2)
        Phi += JITTER * np.eye(len(X))
        self.w = np.linalg.solve(Phi, y)
        return self

    def predict(self, Xstar, return_std=True):
        Xstar = np.atleast_2d(Xstar)
        d2 = np.sum((Xstar[:, None, :] - self.X[None, :, :]) ** 2, axis=-1)
        Phi = np.sqrt(1.0 + d2 / self.epsilon ** 2)
        mu = Phi @ self.w
        if not return_std:
            return mu
        dist = np.sqrt(np.min(d2, axis=1))
        pseudo_std = dist  # exploration proxy, see class docstring
        return mu, pseudo_std


# ================================================================================================
# LOWER-LEVEL SOLVER (ll_solver.py)
# ================================================================================================

# ll_solver.py
# ============
# The nested, "exact" lower-level solve used by every one of the 13
# methods whenever they need a true (x, y*(x)) pair -- the initial LHS
# design, every infill point for surrogate-assisted methods, and every
# population member for the fully nested baselines (M12, M13).
#
# Implementation choice, documented rather than left implicit: this uses
# scipy.optimize.differential_evolution (a real, independently-validated
# global optimiser) with constraints handled by a quadratic penalty on
# g_l(x,y) > 0, uniformly across all 44 instances and all 13 methods. The
# manuscript does not specify what solves the LL problem "exactly"; using
# one consistent, off-the-shelf, non-fabricated solver for this shared
# subroutine keeps the comparison between M1-M13 fair (they differ only
# in their UL search / constraint-handling / infill choices, which is
# what Section 5 actually studies), and is stated here explicitly rather
# than silently assumed.
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


# ================================================================================================
# CONFIGURATIONS M1-M13 (methods.py)
# ================================================================================================

# methods.py
# ==========
# Real implementations of all 13 Table-1 method configurations.
#
# Budget note (per user sign-off, "shrink the evaluation budget"): the
# manuscript's placeholder budget rule (N0=11*dim-1, T_max=6*dim) was
# extrapolated, from real timing data on this sandbox's 2 CPU cores, to
# require 500+ hours for the full 30-run x 13-method x 44-instance sweep
# -- infeasible. Per explicit user sign-off, this is replaced with a
# SMALLER, FIXED, dimension-independent budget: N0=25 initial LHS points,
# T_max=35 infill/generations, i.e. 60 expensive (nested LL-solve)
# evaluations per run for EVERY instance regardless of scale, matched for
# nested (non-surrogate) methods as N_ul*T_ul=60 (population 6 x 10
# generations). This is a genuine, documented DEVIATION from the
# [FILL IN]-flagged placeholder rule -- not a return to synthetic data.
# Every number produced by this module is a real optimisation-run output.
#
# Every method returns a per-run trajectory: list of dicts with keys
# eval_count, iteration, best_feasible_obj, is_feasible, plus final
# wall-clock CPU time in seconds.
N0 = 25
T_MAX = 35
NEST_POP = 6
NEST_GEN = 10  # NEST_POP * NEST_GEN = 60, matched budget for M12/M13

N_UL_POP = 15  # UL population size for population-based engines


def _stable_seed(*parts, mod=2 ** 31 - 1):
    import hashlib
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % mod


def _lhs(bounds, n, seed):
    d = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    u = sampler.random(n=n)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def _uniform(bounds, n, rng):
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))


def _feasible(Gv, tol=1e-6):
    return bool(np.all(Gv <= tol)) if Gv.size else True


def _ei(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _pof(mu_list, sigma_list):
    if not mu_list:
        return np.ones_like(mu_list[0]) if mu_list else 1.0
    p = 1.0
    for mu, sigma in zip(mu_list, sigma_list):
        sigma = np.maximum(sigma, 1e-12)
        p = p * norm.cdf(-mu / sigma)
    return p


class Trajectory:
    def __init__(self):
        self.rows = []
        self._best = np.inf
        self._found_feasible = False

    def record(self, eval_count, iteration, obj, feasible):
        if feasible and obj < self._best:
            self._best = obj
            self._found_feasible = True
        self.rows.append(dict(
            eval_count=eval_count, iteration=iteration,
            best_feasible_obj=(self._best if self._found_feasible else np.nan),
            is_feasible=self._found_feasible,
        ))

    @property
    def best(self):
        return self._best if self._found_feasible else np.nan

    @property
    def found_feasible(self):
        return self._found_feasible


def _initial_design(inst, seed, init="lhs"):
    """N0 initial UL points, each with an exact nested LL solve. Returns
    Xtr (N0,n_ul), Ftr, Gtr(list of arrays), feas(bool array)."""
    if init == "lhs":
        X = _lhs(inst.x_bounds, N0, seed)
    else:
        rng = np.random.default_rng(seed)
        X = _uniform(inst.x_bounds, N0, rng)
    Ftr, Gtr, feas = [], [], []
    for i, x in enumerate(X):
        y, F, G, ll_feas = ll_solve(inst, x, seed=_stable_seed(seed, "init", i) % (2 ** 31 - 1),
                                     popsize=6, maxiter=20)
        Ftr.append(F)
        Gtr.append(G)
        feas.append(_feasible(G))
    return X, np.array(Ftr), Gtr, np.array(feas)


def _fit_surrogates(X, Ftr, Gtr, m_ul, surrogate_cls=GPSurrogate):
    theta_gp = surrogate_cls().fit(X, Ftr)
    phi_gps = []
    if m_ul > 0:
        Garr = np.array(Gtr)  # (n, m_ul)
        for j in range(m_ul):
            phi_gps.append(surrogate_cls().fit(X, Garr[:, j]))
    return theta_gp, phi_gps


def _predict_cei(cands, theta_gp, phi_gps, f_best):
    mu, sigma = theta_gp.predict(cands)
    ei = _ei(mu, sigma, f_best)
    if phi_gps:
        mus, sigmas = zip(*[g.predict(cands) for g in phi_gps])
        pof = _pof(list(mus), list(sigmas))
    else:
        pof = np.ones_like(mu)
    return ei * pof, mu, sigma


def _predict_penalized_ei(cands, theta_gp, phi_gps, f_best, t, t_max):
    mu, sigma = theta_gp.predict(cands)
    ei = _ei(mu, sigma, f_best)
    if phi_gps:
        pen = np.zeros(len(cands))
        for g in phi_gps:
            mu_g, _ = g.predict(cands)
            pen += np.maximum(mu_g, 0.0) ** 2
        alpha = (t / max(t_max, 1)) ** 2
        return ei - alpha * pen, mu, sigma
    return ei, mu, sigma


# ---------------------------------------------------------------------------
# APPA (and GA/DE/PSO engine variants), used by M1-M11
# ---------------------------------------------------------------------------

def _clip(X, bounds):
    return np.clip(X, bounds[:, 0], bounds[:, 1])


def _appa_step(X, best_idx, worst_idx, rng, lam):
    N, d = X.shape
    Xnew = X.copy()
    p_f = lam["p_f"]
    for i in range(N):
        if i == worst_idx:
            continue
        if rng.uniform() <= p_f:
            r1 = rng.uniform()
            diff = X[best_idx] - X[i]
            n = np.linalg.norm(diff) + 1e-12
            r2 = rng.uniform(size=d)
            Xnew[i] = X[i] + lam["explore"] * r1 * diff / n + lam["local"] * (rng.uniform(size=d) * 2 - 1)
        else:
            r4 = rng.uniform()
            diff = X[i] - X[worst_idx]
            n = np.linalg.norm(diff) + 1e-12
            Xnew[i] = X[i] + lam["evade"] * r4 * diff / n
    r5 = rng.uniform()
    diff = X[best_idx] - X[worst_idx]
    Xnew[worst_idx] = X[worst_idx] + lam["chase"] * r5 * diff + lam["global"] * (rng.uniform(size=d) * 2 - 1)
    return Xnew


def _ga_hybridize(X, fitness, rng, Pc=0.8, Pm=0.1, t=0, t_max=1, sigma0=0.3):
    N, d = X.shape
    order = np.argsort(fitness)
    Xn = X.copy()
    for _ in range(N // 3):
        i, j = rng.choice(N, size=2, replace=False)
        if rng.uniform() < Pc:
            alpha = rng.uniform(size=d)
            off = alpha * X[i] + (1 - alpha) * X[j]
            if rng.uniform() < Pm:
                sigma = sigma0 * (1 - t / max(t_max, 1))
                off = off + rng.normal(0, sigma, size=d)
            worst = order[-1]
            Xn[worst] = off
            order = order[:-1]
    return Xn


def _de_step(X, fitness_fn, rng, F=0.6, CR=0.8):
    N, d = X.shape
    Xn = X.copy()
    for i in range(N):
        idxs = [k for k in range(N) if k != i]
        a, b, c = X[rng.choice(idxs, 3, replace=False)]
        mutant = a + F * (b - c)
        cross = rng.uniform(size=d) < CR
        if not cross.any():
            cross[rng.integers(d)] = True
        trial = np.where(cross, mutant, X[i])
        Xn[i] = trial
    return Xn


def _pso_step(X, V, Pbest, gbest, rng, w=0.7, c1=1.5, c2=1.5):
    N, d = X.shape
    r1, r2 = rng.uniform(size=(N, d)), rng.uniform(size=(N, d))
    Vn = w * V + c1 * r1 * (Pbest - X) + c2 * r2 * (gbest - X)
    Xn = X + Vn
    return Xn, Vn


def run_ablation_method(inst, method_id, engine, constraint_mode, adaptation, init,
                         seed=0, pf_eta=None, pf_min=0.1, pf_max=0.9,
                         beta=None, gamma=None, delta=None, hook=None):
    """Unifies M1-M9 and, with engine set appropriately, M4-M6: one
    surrogate-assisted optimiser parameterised by:
      engine: 'appa' | 'ga' | 'de' | 'pso'
      constraint_mode: 'cei' | 'penalty' | 'none'  (none = unconstrained EI, M8)
      adaptation: 'adaptive' | 'fixed'   (APPA-only; ignored for ga/de/pso)
      init: 'lhs' | 'uniform'
      ga_hybrid: whether periodic GA hybridisation is applied (APPA-only)

    pf_eta, pf_min, pf_max, beta, gamma, delta: OPTIONAL diagnostic-only
    overrides, added for the Scenario 5 tuning diagnostic (Task 29,
    post-sign-off supplementary analysis). Every call made via
    run_method()/METHOD_SPECS (i.e. every run that produced the locked
    Scenario 1-7 results in results/summary.csv) leaves these at their
    defaults, so the original M1-M13 behaviour is completely unchanged.
    When pf_eta is not None AND adaptation=='adaptive', the follow-up
    probability p_f is additionally updated each generation via Eq. (pf):
    p_f <- clip(p_f + pf_eta*(n_success/N_p - p_f), pf_min, pf_max), where
    n_success counts prey whose surrogate-predicted acquisition improved
    that generation -- this closes a real gap versus the manuscript, which
    specifies this update (Eq. 11) but whose original implementation never
    varied p_f at all (see task7 notes / Task 29). beta/gamma/delta, if
    given, override the hardcoded 0.2/0.3/0.3 sensitivity coefficients for
    the lambda_explore/evade/chase updates, for the same diagnostic.
    """
    t0 = time.time()
    ga_hybrid = (method_id != "M3")
    X, Ftr, Gtr, feas = _initial_design(inst, seed, init=init)
    traj = Trajectory()
    eval_count = 0
    for i in range(N0):
        eval_count += 1
        traj.record(eval_count, 0, Ftr[i], feas[i])
    f_best = np.min(Ftr[feas]) if feas.any() else np.inf

    surrogate_cls = RBFSurrogate if method_id == "M11" else GPSurrogate
    m_ul = len(Gtr[0]) if len(Gtr[0]) else 0
    theta_gp, phi_gps = _fit_surrogates(X, Ftr, Gtr, m_ul, surrogate_cls=surrogate_cls)
    if hook is not None:   # read-only snapshot for figures (Phase 7); None in every locked run
        hook("init", X.copy(), Ftr.copy(), [g.copy() for g in Gtr], theta_gp, phi_gps)

    rng = np.random.default_rng(_stable_seed(seed, method_id, inst.id, "pop") % (2 ** 31 - 1))
    Pop = _lhs(inst.x_bounds, N_UL_POP, _stable_seed(seed, method_id, inst.id, "poplhs") % (2 ** 31 - 1)) \
        if init == "lhs" else _uniform(inst.x_bounds, N_UL_POP, rng)
    V = np.zeros_like(Pop)
    Pbest = Pop.copy()
    Pbest_val = None

    lam = dict(explore=0.3, local=0.1, evade=0.3, chase=0.3, global_=0.2, p_f=0.6)
    lam["global"] = lam.pop("global_")
    sigma_init = np.std(Pop)
    tau_div = 0.3
    beta = 0.2 if beta is None else beta
    gamma = 0.3 if gamma is None else gamma
    delta = 0.3 if delta is None else delta

    d_range = inst.x_bounds[:, 1] - inst.x_bounds[:, 0]

    for t in range(T_MAX):
        # score current population via surrogate
        if constraint_mode == "cei":
            scores, mu, sigma = _predict_cei(Pop, theta_gp, phi_gps, f_best)
            acq = scores  # maximize
        elif constraint_mode == "penalty":
            scores, mu, sigma = _predict_penalized_ei(Pop, theta_gp, phi_gps, f_best, t, T_MAX)
            acq = scores
        else:  # none -> plain EI
            mu, sigma = theta_gp.predict(Pop)
            acq = _ei(mu, sigma, f_best)

        best_idx = int(np.argmax(acq))
        worst_idx = int(np.argmin(acq))
        fitness_for_ga = -acq  # lower is better for GA replace-worst logic

        if engine == "appa":
            sigma_pop = float(np.std(Pop))
            if adaptation == "adaptive":
                lam["explore"] *= np.exp(beta * (tau_div - sigma_pop / (sigma_init + 1e-9)))
                delta_fit = (np.max(mu) - np.min(mu)) / (abs(np.max(mu)) + 1e-6)
                delta_fit = np.clip(delta_fit, -1.0, 1.0)
                lam["evade"] *= (1 - gamma * delta_fit)
                lam["chase"] *= (1 + delta * delta_fit)
                lam["explore"] = float(np.clip(lam["explore"], 0.02, 1.0))
                lam["evade"] = float(np.clip(lam["evade"], 0.02, 1.0))
                lam["chase"] = float(np.clip(lam["chase"], 0.02, 1.0))
            else:  # fixed schedule
                frac = 1 - t / max(T_MAX, 1)
                lam["explore"] = 0.3 * frac + 0.02
                lam["evade"] = 0.3 * frac + 0.02
                lam["chase"] = 0.3 * (1 - frac) + 0.02
            lam_scaled = {k: v * (d_range.mean()) for k, v in lam.items() if k != "p_f"}
            lam_scaled["p_f"] = lam["p_f"]
            Pop = _appa_step(Pop, best_idx, worst_idx, rng, lam_scaled)
            if pf_eta is not None and adaptation == "adaptive":
                # Eq. (pf): p_f <- clip(p_f + eta*(n_success/N_p - p_f), pf_min, pf_max).
                # n_success = number of prey whose surrogate-predicted acquisition
                # (mu, minimisation) improved this generation. Diagnostic-only path
                # (Task 29): inactive (pf_eta is None) on every call made through
                # run_method()/METHOD_SPECS, so the locked Scenario 1-7 results are
                # unaffected by this branch's existence.
                mu_new, _ = theta_gp.predict(Pop)
                n_success = int(np.sum(mu_new < mu))
                lam["p_f"] = float(np.clip(
                    lam["p_f"] + pf_eta * (n_success / len(Pop) - lam["p_f"]),
                    pf_min, pf_max))
            if ga_hybrid and (t % 5 == 0):
                rho = 0.3 * np.exp(-float(np.std(Pop)) / (sigma_init + 1e-9))
                if rng.uniform() < rho:
                    Pop = _ga_hybridize(Pop, fitness_for_ga, rng, t=t, t_max=T_MAX,
                                         sigma0=0.15 * d_range.mean())
        elif engine == "ga":
            Pop = _ga_hybridize(Pop, fitness_for_ga, rng, Pc=0.8, Pm=0.2, t=t, t_max=T_MAX,
                                 sigma0=0.2 * d_range.mean())
        elif engine == "de":
            Pop = _de_step(Pop, None, rng, F=0.6, CR=0.8)
        elif engine == "pso":
            if Pbest_val is None:
                Pbest_val = mu.copy()
            improved = mu < Pbest_val
            Pbest[improved] = Pop[improved]
            Pbest_val[improved] = mu[improved]
            gbest = Pop[int(np.argmin(mu))]
            Pop, V = _pso_step(Pop, V, Pbest, gbest, rng)

        Pop = _clip(Pop, inst.x_bounds)

        # infill: pick the current best-acquisition candidate for a true eval
        if constraint_mode == "cei":
            acq2, _, _ = _predict_cei(Pop, theta_gp, phi_gps, f_best)
        elif constraint_mode == "penalty":
            acq2, _, _ = _predict_penalized_ei(Pop, theta_gp, phi_gps, f_best, t, T_MAX)
        else:
            mu2, sigma2 = theta_gp.predict(Pop)
            acq2 = _ei(mu2, sigma2, f_best)
        infill_idx = int(np.argmax(acq2))
        x_infill = Pop[infill_idx]

        y, F, G, ll_feas = ll_solve(inst, x_infill, seed=_stable_seed(seed, method_id, inst.id, t) % (2 ** 31 - 1),
                                     popsize=6, maxiter=20)
        feas_x = _feasible(G)
        eval_count += 1
        if feas_x and F < f_best:
            f_best = F
        traj.record(eval_count, t + 1, F, feas_x)

        X = np.vstack([X, x_infill])
        Ftr = np.append(Ftr, F)
        Gtr = Gtr + [G]
        theta_gp, phi_gps = _fit_surrogates(X, Ftr, Gtr, len(G), surrogate_cls=surrogate_cls)

    if hook is not None:
        hook("final", X.copy(), Ftr.copy(), [g.copy() for g in Gtr], theta_gp, phi_gps)
    cpu_time = time.time() - t0
    return traj, cpu_time


def run_nested_method(inst, method_id, seed=0, constraint_mode="penalty"):
    """M12 (feasibility-rules GA, 'Nested-NSGA-II') and M13 (co-evolutionary
    GA with dynamic penalty, 'CoEA'). No surrogate: every population
    member gets a real, exact nested LL solve every generation. Budget
    matched to NEST_POP*NEST_GEN = 60 expensive evals, same as the
    surrogate methods' N0+T_max."""
    t0 = time.time()
    rng = np.random.default_rng(_stable_seed(seed, method_id, inst.id) % (2 ** 31 - 1))
    X = _lhs(inst.x_bounds, NEST_POP, _stable_seed(seed, method_id, inst.id, "init") % (2 ** 31 - 1))
    traj = Trajectory()
    eval_count = 0
    f_best = np.inf

    def evaluate(Xpop, gen):
        nonlocal eval_count, f_best
        Fs, Gs, feass = [], [], []
        for i, x in enumerate(Xpop):
            y, F, G, _ = ll_solve(inst, x, seed=_stable_seed(seed, method_id, inst.id, gen, i) % (2 ** 31 - 1),
                                   popsize=6, maxiter=20)
            feasx = _feasible(G)
            eval_count += 1
            if feasx and F < f_best:
                f_best = F
            traj.record(eval_count, gen, F, feasx)
            Fs.append(F)
            Gs.append(G)
            feass.append(feasx)
        return np.array(Fs), Gs, np.array(feass)

    Fs, Gs, feass = evaluate(X, 0)

    for gen in range(1, NEST_GEN):
        # feasibility-rules ranking (Deb 2000): feasible-and-better beats infeasible;
        # among infeasible, lower total violation wins
        viol = np.array([np.sum(np.maximum(g, 0.0)) if g.size else 0.0 for g in Gs])
        rank_key = np.where(feass, Fs, 1e12 + viol)
        order = np.argsort(rank_key)
        Xr = X[order]
        Xnew = _ga_hybridize(Xr, rank_key[order], rng, Pc=0.9, Pm=0.15, t=gen, t_max=NEST_GEN,
                              sigma0=0.2 * (inst.x_bounds[:, 1] - inst.x_bounds[:, 0]).mean())
        Xnew = _clip(Xnew, inst.x_bounds)
        X = Xnew
        Fs, Gs, feass = evaluate(X, gen)

    cpu_time = time.time() - t0
    return traj, cpu_time


METHOD_SPECS = {
    "M1": dict(kind="ablation", engine="appa", constraint_mode="cei", adaptation="adaptive", init="lhs"),
    "M2": dict(kind="ablation", engine="appa", constraint_mode="cei", adaptation="adaptive", init="uniform"),
    "M3": dict(kind="ablation", engine="appa", constraint_mode="cei", adaptation="adaptive", init="lhs"),  # ga_hybrid disabled internally
    "M4": dict(kind="ablation", engine="ga", constraint_mode="cei", adaptation="adaptive", init="lhs"),
    "M5": dict(kind="ablation", engine="de", constraint_mode="cei", adaptation="adaptive", init="lhs"),
    "M6": dict(kind="ablation", engine="pso", constraint_mode="cei", adaptation="adaptive", init="lhs"),
    "M7": dict(kind="ablation", engine="appa", constraint_mode="penalty", adaptation="adaptive", init="lhs"),
    "M8": dict(kind="ablation", engine="appa", constraint_mode="none", adaptation="adaptive", init="lhs"),
    "M9": dict(kind="ablation", engine="appa", constraint_mode="cei", adaptation="fixed", init="lhs"),
    "M10": dict(kind="ablation", engine="de", constraint_mode="penalty", adaptation="adaptive", init="lhs"),
    "M11": dict(kind="ablation", engine="pso", constraint_mode="penalty", adaptation="adaptive", init="lhs"),
    "M12": dict(kind="nested"),
    "M13": dict(kind="nested"),
}


def run_method(inst, method_id, seed):
    spec = METHOD_SPECS[method_id]
    if spec["kind"] == "nested":
        return run_nested_method(inst, method_id, seed=seed)
    return run_ablation_method(inst, method_id, spec["engine"], spec["constraint_mode"],
                                spec["adaptation"], spec["init"], seed=seed)


# ================================================================================================
# CONFIGURATION M14, BL-SAEA RE-IMPLEMENTATION (methods_m14.py)
# ================================================================================================

# methods_m14.py
# ==============
# M14: a matched-budget re-implementation of BL-SAEA,
#
#     H. Jiang, K. Chou, Y. Tian, X. Zhang, Y. Jin, "Efficient surrogate
#     modeling method for evolutionary algorithm to solve bilevel
#     optimization problems", IEEE Transactions on Cybernetics 54(7)
#     (2024) 4335-4347, doi:10.1109/TCYB.2023.3309598.
#
# Source used for the algorithmic details: the authors' group's own
# reference implementation in PlatEMO (BIMK/PlatEMO, file
# Algorithms/Multi-objective optimization/BL-SAEA/BLSAEA.m and llSearch.m),
# read directly. The published paper was checked afterwards (October 2026):
# it specifies LL mapping models from x_u to x_l* (Section III-C, Algorithm 4),
# which is what is implemented here; its per-UL-point top-K training set is
# available through run_m14(..., ll_topk=True) (sensitivity run M14K).
#
# This module is ADDITIVE. It does not import-modify methods.py, gp.py or
# ll_solver.py, so the locked M1-M13 results in results/summary.csv are
# untouched (verified by the bit-for-bit re-run check in the Task 14 notes).
#
# What is kept from BL-SAEA (PlatEMO BLSAEA.m):
#   * UL search operator: binary tournament selection, simulated binary
#     crossover (p_c = 1, eta_c = 20) and polynomial mutation
#     (p_m = 1/D, eta_m = 20), producing 2 x N_ul offspring per generation.
#   * UL surrogate: a Kriging (GP) model of the UL objective F trained on the
#     archive of truly evaluated UL points. Offspring are pre-screened by
#     non-dominated sorting on (predicted mean, predicted variance), both
#     minimised, and only the first front receives a full nested LL solve.
#   * LL surrogates: one Kriging model per LL variable, mapping x_u -> x_l*
#     over the archive, used to predict a starting point x_l0 for the LL
#     search at each screened offspring.
#   * LL warm start (llSearch.m, flag=1): the LL initial population is half
#     Latin-hypercube points and half seeded around the predictions: the LL
#     optimum of the archive point closest in cosine similarity (itself and
#     one Gaussian perturbation), Gaussian perturbations of x_l0 (sigma =
#     0.2 x LL range, about half of the coordinates perturbed), and x_l0.
#   * Fitness: feasibility penalty (objective if UL and LL feasible, else
#     total violation + 1e10); (mu+lambda) truncation to N_ul.
#   * Quadratic-approximation local search around the current best UL point
#     each generation, followed by one full nested LL solve at its result,
#     replacing the worst population member.
#
# Deliberate, documented deviations (needed to run under this study's
# matched budget and shared LL solver; none of them favours M14 or M1):
#   1. Budget: 60 full nested LL solves per run, as for every other method
#      (25 LHS points shared with M1's initial design + 35 further solves).
#      The screened first front is truncated, by predicted mean, to the
#      remaining budget.
#   2. LL solver: the same scipy differential_evolution call used by every
#      other method (popsize=6, maxiter=20, polish=True), with only its
#      initial population replaced by the BL-SAEA warm start. PlatEMO's
#      llSearch runs its own GA for up to 2000 generations; that is not
#      comparable with the other 13 methods' LL effort and is not used.
#   3. Local search: PlatEMO minimises a quadratic approximation of the UL
#      objective with SQP while calling the true UL objective at
#      approximated LL responses. Those true calls are not counted as
#      expensive in BL-SAEA but would be uncounted extra information here,
#      so this re-implementation minimises the quadratic approximations of
#      the UL objective and UL constraints only (scipy SLSQP inside the UL
#      box), then spends one counted nested LL solve at the result.
#   4. LL mapping models: PlatEMO's code fits the per-LL-variable models on
#      (archive x_u, archive F), which appears to be an indexing slip, since
#      the paper describes the models as mapping x_u to the LL optimum. This
#      re-implementation fits them on (archive x_u, archive x_l*), as the
#      paper describes (confirmed against Jiang et al. 2024, Section III-C and Algorithm 4).
#   5. Kriging: the study's own GPSurrogate (squared-exponential kernel) is
#      used, but with its lengthscale chosen by maximum marginal likelihood
#      over a log-spaced grid at every refit, since DACE Kriging (used by
#      BL-SAEA) estimates its correlation parameter by maximum likelihood.
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


# ================================================================================================
# THE EIGHT SCENARIOS (scenarios.py)
# ================================================================================================

# scenarios.py
# ============
# The 7 experimental scenarios exactly as specified by the user, each an
# independent comparison (own Friedman test, not pooled into one omnibus
# test across all 13 methods).
#
# "Reconciled problem instances relevant to that scenario's benchmark
# group" is read as: the full 44-instance reconciled suite for every
# scenario, since Phase I / Phase II in the manuscript split the methods
# into two research questions, not the instances into two benchmark
# groups -- every method (M1-M13) is defined to run on the same suite.
# Flagged explicitly here since it's an interpretive choice, not a
# recovered fact.
SCENARIOS = [
    dict(num=1, phase="I", name="Initial design",
         methods=["M1", "M2"], isolates="LHS vs uniform init"),
    dict(num=2, phase="I", name="GA hybridisation",
         methods=["M1", "M3"], isolates="crossover/mutation contribution"),
    dict(num=3, phase="I", name="Search engine",
         methods=["M3", "M4", "M5", "M6"],
         isolates="PPA engine vs GA/PSO/DE (M3, bare PPA, used as the PPA "
                   "representative so Scenario 2's genetic-operator effect "
                   "is not confounded with the engine comparison)"),
    dict(num=4, phase="I", name="Constraint handling",
         methods=["M1", "M7", "M8"], isolates="CEI vs dynamic penalty vs unconstrained EI"),
    dict(num=5, phase="I", name="Step-size rule",
         methods=["M1", "M9"], isolates="adaptive vs fixed parameter schedule"),
    dict(num=6, phase="II", name="Surrogate-assisted engine baselines",
         methods=["M1", "M10", "M11"],
         isolates="SA-APPA vs in-house surrogate-assisted DE (GP) and PSO (RBF) baselines "
                  "with penalised-EI constraint handling (M10/M11 are NOT re-implementations "
                  "of published methods; see methods.py METHOD_SPECS)"),
    dict(num=7, phase="II", name="Fully nested baseline",
         methods=["M1", "M12", "M13"],
         isolates="SA-APPA vs a fully nested GA with feasibility rules; M12 and M13 run the "
                  "SAME algorithm (methods.run_nested_method) with different seed streams"),
    dict(num=8, phase="II", name="Published surrogate-assisted bilevel method",
         methods=["M1", "M14"],
         isolates="SA-APPA vs a matched-budget re-implementation of BL-SAEA "
                  "(Jiang et al., IEEE TCYB 2024)"),
]

SCENARIO_BY_NUM = {s["num"]: s for s in SCENARIOS}


# ================================================================================================
# FRIEDMAN / CONOVER / NEMENYI (friedman_test.py)
# ================================================================================================

# friedman_test.py
# =================
# Standalone implementation of the Friedman test with Conover's (1998)
# F-approximation, and its associated post-hoc paired critical-difference
# procedure, exactly as presented in the attached methodology note
# (Villegas, based on Conover 1998, pp. 294-302) -- Task 5.
#
# This module is deliberately independent of the rest of the figures
# package (it only needs a "rank matrix": one row per block/instance, one
# column per treatment/method, entries = within-block ranks). It is used
# by two callers:
#
# 1. The Task-5 analysis in this file's own `__main__` / notes, applied to
#    Phase I and Phase II separately (see task5-friedman-test.md).
# 2. `figures/fig5_cd_diagram.py`, which previously (Task 4) implemented
#    its OWN, INCORRECT, version of this statistic -- see "Bug found and
#    fixed" below. It now imports `friedman_f_statistic` from here instead
#    of recomputing it.
#
# Formulae (Conover 1998, via the Friedmantest.pdf / Villegas note)
# -------------------------------------------------------------------
# Let n = number of blocks (instances), k = number of treatments (methods),
# R(X_ij) = rank of treatment j within block i (average ranks on ties,
# 1 = best), R_j = sum_i R(X_ij) = rank sum for treatment j.
#
#     A2 = sum_i sum_j R(X_ij)^2            ("A_2" in the note)
#     B2 = sum_j R_j^2                      ("B_2" in the note)
#     C1 = n * k * (k+1)^2 / 4               (correction term)
#
#     SST          = A2 - C1                 (total sum of squares of ranks)
#     SS_treatments = B2 / n - C1            (between-treatments sum of squares)
#     SS_error      = SST - SS_treatments = A2 - B2 / n
#
#     T2 = (n - 1) * SS_treatments / SS_error
#        = (n - 1) * (B2/n - C1) / (A2 - B2/n)
#
# T2 is referred to the F distribution with df1 = k-1, df2 = (n-1)(k-1)
# (Conover's F-approximation -- preferred over the older chi-square
# approximation because it is more accurate for typical n, k; the note is
# explicit about this preference).
#
# Post-hoc paired comparison (same note): two treatments i, j are declared
# significantly different at level alpha if
#
#     |R_i - R_j| > t_{1-alpha/2, (n-1)(k-1)} * sqrt( 2*n*SS_error / ((n-1)(k-1)) )
#
# i.e. CD_conover (in RANK-SUM units, matching the note's own worked
# example, which reports and compares raw rank-sum differences like
# |80-48|=32, not mean-rank differences).
#
# IMPORTANT -- this is NOT the Nemenyi critical distance used by the CD
# diagram in Figure 5 (Task 4). Nemenyi's CD is a different, more
# conservative, procedure conventionally used to draw CD diagrams
# (Demsar, 2006), operating on MEAN ranks rather than rank sums and using
# the studentized range distribution rather than the t distribution. The
# two are related (both test rank-sum/mean-rank differences under the
# Friedman omnibus model) but are not the same number and must not be
# reported as if interchangeable. Per the user's explicit Task 5
# instruction: the omnibus test reported here uses Conover's
# F-approximation; the CD diagram's post-hoc bars use Nemenyi. See
# `nemenyi_cd()` below for the (unchanged, already-correct) Nemenyi
# formula, included here only for the side-by-side comparison in
# task5-friedman-test.md -- `figures/fig5_cd_diagram.py` keeps its own
# copy for the diagram itself, unchanged.
#
# Bug found and fixed
# --------------------
# Task 4's `figures/fig5_cd_diagram.py._friedman_stat()` computed:
#
#     denom = A1 - C1                        # = SST, not SS_error
#     T = (n_instances - 1) * ss_ranks / denom     # ss_ranks = B2 - n*C1*4/(k(k+1)^2)... actually
#                                                    # ss_ranks as coded = sum_j (R_j - n(k+1)/2)^2
#                                                    # which equals B2 - n(k+1)*sum(R_j) + n_k*(n(k+1)/2)^2
#                                                    # and, since sum(R_j) = n*k*(k+1)/2 identically,
#                                                    # reduces to B2 - n*k*(k+1)^2/4 = B2 - C1,
#                                                    # i.e. n*SS_treatments, NOT SS_treatments itself.
#
# So the old code's numerator was n times too large AND its denominator
# was SST instead of SS_error -- on the worked example (n=21, k=4) this
# produced T=236.38 against the correct T2=25.74, an ~9.2x inflation.
# Validated below against the note's own worked example.
def rank_matrix_from_wide(wide):
    """
    wide: 2D array-like, shape (n_blocks, k_treatments), raw scores (lower
    = better, i.e. minimisation -- matching SA-APPA's objective
    convention). Returns the within-block rank matrix (average ranks on
    ties, 1 = best), same shape.
    """
    wide = np.asarray(wide, dtype=float)
    return np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, wide)


def friedman_f_statistic(rank_matrix):
    """
    rank_matrix: 2D array, shape (n_blocks, k_treatments), already ranks
    (1 = best) within each row/block.

    Returns a dict with A2, B2, C1, SST, SS_treatments, SS_error, T2,
    df1, df2, p_value, rank_sums (length k), n, k -- everything needed to
    reproduce the note's worked-example presentation.
    """
    rank_matrix = np.asarray(rank_matrix, dtype=float)
    n, k = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)

    A2 = float((rank_matrix ** 2).sum())
    B2 = float((rank_sums ** 2).sum())
    C1 = n * k * (k + 1) ** 2 / 4.0

    SST = A2 - C1
    SS_treatments = B2 / n - C1
    SS_error = SST - SS_treatments  # == A2 - B2/n

    if SS_error <= 0:
        T2, p_value = np.nan, np.nan
    else:
        T2 = (n - 1) * SS_treatments / SS_error
        df1, df2 = k - 1, (n - 1) * (k - 1)
        p_value = 1 - f_dist.cdf(T2, df1, df2)

    df1, df2 = k - 1, (n - 1) * (k - 1)
    return {
        "n": n, "k": k, "rank_sums": rank_sums,
        "A2": A2, "B2": B2, "C1": C1,
        "SST": SST, "SS_treatments": SS_treatments, "SS_error": SS_error,
        "T2": T2, "df1": df1, "df2": df2, "p_value": p_value,
    }


def f_critical(df1, df2, alpha):
    return f_dist.ppf(1 - alpha, df1, df2)


def conover_posthoc_cd(stat, alpha=0.05):
    """
    Conover's post-hoc critical difference on |R_i - R_j| (RANK-SUM
    units, matching the note's worked example), using the SS_error from
    `friedman_f_statistic()`'s output dict (`stat`).
    """
    n, k = stat["n"], stat["k"]
    df2 = stat["df2"]
    t_crit = t_dist.ppf(1 - alpha / 2, df2)
    cd = t_crit * np.sqrt(2 * n * stat["SS_error"] / df2)
    return cd, t_crit


def nemenyi_cd(k, n, alpha=0.05):
    """
    Nemenyi critical distance on MEAN ranks (the convention used by
    figures/fig5_cd_diagram.py for the CD diagram) -- included here only
    for side-by-side documentation; NOT used to compute conover_posthoc_cd
    above, and not a substitute for it. q_alpha via the exact studentized
    range distribution (scipy), not a hand-typed table.
    """
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def pairwise_table(stat, alpha_list=(0.05, 0.01)):
    """
    Build the note's-style pairwise |R_i - R_j| summary: for every method
    pair, the rank-sum difference and whether it exceeds the Conover CD
    at each alpha in alpha_list. Returns a list of dicts.
    method_labels: optional; if omitted, treatments are 0..k-1.
    """
    k = stat["k"]
    rs = stat["rank_sums"]
    rows = []
    cds = {a: conover_posthoc_cd(stat, a)[0] for a in alpha_list}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(rs[i] - rs[j])
            row = {"i": i, "j": j, "abs_diff": diff}
            for a in alpha_list:
                row[f"significant_{a}"] = diff > cds[a]
            rows.append(row)
    rows.sort(key=lambda r: -r["abs_diff"])
    return rows, cds


# ---------------------------------------------------------------------------
# Validation against the Friedmantest.pdf worked example
# (n=21 instances/blocks, k=4 metaheuristics, rank sums R=[80,48,51,31])
# ---------------------------------------------------------------------------

def _validate_against_worked_example(verbose=True):
    n, k = 21, 4
    rank_sums = np.array([80.0, 48.0, 51.0, 31.0])
    A2_expected = 630.0
    B2_expected = 12266.0

    # We don't have the raw per-block rank matrix (not reproduced in the
    # extracted text), but A2 and B2 are given directly in the worked
    # example, so validate the formula algebraically from A2/B2 rather
    # than from a synthesised rank matrix.
    C1 = n * k * (k + 1) ** 2 / 4.0
    SST = A2_expected - C1
    SS_treatments = B2_expected / n - C1
    SS_error = SST - SS_treatments
    T2 = (n - 1) * SS_treatments / SS_error
    df1, df2 = k - 1, (n - 1) * (k - 1)
    p_value = 1 - f_dist.cdf(T2, df1, df2)

    f_crit_05 = f_critical(df1, df2, 0.05)
    f_crit_01 = f_critical(df1, df2, 0.01)

    stat = {"n": n, "k": k, "rank_sums": rank_sums, "SS_error": SS_error, "df2": df2}
    # NOTE: the worked example's post-hoc CD (~15.08, t_crit~2.660) turns
    # out to correspond to alpha=0.01 two-sided (t_{0.995,60}), not
    # alpha=0.05 (t_{0.975,60}=2.000) -- i.e. the note applies the
    # post-hoc comparison at the *stricter* of the two levels it tested
    # the omnibus statistic against, having rejected H0 at both 0.05 and
    # 0.01. Validate against alpha=0.01 accordingly.
    cd_01, t_crit_01 = conover_posthoc_cd(stat, 0.01)

    if verbose:
        print("=== Validation against Friedmantest.pdf worked example ===")
        print(f"C1 = {C1} (expect 525)")
        print(f"SST = {SST} (expect 105)")
        print(f"SS_treatments = {SS_treatments:.3f} (expect 59.095)")
        print(f"SS_error = {SS_error:.3f} (expect 45.905)")
        print(f"T2 = {T2:.3f} (expect approx 25.74)")
        print(f"df1={df1}, df2={df2}")
        print(f"F_crit(0.05) = {f_crit_05:.3f} (expect approx 2.76)")
        print(f"F_crit(0.01) = {f_crit_01:.3f} (expect approx 4.13)")
        print(f"p_value = {p_value:.6g}")
        print(f"t_crit(alpha=0.01 two-sided, df2={df2}) = {t_crit_01:.3f} (expect approx 2.660)")
        print(f"CD (rank-sum units, alpha=0.01) = {cd_01:.3f} (expect approx 15.08)")
        print("Worked-example pairwise |R_i-R_j|: 32,29,49,3,17,20 "
              "-> significant except |R2-R3|=3")

    assert abs(C1 - 525) < 1e-9
    assert abs(SST - 105) < 1e-6
    assert abs(SS_treatments - 59.095) < 0.01
    assert abs(SS_error - 45.905) < 0.01
    assert abs(T2 - 25.74) < 0.1
    assert abs(f_crit_05 - 2.76) < 0.05
    assert abs(f_crit_01 - 4.13) < 0.05
    assert abs(t_crit_01 - 2.660) < 0.01
    assert abs(cd_01 - 15.08) < 0.1
    return T2, cd_01


# ================================================================================================
# SCENARIO STATISTICS (analysis.py)
# ================================================================================================

# analysis.py
# ===========
# Reads the real experiment results (results/summary.csv, produced by
# run_all.py) and, for each of the 7 scenarios (scenarios.py):
#   1. builds the Best/Mean/Std/CPU-Time table (Appendix B), one row per
#      reconciled instance, one column-group per method in the scenario;
#   2. ranks methods within each instance on the mean-of-30-runs feasible
#      objective (methods with zero feasible runs on an instance ranked
#      worst, tied -- same rule as Task 4/5's fig5_cd_diagram.py);
#   3. runs the Friedman test (Conover's F-approximation) on that scenario's
#      rank matrix, via the validated friedman_test.py;
#   4. computes the Conover post-hoc paired critical difference (rank-sum
#      units) AND the Nemenyi critical distance (mean-rank units, for the
#      CD diagram) -- kept explicitly distinct per Task 5's standing rule.
#
# Usage: python3 analysis.py            # all 8 scenarios (Scenario 8 needs M14 results)
#        python3 analysis.py --num 3    # just scenario 3
RESULTS_CSV = os.path.join(RESULTS_DIR, "summary.csv")            # locked M1-M13 sweep (never modified)
RESULTS_CSV_M14 = os.path.join(RESULTS_DIR, "summary_m14.csv")    # M14 sweep (run_m14.py), additive


def load_results(include_m14=True):
    RESULTS_CSV = os.path.join(RESULTS_DIR, "summary.csv")
    RESULTS_CSV_M14 = os.path.join(RESULTS_DIR, "summary_m14.csv")
    """Locked M1-M13 results, plus M14 rows when results/summary_m14.csv
    exists. The two files are concatenated in memory only; summary.csv is
    never rewritten."""
    df = pd.read_csv(RESULTS_CSV)
    if include_m14 and _os.path.exists(RESULTS_CSV_M14):
        df = pd.concat([df, pd.read_csv(RESULTS_CSV_M14)], ignore_index=True)
    return df


def per_cell_stats(df, method_id, instance_id):
    sub = df[(df.method_id == method_id) & (df.instance_id == instance_id)]
    if sub.empty:
        return dict(n_runs=0, n_feasible=0, best=np.nan, mean=np.nan, std=np.nan, cpu_mean=np.nan)
    feas = sub[sub.feasible_found]
    n_feas = len(feas)
    if n_feas > 0:
        best = feas.best.min()
        mean = feas.best.mean()
        std = feas.best.std(ddof=1) if n_feas > 1 else 0.0
    else:
        best = mean = std = np.nan
    return dict(n_runs=len(sub), n_feasible=n_feas, best=best, mean=mean, std=std,
                cpu_mean=sub.cpu_time_s.mean())


def scenario_table(df, scenario):
    methods = scenario["methods"]
    rows = []
    for iid in SUITE.keys():
        row = dict(instance_id=iid)
        for m in methods:
            st = per_cell_stats(df, m, iid)
            row[f"{m}_best"] = st["best"]
            row[f"{m}_mean"] = st["mean"]
            row[f"{m}_std"] = st["std"]
            row[f"{m}_cpu"] = st["cpu_mean"]
            row[f"{m}_nfeas"] = st["n_feasible"]
        rows.append(row)
    return pd.DataFrame(rows)


def rank_matrix_for_scenario(table_df, methods):
    """One row per instance, one column per method, rank 1=best. Methods
    with zero feasible runs on an instance are ranked worst (tied)."""
    n = len(table_df)
    k = len(methods)
    ranks = np.zeros((n, k))
    for i, row in table_df.iterrows():
        means = np.array([row[f"{m}_mean"] for m in methods])
        nfeas = np.array([row[f"{m}_nfeas"] for m in methods])
        BIG = (np.nanmax(means) + 1e9) if np.any(~np.isnan(means)) else 1e9
        score = np.where(nfeas > 0, np.nan_to_num(means, nan=BIG), BIG)
        ranks[i, :] = rankdata(score, method="average")
    return ranks


def run_scenario(df, scenario_num, alpha=(0.05, 0.01)):
    scenario = SCENARIO_BY_NUM[scenario_num]
    methods = scenario["methods"]
    table_df = scenario_table(df, scenario)
    rank_matrix = rank_matrix_for_scenario(table_df, methods)
    n, k = rank_matrix.shape

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05)
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01)
    cd_conover_05, t_crit_05 = conover_posthoc_cd(stat, 0.05)
    cd_conover_01, t_crit_01 = conover_posthoc_cd(stat, 0.01)
    cd_nemenyi_05 = nemenyi_cd(k, n, 0.05)
    mean_ranks = rank_matrix.mean(axis=0)

    return dict(
        scenario=scenario, table=table_df, rank_matrix=rank_matrix,
        stat=stat, f_crit_05=f_crit_05, f_crit_01=f_crit_01,
        cd_conover_05=cd_conover_05, cd_conover_01=cd_conover_01,
        cd_nemenyi_05=cd_nemenyi_05, mean_ranks=dict(zip(methods, mean_ranks)),
    )


def print_scenario_report(result):
    s = result["scenario"]
    stat = result["stat"]
    print("=" * 78)
    print(f"Scenario {s['num']} ({s['phase']}): {s['name']}  --  {', '.join(s['methods'])}")
    print(f"Isolates: {s['isolates']}")
    print(f"n={stat['n']} instances, k={stat['k']} methods")
    print("-" * 78)
    print(f"A2={stat['A2']:.3f}  B2={stat['B2']:.3f}  T2={stat['T2']:.4f}  "
          f"F({stat['df1']},{stat['df2']})  p={stat['p_value']:.4g}")
    decision05 = "REJECT H0" if stat["T2"] > result["f_crit_05"] else "fail to reject"
    decision01 = "REJECT H0" if stat["T2"] > result["f_crit_01"] else "fail to reject"
    print(f"F_crit(0.05)={result['f_crit_05']:.3f} -> {decision05}")
    print(f"F_crit(0.01)={result['f_crit_01']:.3f} -> {decision01}")
    print(f"Mean ranks: " + ", ".join(f"{m}={r:.3f}" for m, r in result["mean_ranks"].items()))
    print(f"Conover post-hoc CD (rank-sum units): alpha=0.05 -> {result['cd_conover_05']:.3f}, "
          f"alpha=0.01 -> {result['cd_conover_01']:.3f}")
    print(f"Nemenyi CD (mean-rank units, for CD diagram): alpha=0.05 -> {result['cd_nemenyi_05']:.3f}")
    print("=" * 78)


# ================================================================================================
# FEASIBILITY, CONSTRAINT-DENSITY SUBSET AND HOLM (analysis_extra.py)
# ================================================================================================

# analysis_extra.py
# ==================
# Supplementary Friedman-test analyses requested after the 7 scenario results
# were signed off as final (do NOT touch/re-run/reframe those -- this module
# only reads results/summary.csv, never writes to it, and never re-derives
# Scenario 1-7's objective-value results).
#
# Two additional analyses, both using the same validated friedman_f_statistic
# / conover_posthoc_cd / nemenyi_cd machinery as analysis.py:
#
# 1. Feasibility-rate Friedman test (one per scenario, 7 total): ranks methods
#    within each instance by N-feasible-of-30 (higher = better = rank 1)
#    instead of mean feasible objective. Reported as its own labelled set of
#    7 tests, explicitly separate from the objective-value tests.
#
# 2. Constraint-density subset analysis: the same 7 scenario groupings, same
#    procedure, restricted to the 8 instances belonging to SMD10/SMD11/SMD12
#    (the three benchmark families with the most active/scaling constraints:
#    SMD10@(3,3),(10,10) [30,30 excluded, Appendix C], SMD11@(3,3),(10,10),
#    (30,30), SMD12@(3,3),(10,10),(30,30)). This adds 7 more tests; combined
#    with the 7 full-suite tests that is 14 tests, and a Holm-Bonferroni
#    correction is applied across all 14 (both raw and corrected p reported).
#
# Usage: python3 analysis_extra.py
CONSTRAINT_DENSE_FAMILIES = ("SMD10", "SMD11", "SMD12")
CONSTRAINT_DENSE_INSTANCES = [iid for iid in SUITE.keys()
                              if iid.split("_p")[0] in CONSTRAINT_DENSE_FAMILIES]


# ---------------------------------------------------------------------------
# 1. Feasibility-rate Friedman test
# ---------------------------------------------------------------------------

def feas_rank_matrix_for_scenario(table_df, methods):
    """One row per instance, one column per method. Ranking variable is
    N-feasible-of-30 (higher is better -> rank 1). Ties (e.g. two methods
    both 0/30 or both 30/30 on an instance) get average ranks, same
    convention as the objective-value rank matrix."""
    n = len(table_df)
    k = len(methods)
    ranks = np.zeros((n, k))
    for i, row in table_df.iterrows():
        nfeas = np.array([row[f"{m}_nfeas"] for m in methods], dtype=float)
        # higher nfeas = better = should get rank 1 -> rank the negative
        ranks[i, :] = rankdata(-nfeas, method="average")
    return ranks


def run_scenario_feasibility(df, scenario_num, instance_ids=None, alpha=(0.05, 0.01)):
    scenario = SCENARIO_BY_NUM[scenario_num]
    methods = scenario["methods"]
    table_df = scenario_table(df, scenario)
    if instance_ids is not None:
        table_df = table_df[table_df.instance_id.isin(instance_ids)].reset_index(drop=True)
    rank_matrix = feas_rank_matrix_for_scenario(table_df, methods)
    n, k = rank_matrix.shape

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05) if not np.isnan(stat["T2"]) else np.nan
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01) if not np.isnan(stat["T2"]) else np.nan
    mean_ranks = rank_matrix.mean(axis=0)
    mean_nfeas = {m: table_df[f"{m}_nfeas"].mean() for m in methods}

    return dict(scenario=scenario, table=table_df, rank_matrix=rank_matrix, stat=stat,
                f_crit_05=f_crit_05, f_crit_01=f_crit_01,
                mean_ranks=dict(zip(methods, mean_ranks)), mean_nfeas=mean_nfeas,
                n_instances=n)


# ---------------------------------------------------------------------------
# 2. Constraint-density subset (objective-value ranking, same as analysis.py
#    but restricted to CONSTRAINT_DENSE_INSTANCES)
# ---------------------------------------------------------------------------

def run_scenario_subset(df, scenario_num, instance_ids, alpha=(0.05, 0.01)):
    scenario = SCENARIO_BY_NUM[scenario_num]
    methods = scenario["methods"]
    table_df = scenario_table(df, scenario)
    table_df = table_df[table_df.instance_id.isin(instance_ids)].reset_index(drop=True)
    pass  # (was: from analysis import rank_matrix_for_scenario)
    rank_matrix = rank_matrix_for_scenario(table_df, methods)
    n, k = rank_matrix.shape

    stat = friedman_f_statistic(rank_matrix)
    f_crit_05 = f_critical(stat["df1"], stat["df2"], 0.05) if not np.isnan(stat["T2"]) else np.nan
    f_crit_01 = f_critical(stat["df1"], stat["df2"], 0.01) if not np.isnan(stat["T2"]) else np.nan
    mean_ranks = rank_matrix.mean(axis=0)

    return dict(scenario=scenario, table=table_df, rank_matrix=rank_matrix, stat=stat,
                f_crit_05=f_crit_05, f_crit_01=f_crit_01,
                mean_ranks=dict(zip(methods, mean_ranks)), n_instances=n)


# ---------------------------------------------------------------------------
# Holm-Bonferroni across the 14 tests (7 full-suite objective-value tests
# already computed by analysis.run_scenario, + 7 constraint-density subset
# tests from run_scenario_subset above)
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals):
    """pvals: list of (label, p). Returns list of (label, p_raw, p_holm,
    reject_at_0.05) sorted by ascending raw p, with the standard
    step-down Holm procedure (monotone-adjusted)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    adj = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        label, p = pvals[idx]
        holm_p = (m - rank) * p
        running_max = max(running_max, holm_p)
        adj[idx] = min(running_max, 1.0)
    out = []
    for i, (label, p) in enumerate(pvals):
        out.append((label, p, adj[i], adj[i] < 0.05))
    return out


def analysis_extra_main():
    df = load_results()

    print("\n" + "#" * 78)
    print("# 1. FEASIBILITY-RATE FRIEDMAN TESTS (N-feasible-of-30 as ranking variable)")
    print("#" * 78)
    feas_results = {}
    for s in SCENARIOS:
        r = run_scenario_feasibility(df, s["num"])
        feas_results[s["num"]] = r
        stat = r["stat"]
        print("=" * 78)
        print(f"Scenario {s['num']} feasibility-rate test -- {', '.join(s['methods'])}")
        print(f"n={stat['n']}, k={stat['k']}, T2={stat['T2']:.4f}, p={stat['p_value']:.4g}")
        dec05 = "REJECT H0" if stat["T2"] > r["f_crit_05"] else "fail to reject"
        dec01 = "REJECT H0" if stat["T2"] > r["f_crit_01"] else "fail to reject"
        print(f"  alpha=0.05: F_crit={r['f_crit_05']:.3f} -> {dec05}")
        print(f"  alpha=0.01: F_crit={r['f_crit_01']:.3f} -> {dec01}")
        print(f"  Mean feasibility ranks: " + ", ".join(f"{m}={v:.3f}" for m, v in r["mean_ranks"].items()))
        print(f"  Mean N-feasible/30: " + ", ".join(f"{m}={v:.2f}" for m, v in r["mean_nfeas"].items()))

    print("\n" + "#" * 78)
    print("# 2. CONSTRAINT-DENSITY SUBSET (SMD10/SMD11/SMD12, n=8 instances)")
    print("#" * 78)
    print(f"Subset instances ({len(CONSTRAINT_DENSE_INSTANCES)}): {CONSTRAINT_DENSE_INSTANCES}")
    subset_results = {}
    for s in SCENARIOS:
        r = run_scenario_subset(df, s["num"], CONSTRAINT_DENSE_INSTANCES)
        subset_results[s["num"]] = r
        stat = r["stat"]
        print("=" * 78)
        print(f"Scenario {s['num']} subset test -- {', '.join(s['methods'])}")
        print(f"n={stat['n']}, k={stat['k']}, T2={stat['T2']:.4f}, p={stat['p_value']:.4g}")
        dec05 = "REJECT H0" if stat["T2"] > r["f_crit_05"] else "fail to reject"
        print(f"  alpha=0.05: F_crit={r['f_crit_05']:.3f} -> {dec05}")
        print(f"  Mean ranks (objective): " + ", ".join(f"{m}={v:.3f}" for m, v in r["mean_ranks"].items()))

    print("\n" + "#" * 78)
    print("# 3. HOLM-BONFERRONI ACROSS 14 TESTS (7 full-suite + 7 subset)")
    print("#" * 78)
    full_results = {s["num"]: run_scenario(df, s["num"]) for s in SCENARIOS}
    pvals = []
    for s in SCENARIOS:
        pvals.append((f"S{s['num']}_full", full_results[s["num"]]["stat"]["p_value"]))
    for s in SCENARIOS:
        pvals.append((f"S{s['num']}_subset", subset_results[s["num"]]["stat"]["p_value"]))
    holm = holm_bonferroni(pvals)
    holm_sorted_report = sorted(holm, key=lambda x: x[1])
    for label, p, p_holm, reject in holm_sorted_report:
        print(f"  {label:12s}  p_raw={p:.4g}  p_holm={p_holm:.4g}  reject@0.05={reject}")

    return dict(feas_results=feas_results, subset_results=subset_results,
                full_results=full_results, holm=holm)


# ================================================================================================
# ALL TEST STATISTICS OF THE PAPER (analysis_resub.py)
# ================================================================================================

# analysis_resub.py
# =================
# Single entry point that regenerates every number the resubmitted manuscript
# reports from the scenario tests, so that each figure in the text can be
# traced to code:
#
#   * the eight scenario Friedman tests (Scenarios 1-7 unchanged; Scenario 8 =
#     M1 vs M14, new), with Conover and Nemenyi post-hoc results and the full
#     pairwise decision table for both procedures;
#   * the eight feasibility-rate tests and the eight constraint-density subset
#     tests;
#   * the Holm-Bonferroni correction across the combined 16 tests;
#   * per-scale mean ranks for Scenarios 3, 5, 6 and 8;
#   * results/scenario_tables/scenario_8.csv (Appendix B raw table).
#
# Writes results/resub_numbers.json. Never writes summary.csv.
def scale_of(iid):
    if iid.startswith("TP"):
        return "TP"
    return iid.split("_")[1]


def compute_resub_numbers():
    df = load_results(include_m14=True)
    assert set(df.method_id.unique()) >= {f"M{i}" for i in range(1, 15)}, "M14 results missing"
    n_m14 = (df.method_id == "M14").sum()
    if os.environ.get("RESUB_ALLOW_PARTIAL") != "1":
        assert n_m14 == 44 * 30, f"M14 has {n_m14} rows, expected 1320"

    out = {"scenarios": {}, "feas": {}, "subset": {}, "holm": [], "per_scale": {},
           "methods": {}}

    for s in SCENARIOS:
        num = s["num"]
        r = run_scenario(df, num)
        st = r["stat"]
        methods = s["methods"]
        n = st["n"]
        rank_sums = {m: float(v) for m, v in zip(methods, r["rank_matrix"].sum(axis=0))}
        pairs = []
        for a, b in itertools.combinations(methods, 2):
            d_sum = abs(rank_sums[a] - rank_sums[b])
            d_mean = d_sum / n
            pairs.append(dict(a=a, b=b, rank_sum_diff=d_sum, mean_rank_diff=d_mean,
                              conover_sig05=bool(st["p_value"] < 0.05 and d_sum > r["cd_conover_05"]),
                              nemenyi_sig05=bool(d_mean > r["cd_nemenyi_05"])))
        out["scenarios"][num] = dict(
            name=s["name"], methods=methods, n=n, k=st["k"], A2=st["A2"], B2=st["B2"],
            T2=st["T2"], df1=st["df1"], df2=st["df2"], p=st["p_value"],
            f05=r["f_crit_05"], f01=r["f_crit_01"],
            reject05=bool(st["T2"] > r["f_crit_05"]), reject01=bool(st["T2"] > r["f_crit_01"]),
            mean_ranks={m: float(v) for m, v in r["mean_ranks"].items()},
            rank_sums=rank_sums, cd_conover05=r["cd_conover_05"], cd_conover01=r["cd_conover_01"],
            cd_nemenyi05=r["cd_nemenyi_05"], pairs=pairs)
        if num == 8:
            r["table"].to_csv(os.path.join(RESULTS_DIR, "scenario_tables", "scenario_8.csv"), index=False)
            tab = r["table"]
            wins = ties = losses = 0
            for _, row in tab.iterrows():
                a, b = row["M1_mean"], row["M14_mean"]
                fa, fb = row["M1_nfeas"] > 0, row["M14_nfeas"] > 0
                if not fa and not fb:
                    ties += 1
                elif fa and (not fb or a < b):
                    wins += 1
                elif fb and (not fa or b < a):
                    losses += 1
                else:
                    ties += 1
            out["scenarios"][num]["M1_better_instances"] = wins
            out["scenarios"][num]["M14_better_instances"] = losses
            out["scenarios"][num]["ties_or_no_comparison"] = ties

        # per-scale mean ranks
        tab = r["table"]
        rm = r["rank_matrix"]
        scales = {}
        for sc in ["TP", "p3q3", "p10q10", "p30q30"]:
            idx = [i for i, iid in enumerate(tab.instance_id) if scale_of(iid) == sc]
            scales[sc] = dict(n=len(idx), mean_ranks={m: float(v) for m, v in zip(methods, rm[idx].mean(axis=0))})
        out["per_scale"][num] = scales

        fr = run_scenario_feasibility(df, num)
        out["feas"][num] = dict(T2=fr["stat"]["T2"], p=fr["stat"]["p_value"],
                                reject05=bool(fr["stat"]["T2"] > fr["f_crit_05"]),
                                mean_nfeas={m: float(v) for m, v in fr["mean_nfeas"].items()})
        sr = run_scenario_subset(df, num, CONSTRAINT_DENSE_INSTANCES)
        out["subset"][num] = dict(n=sr["n_instances"], T2=sr["stat"]["T2"], p=sr["stat"]["p_value"],
                                  mean_ranks={m: float(v) for m, v in sr["mean_ranks"].items()})

    pvals = [(f"S{n}_full", out["scenarios"][n]["p"]) for n in out["scenarios"]] + \
            [(f"S{n}_subset", out["subset"][n]["p"]) for n in out["subset"]]
    for label, p, ph, rej in sorted(holm_bonferroni(pvals), key=lambda x: x[1]):
        out["holm"].append(dict(label=label, p_raw=p, p_holm=ph, reject05=bool(rej)))

    for m in sorted(df.method_id.unique(), key=lambda x: int(x[1:])):
        sub = df[df.method_id == m]
        out["methods"][m] = dict(runs=int(len(sub)), feasible_runs=int(sub.feasible_found.sum()),
                                 mean_cpu=float(sub.cpu_time_s.mean()))
    out["total_runs"] = int(len(df))
    out["total_feasible"] = int(df.feasible_found.sum())

    with open(os.path.join(RESULTS_DIR, "resub_numbers.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # console summary
    for n, s in out["scenarios"].items():
        print(f"S{n} {s['name']}: T2={s['T2']:.4f} p={s['p']:.4g} ranks={ {k: round(v,3) for k,v in s['mean_ranks'].items()} }")
    for h in out["holm"]:
        print(h)


# ================================================================================================
# CD-DIAGRAM SUMMARIES (export_scenario_summaries.py)
# ================================================================================================

# export_scenario_summaries.py
# =============================
# Task 20 (CD diagrams per scenario): dumps, for each of the 7 scenarios,
# exactly the numbers already computed and validated by analysis.py's
# run_scenario() (mean ranks, Nemenyi CD, Conover F-statistic/p-value) to a
# single JSON file that sa_appa_figures/figures/fig5_cd_diagram_scenarios.py
# reads to draw the CD diagrams.
#
# Deliberately does NOT recompute ranks/statistics from the raw trajectory
# data via a separate code path (the original fig5_cd_diagram.py did this,
# against the synthetic-data convention) -- that would risk a second,
# independently-computed set of numbers silently drifting from the ones
# already reported in Appendix B's tables and Section 5.7/5.8's prose. This
# script is a thin export of analysis.py's own run_scenario() output, so the
# CD diagrams are guaranteed consistent with the numbers already in the
# manuscript.



def _cd_to_native(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def export_cd_summaries():
    df = load_results()
    out = []
    for s in SCENARIOS:
        result = run_scenario(df, s["num"])
        stat = result["stat"]
        out.append(dict(
            num=s["num"],
            phase=s["phase"],
            name=s["name"],
            methods=s["methods"],
            isolates=s["isolates"],
            mean_ranks={m: _cd_to_native(r) for m, r in result["mean_ranks"].items()},
            n_instances=_cd_to_native(stat["n"]),
            k_methods=_cd_to_native(stat["k"]),
            T2=_cd_to_native(stat["T2"]),
            p_value=_cd_to_native(stat["p_value"]),
            df1=_cd_to_native(stat["df1"]),
            df2=_cd_to_native(stat["df2"]),
            f_crit_05=_cd_to_native(result["f_crit_05"]),
            f_crit_01=_cd_to_native(result["f_crit_01"]),
            cd_nemenyi_05=_cd_to_native(result["cd_nemenyi_05"]),
            cd_conover_05=_cd_to_native(result["cd_conover_05"]),
        ))
    with open(os.path.join(RESULTS_DIR, "scenario_cd_summaries.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote scenario_cd_summaries.json")


# ================================================================================================
# VALUES BELOW THE KNOWN OPTIMUM (analysis_s8_llcheck.py)
# ================================================================================================

# Scenario 8 sensitivity check for inexact LL responses.
#
# On SMD1-SMD9 the UL optimum is F* = 0 and any UL value below it can only arise when the
# returned LL response is not the global LL optimum (e.g. a Rastrigin local minimum in SMD4).
# This script counts such runs per method and repeats the Scenario 8 Friedman test with the
# affected instances removed. It adds the key 's8_llcheck' to results/resub_numbers.json.
def run_llcheck():
    LLCHECK_TOL = 1e-2
    d = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv")), pd.read_csv(os.path.join(RESULTS_DIR, "summary_m14.csv"))])
    f = d[d.feasible_found.astype(bool)]
    smd = f[f.instance_id.str.match(r"SMD[1-9]_")]
    below = smd.assign(b=smd.best < -LLCHECK_TOL).groupby("method_id").b.agg(["sum", "count"])
    pair = f[f.method_id.isin(["M1", "M14"])]
    bad = (pair[pair.instance_id.str.match(r"SMD[1-9]_") & (pair.best < -LLCHECK_TOL)]
           .groupby(["instance_id", "method_id"]).size().unstack(fill_value=0))
    for c in ["M1", "M14"]:
        if c not in bad:
            bad[c] = 0
    m = pair.groupby(["instance_id", "method_id"]).best.mean().unstack()
    m = m.reindex(sorted(d.instance_id.unique())).drop(index=list(bad.index))
    r = m.rank(axis=1, na_option="bottom")
    n, k = r.shape
    A2 = (r ** 2).values.sum(); B2 = (r.sum() ** 2).sum() / n; C1 = n * k * (k + 1) ** 2 / 4
    T2 = (n - 1) * (B2 - C1) / (A2 - B2)
    p = float(1 - stats.f.cdf(T2, k - 1, (n - 1) * (k - 1)))
    out = {
        "tol": LLCHECK_TOL,
        "below_opt_runs": {mid: int(row["sum"]) for mid, row in below.iterrows()},
        "feasible_smd1_9_runs": {mid: int(row["count"]) for mid, row in below.iterrows()},
        "affected": {i: {"M1": int(row["M1"]), "M14": int(row["M14"])} for i, row in bad.iterrows()},
        "n_kept": int(n), "M14_better": int((m.M14 < m.M1).sum()), "M1_better": int((m.M1 < m.M14).sum()),
        "mean_ranks": {c: float(v) for c, v in r.mean().items()}, "T2": float(T2), "p": p,
    }
    J = os.path.join(RESULTS_DIR, "resub_numbers.json")
    js = json.load(open(J)); js["s8_llcheck"] = out
    json.dump(js, open(J, "w"), indent=1)
    print(json.dumps(out, indent=1))


    def drop_affected(methods):
        g = f[f.method_id.isin(methods)]
        b = g[g.instance_id.str.match(r"SMD[1-9]_") & (g.best < -LLCHECK_TOL)].instance_id.unique()
        mm = g.groupby(["instance_id", "method_id"]).best.mean().unstack().reindex(index=sorted(d.instance_id.unique()), columns=methods)
        mm = mm.drop(index=list(b))
        # instances on which no method has a feasible run stay in, all tied (the main ranking rule)
        rr = mm.rank(axis=1, na_option="bottom")
        nn, kk = rr.shape
        a2 = (rr ** 2).values.sum(); b2 = (rr.sum() ** 2).sum() / nn; c1 = nn * kk * (kk + 1) ** 2 / 4
        t2 = (nn - 1) * (b2 - c1) / (a2 - b2)
        return {"dropped": sorted(map(str, b)), "n": int(nn), "mean_ranks": {c: float(v) for c, v in rr.mean().items()},
                "T2": float(t2), "p": float(1 - stats.f.cdf(t2, kk - 1, (nn - 1) * (kk - 1)))}


    js = json.load(open(J))
    js["llcheck_scenarios"] = {"3": drop_affected(["M3", "M4", "M5", "M6"]),
                               "6": drop_affected(["M1", "M10", "M11"]),
                               "8": drop_affected(["M1", "M14"])}
    json.dump(js, open(J, "w"), indent=1)
    print(json.dumps(js["llcheck_scenarios"], indent=1))


# ================================================================================================
# APPENDIX B TABLES AS LATEX (build_appendix_b.py)
# ================================================================================================

r"""
build_appendix_b.py
====================
Generates the Appendix B LaTeX (one longtable per scenario, Best/Mean/Std/CPU
per method, one row per reconciled instance, with a Friedman-test summary
paragraph beneath each table) from the real results/scenario_tables/*.csv and
analysis.py's run_scenario().

Output: appendix_b_scenario_{1..7}.tex, appendix_b_all.tex (all 7, in order,
each starting a new \subsection, ready to \input into the appendix).
"""

def esc(s):
    return str(s).replace("_", "\\_")

def fmt(v, nd=3):
    if pd.isna(v):
        return "--"
    return f"{v:.{nd}f}"

def method_label(m):
    # Table 1 short names, kept consistent with the manuscript's own naming
    names = {
        "M1": "SA-APPA", "M2": "M2 (unif.\\ init)", "M3": "M3 (bare PPA)",
        # M5/M6 corrected 2026-10-02 to match METHOD_SPECS (M5 = DE, M6 = PSO);
        # the earlier dict had the two engine names swapped (display only;
        # every number was always keyed by method ID and is unchanged).
        "M4": "M4 (GA engine)", "M5": "M5 (DE engine)", "M6": "M6 (PSO engine)",
        "M7": "M7 (dyn.\\ penalty)", "M8": "M8 (uncon.\\ EI)", "M9": "M9 (fixed sched.)",
        # Honest labels (2026-10-02): M10-M13 are in-house baselines, not
        # re-implementations of the published methods they were once named after.
        "M10": "M10 (SA-DE-PEN)", "M11": "M11 (SA-PSO-RBF)",
        "M12": "M12 (NEST-GA, seeds A)", "M13": "M13 (NEST-GA, seeds B)",
        "M14": "M14 (BL-SAEA re-impl.)",
    }
    return names.get(m, m)

def build_scenario_tex(scenario_num):
    s = next(x for x in SCENARIOS if x["num"] == scenario_num)
    methods = s["methods"]
    df = load_results()
    table_df = pd.read_csv(os.path.join(RESULTS_DIR, "scenario_tables", f"scenario_{scenario_num}.csv"))
    result = run_scenario(df, scenario_num)
    stat = result["stat"]

    k = len(methods)
    col_spec = "l" + "rrrr" * k
    header1 = " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{method_label(m)}}}" for m in methods) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(k))
    header2 = "Instance & " + " & ".join("Best & Mean & Std & CPU (s)" for _ in methods) + " \\\\"

    lines = []
    lines.append(f"\\subsection{{Scenario {scenario_num} ({s['phase']}): {s['name']}}}")
    lines.append(f"\\label{{app:scenario{scenario_num}}}")
    lines.append(f"Methods compared: {', '.join(method_label(m) for m in methods)}. "
                 f"Isolates: {esc(s['isolates'])}. Each cell reports Best/Mean/Std (feasible runs "
                 f"only, out of 30 seeds) and mean CPU time per run; a dash indicates zero "
                 f"feasible runs on that instance.")
    lines.append("")
    lines.append("\\begingroup\\scriptsize")
    lines.append(f"\\begin{{longtable}}{{{col_spec}}}")
    lines.append(f"\\caption{{Scenario {scenario_num} raw results, all 44 reconciled instances.}}"
                 f"\\label{{tab:scenario{scenario_num}}}\\\\")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append(cmid)
    lines.append(header2)
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append(cmid)
    lines.append(header2)
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\midrule\\multicolumn{" + str(1+4*k) + "}{r}{\\emph{continued on next page}}\\\\\\midrule")
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for _, row in table_df.iterrows():
        cells = [esc(row["instance_id"])]
        for m in methods:
            cells.append(fmt(row[f"{m}_best"]))
            cells.append(fmt(row[f"{m}_mean"]))
            cells.append(fmt(row[f"{m}_std"]))
            cells.append(fmt(row[f"{m}_cpu"], nd=2))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    lines.append("")

    # Friedman summary
    decision05 = "reject" if stat["T2"] > result["f_crit_05"] else "fail to reject"
    decision01 = "reject" if stat["T2"] > result["f_crit_01"] else "fail to reject"
    mean_ranks_str = ", ".join(f"{method_label(m)}={r:.3f}" for m, r in result["mean_ranks"].items())
    lines.append(f"\\paragraph{{Friedman test summary (Scenario {scenario_num}).}}")
    lines.append(f"$n={stat['n']}$ instances, $k={stat['k']}$ methods. "
                 f"$A_2={stat['A2']:.3f}$, $B_2={stat['B2']:.3f}$, "
                 f"$T_2={stat['T2']:.4f}$, distributed $F({stat['df1']},{stat['df2']})$ under "
                 f"$H_0$, $p={stat['p_value']:.4g}$. Critical values: "
                 f"$F_{{0.05}}({stat['df1']},{stat['df2']})={result['f_crit_05']:.3f}$ "
                 f"({decision05} $H_0$ at $\\alpha=0.05$); "
                 f"$F_{{0.01}}({stat['df1']},{stat['df2']})={result['f_crit_01']:.3f}$ "
                 f"({decision01} $H_0$ at $\\alpha=0.01$). Mean ranks: {mean_ranks_str}. "
                 f"Conover post-hoc paired critical difference (rank-sum units): "
                 f"$CD_{{0.05}}={result['cd_conover_05']:.3f}$, $CD_{{0.01}}={result['cd_conover_01']:.3f}$. "
                 f"Nemenyi critical distance (mean-rank units, for the CD diagram, "
                 f"Figure~\\ref{{fig:cd{scenario_num}}}): $CD_{{0.05}}={result['cd_nemenyi_05']:.3f}$.")
    lines.append("")
    return "\n".join(lines)


# ================================================================================================
# FIGURE HELPERS (figures/common.py)
# ================================================================================================

# common.py
# =========
# Shared plotting utilities: consistent styling, the mandatory SYNTHETIC
# watermark, and a single save_figure() helper that writes both the image
# file and a sibling .caption.txt file, so every figure's caption is
# available as plain text for direct use in the paper (and is never only
# inside this Python session).
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


# ================================================================================================
# CRITICAL-DIFFERENCE DIAGRAMS (figures/fig5_cd_diagram_scenarios.py)
# ================================================================================================

# fig5_cd_diagram_scenarios.py
# ==============================
# Task 20: seven critical-difference (CD) diagrams, one per experimental
# scenario, matching the fig:cd1--fig:cd7 labels already referenced from
# Appendix B's per-scenario Friedman summary paragraphs in the manuscript.
#
# Unlike the original fig5_cd_diagram.py (built for the synthetic-data
# Phase I/Phase II convention and its own from-scratch rank computation),
# this module reads its mean ranks, Nemenyi critical difference, and
# Friedman F-statistic directly from
# sa_appa_experiments/results/scenario_cd_summaries.json, produced by
# export_scenario_summaries.py as a thin export of analysis.py's already-
# validated run_scenario() output -- the same numbers already embedded in
# Appendix B's tables and Section 5.7/5.8's prose. This guarantees the CD
# diagrams cannot silently drift from the numbers already reported
# elsewhere in the paper.
#
# The connecting bar that groups methods not significantly different at the
# Nemenyi CD is drawn as a fully opaque line (no alpha), per the project's
# EPS/no-transparency format rule -- the original fig5_cd_diagram.py used
# `alpha=0.7` on this bar, which is fixed here.
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


# ================================================================================================
# PAPER FIGURES 1-4 (figures_resub.py)
# ================================================================================================

# figures_resub.py
# ================
# Figures for the resubmitted manuscript, built from the real trajectories of
# all fourteen configurations (results/trajectories.jsonl for M1-M13, locked;
# results/trajectories_m14.jsonl for M14).
#
# Every pooled figure is split into an internal-ablation view (M1-M9) and a
# Phase II view (M1, M10-M14). Colour follows the configuration, not its
# rank: M1 is slot 1 (blue) in every panel of every figure.
#
# Definitions are unchanged from fig1_boxplot.py, fig2_performance_profile.py
# and fig2b_data_profile.py, except that f_best / f_L / normalisation ranges
# are now taken over all fourteen configurations. The convergence-rate figure
# uses the cumulative number of nested solves as its time axis (rather than
# the iteration counter), because M12-M14 count iterations differently from
# M1-M11.
#
# Outputs (PDF + EPS, opaque fills only) to ../paper_figures_resub/.

ABL = [f"M{i}" for i in range(1, 10)]
EXT = ["M1", "M10", "M11", "M12", "M13", "M14"]
RS_LABEL = {"M1": "M1 SA-APPA", "M2": "M2", "M3": "M3", "M4": "M4 GA", "M5": "M5 DE",
         "M6": "M6 PSO", "M7": "M7", "M8": "M8", "M9": "M9 fixed",
         "M10": "M10 SA-DE-Pen", "M11": "M11 SA-PSO-RBF", "M12": "M12 Nest-GA (A)",
         "M13": "M13 Nest-GA (B)", "M14": "M14 BL-SAEA"}

# Reference categorical palette (dataviz skill, light mode), fixed order.
P = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#52514e"
STYLE = {
    "M1": (P[0], "-", 2.2),
    "M2": (P[1], "-", 1.3), "M3": (P[2], "-", 1.3), "M4": (P[3], "-", 1.3),
    "M5": (P[4], "-", 1.3), "M6": (P[5], "-", 1.3), "M7": (P[6], "-", 1.3),
    "M8": (P[7], "-", 1.3), "M9": (GRAY, "--", 1.3),
    "M10": (P[1], "-", 1.3), "M11": (P[2], "-", 1.3), "M12": (P[3], "-", 1.3),
    "M13": (P[3], "--", 1.3), "M14": (P[7], "-", 1.6),
}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False})


def rs_lighten(c, a=0.82):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a)


def rs_load():
    recs = []
    for fn in ["trajectories.jsonl", "trajectories_m14.jsonl"]:
        with open(os.path.join(RESULTS_DIR, fn)) as fh:
            for line in fh:
                o = json.loads(line)
                for r in o["rows"]:
                    recs.append((o["method_id"], o["instance_id"], o["run_id"], r["eval_count"],
                                 r["iteration"], r["best_feasible_obj"] if r["is_feasible"] else np.nan,
                                 r["is_feasible"]))
    df = pd.DataFrame(recs, columns=["method_id", "instance_id", "run_id", "eval_count",
                                     "iteration", "best_feasible_obj", "is_feasible"])
    assert df.method_id.nunique() == 14
    return df


def final_values(df):
    df = df.sort_values(["method_id", "instance_id", "run_id", "eval_count"])
    return df.groupby(["method_id", "instance_id", "run_id"], as_index=False).last()


def rs_save(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, stem + ".eps"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, stem + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


def group_of(iid):
    return "TP1–TP9" if iid.startswith("TP") else {"p3q3": "SMD (3,3)", "p10q10": "SMD (10,10)",
                                                     "p30q30": "SMD (30,30)"}[iid.split("_")[1]]


# ---------------------------------------------------------------- Figure 1
def fig1(last, methods, stem):
    last = last.copy()
    last["norm"] = np.nan
    for iid, sub in last.groupby("instance_id"):   # normalise over ALL 14 configurations
        v = sub.loc[sub.is_feasible, "best_feasible_obj"]
        if v.empty:
            continue
        lo, hi = v.min(), v.max()
        last.loc[v.index, "norm"] = (v - lo) / (hi - lo) if hi > lo else 0.0
    groups = ["TP1–TP9", "SMD (3,3)", "SMD (10,10)", "SMD (30,30)"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, g in zip(axes, groups):
        sub = last[last.instance_id.map(group_of) == g]
        data = [sub.loc[sub.method_id == m, "norm"].dropna().values for m in methods]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6,
                        medianprops=dict(color="#0b0b0b", linewidth=1.0),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="#0b0b0b", markersize=3),
                        flierprops=dict(marker="o", markersize=2, markerfacecolor="#8a8984",
                                        markeredgecolor="none"))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(rs_lighten(STYLE[m][0], 0.35))
            patch.set_edgecolor(STYLE[m][0])
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=90)
        ax.set_title(g)
        ax.grid(axis="y")
    axes[0].set_ylabel("Normalised final feasible objective\n(0 = best found on instance)")
    fig.tight_layout()
    rs_save(fig, stem)


# ---------------------------------------------------------------- Figure 2
def perf_profile(ax, last, methods, eps=1e-8, tau_max=2.0):
    fb = last[last.is_feasible].groupby("instance_id").best_feasible_obj.min()
    insts = list(fb.index)
    means = last[last.is_feasible].groupby(["method_id", "instance_id"]).best_feasible_obj.mean()
    taus = np.linspace(0, tau_max, 400)
    for m in methods:
        e = np.array([((means.get((m, i), np.nan) - fb[i]) / (abs(fb[i]) + eps))
                      if (m, i) in means.index else np.inf for i in insts])
        e = np.where(np.isnan(e), np.inf, e)
        rho = [(e <= t).mean() for t in taus]
        c, ls, lw = STYLE[m]
        ax.step(taus, rho, where="post", color=c, linestyle=ls, linewidth=lw, label=RS_LABEL[m])
    ax.set_xlabel(r"$\tau$ (relative error to best found)")
    ax.set_ylabel(r"$\rho_m(\tau)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(fontsize=6.5, loc="upper left")
    return len(insts)


def data_profile(ax, df, methods, tau, n0=25, kmax=25.0):
    last = final_values(df)
    fL = last[last.is_feasible].groupby("instance_id").best_feasible_obj.min().rename("fL")
    at0 = df[(df.eval_count == n0) & df.is_feasible]
    f0 = at0.groupby("instance_id").best_feasible_obj.mean().rename("f0")
    sub = df[df.method_id.isin(methods)].merge(fL, on="instance_id").merge(f0, on="instance_id")
    sub["ok"] = sub.is_feasible & (sub.best_feasible_obj <= sub.fL + tau * (sub.f0 - sub.fL))
    first = sub[sub.ok].groupby(["method_id", "instance_id", "run_id"]).eval_count.min()
    keys = df[df.method_id.isin(methods)][["method_id", "instance_id", "run_id"]].drop_duplicates()
    keys = keys.merge(first.rename("ev").reset_index(), how="left").fillna({"ev": np.inf})
    keys["kappa"] = keys.ev / (keys.instance_id.map(lambda i: SUITE[i].n_ul) + 1)
    grid = np.linspace(0, kmax, 500)
    for m in methods:
        k = keys.loc[keys.method_id == m, "kappa"].values
        c, ls, lw = STYLE[m]
        ax.step(grid, [(k <= g).mean() for g in grid], where="post", color=c, linestyle=ls,
                linewidth=lw, label=RS_LABEL[m])
    ax.set_xlabel(r"$\kappa$ (nested solves / $(n_{\mathrm{UL}}+1)$)")
    ax.set_ylabel(rf"$d_m(\kappa)$, $\tau={tau}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(fontsize=6.5, loc="upper left")


# ---------------------------------------------------------------- Figures 3/4
REP = ["TP2", "SMD4_p3q3", "SMD4_p10q10", "SMD4_p30q30"]


def fig3(df, methods, stem):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, iid in zip(axes.ravel(), REP):
        d = df[df.instance_id == iid]
        omitted = []
        for m in methods:
            s = d[d.method_id == m]
            piv = s.pivot_table(index="eval_count", columns="run_id", values="best_feasible_obj",
                                aggfunc="last").sort_index().ffill()
            if piv.dropna(how="all").empty:
                omitted.append(m)
                continue
            c, ls, lw = STYLE[m]
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), facecolor=rs_lighten(c),
                            edgecolor="none", zorder=1)
            ax.plot(piv.index, piv.mean(axis=1), color=c, linestyle=ls, linewidth=lw,
                    label=RS_LABEL[m], zorder=3)
        ax.set_title(iid.replace("_", " "))
        ax.set_xlabel("Cumulative nested LL solves")
        ax.set_ylabel("Best feasible objective so far")
        ax.grid(True)
        if omitted:
            ax.text(0.02, 0.02, "No feasible run: " + ", ".join(omitted), transform=ax.transAxes,
                    fontsize=6, color=GRAY)
    axes[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    rs_save(fig, stem)


def fit_order(t, err, floor=1e-10):
    t, err = np.asarray(t, float), np.asarray(err, float)
    mask = (t >= 1) & (err > floor) & np.isfinite(err)
    if mask.sum() < 3:
        return np.nan
    x, y = np.log(t[mask]), np.log(err[mask])
    if np.ptp(x) == 0:
        return np.nan
    return -np.polyfit(x, y, 1)[0]


def fig4(df, methods, stem, floor=1e-10):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, iid in zip(axes.ravel(), REP):
        d = df[df.instance_id == iid]
        for m in methods:
            s = d[d.method_id == m]
            curves, orders = [], []
            for r, g in s.groupby("run_id"):
                g = g.sort_values("eval_count")
                v = g.best_feasible_obj.ffill()
                if v.isna().all():
                    continue
                err = (v - v.iloc[-1]).abs()
                err.index = g.eval_count.values
                curves.append(err)
                o = fit_order(g.eval_count.values, err.values, floor)
                if np.isfinite(o):
                    orders.append(o)
            if not curves:
                continue
            mean_err = pd.concat(curves, axis=1).mean(axis=1)
            mk = mean_err > floor
            c, ls, lw = STYLE[m]
            lab = f"{RS_LABEL[m]} (p={np.mean(orders):.2f}$\\pm${np.std(orders):.2f})" if orders else f"{RS_LABEL[m]} (p n/a)"
            if mk.sum() >= 2:
                ax.loglog(mean_err.index[mk], mean_err.values[mk], color=c, linestyle=ls,
                          linewidth=lw, label=lab)
        ax.set_title(iid.replace("_", " "))
        ax.set_xlabel("Cumulative nested LL solves (log)")
        ax.set_ylabel(r"$|f_t-f_{best,run}|$ (log)")
        ax.grid(True, which="both")
        ax.legend(fontsize=5.5)
    fig.tight_layout()
    rs_save(fig, stem)


def figures_resub_main():
    df = rs_load()
    last = final_values(df)
    fig1(last, ABL, "fig1_ablation")
    fig1(last, EXT, "fig1_external")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    n = perf_profile(axes[0], last, ABL)
    perf_profile(axes[1], last, EXT)
    axes[0].set_title(f"Internal ablation (M1–M9), {n} instances")
    axes[1].set_title(f"Phase II (M1, M10–M14), {n} instances")
    fig.tight_layout()
    rs_save(fig, "fig2_profiles_split")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, tau in enumerate([0.1, 0.25]):
        data_profile(axes[row, 0], df, ABL, tau)
        data_profile(axes[row, 1], df, EXT, tau)
    axes[0, 0].set_title("Internal ablation (M1–M9)")
    axes[0, 1].set_title("Phase II (M1, M10–M14)")
    fig.tight_layout()
    rs_save(fig, "fig2b_data_profiles_split")
    fig3(df, EXT, "fig3_convergence_external")
    fig4(df, EXT, "fig4_rate_external")
    fig3(df, ABL, "fig3_convergence_ablation")   # code-package copies
    fig4(df, ABL, "fig4_rate_ablation")
    print("figures written to", os.path.abspath(FIG_DIR))


# ================================================================================================
# PHASE 7 DATA: GROUND TRUTH, RUN SELECTION, GP SNAPSHOTS (phase7_data.py)
# ================================================================================================

# phase7_data.py
# ==============
# Data for the Phase 7 surrogate-fidelity figures (Part A). Read-only with respect to the locked
# results: every SA-APPA (M1) run used here is a re-run under its locked seed, checked against the
# locked trajectory before any GP snapshot is used.
#
# Ground truth (never charged to any method):
#   * TP3 and TP8: the LL problem is a strictly convex QP with simple bounds, solved exactly by
#     bounded-variable least squares (scipy.optimize.lsq_linear, method 'bvls') and checked against
#     the closed-form projection.
#   * all other instances: a reference LL solve, SciPy differential evolution with population
#     30 x n_ll, up to 300 generations, tol 1e-10, L-BFGS-B polish, best of two seeds. This is far
#     more effort than the study's LL solver but is still a heuristic, and is described as such.
#
# Run selection rule (fixed before any figure was drawn): for each instance, sort the 30 locked
# M1 runs by final best feasible UL objective (runs without a feasible point last), ties by run id,
# and take the 15th (the lower median).
#
# Outputs: results/phase7/*.npz and results/phase7/fidelity.json
P7_PEN = 1e6
TWO_SIGMA = 2.0
HELDOUT_SEED = 20261007
TP_LOWDIM = [f"TP{k}" for k in range(1, 9)]
SMD_SMALL = [f"SMD{k}_p3q3" for k in range(1, 13)]


# ---------------------------------------------------------------------------- ground truth LL
def ystar_exact(iid, x):
    """Exact LL response for the two instances whose LL is a bound-constrained strictly convex QP."""
    x = np.asarray(x, float)
    if iid == "TP3":   # min ||x - y||^2, y in [0,10]^2
        lo, hi, b = np.zeros(2), np.full(2, 10.0), x
    elif iid == "TP8":  # min ||y - (x - 20)||^2, y_i in [-10, min(20, (x_i - 10)/2)]
        lo = np.full(2, -10.0); hi = np.minimum(20.0, (x - 10.0) / 2.0); b = x - 20.0
    else:
        raise ValueError(iid)
    r = lsq_linear(np.eye(2), b, bounds=(lo, hi), method="bvls")
    closed = np.clip(b, lo, hi)
    assert np.max(np.abs(r.x - closed)) < 1e-9, (iid, x, r.x, closed)
    return r.x


def ystar_reference(inst, x):
    def obj(y):
        g = inst.g(x, y)
        v = np.sum(np.maximum(g, 0) ** 2) if g.size else 0.0
        return inst.f(x, y) + P7_PEN * v
    best = None
    for s in (0, 1):
        r = differential_evolution(obj, list(map(tuple, inst.y_bounds)), seed=s, popsize=30,
                                   maxiter=300, tol=1e-10, polish=True, init="latinhypercube")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def ystar(iid, x):
    if iid in ("TP3", "TP8"):
        return ystar_exact(iid, x)
    return ystar_reference(SUITE[iid], x)


def truth(iid, x):
    inst = SUITE[iid]
    y = ystar(iid, x)
    g = inst.g(x, y)
    llf = bool(np.all(g <= 1e-6)) if g.size else True
    return dict(y=y, theta=float(inst.F(x, y)), phi=np.atleast_1d(inst.G(x, y)).astype(float),
                fstar=float(inst.f(x, y)), ll_feasible=llf)


# ---------------------------------------------------------------------------- run selection
def median_run(iid, method="M1"):
    d = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    m = d[(d.method_id == method) & (d.instance_id == iid)].copy()
    m["key"] = np.where(m.feasible_found.astype(bool), m.best, np.inf)
    m = m.sort_values(["key", "run_id"])
    return int(m.iloc[14].run_id), float(m.iloc[14].best)


def locked_traj(method, iid, run):
    tag = f'"method_id": "{method}", "instance_id": "{iid}", "run_id": {run},'
    with open(os.path.join(RESULTS_DIR, "trajectories.jsonl")) as fh:
        for line in fh:
            if tag in line:
                return [r["best_feasible_obj"] for r in json.loads(line)["rows"]]
    raise KeyError(tag)


def rerun_with_snapshots(iid, run, method="M1"):
    """Re-run under the locked seed, capturing SA-APPA's own GP objects after the initial design
    and at the end of the run. Returns the snapshots and the reproduction check."""
    snaps = {}

    def hook(stage, X, F, G, theta_gp, phi_gps):
        snaps[stage] = dict(X=X, F=F, G=np.array(G, dtype=float).reshape(len(F), -1),
                            theta_gp=theta_gp, phi_gps=phi_gps)
    sp = METHOD_SPECS[method]
    traj, _ = run_ablation_method(SUITE[iid], method, sp["engine"], sp["constraint_mode"],
                                  sp["adaptation"], sp["init"], seed=run, hook=hook)
    new = [r["best_feasible_obj"] for r in traj.rows]
    old = locked_traj(method, iid, run)

    def rel(a, b):
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return 0.0 if (a is None or not np.isfinite(a)) == (b is None or not np.isfinite(b)) else np.inf
        return abs(a - b) / max(1.0, abs(b))
    dev = max(rel(a, b) for a, b in zip(new, old)) if len(new) == len(old) else np.inf
    return snaps, dict(instance=iid, method=method, run=run, n_rows=len(new),
                       max_rel_dev=float(dev), reproduces=bool(dev <= 1e-6))


# ---------------------------------------------------------------------------- metrics
def gp_metrics(gp, Xh, th):
    mu, sd = gp.predict(Xh)
    err = mu - th
    cov = np.mean(np.abs(err) <= TWO_SIGMA * sd) if len(th) else np.nan
    return dict(n=int(len(th)), rmse=float(np.sqrt(np.mean(err ** 2))), max_abs=float(np.max(np.abs(err))),
                coverage_2sd=float(cov), mean_sd=float(np.mean(sd)),
                theta_range=float(np.ptp(th)) if len(th) else np.nan)


def heldout(iid, n):
    inst = SUITE[iid]
    U = qmc.LatinHypercube(d=inst.n_ul, seed=HELDOUT_SEED).random(n)
    X = qmc.scale(U, inst.x_bounds[:, 0], inst.x_bounds[:, 1])
    T = [truth(iid, x) for x in X]
    keep = np.array([t["ll_feasible"] for t in T])
    return X, np.array([t["theta"] for t in T]), np.array([t["phi"] for t in T]), keep


# ================================================================================================
# PHASE 7 RUNNER, PART A (phase7_run.py)
# ================================================================================================

# Compute and cache every quantity behind the Phase 7 Part A figures (see phase7_data.py).
NG = 61                      # wireframe / map grid per axis
NH_TP3 = 2500                # held-out points for the TP3 fidelity table
R_NEAR = 1.5                 # neighbourhood of x* (10% of the TP3 box width)
X_STAR_TP3 = np.array([15.0, 10.0])
P7_LOG = {"reruns": [], "notes": []}


def p7_grid2(iid, n=NG):
    b = SUITE[iid].x_bounds
    g1, g2 = np.linspace(b[0, 0], b[0, 1], n), np.linspace(b[1, 0], b[1, 1], n)
    X1, X2 = np.meshgrid(g1, g2)
    return X1, X2, np.column_stack([X1.ravel(), X2.ravel()])


def part_A1_A4():
    X1, X2, XX = p7_grid2("TP3")
    T = [truth("TP3", x) for x in XX]
    np.savez(os.path.join(P7_DIR, "tp3_truth.npz"), X1=X1, X2=X2,
             theta=np.array([t["theta"] for t in T]).reshape(X1.shape),
             fstar=np.array([t["fstar"] for t in T]).reshape(X1.shape),
             y1=np.array([t["y"][0] for t in T]).reshape(X1.shape),
             y2=np.array([t["y"][1] for t in T]).reshape(X1.shape),
             phi=np.array([t["phi"] for t in T]).reshape(X1.shape + (2,)),
             xstar=X_STAR_TP3, theta_star=325.0)


def snap_pred(snap, XX):
    mu, sd = snap["theta_gp"].predict(XX)
    out = dict(mu=mu, sd=sd, X=snap["X"], F=snap["F"], G=snap["G"],
               ls=snap["theta_gp"].lengthscale)
    for j, g in enumerate(snap["phi_gps"]):
        m, s = g.predict(XX)
        out[f"phi{j}_mu"], out[f"phi{j}_sd"], out[f"phi{j}_ls"] = m, s, g.lengthscale
    return out


def part_A2():
    run, best = median_run("TP3")
    snaps, chk = rerun_with_snapshots("TP3", run)
    P7_LOG["reruns"].append(chk)
    if not chk["reproduces"]:
        raise SystemExit(f"TP3 run {run} does not reproduce: {chk}")
    X1, X2, XX = p7_grid2("TP3")
    # held-out sets, fixed seeds
    from scipy.stats import qmc
    Xh = qmc.scale(qmc.LatinHypercube(d=2, seed=HELDOUT_SEED).random(NH_TP3), [0, 0], [15, 15])
    th = np.array([truth("TP3", x)["theta"] for x in Xh])
    rng = np.random.default_rng(HELDOUT_SEED)
    near = []
    while len(near) < 500:
        z = X_STAR_TP3 + rng.uniform(-R_NEAR, R_NEAR, 2)
        if np.linalg.norm(z - X_STAR_TP3) <= R_NEAR and np.all((z >= 0) & (z <= 15)):
            near.append(z)
    Xn = np.array(near); tn = np.array([truth("TP3", x)["theta"] for x in Xn])
    feas = (Xh[:, 0] + 2 * Xh[:, 1] >= 30) & (Xh[:, 0] + Xh[:, 1] <= 25)
    res = {"run": run, "locked_best": best, "check": chk}
    for stage in ("init", "final"):
        s = snaps[stage]
        pr = snap_pred(s, XX)
        np.savez(os.path.join(P7_DIR, f"tp3_gp_{stage}.npz"), X1=X1, X2=X2,
                 mu=pr["mu"].reshape(X1.shape), sd=pr["sd"].reshape(X1.shape),
                 phi0_mu=pr["phi0_mu"].reshape(X1.shape), phi1_mu=pr["phi1_mu"].reshape(X1.shape),
                 phi0_sd=pr["phi0_sd"].reshape(X1.shape), phi1_sd=pr["phi1_sd"].reshape(X1.shape),
                 Xtr=s["X"], Ftr=s["F"], Gtr=s["G"], lengthscale=pr["ls"])
        gp = s["theta_gp"]
        res[stage] = {"n_train": int(len(s["F"])), "lengthscale": float(gp.lengthscale),
                      "signal_var": gp.signal_var, "noise_var": gp.noise_var,
                      "whole_box": gp_metrics(gp, Xh, th),
                      "near_optimum": gp_metrics(gp, Xn, tn),
                      "feasible_region": gp_metrics(gp, Xh[feas], th[feas])}
        # SA-APPA's incumbent: best feasible truly evaluated point in the archive
        Ftr, Gtr = s["F"], s["G"]
        ok = np.all(Gtr <= 1e-6, axis=1)
        k = int(np.argmin(np.where(ok, Ftr, np.inf)))
        res[stage]["incumbent_x"] = s["X"][k].tolist(); res[stage]["incumbent_F"] = float(Ftr[k])
    np.savez(os.path.join(P7_DIR, "tp3_heldout.npz"), Xh=Xh, th=th, Xn=Xn, tn=tn, feas=feas)
    return res


def part_A3():
    out = {}
    for iid in ["TP3", "TP4", "TP5", "TP7", "TP8"]:
        run, best = median_run(iid)
        snaps, chk = rerun_with_snapshots(iid, run)
        P7_LOG["reruns"].append(chk)
        if not chk["reproduces"]:
            raise SystemExit(f"{iid} run {run} does not reproduce: {chk}")
        X1, X2, XX = p7_grid2(iid)
        inst = SUITE[iid]
        if iid == "TP8":          # phi depends on y*: exact LL response
            PHI = np.array([truth(iid, x)["phi"] for x in XX])
        else:                     # phi depends on x only
            PHI = np.array([np.atleast_1d(inst.G(x, np.mean(inst.y_bounds, axis=1))) for x in XX])
        s = snaps["final"]
        pr = snap_pred(s, XX)
        m = PHI.shape[1]
        d = dict(X1=X1, X2=X2, Xtr=s["X"], Gtr=s["G"])
        for j in range(m):
            d[f"true{j}"] = PHI[:, j].reshape(X1.shape)
            d[f"mu{j}"] = pr[f"phi{j}_mu"].reshape(X1.shape)
            d[f"sd{j}"] = pr[f"phi{j}_sd"].reshape(X1.shape)
        Ftr, Gtr = s["F"], s["G"]
        ok = np.all(Gtr <= 1e-6, axis=1)
        k = int(np.argmin(np.where(ok, Ftr, np.inf))) if ok.any() else -1
        d["incumbent"] = s["X"][k] if k >= 0 else np.full(2, np.nan)
        np.savez(os.path.join(P7_DIR, f"{iid.lower()}_phi_final.npz"), **d)
        met = {}
        for j in range(m):
            err = pr[f"phi{j}_mu"] - PHI[:, j]
            met[f"phi{j}"] = dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                                  max_abs=float(np.max(np.abs(err))),
                                  coverage_2sd=float(np.mean(np.abs(err) <= 2 * pr[f"phi{j}_sd"])),
                                  sign_agreement=float(np.mean(np.sign(pr[f"phi{j}_mu"]) == np.sign(PHI[:, j]))))
        out[iid] = dict(run=run, locked_best=best, m_ul=m, metrics=met)
    return out


def part_A5():
    path = os.path.join(P7_DIR, "fidelity.json")
    out = (json.load(open(path)).get("A5", {}) if os.path.exists(path) else {})
    for iid in TP_LOWDIM + SMD_SMALL:
        if iid in out and os.path.exists(os.path.join(P7_DIR, f"a5_{iid}.npz")):
            continue
        t0 = time.time()
        run, best = median_run(iid)
        snaps, chk = rerun_with_snapshots(iid, run)
        P7_LOG["reruns"].append(chk)
        if not chk["reproduces"]:
            out[iid] = dict(run=run, error="does not reproduce", check=chk); continue
        n = 400 if iid.startswith("TP") else 200
        Xh, th, phih, keep = heldout(iid, n)
        gp = snaps["final"]["theta_gp"]
        mu, sd = gp.predict(Xh[keep])
        np.savez(os.path.join(P7_DIR, f"a5_{iid}.npz"), Xh=Xh[keep], th=th[keep], mu=mu, sd=sd)
        met = gp_metrics(gp, Xh[keep], th[keep])
        met.update(run=run, locked_best=best, n_dropped_ll_infeasible=int((~keep).sum()),
                   lengthscale=float(gp.lengthscale),
                   truth="exact QP" if iid in ("TP3", "TP8") else "reference DE (pop 30 n_ll, 300 gen, 2 seeds)")
        out[iid] = met
        cur = json.load(open(path)) if os.path.exists(path) else {}
        cur["A5"] = out
        cur.setdefault("reruns", []).append(chk)
        json.dump(cur, open(path, "w"), indent=1, default=float)   # checkpoint after every instance
        print(iid, f"{time.time()-t0:.0f}s", met["rmse"], met["coverage_2sd"], flush=True)
    return out


def phase7_run_main(parts=("A1", "A2", "A3", "A5")):
    parts = list(parts)
    path = os.path.join(P7_DIR, "fidelity.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if "A1" in parts:
        part_A1_A4(); res["A1"] = {"grid": f"{NG}x{NG}", "solver": "scipy lsq_linear (bvls), checked against closed-form projection",
                                   "x_star": X_STAR_TP3.tolist(), "theta_star": 325.0, "y_star": [10.0, 10.0], "fstar_star": 25.0}
    if "A2" in parts:
        res["A2"] = part_A2()
    if "A3" in parts:
        res["A3"] = part_A3()
    if "A5" in parts:
        res["A5"] = part_A5()
    res.setdefault("reruns", [])
    res["reruns"] = [r for r in res["reruns"] if (r["instance"], r["run"]) not in
                     {(c["instance"], c["run"]) for c in P7_LOG["reruns"]}] + P7_LOG["reruns"]
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("done", parts)


# ================================================================================================
# PHASE 7 PART B1: INSTRUMENTED RE-RUN (instrument_ll.py)
# ================================================================================================

# instrument_ll.py
# ================
# Instrumented re-run for the evaluation-count figures (Phase 7, Part B1).
#
# Re-runs runs 0-4 (seed = run_id, the locked seed convention) of all 14 configurations on all
# 44 instances with a counter wrapped around the lower-level objective f(x, y). The wrapper
# returns exactly the value of f, so the run is unchanged; this is verified by comparing every
# re-run with the locked results (summary.csv / summary_m14.csv and the trajectory files).
#
# Counted quantities per run:
#   n_ul      number of full nested LL solves (= UL evaluations), from the trajectory length
#   n_ll_f    number of calls to the LL objective f(x, y), over all nested solves
#
# Writes results/ll_counts_subset.csv (resumable) and nothing else; locked files are only read.

LL_FIELDS = ["method_id", "instance_id", "run_id", "n_ul", "n_ll_f", "best", "feasible_found",
          "locked_best", "best_match", "traj_match", "traj_max_rel_dev", "cpu_time_s"]
ALL_METHODS = list(METHOD_SPECS.keys()) + ["M14"]


class _LLCounter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, x, y):
        self.n += 1
        return self.f(x, y)


def _ll_load_locked():
    import pandas as pd
    d = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv")),
                   pd.read_csv(os.path.join(RESULTS_DIR, "summary_m14.csv"))])
    return {(r.method_id, r.instance_id, int(r.run_id)): (r.best, bool(r.feasible_found))
            for r in d.itertuples()}


def _ll_load_locked_traj(keys):
    want = set(keys); out = {}
    for fn in ("trajectories.jsonl", "trajectories_m14.jsonl"):
        with open(os.path.join(RESULTS_DIR, fn)) as fh:
            for line in fh:
                j = json.loads(line)
                k = (j["method_id"], j["instance_id"], int(j["run_id"]))
                if k in want:
                    out[k] = [round(r["best_feasible_obj"], 9) if r["best_feasible_obj"] is not None
                              and np.isfinite(r["best_feasible_obj"]) else None for r in j["rows"]]
    return out


REPRO_TOL = 1e-6   # relative tolerance for "reproduces"; the exact deviation is recorded as well


def _rel(a, b):
    if a is None or b is None:
        return 0.0 if (a is None and b is None) else float("inf")
    return abs(a - b) / max(1.0, abs(b))


def _same(a, b, tol=REPRO_TOL):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)):
        return (a is None or (isinstance(a, float) and np.isnan(a))) and \
               (b is None or (isinstance(b, float) and np.isnan(b)))
    return abs(a - b) <= tol * max(1.0, abs(b))


def _ll_worker(args):
    mid, iid, r, locked, ltraj = args
    inst = copy.copy(SUITE[iid])
    cnt = _LLCounter(inst.f)
    inst.f = cnt
    if mid == "M14":
        traj, cpu = run_m14(inst, seed=r)
    else:
        traj, cpu = run_method(inst, mid, seed=r)
    rows = [round(x["best_feasible_obj"], 9) if x["best_feasible_obj"] is not None
            and np.isfinite(x["best_feasible_obj"]) else None for x in traj.rows]
    if ltraj is not None and len(rows) == len(ltraj):
        dev = max((_rel(a, b) for a, b in zip(rows, ltraj)), default=0.0)
    else:
        dev = float("inf")
    tm = dev <= REPRO_TOL
    lb = locked[0]
    return dict(method_id=mid, instance_id=iid, run_id=r, n_ul=len(traj.rows), n_ll_f=cnt.n,
                best=traj.best, feasible_found=traj.found_feasible, locked_best=lb,
                best_match=_same(traj.best if traj.found_feasible else np.nan,
                                 lb if locked[1] else np.nan),
                traj_match=tm, traj_max_rel_dev=dev, cpu_time_s=cpu)


def instrument_ll_main(runs=5, workers=2):
    import types
    a = types.SimpleNamespace(runs=runs, workers=workers)
    order = ([i for i in SUITE if i.startswith("TP")] + [i for i in SUITE if i.endswith("p3q3")] +
             [i for i in SUITE if i.endswith("p30q30")] + [i for i in SUITE if i.endswith("p10q10")])
    done = set()
    if os.path.exists(LL_COUNTS_CSV):
        with open(LL_COUNTS_CSV) as fh:
            done = {(r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(fh)}
    keys = [(m, i, r) for i in order for m in ALL_METHODS for r in range(a.runs) if (m, i, r) not in done]
    locked = _ll_load_locked(); ltraj = _ll_load_locked_traj(keys)
    work = [(m, i, r, locked[(m, i, r)], ltraj.get((m, i, r))) for (m, i, r) in keys]
    print(f"{len(work)} runs to do ({len(done)} done)", flush=True)
    new = not os.path.exists(LL_COUNTS_CSV)
    fh = open(LL_COUNTS_CSV, "a", newline=""); w = csv.DictWriter(fh, fieldnames=LL_FIELDS)
    if new:
        w.writeheader()
    t0 = time.time(); n = 0; bad = 0
    with mp.Pool(a.workers) as pool:
        for row in pool.imap_unordered(_ll_worker, work, chunksize=1):
            w.writerow(row); fh.flush(); n += 1
            bad += (not row["best_match"]) or (not row["traj_match"])
            if n % 50 == 0 or n == len(work):
                print(f"{n}/{len(work)} done, {bad} mismatches, {(time.time()-t0)/60:.1f} min", flush=True)
    fh.close()


# ================================================================================================
# PHASE 7 FIGURES (figures_phase7.py)
# ================================================================================================

# figures_phase7.py
# =================
# Phase 7 figures, modelled on the visual ideas of Sinha & Shaikh (IEEE TCYB 52(10), 2022,
# Figs 4-9) but drawn from this study's own data and notation. No numbers from that paper are
# used or compared.
#
# Part A (surrogate fidelity) reads results/phase7/*.npz and fidelity.json written by
# sa_appa_experiments/phase7_run.py. SA-APPA's GPs model theta(x) = F(x, y*(x)) and the UL
# constraint maps phi_j(x); nothing here is drawn or described as an approximation of the
# reaction set Omega(x), the follower response y*(x) or the LL value function f*(x).
#
# Part B (evaluation counts) reads the locked trajectories (all 30 runs) and
# results/ll_counts_subset.csv (instrumented re-run of runs 0-4, instrument_ll.py).
#
# Every figure is its own file, written as .pdf and .eps with opaque fills only; a .png preview
# and a .caption.txt are written beside it, and every figure is listed in manifest.json.

# ----------------------------------------------------------------------------- style
# Colour and line style are the map used by every resubmission figure (figures_resub.STYLE);
# Phase 7 adds a marker and a bar hatch per configuration so that identity survives greyscale.
P = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, INK, MUTED = "#52514e", "#0b0b0b", "#8a8984"
METHOD_STYLE = {   # colour, line style, marker, hatch
    "M1": (P[0], "-", "o", ""),     "M2": (P[1], "-", "s", "////"), "M3": (P[2], "-", "^", "\\\\\\\\"),
    "M4": (P[3], "-", "D", "xxxx"), "M5": (P[4], "-", "v", "...."), "M6": (P[5], "-", "P", "----"),
    "M7": (P[6], "-", "X", "++++"), "M8": (P[7], "-", "<", "||||"), "M9": (GRAY, "--", ">", "oo"),
    "M10": (P[1], "-", "s", "\\\\||"), "M11": (P[2], "-", "^", "//--"), "M12": (P[3], "-", "D", "x.."),
    "M13": (P[3], "--", "d", "**"), "M14": (P[7], "-", "*", "OO"),
}
P7_LABEL = {"M1": "M1 SA-APPA", "M2": "M2 uniform init.", "M3": "M3 no GA", "M4": "M4 GA engine",
         "M5": "M5 DE engine", "M6": "M6 PSO engine", "M7": "M7 penalised EI", "M8": "M8 plain EI",
         "M9": "M9 fixed schedule", "M10": "M10 SA-DE-Pen", "M11": "M11 SA-PSO-RBF",
         "M12": "M12 Nest-GA (A)", "M13": "M13 Nest-GA (B)", "M14": "M14 BL-SAEA"}
GROUPS = {"ablA": ["M1", "M2", "M3", "M4", "M5", "M6"],
          "ablB": ["M1", "M7", "M8", "M9"],
          "ext": ["M1", "M10", "M11", "M12", "M13", "M14"],
          "all": [f"M{i}" for i in range(1, 15)]}
GROUP_TXT = {"ablA": "internal ablation, design and engine (M1-M6)",
             "ablB": "internal ablation, constraint handling and step-size rule (M1, M7-M9)",
             "ext": "Phase II comparators (M1, M10-M14)",
             "all": "all fourteen configurations"}
SETS = {"tp": [f"TP{k}" for k in range(1, 10)],
        "smd_s": [f"SMD{k}_p3q3" for k in range(1, 13)],
        "smd_m": [f"SMD{k}_p10q10" for k in range(1, 13)],
        "smd_l": [f"SMD{k}_p30q30" for k in range(1, 13)]}
SET_TXT = {"tp": "TP1-TP9", "smd_s": "SMD1-SMD12 at (p, q) = (3, 3)",
           "smd_m": "SMD1-SMD12 at (p, q) = (10, 10)", "smd_l": "SMD1-SMD12 at (p, q) = (30, 30)"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False,
                     "hatch.linewidth": 0.6, "legend.framealpha": 1.0, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": GRAY, "ytick.color": GRAY, "ps.fonttype": 42, "pdf.fonttype": 42})
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#123e73"])
P7_MANIFEST = []


def p7_lighten(c, a=0.6):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a)


def p7_save(fig, stem, caption, meta):
    os.makedirs(P7_FIG_DIR, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(P7_FIG_DIR, f"{stem}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(P7_FIG_DIR, f"{stem}.png"), dpi=150, bbox_inches="tight")
    open(os.path.join(P7_FIG_DIR, f"{stem}.caption.txt"), "w").write(caption + "\n")
    plt.close(fig)
    P7_MANIFEST.append(dict(file=stem, caption=caption, **meta))


def ax3d(figsize=(4.6, 3.8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=26, azim=-62)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(1.0); a.pane.set_facecolor((1, 1, 1, 1)); a.pane.set_edgecolor(to_rgb("#e4e3df") + (1,))
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True)); ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    return fig, ax


def zlabel(ax, txt):
    ax.set_zlabel(txt, rotation=90, labelpad=6)


def p7_J():
    return json.load(open(os.path.join(P7_DIR, "fidelity.json")))


# ============================================================================ PART A
def fig_A1():
    d = np.load(os.path.join(P7_DIR, "tp3_truth.npz"))
    xs, ts = d["xstar"], float(d["theta_star"])
    for key, zlab, stem, what in [
            ("fstar", "$f^*(\\mathbf{x})$", "A1a_tp3_llvalue_truth",
             "the LL optimal value function $f^*(\\mathbf{x})=f(\\mathbf{x},\\mathbf{y}^*(\\mathbf{x}))$"),
            ("theta", "$\\theta(\\mathbf{x})$", "A1b_tp3_theta_truth",
             "the UL objective on the follower's response, $\\theta(\\mathbf{x})=F(\\mathbf{x},\\mathbf{y}^*(\\mathbf{x}))$")]:
        fig, ax = ax3d()
        Z = d[key]
        ax.plot_wireframe(d["X1"], d["X2"], Z, rstride=2, cstride=2, color=GRAY, linewidth=0.45)
        zs = float(d["fstar"][0, 0] * 0 + (25.0 if key == "fstar" else ts))
        ax.scatter([xs[0]], [xs[1]], [zs], s=46, marker="o", facecolor="white", edgecolor=P[7],
                   linewidth=1.6, depthshade=False, zorder=10, label="bilevel optimum $\\mathbf{x}^*=(15,10)$")
        # UL feasible region (x1 + 2 x2 >= 30, x1 + x2 <= 25) drawn on the floor
        z0 = float(np.min(Z))
        g = np.linspace(0, 15, 200)
        ax.plot(g, (30 - g) / 2, z0, color=P[0], linewidth=1.0, label="UL constraint boundaries")
        ax.plot(g, 25 - g, z0, color=P[0], linewidth=1.0, linestyle="--")
        ax.set_xlim(0, 15); ax.set_ylim(0, 15)
        zlabel(ax, zlab)
        ax.legend(loc="upper left", fontsize=7)
        cap = (f"TP3 (Appendix A), ground truth: {what} on a {d['X1'].shape[0]} x {d['X1'].shape[1]} grid "
               "over the UL box [0, 15]^2. The LL problem min ||x - y||^2 over y in [0, 10]^2 is a strictly "
               "convex quadratic programme, so y*(x) = min(x, 10) is single-valued, as Assumption A2 requires; "
               "it was solved by bounded-variable least squares (SciPy lsq_linear, 'bvls') and checked "
               "against the closed-form projection. The kinks at x_i = 10 are where the LL bound becomes "
               "active. The open circle marks the bilevel optimum x* = (15, 10), y* = (10, 10), "
               "theta = 325, f* = 25; the blue lines on the floor are the UL constraint boundaries "
               "x1 + 2 x2 = 30 (solid) and x1 + x2 = 25 (dashed). Ground truth only; no method was charged.")
        p7_save(fig, stem, cap, dict(part="A1", instance="TP3", run=None, seed=None))


def fig_A2():
    fj = p7_J()["A2"]
    for stage, n_tr in (("init", 25), ("final", 60)):
        g = np.load(os.path.join(P7_DIR, f"tp3_gp_{stage}.npz"))
        t = np.load(os.path.join(P7_DIR, "tp3_truth.npz"))
        X1, X2, TH, MU, SD = g["X1"], g["X2"], t["theta"], g["mu"], g["sd"]
        Xtr, Ftr = g["Xtr"], g["Ftr"]
        inc = np.array(fj[stage]["incumbent_x"]); incF = fj[stage]["incumbent_F"]
        m = fj[stage]
        stg = ("after the 25-point initial design" if stage == "init"
               else "at the end of the run (60 nested solves)")
        meta = dict(part="A2", instance="TP3", run=fj["run"], seed=fj["run"], stage=stage)
        common = (f"TP3, SA-APPA (M1), run {fj['run']} (seed {fj['run']}), selected by a rule fixed in "
                  "advance: the lower median (15th of 30) of the locked final UL objectives. The GP is the "
                  f"one SA-APPA held {stg}, trained on {n_tr} truly evaluated points with its own settings "
                  f"(squared-exponential kernel, lengthscale {float(g['lengthscale']):.2f} from the median "
                  "heuristic, signal variance 1, outputs mean-centred and not rescaled); no hyperparameter "
                  "was tuned for this figure.")
        # (i) true theta vs GP mean
        fig, ax = ax3d()
        ax.plot_wireframe(X1, X2, TH, rstride=3, cstride=3, color=GRAY, linewidth=0.45, label="true $\\theta(\\mathbf{x})$")
        ax.plot_wireframe(X1, X2, MU, rstride=3, cstride=3, color=P[0], linewidth=0.55, label="GP posterior mean $\\mu_\\theta(\\mathbf{x})$")
        ax.scatter(Xtr[:, 0], Xtr[:, 1], Ftr, s=10, color=INK, depthshade=False, label="training points")
        ax.scatter([15], [10], [325], s=46, facecolor="white", edgecolor=P[7], linewidth=1.6,
                   depthshade=False, label="optimum $\\mathbf{x}^*$")
        ax.scatter([inc[0]], [inc[1]], [incF], s=50, marker="X", color=P[7], depthshade=False,
                   label="SA-APPA incumbent")
        zlabel(ax, "$\\theta$"); ax.legend(loc="upper left", fontsize=6.5)
        cap = (common + f" Grey: true theta(x); blue: GP posterior mean; black: training points; open circle: "
               f"optimum x* = (15, 10), theta = 325; cross: SA-APPA's best feasible evaluated point at this "
               f"stage, x = ({inc[0]:.2f}, {inc[1]:.2f}), theta = {incF:.2f}. On 2,500 held-out points over "
               f"the box the mean has RMSE {m['whole_box']['rmse']:.2f} and maximum absolute error "
               f"{m['whole_box']['max_abs']:.1f} (theta spans {m['whole_box']['theta_range']:.0f}); within "
               f"1.5 of x* the RMSE is {m['near_optimum']['rmse']:.2f}.")
        p7_save(fig, f"A2{'a' if stage == 'init' else 'd'}_tp3_theta_gp_mean_{stage}", cap, meta)
        # (ii) mean +/- 2 sd wireframes
        fig, ax = ax3d()
        ax.plot_wireframe(X1, X2, TH, rstride=3, cstride=3, color=GRAY, linewidth=0.45, label="true $\\theta$")
        ax.plot_wireframe(X1, X2, MU + 2 * SD, rstride=3, cstride=3, color=P[1], linewidth=0.5, label="$\\mu_\\theta+2\\sigma_\\theta$")
        ax.plot_wireframe(X1, X2, MU - 2 * SD, rstride=3, cstride=3, color=P[6], linewidth=0.5, linestyle="--", label="$\\mu_\\theta-2\\sigma_\\theta$")
        zlabel(ax, "$\\theta$"); ax.legend(loc="upper left", fontsize=6.5)
        cap = (common + f" Uncertainty of the GP: the surfaces mu +/- 2 sigma (multiplier 2) and the true "
               f"theta. Because the signal variance is fixed at 1 while theta varies by about "
               f"{m['whole_box']['theta_range']:.0f}, sigma is at most 1 and averages "
               f"{m['whole_box']['mean_sd']:.4f}; the two surfaces are indistinguishable at this scale. The "
               f"empirical coverage of the +/- 2 sigma interval on 2,500 held-out points is "
               f"{100 * m['whole_box']['coverage_2sd']:.1f}% over the box and "
               f"{100 * m['near_optimum']['coverage_2sd']:.1f}% within 1.5 of x*, against a nominal 95%: "
               "the GP's predictive uncertainty is not calibrated on this problem.")
        p7_save(fig, f"A2{'b' if stage == 'init' else 'e'}_tp3_theta_gp_band_{stage}", cap, meta)
        # (iii) error map
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        E = np.abs(MU - TH)
        from matplotlib.ticker import MaxNLocator
        lv = MaxNLocator(10).tick_values(0, np.nanmax(E))
        cs = ax.contourf(X1, X2, E, levels=lv, cmap=SEQ)
        cb = fig.colorbar(cs, ax=ax); cb.set_label("$|\\mu_\\theta(\\mathbf{x})-\\theta(\\mathbf{x})|$")
        g1 = np.linspace(0, 15, 100)
        ax.plot(g1, (30 - g1) / 2, color=INK, linewidth=0.8); ax.plot(g1, 25 - g1, color=INK, linewidth=0.8, linestyle="--")
        ax.axvline(10, color=MUTED, linewidth=0.6, linestyle=":"); ax.axhline(10, color=MUTED, linewidth=0.6, linestyle=":")
        ax.scatter(Xtr[:, 0], Xtr[:, 1], s=9, color=INK, zorder=3)
        ax.scatter([15], [10], s=40, facecolor="white", edgecolor=P[7], linewidth=1.5, zorder=4, clip_on=False)
        ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        imax = np.unravel_index(np.nanargmax(E), E.shape)
        cap = (common + f" Absolute error of the GP mean on a 61 x 61 grid, with training points (dots), the "
               f"optimum (open circle), the UL constraint boundaries (black) and the LL kink lines x_i = 10 "
               f"(dotted). The error is largest at ({X1[imax]:.1f}, {X2[imax]:.1f}), "
               f"{'a corner of the box far from the training points' if (X1[imax] in (0, 15) and X2[imax] in (0, 15)) else 'away from the training points'}; "
               "inside the feasible region it is smaller but does not vanish near the kink x_2 = 10, on which "
               "the optimum lies, because the smooth kernel cannot represent the change of slope there.")
        p7_save(fig, f"A2{'c' if stage == 'init' else 'f'}_tp3_theta_gp_error_{stage}", cap, meta)


def fig_A3():
    fj = p7_J()["A3"]
    names = {"TP3": ["$-(x_1+2x_2-30)$", "$x_1+x_2-25$"], "TP4": ["$x_1^2+2x_2-4$"],
             "TP5": ["$x_1-x_2$", "$x_1^2+x_2^2-100$"], "TP7": ["$x_1^2+x_2^2-1$"],
             "TP8": ["$x_1+x_2+y^*_1(\\mathbf{x})-2y^*_2(\\mathbf{x})-40$"]}
    for iid, info in fj.items():
        d = np.load(os.path.join(P7_DIR, f"{iid.lower()}_phi_final.npz"))
        for j in range(info["m_ul"]):
            fig, ax = ax3d()
            ax.plot_wireframe(d["X1"], d["X2"], d[f"true{j}"], rstride=3, cstride=3, color=GRAY, linewidth=0.45,
                              label=f"true $\\varphi_{j+1}(\\mathbf{{x}})$")
            ax.plot_wireframe(d["X1"], d["X2"], d[f"mu{j}"], rstride=3, cstride=3, color=P[0], linewidth=0.55,
                              label=f"GP mean $\\mu_{{\\varphi_{j+1}}}$")
            ax.scatter(d["Xtr"][:, 0], d["Xtr"][:, 1], d["Gtr"][:, j], s=9, color=INK, depthshade=False,
                       label="training points")
            inc = d["incumbent"]
            if np.all(np.isfinite(inc)):
                zi = float(np.interp(0, [0, 1], [0, 0]))
                ax.scatter([inc[0]], [inc[1]], [0.0], s=46, marker="X", color=P[7], depthshade=False,
                           label="SA-APPA incumbent (at $\\varphi=0$ level)")
            zlabel(ax, f"$\\varphi_{j+1}$"); ax.legend(loc="upper left", fontsize=6.5)
            mt = info["metrics"][f"phi{j}"]
            dep = ("depends on y*(x); its true values use the exact LL response (bound-constrained QP)"
                   if iid == "TP8" else "depends on x only")
            cap = (f"{iid}, UL constraint phi_{j+1}(x) = {names[iid][j].replace('$', '')} ({dep}): true map "
                   f"(grey) and the GP SA-APPA held at the end of run {info['run']} (seed {info['run']}; lower-median "
                   f"run of 30, rule fixed in advance), trained on 60 truly evaluated points. On the 61 x 61 grid "
                   f"the GP mean has RMSE {mt['rmse']:.3g}, maximum absolute error {mt['max_abs']:.3g}, "
                   f"+/- 2 sigma coverage {100 * mt['coverage_2sd']:.0f}% and predicts the sign of phi_{j+1} "
                   f"(feasible or not) correctly at {100 * mt['sign_agreement']:.1f}% of grid points."
                   + (" No run of the median rank found a feasible point on TP4, so no incumbent is marked."
                      if not np.all(np.isfinite(d['incumbent'])) else ""))
            p7_save(fig, f"A3_{iid.lower()}_phi{j+1}", cap, dict(part="A3", instance=iid, run=info["run"], seed=info["run"]))


def fig_A4():
    t = np.load(os.path.join(P7_DIR, "tp3_truth.npz"))
    for k, key in ((1, "y1"), (2, "y2")):
        fig, ax = ax3d()
        ax.plot_wireframe(t["X1"], t["X2"], t[key], rstride=2, cstride=2, color=GRAY, linewidth=0.45)
        zlabel(ax, f"$y^*_{k}(\\mathbf{{x}})$"); ax.set_xlim(0, 15); ax.set_ylim(0, 15)
        cap = (f"TP3, true follower response y*_{k}(x) = min(x_{k}, 10) over the UL box, from the exact "
               "solution of the strictly convex LL quadratic programme (single-valued, as Assumption A2 "
               "requires). SA-APPA does not model y*(x); no approximation is drawn. A diagnostic GP of y* "
               "was not produced, because the nested solves of the selected run do not store y and adding "
               "it would require new evaluations.")
        p7_save(fig, f"A4_tp3_ystar{k}", cap, dict(part="A4", instance="TP3", run=None, seed=None))


def fig_A5():
    fj = p7_J()["A5"]
    for iid, m in fj.items():
        if "error" in m:
            continue
        d = np.load(os.path.join(P7_DIR, f"a5_{iid}.npz"))
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        th, mu, sd = d["th"], d["mu"], d["sd"]
        ax.errorbar(th, mu, yerr=2 * sd, fmt="o", ms=2.6, color=P[0], ecolor=p7_lighten(P[0], 0.45),
                    elinewidth=0.6, capsize=0)
        lo, hi = float(min(th.min(), mu.min())), float(max(th.max(), mu.max()))
        ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.8, linestyle="--")
        ax.set_xlabel("true $\\theta(\\mathbf{x})$"); ax.set_ylabel("GP mean $\\mu_\\theta(\\mathbf{x})$ $\\pm 2\\sigma$")
        ax.set_title(iid.replace("_p3q3", " (3, 3)"), fontsize=8)
        cap = (f"{iid}: GP posterior mean of theta against the true theta on {m['n']} held-out LHS points, with "
               f"+/- 2 sigma bars, for the GP SA-APPA held at the end of run {m['run']} (lower-median run, rule "
               f"fixed in advance). True values from {m['truth']}"
               + (f"; {m['n_dropped_ll_infeasible']} points without an LL-feasible response were dropped" if m['n_dropped_ll_infeasible'] else "")
               + f". RMSE {m['rmse']:.3g}, coverage {100 * m['coverage_2sd']:.0f}% (nominal 95%). The mean "
               f"predictive standard deviation is {m['mean_sd']:.2g}, so most bars are narrower than the markers.")
        p7_save(fig, f"A5_{iid.lower()}", cap, dict(part="A5", instance=iid, run=m["run"], seed=m["run"]))


# ============================================================================ PART B
def load_traj():
    out = {}
    for fn in ("trajectories.jsonl", "trajectories_m14.jsonl"):
        with open(os.path.join(RESULTS_DIR, fn)) as fh:
            for line in fh:
                j = json.loads(line)
                b = [r["best_feasible_obj"] for r in j["rows"]]
                out[(j["method_id"], j["instance_id"], int(j["run_id"]))] = np.array(
                    [np.inf if v is None or not np.isfinite(v) else v for v in b])
    return out


def reference_values(T):
    """F_ref(i): best final feasible value over all configurations and runs, except that on
    SMD1-SMD9, whose UL optimum is F* = 0, F_ref = 0 (values below zero come from inexact LL
    responses, Section 5.6). Fixed before any evaluations-to-target figure was drawn."""
    ref = {}
    for i in SUITE:
        v = min(tr[-1] for (m, ii, r), tr in T.items() if ii == i)
        if i.startswith("SMD") and int(i.split("_")[0][3:]) <= 9:
            v = 0.0
        ref[i] = v
    return ref


def evals_to_target(T, ref, tol):
    rows = []
    for (m, i, r), tr in T.items():
        hit = np.nonzero(np.abs(tr - ref[i]) <= tol)[0] if np.isfinite(ref[i]) else []
        hit = [k for k in hit if np.isfinite(tr[k])]
        rows.append(dict(method_id=m, instance_id=i, run_id=r, hit=int(hit[0]) + 1 if len(hit) else np.inf))
    d = pd.DataFrame(rows)

    def lower_median(x):
        x = np.sort(x.values)
        return x[14] if len(x) == 30 else np.nan       # 15th of 30: finite iff >= 15 runs reach
    g = d.groupby(["instance_id", "method_id"]).hit
    return pd.DataFrame({"median": g.apply(lower_median), "n_reached": g.apply(lambda x: int(np.isfinite(x).sum())),
                         "n_runs": g.size()}).reset_index()


def legend_handles(methods, stacked=False):
    from matplotlib.patches import Patch
    hs = []
    for m in methods:
        c, ls, mk, h = METHOD_STYLE[m]
        hs.append(Patch(facecolor=p7_lighten(c, 0.15), edgecolor=INK if m == "M1" else c, hatch=h,
                        linewidth=0.5, label=P7_LABEL[m]))
    if stacked:
        hs.append(Patch(facecolor=MUTED, edgecolor=MUTED, label="lower segment: UL evaluations"))
        hs.append(Patch(facecolor="white", edgecolor=GRAY, hatch="////", label="upper segment: LL evaluations"))
    return hs


def bar_chart(table, value, inst, methods, ylabel, logy, annotate_missing, stacked=None):
    """Grouped bars, one group per instance, one bar per configuration (at most seven per group
    except in the all-configurations appendix version)."""
    width = 0.84 / len(methods)
    fig, ax = plt.subplots(figsize=(max(5.6, 0.16 * len(inst) * len(methods) + 1.2), 3.0))
    x0 = np.arange(len(inst))
    reached_any = {i: False for i in inst}
    for k, m in enumerate(methods):
        c, ls, mk, h = METHOD_STYLE[m]
        sub = table[table.method_id == m].set_index("instance_id").reindex(inst)
        xs = x0 - 0.42 + width * (k + 0.5)
        vals = sub[value].values.astype(float)
        ok = np.isfinite(vals)
        for i, o in zip(inst, ok):
            reached_any[i] |= bool(o)
        if stacked:
            lo = sub[stacked[0]].values.astype(float)
            ax.bar(xs[ok], lo[ok], width * 0.9, color=c, edgecolor=c, linewidth=0.4)
            ax.bar(xs[ok], (vals - lo)[ok], width * 0.9, bottom=lo[ok], color=p7_lighten(c, 0.6),
                   edgecolor=INK if m == "M1" else c, hatch=h, linewidth=0.4)
        else:
            ax.bar(xs[ok], vals[ok], width * 0.9, color=p7_lighten(c, 0.15), edgecolor=INK if m == "M1" else c,
                   hatch=h, linewidth=0.5)
        if annotate_missing:
            for xx, i, o in zip(xs, inst, ok):
                if not o:
                    ax.text(xx, 0.8, "target not reached", rotation=90, ha="center", va="bottom",
                            fontsize=5.0, color=GRAY, clip_on=False)
    ax.set_xticks(x0)
    ax.set_xticklabels([i.split("_")[0] for i in inst], fontsize=7)
    ax.set_xlim(-0.55, len(inst) - 0.45)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(handles=legend_handles(methods, bool(stacked)), ncol=4, fontsize=6, loc="lower left",
              bbox_to_anchor=(0, 1.01), handlelength=2.2, handleheight=1.1)
    return fig, ax


def fig_B2(T=None):
    T = T or load_traj()
    ref = reference_values(T)
    tables = {}
    for tag, tol in (("t1", 1e-2), ("t2", 1.0)):
        e = evals_to_target(T, ref, tol)
        e.to_csv(os.path.join(P7_FIG_DIR, f"B2_success_counts_{tag}.csv"), index=False)
        tables[tag] = e
    json.dump({k: float(v) for k, v in ref.items()}, open(os.path.join(P7_FIG_DIR, "B2_reference_values.json"), "w"), indent=1)
    # ---- B2
    for tag, tol in (("t1", 1e-2), ("t2", 1.0)):
        for sk, inst in SETS.items():
            if tag == "t2" and sk == "tp":
                continue
            inst = [i for i in inst if i in SUITE]
            for gk, methods in GROUPS.items():
                tb = tables[tag]
                fig, ax = bar_chart(tb, "median", inst, methods, "median nested solves to target",
                                    logy=False, annotate_missing=True)
                ax.set_ylim(0, 62)
                place = "appendix" if (sk == "smd_m" or gk == "all" or tag == "t2") else "main or appendix"
                excl = (" SMD10 at (30, 30) is excluded (no feasible point exists within the box; Appendix C)."
                        if sk == "smd_l" else "")
                cap = (f"Evaluations to target, {SET_TXT[sk]}, {GROUP_TXT[gk]}. Bar: median over the 30 runs "
                       f"(not 31) of the number of full nested LL solves after which the best feasible UL objective "
                       f"first lies within {tol:g} (absolute) of the reference value F_ref; F_ref is the best value "
                       "found by any configuration in any run, except on SMD1-SMD9 where F_ref = 0, the known UL "
                       "optimum. Runs that never reach the target are censored; where fewer than 15 of the 30 runs "
                       "reach it the median is undefined and 'target not reached' is written instead of a bar. "
                       "The budget is 60 nested solves. Success counts are in the companion table. Counts are in "
                       "nested solves, the study's cost unit; LL function evaluations per solve are in the "
                       "budget-composition figure." + excl)
                p7_save(fig, f"B2_{tag}_{sk}_{gk}", cap, dict(part="B2", instance_set=sk, group=gk, target=tol,
                                                         runs="0-29 (all, locked)", placement=place))
    return tables


def fig_B1():
    ll = pd.read_csv(os.path.join(RESULTS_DIR, "ll_counts_subset.csv"))
    ll["ok"] = ll.best_match & ll.traj_match
    good = ll[ll.ok]
    b1 = good.groupby(["instance_id", "method_id"]).agg(n_ul=("n_ul", "median"), n_ll=("n_ll_f", "median"),
                                                         n_runs=("run_id", "size")).reset_index()
    b1["total"] = b1.n_ul + b1.n_ll
    b1.to_csv(os.path.join(P7_FIG_DIR, "B1_budget_composition.csv"), index=False)
    for sk, inst in SETS.items():
        inst = [i for i in inst if i in SUITE]
        for gk, methods in GROUPS.items():
            fig, ax = bar_chart(b1, "total", inst, methods, "median evaluations (log scale)", logy=True,
                                annotate_missing=False, stacked=("n_ul",))
            nexcl = int((~ll.ok & ll.instance_id.isin(inst) & ll.method_id.isin(methods)).sum())
            excl = (" SMD10 at (30, 30) is excluded (no feasible point exists within the box; Appendix C)."
                    if sk == "smd_l" else "")
            cap = (f"Budget composition, {SET_TXT[sk]}, {GROUP_TXT[gk]}. Each bar is the median over runs 0-4 "
                   "(5 of the 30 locked runs, re-run under the same seeds with a counter on the LL objective f) "
                   "of the UL evaluations (solid lower segment; 60 full nested solves for every configuration) "
                   "plus the LL objective evaluations made inside those solves (hatched upper segment). Every "
                   "LL solve is a nested heuristic solve: SciPy differential evolution with population "
                   "N_ll = 6 n_ll, at most T_ll = 20 generations and L-BFGS-B polishing (M14 changes only its "
                   "initial population); none is exact. The y axis is logarithmic, so segment heights are not "
                   "proportional to counts; the UL share is under 1% of the total in every case."
                   + (f" {nexcl} re-run(s) that did not reproduce the locked trajectory within 1e-6 (relative) "
                      "were left out of the median." if nexcl else "") + excl)
            place = "appendix" if (sk == "smd_m" or gk == "all") else "main or appendix"
            p7_save(fig, f"B1_{sk}_{gk}", cap, dict(part="B1", instance_set=sk, group=gk, runs="0-4 (instrumented)",
                                                 placement=place))
    return b1


def figures_phase7_main(which=("A1", "A2", "A3", "A4", "A5", "B2", "B1")):
    for w in which:
        globals()[f"fig_{w}"]()
    mp_ = os.path.join(P7_FIG_DIR, "manifest.json")
    old = json.load(open(mp_)) if os.path.exists(mp_) else []
    keep = {m["file"] for m in P7_MANIFEST}
    json.dump([m for m in old if m["file"] not in keep] + P7_MANIFEST, open(mp_, "w"), indent=1, default=str)
    print(len(P7_MANIFEST), "figures written")



# ================================================================================================
# DRIVERS (run_all.py + run_m14.py, statistics, figures, Phase 7, self-validation)
# ================================================================================================
SWEEP_FIELDS = ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"]
ALL_CONFIGS = [f"M{i}" for i in range(1, 15)]


def _summary_paths(method_id):
    tag = "_m14" if method_id == "M14" else ""
    return (os.path.join(RESULTS_DIR, f"summary{tag}.csv"), os.path.join(RESULTS_DIR, f"trajectories{tag}.jsonl"))


def _sweep_worker(args):
    method_id, instance_id, run_id = args
    t0 = time.time()
    try:
        if method_id == "M14":
            traj, cpu = run_m14(SUITE[instance_id], seed=run_id)
        else:
            traj, cpu = run_method(SUITE[instance_id], method_id, seed=run_id)
        return dict(method_id=method_id, instance_id=instance_id, run_id=run_id, best=traj.best,
                    feasible_found=traj.found_feasible, cpu_time_s=cpu, rows=traj.rows, error=None)
    except Exception as e:  # recorded, never hidden
        return dict(method_id=method_id, instance_id=instance_id, run_id=run_id, best=np.nan,
                    feasible_found=False, cpu_time_s=time.time() - t0, rows=[], error=f"{type(e).__name__}: {e}")


def run_sweep(method_ids, instance_ids, n_runs, workers):
    """All (configuration, instance, run) triples, in parallel, resumable. Same file formats as
    run_all.py (M1-M13) and run_m14.py (M14)."""
    done = set()
    for m in set(method_ids):
        sp, _ = _summary_paths(m)
        if os.path.exists(sp):
            with open(sp, newline="") as fh:
                done |= {(r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(fh)}
    work = [(m, i, r) for m in method_ids for i in instance_ids for r in range(n_runs) if (m, i, r) not in done]
    print(f"sweep: {len(work)} runs to do, {len(done)} already done, {workers} workers", flush=True)
    if not work:
        return
    handles = {}
    for m in set(method_ids):
        sp, tp = _summary_paths(m)
        if sp not in handles:
            new = not os.path.exists(sp)
            sf = open(sp, "a", newline="")
            w = csv.DictWriter(sf, fieldnames=SWEEP_FIELDS)
            if new:
                w.writeheader()
            handles[sp] = (sf, w, open(tp, "a"))
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_sweep_worker, work, chunksize=1), 1):
            sf, w, tf = handles[_summary_paths(res["method_id"])[0]]
            w.writerow({k: res[k] for k in SWEEP_FIELDS}); sf.flush()
            tf.write(json.dumps(dict(method_id=res["method_id"], instance_id=res["instance_id"],
                                     run_id=res["run_id"], rows=res["rows"], error=res["error"])) + "\n")
            tf.flush()
            if res["error"]:
                print("ERROR", res["method_id"], res["instance_id"], res["run_id"], res["error"], flush=True)
            if n % 25 == 0 or n == len(work):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(work)} runs, "
                      f"ETA {(len(work) - n) * el / n / 3600:.2f} h", flush=True)
    for sf, w, tf in handles.values():
        sf.close(); tf.close()


def bootstrap_stored_results():
    """Copy the stored results from the repository, if present, into RESULTS_DIR (read-only use)."""
    if all(os.path.exists(os.path.join(RESULTS_DIR, f)) for f in STORED_FILES):
        return True
    if not os.path.exists(os.path.join(REPO_RESULTS, "summary.csv")):
        return False
    import shutil
    for f in STORED_FILES:
        src = os.path.join(REPO_RESULTS, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(RESULTS_DIR, f)):
            shutil.copy2(src, os.path.join(RESULTS_DIR, f))
    ref = os.path.join(REPO_RESULTS, "resub_numbers.json")
    if os.path.exists(ref):
        shutil.copy2(ref, os.path.join(RESULTS_DIR, "resub_numbers_reference.json"))
    print("stored results copied from", REPO_RESULTS)
    return True


def run_statistics(quick=False):
    """Scenario tables, every test statistic of the paper, CD summaries, Appendix B LaTeX."""
    df = load_results(include_m14=True)
    for s in SCENARIOS:
        r = run_scenario(df, s["num"])
        r["table"].to_csv(os.path.join(RESULTS_DIR, "scenario_tables", f"scenario_{s['num']}.csv"), index=False)
        print_scenario_report(r)
    if quick:
        os.environ["RESUB_ALLOW_PARTIAL"] = "1"
    try:
        compute_resub_numbers()
        if not quick:
            run_llcheck()
    except Exception as e:
        print(f"[statistics] resub numbers skipped: {type(e).__name__}: {e}")
    export_cd_summaries()
    cwd = os.getcwd()
    try:
        os.chdir(RESULTS_DIR)
        build_appendix_b_main_tex()
    finally:
        os.chdir(cwd)


def build_appendix_b_main_tex():
    all_tex = ["\\section{Scenario Results (Appendix B)}", "\\label{app:scenarios}", ""]
    for s in SCENARIOS:
        tex = build_scenario_tex(s["num"])
        with open(f"appendix_b_scenario_{s['num']}.tex", "w") as fh:
            fh.write(tex)
        all_tex.append(tex)
    with open("appendix_b_all.tex", "w") as fh:
        fh.write("\n".join(all_tex))
    print("Appendix B LaTeX written to", RESULTS_DIR)


def run_figures():
    figures_resub_main()
    make_all_cd_diagrams(os.path.join(RESULTS_DIR, "scenario_cd_summaries.json"), FIG_DIR)


def run_phase7(workers):
    """MAKE_EXTRA_FIGS: Part A data (re-runs under locked seeds, checked before use), the
    instrumented runs 0-4 for B1, then every Phase 7 figure."""
    phase7_run_main(("A1", "A2", "A3", "A5"))
    instrument_ll_main(runs=5, workers=workers)
    figures_phase7_main()


# ------------------------------------------------------------------------------ self-validation
def self_validate(workers, report_path):
    lines = [f"SA_APPA_4nd.py self-validation, {time.strftime('%Y-%m-%d %H:%M')}",
             f"numpy {np.__version__}, scipy {__import__('scipy').__version__}, pandas {pd.__version__}", ""]
    ok_all = True
    # 1. benchmark sanity and a known optimum
    bad = []
    for iid, inst in SUITE.items():
        x = inst.x_bounds.mean(axis=1); y = inst.y_bounds.mean(axis=1)
        vals = np.concatenate([[inst.F(x, y), inst.f(x, y)], np.atleast_1d(inst.G(x, y)), np.atleast_1d(inst.g(x, y))])
        if not np.all(np.isfinite(vals)):
            bad.append(iid)
    lines.append(f"[1] {len(SUITE)} instances evaluate to finite values at the box centre: {'yes' if not bad else bad}")
    ok_all &= not bad and len(SUITE) == 44
    t = truth("TP3", np.array([15.0, 10.0]))
    tp3_ok = abs(t["theta"] - 325.0) < 1e-9 and np.allclose(t["y"], [10, 10]) and abs(t["fstar"] - 25.0) < 1e-9
    lines.append(f"[2] TP3 at x = (15, 10): theta = {t['theta']:.6g}, y* = {t['y'].tolist()}, f* = {t['fstar']:.6g} "
                 f"(expected 325, [10, 10], 25): {'ok' if tp3_ok else 'MISMATCH'}")
    ok_all &= tp3_ok
    # 2. locked runs re-run
    locked = _ll_load_locked()
    keys = [(m, i, r) for (i, r) in VALIDATION_RUNS for m in ALL_CONFIGS if (m, i, r) in locked]
    ltraj = _ll_load_locked_traj(keys)
    lines.append(f"[3] re-run of {len(keys)} locked runs under their seeds (tolerance {REPRO_TOL:g} relative):")
    with mp.Pool(processes=max(1, min(workers, len(keys)))) as pool:
        res = pool.map(_ll_worker, [(m, i, r, locked[(m, i, r)], ltraj.get((m, i, r))) for (m, i, r) in keys])
    n_ok = 0
    for row in res:
        good = bool(row["best_match"] and row["traj_match"])
        n_ok += good
        if not good:
            lines.append(f"    {row['method_id']} {row['instance_id']} run {row['run_id']}: best {row['best']} vs "
                         f"stored {row['locked_best']}, max trajectory deviation {row['traj_max_rel_dev']:.3g}")
    lines.append(f"    {n_ok} of {len(keys)} reproduce the stored best value and trajectory")
    ok_all &= n_ok == len(keys)
    # 3. statistics against the stored reference
    ref_p = os.path.join(RESULTS_DIR, "resub_numbers_reference.json")
    new_p = os.path.join(RESULTS_DIR, "resub_numbers.json")
    if os.path.exists(ref_p) and os.path.exists(new_p):
        ref, new = json.load(open(ref_p)), json.load(open(new_p))
        dmax = 0.0
        for n, s in ref["scenarios"].items():
            t = new["scenarios"][n]
            dmax = max(dmax, abs(s["T2"] - t["T2"]), abs(s["p"] - t["p"]),
                       max(abs(s["mean_ranks"][m] - t["mean_ranks"][m]) for m in s["mean_ranks"]))
        hmax = max(abs(a["p_holm"] - b["p_holm"]) for a, b in zip(
            sorted(ref["holm"], key=lambda h: h["label"]), sorted(new["holm"], key=lambda h: h["label"])))
        st_ok = dmax < 1e-9 and hmax < 1e-9
        lines.append(f"[4] eight scenario tests and the 16-test Holm correction recomputed from the stored CSVs: "
                     f"largest difference from the reference {max(dmax, hmax):.2g}: {'ok' if st_ok else 'MISMATCH'}")
        ok_all &= st_ok
    else:
        lines.append("[4] statistics check skipped (no reference resub_numbers.json, e.g. after a fresh sweep)")
    lines += ["", "OVERALL: " + ("PASS" if ok_all else "SEE DEVIATIONS ABOVE")]
    open(report_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok_all


# ------------------------------------------------------------------------------ command line
def main():
    global QUICK_TEST, RUN_SWEEP, MAKE_EXTRA_FIGS, SELF_VALIDATE, N_WORKERS
    ap = argparse.ArgumentParser(description="SA-APPA single-file experiment code")
    ap.add_argument("--quick", action="store_true", help="3 instances x 3 runs smoke test (own output folder)")
    ap.add_argument("--run-sweep", action="store_true", help="run the experiment instead of using stored results")
    ap.add_argument("--extra-figs", action="store_true", help="MAKE_EXTRA_FIGS: Phase 7 figures")
    ap.add_argument("--validate-only", action="store_true", help="only the self-validation report")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--output", default=None, help="output folder (default ./sa_appa_4nd_output)")
    a = ap.parse_args()
    QUICK_TEST |= a.quick; RUN_SWEEP |= a.run_sweep; MAKE_EXTRA_FIGS |= a.extra_figs
    SELF_VALIDATE = SELF_VALIDATE and not a.no_validate; N_WORKERS = a.workers
    out = a.output or os.path.join(HERE, "sa_appa_4nd_quick_output" if QUICK_TEST else "sa_appa_4nd_output")
    configure_paths(out)
    print(f"SA_APPA_4nd.py  quick={QUICK_TEST} run_sweep={RUN_SWEEP} extra_figs={MAKE_EXTRA_FIGS} "
          f"workers={N_WORKERS} output={OUTPUT_DIR}", flush=True)

    if a.validate_only:
        bootstrap_stored_results()
        if not os.path.exists(os.path.join(RESULTS_DIR, "resub_numbers.json")):
            compute_resub_numbers()
        self_validate(N_WORKERS, os.path.join(OUTPUT_DIR, "validation_report.txt"))
        return
    if QUICK_TEST:
        insts, runs = ["TP1", "TP3", "SMD1_p3q3"], 3
        run_sweep(ALL_CONFIGS, insts, runs, N_WORKERS)
    else:
        have = bootstrap_stored_results() if not RUN_SWEEP else False
        if RUN_SWEEP or not have:
            run_sweep(ALL_CONFIGS, list(SUITE.keys()), 30, N_WORKERS)
    run_statistics(quick=QUICK_TEST)
    try:
        run_figures()
    except Exception as e:
        print(f"[figures] {type(e).__name__}: {e}")
    if MAKE_EXTRA_FIGS:
        if QUICK_TEST:
            print("[Phase 7] needs the full 44-instance, 30-run results; skipped in quick mode.")
        else:
            run_phase7(N_WORKERS)
    if SELF_VALIDATE and not QUICK_TEST:
        self_validate(N_WORKERS, os.path.join(OUTPUT_DIR, "validation_report.txt"))
    print("done; outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
