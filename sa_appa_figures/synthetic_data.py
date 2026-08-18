"""
synthetic_data.py
==================
********************  THIS MODULE PRODUCES FAKE DATA.  ********************

No real SA-APPA experiments have been run to produce this repository. The
paper's own Section 5 (as drafted) does not yet contain a reported results
table, and Task 4's instructions explicitly authorise shipping clearly
labelled placeholder data instead of inventing numbers that could be
mistaken for real findings.

Every array, DataFrame, and figure produced from this module is tagged
SYNTHETIC (function names, DataFrame columns, filenames, and on-figure
watermarks) so that nobody -- including future-you, six months from now --
mistakes a placeholder plot for a real result.

--------------------------------------------------------------------------
HOW TO REPLACE THIS WITH REAL DATA
--------------------------------------------------------------------------
Every figure function in `figures/` consumes one canonical long-format
DataFrame with exactly these columns:

    method_id        str   e.g. "M1"            (must be in config.METHOD_IDS)
    instance_id       str   e.g. "SMD3_p10q10"    (must be in config.INSTANCE_IDS)
    run_id            int   0 .. N_RUNS-1
    eval_count        int   cumulative number of *expensive* evaluations
                            (nested LL solves) spent so far
    iteration         int   generation index t (>= eval_count in general
                            since not every generation spends an expensive
                            evaluation, e.g. SA-APPA spends one per
                            generation after the initial LHS design)
    best_feasible_obj float running best feasible objective found so far;
                            NaN if no feasible point has been found yet
    is_feasible       bool  True once >=1 feasible point has been found

To use real results, write a loader that returns a DataFrame with this
schema (e.g. reading your experiment logs / CSVs) and pass it to the
figure functions in `main.py` instead of calling
`generate_synthetic_results()`. Nothing else needs to change -- that is
the entire point of the schema boundary.
"""

import hashlib
import numpy as np
import pandas as pd

from config import METHODS, METHOD_IDS, INSTANCES, N_RUNS, RANDOM_SEED

SYNTHETIC_DATA_WARNING = (
    "SYNTHETIC PLACEHOLDER DATA -- NOT REAL SA-APPA EXPERIMENTAL RESULTS"
)


def _stable_hash(*parts, mod=1_000_000):
    """
    Deterministic string -> int hash for seeding, stable ACROSS Python
    processes (unlike the builtin `hash()`, which is randomised per
    process via PYTHONHASHSEED for security reasons and therefore
    silently breaks the "seeded for reproducibility" requirement --
    identical `seed` values produced different synthetic data on every
    fresh `python3` invocation until this was found and fixed while
    validating Task 5's Friedman-test results, which need the same
    (instance, method) mean objectives on every run to be checked against
    each other and against the CD-diagram figure).
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % mod


def _synthetic_method_profile(rng, method_id):
    """
    Draw a *fixed-per-method* (quality, speed) profile, seeded, so that a
    given method behaves consistently (but still randomly and non
    dominantly) across problems within one call to
    generate_synthetic_results(). No method is constructed to "win" --
    quality and speed are independent random draws per method, consistent
    with the paper's own hedged framing (competitive with, not superior
    to, the field).
    """
    quality = rng.uniform(0.15, 0.85)   # lower = converges closer to the
                                         # instance's synthetic optimum
    speed = rng.uniform(0.05, 0.35)     # higher = fewer evaluations to
                                         # approach its own limit
    noise = rng.uniform(0.02, 0.12)     # per-run stochastic spread
    feas_prob_floor = 0.85 if method_id.startswith("M1") or "SA-" in method_id else 0.7
    return dict(quality=quality, speed=speed, noise=noise,
                feas_prob_floor=feas_prob_floor)


def _budget_for_instance(inst):
    """
    Synthetic per-instance evaluation budget. Scales with (n_ul + n_ll),
    consistent with the common expensive-optimisation convention flagged
    as a [FILL IN] in Task 3's Section 5 draft -- substitute the real
    budget rule here once you have it.
    """
    dim = inst.n_ul + inst.n_ll
    n0 = max(10, 11 * dim - 1)     # initial LHS-style design
    t_max = max(30, 6 * dim)       # generations / infill points
    return n0, t_max


def generate_synthetic_results(methods=None, instances=None,
                                n_runs=N_RUNS, seed=RANDOM_SEED):
    """
    Build a SYNTHETIC results table with the canonical schema described in
    this module's docstring.

    Convergence model (per method x instance x run), purely illustrative:

        best_feasible_obj(t) = f_inf + (f_0 - f_inf) * exp(-speed * t) + eps_t

    where f_inf, f_0 are synthetic per-instance normalising constants,
    `speed` and `quality` (which sets f_inf's offset) are per-method random
    draws (see `_synthetic_method_profile`), and eps_t is i.i.d. Gaussian
    noise scaled by the method's `noise` parameter. Feasibility is modelled
    as "not yet found" for the first few evaluations with a per-method
    random delay, then permanently feasible thereafter (a simplification;
    real runs can lose and regain feasibility, which this placeholder does
    not attempt to model).

    Returns
    -------
    pandas.DataFrame with columns
        method_id, instance_id, run_id, eval_count, iteration,
        best_feasible_obj, is_feasible
    """
    methods = methods or METHOD_IDS
    instances = instances or INSTANCES
    master_rng = np.random.default_rng(seed)

    # Fixed per-method profile, independent of instance, drawn once.
    method_profiles = {
        m: _synthetic_method_profile(
            np.random.default_rng(seed + _stable_hash(m, mod=10_000)), m
        )
        for m in methods
    }

    rows = []
    for inst in instances:
        n0, t_max = _budget_for_instance(inst)
        inst_rng = np.random.default_rng(seed + _stable_hash(inst.id, mod=10_000))
        f0 = inst_rng.uniform(50, 500)          # synthetic starting objective
        f_true_inf = inst_rng.uniform(0, 5)     # synthetic "true" optimum region

        for m in methods:
            prof = method_profiles[m]
            f_inf_m = f_true_inf + prof["quality"] * inst_rng.uniform(0, 20)
            for run_id in range(n_runs):
                run_rng = np.random.default_rng(
                    seed + _stable_hash(inst.id, m, run_id)
                )
                feas_delay = run_rng.integers(0, max(1, n0 // 4))
                best_so_far = np.inf
                found_feasible = False
                for t in range(n0 + t_max):
                    eval_count = min(t + 1, n0) if t < n0 else n0 + (t - n0 + 1)
                    raw = (f_inf_m + (f0 - f_inf_m) * np.exp(-prof["speed"] * t)
                           + run_rng.normal(0, prof["noise"] * (f0 - f_inf_m) * 0.05))
                    if t >= feas_delay:
                        found_feasible = True
                        best_so_far = min(best_so_far, max(raw, f_true_inf * 0.98))
                    rows.append(dict(
                        method_id=m, instance_id=inst.id, run_id=run_id,
                        eval_count=eval_count, iteration=t,
                        best_feasible_obj=(best_so_far if found_feasible else np.nan),
                        is_feasible=found_feasible,
                    ))

    df = pd.DataFrame(rows)
    df.attrs["SYNTHETIC"] = True
    df.attrs["warning"] = SYNTHETIC_DATA_WARNING
    return df
