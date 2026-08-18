# SA-APPA MATLAB translation

## What this is, plainly

This is a MATLAB/Octave translation of the core SA-APPA algorithm
(Gaussian-process surrogates, the CEI infill criterion, the adaptive
Prey-Predator update law) from the validated Python implementation in
`sa_appa_experiments/`. It has **not been executed**: no MATLAB or Octave
interpreter was available in the sandboxed environment this translation
was produced in. `apt-get install octave` was attempted and failed --
the package mirror required for several of Octave's dependencies
returned HTTP 403 from that sandbox's network allowlist, and no MATLAB
license/installation was available at all. This is stated here directly,
per the project's standing rule against presenting unexecuted work as
validated: **do not treat any file in this package as numerically
verified until you have run `validate_against_python.m` yourself and
seen it pass.**

## What is and is not translated

Translated and covered by `validate_against_python.m`'s deterministic
checks (fixed inputs, exact expected outputs from the Python
implementation):

- `gp_surrogate.m` / `gp_predict.m` -- the GP surrogate (`gp.py`'s
  `GPSurrogate`), squared-exponential kernel, Cholesky training,
  predictive mean/variance (manuscript Eqs. 5-6).
- `expected_improvement.m` -- the EI acquisition (`methods.py::_ei`).
- `probability_of_feasibility.m` -- the PoF term of CEI
  (`methods.py::_pof`).
- `appa_step_deterministic.m` -- the adaptive Prey-Predator update
  formula, given explicit (not internally-drawn) random numbers so the
  comparison tests the formula, not two different RNGs (see that file's
  header for why).
- `benchmark_tp1.m`, `benchmark_tp2.m` -- two of the nine TP problems
  (Appendix A.1), transcribed directly from `benchmark.py`.

Translated but **not** independently checkable this way (composed from
the pieces above into a full stochastic optimisation loop, where
"correct" output is a random search trajectory, not a fixed number):

- `appa_step.m` -- production version of the PPA update, drawing from
  MATLAB/Octave's own RNG.
- `ll_solve_simple.m` -- a self-contained differential-evolution solver
  for the nested lower-level problem. This is a plain, dependency-free
  DE, not a translation of `scipy.optimize.differential_evolution`
  (which has its own specific internal algorithm); the two will not
  produce identical LL solves even given "the same seed," because they
  are different optimisers achieving the same goal, not the same code.
- `run_sa_appa.m` -- the full SA-APPA (M1) loop on TP1, composing all of
  the above: uniform-random initial design, GP surrogates, CEI-scored
  adaptive PPA search, infill, retraining, for `t_max` generations.
  Scoped to one representative problem (TP1) rather than the full
  44-instance suite, since translating and (once you have an
  interpreter) validating all 44 problems x 13 methods was out of scope
  for what could be produced without being able to run any of it here.
  Note this uses uniform-random sampling for the initial design, not
  Latin Hypercube Sampling like `methods.py::_lhs()` (which uses
  `scipy.stats.qmc.LatinHypercube`) -- a further scope reduction, stated
  here rather than left for you to discover.

**Why the full stochastic loop can't be validated the same way as the
building blocks**: NumPy's and MATLAB/Octave's random number generators
are different algorithms. Even with "the same seed," a Python run of
`run_ablation_method(..., method_id='M1', seed=0)` and a MATLAB run of
`run_sa_appa(0)` will diverge from the first random draw onward -- this
is an inherent property of comparing two different RNGs, not a
translation defect. What CAN be, and is, checked exactly is that each
deterministic building block computes the same formula the same way
given the same inputs. `run_sa_appa.m` is offered as a structural
translation -- correct algorithm shape, correct formulas -- not as
something whose search trajectory should match Python's.

## How to validate

```
cd sa_appa_matlab
octave validate_against_python.m     % or: run inside MATLAB
```

This loads `reference_values.json` (produced by
`generate_reference_values.py` against the real, validated Python code)
and checks every deterministic function above against it, printing
PASS/FAIL per check at a default tolerance of 1e-6. **Run this before
using any of these files for real work**, and if anything fails, treat
it as a translation bug to fix, not a tolerance to loosen.

To regenerate `reference_values.json` yourself (e.g. after changing the
Python source):

```
python3 sa_appa_matlab/generate_reference_values.py
```

## Requirements

Base MATLAB or GNU Octave (tested design target: Octave >= 7.1, for
`jsondecode` support; no toolboxes required -- `chol`, `erf`, `median`
and basic array operations are all that is used, and the normal
CDF/PDF are implemented directly via `erf` rather than relying on the
Statistics Toolbox's `normcdf`/`normpdf`).

## Files

| File | Role |
|---|---|
| `gp_surrogate.m`, `gp_predict.m`, `se_kernel.m`, `pairwise_dist.m` | GP surrogate |
| `expected_improvement.m`, `std_normal_cdf.m`, `std_normal_pdf.m` | EI acquisition |
| `probability_of_feasibility.m` | PoF / CEI's constraint term |
| `appa_step_deterministic.m` | PPA update, explicit-random-input validation variant |
| `appa_step.m` | PPA update, production (internal RNG) variant |
| `benchmark_tp1.m`, `benchmark_tp2.m` | Two of the nine TP benchmark problems |
| `ll_solve_simple.m` | Self-contained DE solver for the nested LL problem |
| `run_sa_appa.m` | Full SA-APPA (M1) loop on TP1 |
| `seed_rng.m` | Cross-MATLAB/Octave RNG seeding helper |
| `generate_reference_values.py` | Produces `reference_values.json` from the Python implementation |
| `reference_values.json` | Fixed input/output pairs to validate against |
| `validate_against_python.m` | Runs every deterministic check and reports PASS/FAIL |
