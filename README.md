# SA-APPA supplementary codebase

This is the code and data package accompanying the manuscript "A Surrogate-Assisted
Prey-Predator Algorithm for Constrained Bilevel Problems: A Controlled Component Study
under a Matched Budget". It reproduces every number, table and figure reported in the
paper from either (a) the raw results included in this package, or (b) a full
from-scratch run of the 44-instance x 14-configuration x 30-run experiment.

Version 2.1.0 applies the strong Stackelberg (optimistic) selection among tied optimal
lower-level responses after every nested lower-level solve (`sa_appa_experiments/strong_stackelberg.py`).
The results without the selection are kept in `sa_appa_experiments/results/no_tie_selection/`
for comparison. See `CHANGELOG.md` for the files added in versions 2.0.0 and 2.1.0 and
the order in which to run them; `run_pipeline.py` regenerates the statistics and figures,
and `SA_APPA_4nd.py` (Python) and `sa_appa_matlab/SA_APPA_2nd.m` (MATLAB/Octave) are
single-file versions of the method and experiment.

## Layout

```
run_pipeline.py          single entry point (see below)
requirements.txt         numpy, pandas, scipy, matplotlib
sa_appa_experiments/      algorithms, benchmark suite, experiment runner,
                           statistical tests, raw results
    benchmark.py           the 44 reconciled problem instances (TP1-TP9,
                            SMD1-SMD12 x 3 scales, SMD10@(30,30) excluded
                            -- see Appendix C)
    methods.py              the 13 algorithm variants (M1-M13, Table 1)
    gp.py, ll_solver.py     Gaussian-process surrogate and lower-level
                            nested solver
    run_all.py               the experiment runner (resumable, seeded)
    scenarios.py             the 7 experimental scenarios (Appendix B)
    analysis.py               per-scenario Friedman/Conover/Nemenyi tests
    analysis_extra.py         feasibility-rate Friedman test and
                              constraint-density subset analysis +
                              Holm-Bonferroni (Appendix D.5-D.6)
    diagnostic_scenario5.py   the eta/beta/gamma/delta tuning diagnostic
                              (Appendix D.7)
    rerun_scenario5_tuned.py  the one-time full-suite re-run with the
                              diagnostic's best configuration
    friedman_test.py           Conover's (1998) F-approximation Friedman
                              test, unit-tested against Friedmantest.pdf's
                              worked example
    export_scenario_summaries.py  exports the numbers the CD diagrams
                              plot, as a thin wrapper over analysis.py's
                              own run_scenario() (guarantees the CD
                              diagrams cannot silently diverge from the
                              numbers already in Appendix B's tables)
    results/                   RAW RESULTS: summary.csv (one row per
                              method/instance/run), trajectories.jsonl
                              (full per-evaluation trajectories, 17,160
                              runs), scenario5_tuned_*.csv/.jsonl
                              (Appendix D.7's re-run), scenario tables
sa_appa_figures/           figure-generation code, reading results from
                           ../sa_appa_experiments/results/
    config.py                 method/instance metadata (Table 1's 13
                              variants, the 44-instance suite structure)
    real_data.py               loads trajectories.jsonl into the long-
                              format DataFrame the figure code expects
    figures/fig1_boxplot.py           Figure 1
    figures/fig2_performance_profile.py  Figure 2 (Dolan-More profile)
    figures/fig2b_data_profile.py     Figure 2b (More-Wild data profile)
    figures/fig3_convergence_curves.py   Figure 3 (44 per-instance files)
    figures/fig4_convergence_rate.py     Figure 4 (44 per-instance files)
    figures/fig5_cd_diagram_scenarios.py  the 7 CD diagrams (fig:cd1-cd7)
    output_real/                REGENERATED figure output (PNG/PDF/EPS +
                              a .caption.txt per figure)
```

Two directories rather than one because they were developed as separate
concerns (experiment execution vs. figure generation) and `sa_appa_figures`
reads `sa_appa_experiments`'s results as data rather than importing its
code, except for `benchmark.SUITE` (needed for per-instance dimension
lookups in the data profile) -- `run_pipeline.py` handles that one
cross-package import via `PYTHONPATH` rather than merging the two
directories.

## Quick start

```
pip install -r requirements.txt
python3 run_pipeline.py figures
```

This regenerates every figure in the paper (Figures 1-4 and the 7
critical-difference diagrams) from the raw results already included in
`sa_appa_experiments/results/` -- takes under two minutes and does not
re-run any optimisation. This is almost always what you want.

To also re-run the statistical tests (Appendix B/D) against the included
raw results:

```
python3 run_pipeline.py analysis
```

## Full reproduction from scratch

```
python3 run_pipeline.py experiments   # SLOW: ~17,160 runs, several hours
                                       # of wall time on the machine this
                                       # was developed on; resumable --
                                       # safe to Ctrl-C and re-invoke,
                                       # already-completed (method,
                                       # instance, run) triples are
                                       # skipped on restart
python3 run_pipeline.py analysis
python3 run_pipeline.py figures
```

or simply `python3 run_pipeline.py all`.

Every run is seeded deterministically (`hashlib.md5`-based
`_stable_seed()` keyed on method/instance/run id), so a from-scratch
re-run should reproduce the included `results/summary.csv` bit-for-bit
under the same NumPy/SciPy versions; minor floating-point differences
across NumPy versions are possible but should not change any Friedman
test's reject/fail-to-reject decision at the reported instance counts.

## What is NOT included

The MATLAB translation (validated against this Python implementation) and
the presentation materials are packaged separately -- see the manuscript's
Code and Data Availability section.

## Correspondence to the manuscript

| Manuscript item | Source |
|---|---|
| Table 1 (algorithm variants) | `sa_appa_figures/config.py::METHODS` |
| Appendix A (benchmark formulations) | `sa_appa_experiments/benchmark.py` |
| Appendix B (scenario tables) | `sa_appa_experiments/build_appendix_b.py`, `analysis.py` |
| Appendix C (SMD10 feasibility) | `sa_appa_experiments/benchmark.py` (constraint count), `results/summary.csv` |
| Appendix D.1-D.4 (statistical methodology) | `sa_appa_experiments/friedman_test.py` |
| Appendix D.5 (feasibility-rate Friedman test) | `sa_appa_experiments/analysis_extra.py::run_scenario_feasibility` |
| Appendix D.6 (constraint-density subset + Holm-Bonferroni) | `sa_appa_experiments/analysis_extra.py::run_scenario_subset`, `holm_bonferroni` |
| Appendix D.7 (Scenario 5 tuning diagnostic) | `sa_appa_experiments/diagnostic_scenario5.py`, `rerun_scenario5_tuned.py` |
| Appendix E (flowchart) | native TikZ in the manuscript source, no code dependency |
| Figures 1-4 | `sa_appa_figures/figures/fig{1,2,2b,3,4}*.py` |
| Figures fig:cd1-fig:cd7 | `sa_appa_figures/figures/fig5_cd_diagram_scenarios.py` |
| Section 5.8 (scale sensitivity) | Scenario 3's per-scale breakdown, computed from `results/summary.csv` |
| Scenario 8 and the 16-test Holm correction | `sa_appa_experiments/analysis_paper.py` (writes `results/paper_numbers.json`) |
| Strong Stackelberg selection (Section 5, Appendix F) | `sa_appa_experiments/strong_stackelberg.py`, `analysis_ss_compare.py` |
| Surrogate fidelity, budget composition, evaluations to target (Section 5.11, Appendix F) | `sa_appa_experiments/phase7_*.py`, `sa_appa_figures/figures_phase7.py` |

## Known simplifications

The data profile normalises by $n_{UL}$ (`fig2b_data_profile.py`), and the CD-diagram
ranking rule uses a simplified tie-break for configurations that are infeasible in every
run (`fig5_cd_diagram_scenarios.py`). Both choices are documented in the docstrings of
those files and in the manuscript.
