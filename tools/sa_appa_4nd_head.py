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
# @@IMPORTS@@

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
