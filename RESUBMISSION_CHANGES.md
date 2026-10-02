# Code changes for the resubmission (October 2026)

The 17,160 runs of M1–M13 (`results/summary.csv`, `results/trajectories.jsonl`) are unchanged.
`methods.py`, `gp.py`, `ll_solver.py` and `benchmark.py` were not modified.

New files
- `sa_appa_experiments/methods_m14.py`: matched-budget re-implementation of BL-SAEA
  (Jiang et al., IEEE TCYB 54(7), 2024), based on PlatEMO's BLSAEA.m / llSearch.m.
- `sa_appa_experiments/run_m14.py`: resumable runner (writes `results/summary_m14.csv`,
  `results/trajectories_m14.jsonl`; 1,320 runs, 0 errors).
- `sa_appa_experiments/analysis_resub.py`: all test statistics for the 8 scenarios, the
  feasibility and constraint-density tests and the 16-test Holm correction
  (writes `results/resub_numbers.json`, `results/scenario_tables/scenario_8.csv`).
- `sa_appa_experiments/analysis_s8_llcheck.py`: below-known-optimum check on SMD1–SMD9 and
  re-tests of Scenarios 3, 6 and 8 without the affected instances (adds keys to
  `resub_numbers.json`). Run it after `analysis_resub.py`.
- `sa_appa_figures/figures_resub.py`: split ablation / Phase II figures and CD diagrams
  (output in `paper_figures_resub/`).

Modified files
- `scenarios.py`: scenario names; Scenario 8 (M1 vs M14).
- `analysis.py`: `load_results(include_m14=True)`.
- `build_appendix_b.py`: corrected display labels (M5 = DE engine, M6 = PSO engine; M10–M13
  described as in-house baselines; M14 added).
- `sa_appa_figures/config.py`: labels and colours, M14 added.

Order to reproduce: `python run_m14.py` → `python analysis_resub.py` →
`python analysis_s8_llcheck.py` → `python export_scenario_summaries.py` →
`python ../sa_appa_figures/figures_resub.py`.

Not updated in this package: the standalone merge file `SA_APPA_4nd.py` and the MATLAB
port `SA_APPA_2nd.m` (not available in this session).
