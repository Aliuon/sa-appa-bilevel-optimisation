# Changelog

## v2.0.0 (October 2026)

The 17,160 runs of M1–M13 (`results/summary.csv`, `results/trajectories.jsonl`) are unchanged.
`methods.py`, `gp.py`, `ll_solver.py` and `benchmark.py` were not modified.

New files
- `sa_appa_experiments/methods_m14.py`: matched-budget re-implementation of BL-SAEA
  (Jiang et al., IEEE TCYB 54(7), 2024), based on PlatEMO's BLSAEA.m / llSearch.m.
- `sa_appa_experiments/run_m14.py`: resumable runner (writes `results/summary_m14.csv`,
  `results/trajectories_m14.jsonl`; 1,320 runs, 0 errors).
- `sa_appa_experiments/analysis_paper.py`: all test statistics for the 8 scenarios, the
  feasibility and constraint-density tests and the 16-test Holm correction
  (writes `results/paper_numbers.json`, `results/scenario_tables/scenario_8.csv`).
- `sa_appa_experiments/analysis_s8_llcheck.py`: below-known-optimum check on SMD1–SMD9 and
  re-tests of Scenarios 3, 6 and 8 without the affected instances (adds keys to
  `paper_numbers.json`). Run it after `analysis_paper.py`.
- `sa_appa_figures/figures_paper.py`: split ablation / Phase II figures and CD diagrams
  (output in `paper_figures_main/`).

Modified files
- `scenarios.py`: scenario names; Scenario 8 (M1 vs M14).
- `analysis.py`: `load_results(include_m14=True)`.
- `build_appendix_b.py`: corrected display labels (M5 = DE engine, M6 = PSO engine; M10–M13
  described as in-house baselines; M14 added).
- `sa_appa_figures/config.py`: labels and colours, M14 added.

Order to reproduce: `python run_m14.py` → `python analysis_paper.py` →
`python analysis_s8_llcheck.py` → `python export_scenario_summaries.py` →
`python ../sa_appa_figures/figures_paper.py`.

M14K option
- `methods_m14.py`: new option `ll_topk` (default False, so M14 is unchanged; verified bit-for-bit
  on TP3 and SMD1 (3,3), run 0). True trains the LL mapping models on the K most cosine-similar
  archive points, as in Jiang et al. (2024), Algorithm 4.
- `run_m14k.py` / `analysis_m14k.py`: M14K sensitivity run on the 20 instances where K < 60
  (600 runs; results/summary_m14k.csv); adds 's8_m14k' to paper_numbers.json.

## v2.1.0 (October 2026): strong Stackelberg selection as the main lower-level protocol

- `strong_stackelberg.py`: after every nested LL solve, replaces the returned reply by the
  follower's tied optimal reply that is best for the leader (D'Amato et al. 2012), with an accuracy
  gate, validation, materiality and tie tests (see the module docstring). Replaces the SMD6-only
  `optimistic_ll.py`, which could exploit inaccurate LL solves (SMD10).
- `ss_diagnostic.py`: 20 UL points per instance; the selection changes the reply on TP2, TP4,
  SMD6 and SMD11 (all scales) and on no other instance.
- `run_strong_stackelberg.py`: a one-run-per-configuration scan of the other instances and full
  runs (14 configurations x 30 runs) of the 11 instances where the selection acts: TP2, TP4,
  SMD6, SMD11 and SMD12 at three scales (SMD12 was added after the scan found selections in 9 of
  its 42 runs).
- `run_ss_extras.py`: Scenario 5 tuned diagnostic, M14K and the B1 instrumented counts on the 11
  instances with the selection.
- `merge_ss.py`: archives the no-selection result files in `results/no_tie_selection/` and
  merges the strong Stackelberg rows into the canonical files, which are now the main results.
- `analysis_ss_compare.py`: scenario tests under both protocols and selection statistics (adds
  key 'ss' to `paper_numbers.json`).
- `phase7_data.py`: re-runs and the ground truth use the selection (SA_APPA_LL_PROTOCOL=none
  restores the no-selection protocol); `phase7_smd6_check.py` describes the no-selection protocol and
  reads `results/no_tie_selection/`.
- `SA_APPA_4nd.py` and `SA_APPA_2nd.m` apply the selection after every LL solve (MATLAB needs
  fmincon for it; Octave uses sqp).

Order after the runs: `python merge_ss.py` -> `python analysis_paper.py` ->
`python analysis_s8_llcheck.py` -> `python analysis_m14k.py` -> `python analysis_ss_compare.py`
-> `python export_scenario_summaries.py` -> `python build_appendix_b.py` -> Phase 7 Part A for
the affected instances -> figures.
- File names: `analysis_paper.py`, `figures_paper.py`, `paper_figures_main/` and
  `results/paper_numbers.json` replace the earlier names of the same files; their content is
  unchanged.
- `paper_figures_phase7/PHASE7_REPORT.md` removed: its numbers predate the selection. The
  Phase 7 JSON files in `sa_appa_experiments/results/phase7/` and the manuscript are the reference.

### Carbon-tax case study (DBOP), added to v2.1.0
- `sa_appa_experiments/dbop_case.py`: the dynamic bilevel carbon policy problem of Haroon,
  Ngnotchouye and Tilahun (Engineering Letters 34(7), 2026) as an SA-APPA instance, recalibrated as
  agreed by the authors (production in GW, emissions in Mt CO2, revenue neutrality and Lemma 1 as 22
  hard UL constraints, per-technology capacity limits, closed-form follower response). Where
  D(t) <= 0 the MCE response is the capacity (IMCE.m sets 0 there, which is not the optimum).
  `patch_exact_ll()` routes the nested LL solve to the closed form for this instance only.
- `run_dbop.py`: reference optimum (constrained DE, 5 seeds), the 14 configurations x 30 runs x
  60 nested solves, and the source paper's AGA (100 x 100) x 30 runs.
- `analysis_dbop.py`: Friedman/Conover over the 30 seed-paired runs, Mann-Whitney SA-APPA vs AGA,
  `results/dbop/dbop_numbers.json` and `paper_figures_main/fig_dbop.{pdf,png}`.
- Order: `python run_dbop.py` -> `python analysis_dbop.py` (about 10 minutes on two cores).
