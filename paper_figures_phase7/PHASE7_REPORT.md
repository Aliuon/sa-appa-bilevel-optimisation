# Phase 7 report: Part A5 (held-out surrogate fidelity)

Status, 7 October 2026. A5 is complete (20 instances). B1 is still awaiting the author's decision. The manuscript has not been edited.

## What was measured

For each of TP1 to TP8 and SMD1 to SMD12 at (3, 3), the lower-median SA-APPA run (M1, rule fixed in advance: the 15th of 30 runs by final θ) was re-run with the same seed to recover the GP of θ that SA-APPA held at the end of the run. That GP was then evaluated on held-out Latin hypercube points (400 for TP, 200 for SMD; seed 20261007) that were never used in any run. The true θ at each point comes from the exact QP solution (TP3, TP8) or from a reference DE solve of the LL (population 30·n_ll, 300 generations, best of 2 seeds). Points with no LL-feasible response were dropped.

All 20 re-runs reproduce the stored trajectories (largest relative deviation 4.8e-8, tolerance 1e-6). No locked result was changed.

## A5 coverage table

| Instance | Run | Locked best θ | n | Dropped | Truth | θ range | RMSE | RMSE / range | Max abs error | Mean σ | ±2σ coverage (%) | Lengthscale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP1 | 19 | 0.002581 | 340 | 60 | reference DE | 7.707 | 0.0466 | 0.006 | 0.318 | 0.0043 | 52.4 | 1.64 |
| TP2 | 17 | -90.65 | 90 | 310 | reference DE | 298.8 | 58.7 | 0.196 | 161 | 0.0029 | 0.0 | 5.21 |
| TP3 | 5 | 325.2 | 400 | 0 | exact QP | 1018 | 10.8 | 0.011 | 86.5 | 0.0027 | 0.0 | 7.81 |
| TP4 | 1 | none feasible | 396 | 4 | reference DE | 377.4 | 0.447 | 0.001 | 3.49 | 0.0026 | 1.3 | 5.04 |
| TP5 | 0 | -48.98 | 369 | 31 | reference DE | 95.08 | 0.293 | 0.003 | 1.61 | 0.0019 | 1.4 | 5.09 |
| TP6 | 14 | 0.1 | 210 | 190 | reference DE | 1.971 | 0.000658 | 0.000 | 0.0015 | 0.00039 | 78.1 | 1.29 |
| TP7 | 10 | 0.2008 | 400 | 0 | reference DE | 1.909 | 0.841 | 0.441 | 4.51 | 0.0063 | 0.8 | 0.405 |
| TP8 | 21 | 0 | 400 | 0 | exact QP | 37.19 | 2.12 | 0.057 | 17.6 | 0.0021 | 1.0 | 27.4 |
| SMD1 (3,3) | 19 | 8.361 | 200 | 0 | reference DE | 257.4 | 17 | 0.066 | 63.7 | 0.098 | 0.0 | 12.9 |
| SMD2 (3,3) | 12 | 5.9 | 200 | 0 | reference DE | 245.8 | 8.21 | 0.033 | 31.9 | 0.082 | 0.5 | 10.1 |
| SMD3 (3,3) | 9 | 10.28 | 200 | 0 | reference DE | 257.4 | 22.9 | 0.089 | 117 | 0.11 | 1.0 | 12.2 |
| SMD4 (3,3) | 14 | 1.852 | 200 | 0 | reference DE | 245.1 | 11.5 | 0.047 | 82.7 | 0.074 | 2.5 | 8.38 |
| SMD5 (3,3) | 14 | 8.144 | 200 | 0 | reference DE | 252.9 | 12.7 | 0.050 | 42 | 0.093 | 1.5 | 13.3 |
| SMD6 (3,3) | 1 | 25 | 200 | 0 | reference DE | 363.5 | 194 | 0.533 | 616 | 0.065 | 0.0 | 14.6 |
| SMD7 (3,3) | 29 | 0.703 | 200 | 0 | reference DE | 44.29 | 3.03 | 0.068 | 12.5 | 0.048 | 1.5 | 11.3 |
| SMD8 (3,3) | 3 | 8.716 | 200 | 0 | reference DE | 115.2 | 7.55 | 0.066 | 29.6 | 0.08 | 1.5 | 13.6 |
| SMD9 (3,3) | 29 | 7.738 | 199 | 1 | reference DE | 245.8 | 6.77 | 0.028 | 33 | 0.049 | 1.5 | 11.6 |
| SMD10 (3,3) | 6 | 128.7 | 142 | 58 | reference DE | 157.4 | 20.6 | 0.131 | 86.6 | 0.071 | 0.0 | 14.2 |
| SMD11 (3,3) | 10 | none feasible | 200 | 0 | reference DE | 245.1 | 7.81 | 0.032 | 44.5 | 0.048 | 1.0 | 8.91 |
| SMD12 (3,3) | 6 | none feasible | 108 | 92 | reference DE | 489.9 | 59.3 | 0.121 | 327 | 0.11 | 0.0 | 15.8 |

*Locked best θ* is the stored final value of the selected run; "none feasible" means that run ended without a feasible incumbent. *RMSE / range* divides the RMSE by the range of the true θ over the held-out points.

## Reading

1. The ±2σ intervals are far too narrow. Coverage is at most 2.5% on 18 of 20 instances against a nominal 95%. The exceptions are TP1 (52%) and TP6 (78%), where θ varies over a small range. On the other 18 instances the RMSE is between 60 and 20,000 times the mean predictive standard deviation. The cause is the one found in A2: the GP uses signal variance 1 on unscaled outputs, so σ does not carry the scale of θ.
2. The posterior mean is reasonable on most instances. RMSE is below 7% of the θ range on 14 of 20 instances.
3. Four instances are poor: TP7 (44% of range), SMD6 (53%), TP2 (20%) and SMD10/SMD12 (12 to 13%). On SMD6 the GP mean spans roughly −250 to 800 while the true θ lies in 0 to 400 (figure A5_smd6_p3q3).
4. The lengthscale shown is the value at the end of the run. It is re-estimated by the median heuristic at every refit (see the open Table 3 wording item).

## Flags

- [NEEDS AUTHOR CHECK] SMD6 (3,3): a possible explanation is that SA-APPA's training targets came from the 60-solve nested DE, whose LL responses on SMD6 may differ from the reference DE. A difference in the LL responses would make the GP fit a different θ surface. This has not been tested.
- [NEEDS AUTHOR CHECK] TP4, SMD11 and SMD12: the lower-median run has no feasible incumbent, so at least 16 of the 30 runs are infeasible or the rule placed infeasible runs before feasible ones. The fidelity numbers are still valid (they measure the GP, not the incumbent). Please confirm the rule is acceptable before any A5 figure for these instances is cited.
- Drops are large on TP2 (310 of 400) and TP6 (190 of 400) because much of the box has no LL-feasible response; those RMSEs describe the feasible part only.

## Figures

There are 20 files, A5_<instance>.{eps,pdf,png}, each with a .caption.txt, listed in manifest.json. Each figure is a single panel with opaque fills. The captions now state the mean σ, because the error bars are mostly hidden behind the markers.

## Code propagation

- `sa_appa_figures/figures_phase7.py`: the A5 caption now states the mean σ.
- `SA_APPA_4nd.py`: rebuilt (4,109 lines), compiles.
- `SA_APPA_2nd.m`: the same caption change.

## Still pending

- B1 (budget composition): the author must choose one of three options.
  1. Draw it from the 1,224 reproducing runs (TP1 to TP9 and SMD at (3,3)), with a caption note.
  2. Resume the larger SMD scales.
  3. Drop B1.

  The instrumented re-run (`instrument_ll.py`) has not been restarted.
- The full Phase 7 deliverables pack (manifest, fidelity table, B2 tables, diff report, figure text drafts) follows once B1 is decided.
