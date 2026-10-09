# Phase 7 report (final)

Status, 8 October 2026: Phase 7 is complete and the manuscript has been updated (author approved). The PDF is 56 pages, with 0 LaTeX errors and 0 undefined references. One [NEEDS AUTHOR CHECK] remains: the release DOI.

## 1. Manifest

`paper_figures_phase7/manifest.json` lists 84 figures. Each figure is a single panel, saved as .eps, .pdf and .png with a caption file, and all fills are opaque.

| Part | Figures | What they show |
|---|---|---|
| A1 | 2 | TP3 ground truth (exact QP) |
| A2 | 6 | SA-APPA's GP of θ on TP3 after the initial design and at the end of the run |
| A3 | 7 | Constraint GPs on TP3, TP4, TP5, TP7 and TP8 |
| A4 | 2 | TP3 y*(x) |
| A5 | 20 | Held-out fidelity |
| A6 | 3 | SMD6 diagnosis and the tie-break result |
| B1 | 16 | Budget composition |
| B2 | 28 | Evaluations to target |

### Placement in the manuscript

The main text (Section 5.11) has three figures:

- TP3 truth beside the final GP mean (A1b and A2d, two separate files);
- B1 on TP1–TP9 for M1 and M10–M14;
- B2 at the 1e-2 target on TP1–TP9 for M1 and M10–M14.

Appendix F has nine figures:

- A2e, A2f and A3 (TP3, constraint 1);
- A6a, A6b and A6c;
- the three SMD B1 figures;
- B2 at the target of 1 on SMD (3,3).

Appendix F also has five tables:

- A5 fidelity;
- SMD6 means with the locked and the tie-break solve;
- scenario tests with the tie-break;
- B1 by family;
- B2 success counts.

All other figures are in the code package.

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


## 2. SMD6

### Diagnosis

The LL of SMD6 is optimal on a whole line t1 = t2, and F adds 2t². The nested DE solve therefore returns θ_opt + 2t² with t arbitrary.

- **Training targets:** the gap at SA-APPA's training points has median 32.7 and maximum 180.7.
- **SA-APPA's GP:** RMSE 204 against the optimistic θ.
- **Same GP rule, consistent targets:** RMSE 10.0. This is a diagnostic, not an SA-APPA run.

### Fix chosen: optimistic tie-break after each LL solve

After each LL solve returns a point, SLSQP minimises F subject to f ≤ f0 + 1e-6·max(1, |f0|), starting from that point. The result is kept only if it lowers F.

- This is the strong Stackelberg selection of D'Amato et al. (2012), verified from the full text you supplied and now in the bibliography.
- **Cost:** a median of 9471 F evaluations and 10028 f evaluations per run.
- **Applied to SMD6 only.** On an inaccurate LL solve, the tie-break exploits the sub-optimal level set: on SMD10 (3,3) a single test moved F from 2.01 to 0.29. SMD11's F is constant over its tie, so it is unaffected.

### Sensitivity run

All 14 configurations × 3 SMD6 scales × 30 runs = 1,260 runs, with the same seeds and budget. The locked results are unchanged.

- **SA-APPA's own GP on SMD6 (3,3):** RMSE 7.5 (3% of the range), correlation 0.99, coverage 1.5%. Calibration is unchanged.
- **Rank of M1 on SMD6:**
  - (3,3): 10 → 8;
  - (10,10): 8 → 11;
  - (30,30): 8 → 11.

  Most configurations gain too; M14's mean falls to 0.39, 3.57 and 0.07 at the three scales.
- **Scenario tests (Conover F-approximation p-values):** no decision changes, before or after the Holm correction, and every scenario keeps its best-ranked configuration. Scenario 4's raw p rises from 0.0147 to 0.0359; it is still not significant after Holm.

| Scenario | Question | p locked | p tie-break | best locked | best tie-break |
|---|---|---|---|---|---|
| 1 | Initial design | 0.173 | 0.452 | M1 | M1 |
| 2 | GA hybridisation | 0.279 | 0.441 | M1 | M1 |
| 3 | Search engine | 8.6e-10 | 8.6e-10 | M6 | M6 |
| 4 | Constraint handling | 0.0147 | 0.0359 | M1 | M1 |
| 5 | Step-size rule | 2.71e-06 | 2.71e-06 | M9 | M9 |
| 6 | Surrogate-assisted engine baselines | 8.19e-14 | 8.19e-14 | M11 | M11 |
| 7 | Fully nested baseline | 0.761 | 0.743 | M1 | M1 |
| 8 | Published surrogate-assisted bilevel method | 3.98e-08 | 3.98e-08 | M14 | M14 |

## 3. TP4, SMD11 and SMD12

The rule fixed in advance was kept: the lower median, with runs that found no feasible point placed last. On these three instances fewer than 15 of the 30 runs are feasible.

- They are marked † in the A5 table and are not discussed individually in the main text.
- Their fidelity values are typical of the table.

## 4. B1 (budget composition)

All 3,080 instrumented runs are complete: runs 0–4 for every configuration on every instance, with the same seeds, the LL objective counted, and a relative tolerance of 1e-6.

- 74 runs (2.4%) did not reproduce their stored trajectory, all with the PPA engine (M1, M2, M3, M7, M8). They are listed in `B1_nonreproducing_runs.csv` and excluded from the medians.
- The counter itself is neutral: a run gives the same result with and without it.
- UL evaluations are at most 0.77% of the total.

## 5. B2 (evaluations to target)

The success counts are in Appendix F, Table 26, and in `B2_success_counts_t1.csv` and `B2_success_counts_t2.csv`.

## 6. Code propagation

**Python modules:**

- `phase7_smd6_check.py`
- `optimistic_ll.py`
- `run_optimistic_ll.py`
- `analysis_optll.py`
- `phase7_smd6_optll_fidelity.py`
- `figures_phase7.py` (new figures A6 and A6c; B1 skips families with no data)
- `instrument_ll.py` (order changed only)

**Single file:** `SA_APPA_4nd.py` (4,594 lines) contains all of the above. With the tie-break on, it reproduces the module results exactly (M1 and M14 on SMD6, run 0).

**MATLAB:** `SA_APPA_2nd.m` has the A6 diagnosis (`phase7PartA6`), tested in Octave. The tie-break sensitivity run is not ported, because MATLAB's constrained local solver (`fmincon`) needs the Optimization Toolbox; the Python files are the reference.

## 7. Flags

- [NEEDS AUTHOR CHECK] Release DOI in "Code and data availability" (unchanged).
- The manuscript cites Sinha and Shaikh only as "similar in kind" for these analyses; it makes no claim about their figures.
