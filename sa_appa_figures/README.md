# SA-APPA results figures

Modular figure-generation code for the SA-APPA paper's Section 5 results,
producing the five figure types requested in Task 4:

1. `figures/fig1_boxplot.py` -- solution-vs-method comparison (min-max
   normalised feasible objective, boxplotted by method, one panel per
   benchmark group).
2. `figures/fig2_performance_profile.py` -- Dolan-Moré-style performance
   profile across all reconciled instances (explicitly states on its face
   that "best" means empirical best found, not a certified optimum).
3. `figures/fig3_convergence_curves.py` -- mean best-feasible-objective-so-
   far vs. evaluation count, with a min/max band across runs, one figure
   per problem instance.
4. `figures/fig4_convergence_rate.py` -- log-log |f_t - f_best,run| vs.
   iteration, with empirical convergence order fitted by log-log linear
   regression, one figure per problem instance (all methods overlaid; see
   that file's docstring for why this isn't exploded into one file per
   method-instance pair, and how to do that if you want it).
5. `figures/fig5_cd_diagram.py` -- critical-difference diagram from the
   Friedman/Nemenyi ranking. Produces TWO diagrams on a full run
   (`fig5_cd_diagram_phase1`, `fig5_cd_diagram_phase2`), one per Task 5's
   Phase I / Phase II method sets, since Friedman's test compares a fixed
   set of treatments across blocks and the paper keeps those as two
   separate research questions.

`friedman_test.py` (package root) is a separate, standalone module (Task
5): Conover's (1998) F-approximation omnibus Friedman test plus its
paired-t post-hoc critical difference, unit-tested against the attached
Friedmantest.pdf methodology note's own worked example (n=21, k=4 ->
T2=25.74, CD=15.08 at alpha=0.01; run `python3 friedman_test.py` to
re-verify). `figures/fig5_cd_diagram.py` imports its F-statistic from
here rather than computing its own -- see "Bug found and fixed" below.
`task5_analysis.py` runs it on Phase I and Phase II separately and prints
the full A2/B2/T2/critical-value/pairwise-CD report in the worked
example's own presentation style.

**Nemenyi (CD diagram) vs. Conover (text report) are DIFFERENT
procedures, both reported, not to be conflated**: the CD diagram's
connecting bars use the Nemenyi critical distance (the conventional
basis for CD diagrams, Demsar 2006) on MEAN ranks via the studentized
range distribution; `task5_analysis.py`'s pairwise table uses Conover's
paired-t critical difference on RANK SUMS, per the Friedmantest.pdf
note. Both are valid post-hoc procedures for the same Friedman omnibus
result but are numerically different quantities -- see
`friedman_test.py`'s module docstring.

## IMPORTANT: this ships with synthetic placeholder data

No real SA-APPA experiments have been run to produce any figure in this
package's default `output/` folder. `synthetic_data.py` generates clearly
labelled fake data (every filename, caption, and figure watermark says so)
because the paper does not yet have a reported results table to draw real
numbers from, and Task 4 explicitly asked for this to be handled by marking
placeholder data rather than inventing numbers.

**To use real results**: write a loader that returns a DataFrame with the
schema documented at the top of `synthetic_data.py` (method_id,
instance_id, run_id, eval_count, iteration, best_feasible_obj,
is_feasible), and pass it to `run_all()` in `main.py` in place of
`generate_synthetic_results(...)`. No other file needs to change -- every
figure function only ever touches that one schema.

## Structure

```
config.py             Table 1 (13 methods) and the Task-2-reconciled
                       44-instance benchmark suite, as structured data --
                       nothing here is experimental output.
synthetic_data.py      Fake-data generator (SYNTHETIC, see above).
friedman_test.py        Task 5: Conover F-approximation Friedman test +
                        post-hoc critical difference, validated against
                        the Friedmantest.pdf worked example.
task5_analysis.py       Task 5: runs friedman_test.py on Phase I and
                        Phase II separately, prints the full report.
figures/
  common.py            save_figure() -- writes PNG+PDF+caption.txt per
                        figure, applies the SYNTHETIC watermark.
  fig1_boxplot.py
  fig2_performance_profile.py
  fig3_convergence_curves.py
  fig4_convergence_rate.py
  fig5_cd_diagram.py    Imports its F-statistic from friedman_test.py.
main.py                Orchestrates: builds results data, calls all five
                        figure functions, writes ./output/.
```

## Running it

```bash
python3 main.py              # full run: 13 methods x 44 instances
python3 main.py --quick      # 4 methods x 4 instances, for fast iteration
python3 friedman_test.py     # unit-test the Friedman/CD formulas against
                              # the Friedmantest.pdf worked example
python3 task5_analysis.py    # Task 5: full Phase I / Phase II Friedman
                              # report (text) on the synthetic data
```

Full run takes ~2 minutes on a single core (mostly matplotlib savefig
overhead across ~90 figures) and writes 92 figures (fig5 is now two
files, one per phase; 276 files counting PNG+PDF+caption per figure) to
`./output/`.

## Bug found and fixed (Task 5)

Two bugs were found while implementing Task 5 and are now fixed:

1. **Friedman F-statistic (`figures/fig5_cd_diagram.py`)**: Task 4's
   inline `_friedman_stat()` used the total sum of squares (`A2-C1`) as
   its denominator instead of the error sum of squares (`A2-B2/n`), and
   its numerator was `n` times too large. On the Friedmantest.pdf worked
   example this produced T=236.38 against the correct T2=25.74 (~9x
   inflated), which would have overstated significance throughout.
   Fixed by routing through the new, validated `friedman_test.py`
   instead of duplicating the formula inline.
2. **Synthetic-data seeding (`synthetic_data.py`)**: per-method and
   per-instance RNG seeds were derived using Python's builtin `hash()`
   on strings, which is randomised per process (`PYTHONHASHSEED`) unless
   explicitly disabled -- so `generate_synthetic_results(seed=X)`
   silently produced DIFFERENT data on every fresh `python3` invocation
   despite an identical seed, breaking the "seeded for reproducibility"
   requirement from Task 4. Fixed with a deterministic MD5-based string
   hash (`_stable_hash()`); verified identical output across separate
   process invocations after the fix.

## Known limitations / things flagged for author attention

- Figure 5's ranking rule ties all-infeasible methods on an instance
  rather than breaking ties by constraint violation (Task 3's full rule
  does the latter; the canonical schema here has no violation column --
  see `fig5_cd_diagram.py`'s docstring for how to extend it).
- The Friedman-test methodology (Figure 5) and the CD-diagram convention
  (Demšar, 2006) are cited inline as `[41]` / "Demšar, 2006" respectively
  -- neither is yet in the paper's bibliography; see
  `claude/task3-section5-rewrite.md` for the citation-pool caveat already
  raised on the Friedman-test source, which applies here too.
- `_budget_for_instance()` in `synthetic_data.py` uses a placeholder
  evaluation-budget rule (`11*dim-1` initial design, `6*dim` generations)
  matching the `[FILL IN]` flagged in Task 3's Section 5 draft -- replace
  with the real budget rule once you have it, so synthetic (or real)
  x-axes in Figures 3/4 are dimensioned correctly.
