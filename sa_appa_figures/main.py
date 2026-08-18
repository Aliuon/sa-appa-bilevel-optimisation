"""
main.py
=======
Entry point: builds (synthetic, by default) results data and produces all
five figure types into ./output.

Usage
-----
    python3 main.py                    # full run, all 44 instances (slow-ish
                                        # for figures 3/4, which are one file
                                        # per instance)
    python3 main.py --quick            # small subset of methods/instances,
                                        # for fast smoke-testing

To use real results instead of the synthetic placeholder generator: write a
function returning a DataFrame with the schema documented at the top of
synthetic_data.py, and pass it as `results_df` to `run_all()` in place of
`generate_synthetic_results(...)`. Nothing else in this file, or in
figures/, needs to change.
"""

import argparse
import sys

from config import METHOD_IDS, INSTANCES, N_RUNS, PHASE_I_METHODS, PHASE_II_METHODS
from synthetic_data import generate_synthetic_results, SYNTHETIC_DATA_WARNING
from figures import (
    make_figure1, make_figure2, make_figure3_all, make_figure4_all,
    make_figure5,
)

OUT_DIR = "output"


def run_all(results_df, out_dir=OUT_DIR, method_ids=None, instance_ids=None,
            is_synthetic=None):
    _is_full_run = method_ids is None  # before reassignment below; used to
                                        # decide Figure 5's phase-split (see
                                        # Figure 5 block)
    method_ids = method_ids or METHOD_IDS
    instance_ids = instance_ids or [inst.id for inst in INSTANCES]

    written = {}
    print("[1/5] Figure 1: solution-vs-method boxplots ...")
    written["figure1"] = make_figure1(results_df, out_dir, method_ids=method_ids,
                                       is_synthetic=is_synthetic, n_runs=N_RUNS)

    print("[2/5] Figure 2: performance profile ...")
    written["figure2"] = make_figure2(results_df, out_dir, method_ids=method_ids,
                                       is_synthetic=is_synthetic)

    print(f"[3/5] Figure 3: convergence curves ({len(instance_ids)} instances) ...")
    written["figure3"] = make_figure3_all(results_df, out_dir, method_ids=method_ids,
                                           is_synthetic=is_synthetic,
                                           instances=instance_ids, n_runs=N_RUNS)

    print(f"[4/5] Figure 4: convergence-rate plots ({len(instance_ids)} instances) ...")
    written["figure4"] = make_figure4_all(results_df, out_dir, method_ids=method_ids,
                                           is_synthetic=is_synthetic,
                                           instances=instance_ids, n_runs=N_RUNS)

    print("[5/5] Figure 5: critical-difference diagram(s) ...")
    if _is_full_run:
        # Full run: Task 5 requires Friedman's test to compare the same
        # set of treatments across every block, and the paper's own
        # Phase I (internal ablation) / Phase II (external comparison)
        # split is two separate research questions -- so produce two
        # phase-specific CD diagrams rather than one 13-method diagram
        # that mixes them. See task5_analysis.py for the accompanying
        # Conover-F omnibus test + post-hoc text report.
        written["figure5_phase1"] = make_figure5(
            results_df, out_dir, method_ids=PHASE_I_METHODS,
            is_synthetic=is_synthetic, phase_label="Phase I (internal ablation, M1-M9)",
            filename_stem="fig5_cd_diagram_phase1",
        )
        written["figure5_phase2"] = make_figure5(
            results_df, out_dir, method_ids=PHASE_II_METHODS,
            is_synthetic=is_synthetic, phase_label="Phase II (external comparison, M1,M10-M13)",
            filename_stem="fig5_cd_diagram_phase2",
        )
    else:
        # --quick smoke test: method_ids is a deliberately mixed small
        # subset, not a real phase, so just produce one diagram over it.
        written["figure5"] = make_figure5(results_df, out_dir, method_ids=method_ids,
                                           is_synthetic=is_synthetic)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Smoke-test on a small method/instance subset.")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    if args.quick:
        method_ids = ["M1", "M4", "M10", "M12"]
        instance_ids = ["TP1", "TP3", "SMD1_p3q3", "SMD3_p10q10"]
    else:
        method_ids = None
        instance_ids = None

    print("=" * 70)
    print(SYNTHETIC_DATA_WARNING)
    print("Replace synthetic_data.generate_synthetic_results() with a real")
    print("results loader before using any of this output in the paper.")
    print("=" * 70)

    results_df = generate_synthetic_results(
        methods=method_ids, instances=(
            None if instance_ids is None
            else [i for i in INSTANCES if i.id in instance_ids]
        ),
    )

    written = run_all(results_df, out_dir=args.out, method_ids=method_ids,
                       instance_ids=instance_ids, is_synthetic=True)

    n_files = sum(
        len(v) if isinstance(v, list) else 1 for v in written.values()
    )
    print(f"\nDone. {n_files} figure(s) written under ./{args.out}/")
    print("Every filename and caption is prefixed SYNTHETIC_ -- see", args.out)


if __name__ == "__main__":
    sys.exit(main())
