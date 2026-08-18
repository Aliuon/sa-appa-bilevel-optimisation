"""
run_all.py
==========
Real experiment runner: all 13 methods x all 44 reconciled instances x
30 independent runs (per the user's parameter change superseding the
earlier 5-run pipeline test). Uses multiprocessing across this sandbox's
CPU cores (independent (method, instance, run) triples are embarrassingly
parallel). Resumable: each completed run's summary row and full
trajectory are appended to disk immediately, and already-completed
triples are skipped on restart, so a multi-hour job surviving across
checkpoints doesn't lose progress or duplicate work.

Budget: N0=25, T_max=35 (60 expensive evals/run), a fixed,
dimension-independent budget substituted for the manuscript's
unconfirmed placeholder rule -- see methods.py's module docstring and
the user's explicit sign-off on "shrink the evaluation budget."

Outputs (under results/):
    summary.csv       one row per (method_id, instance_id, run_id):
                       best, feasible_found, cpu_time_s
    trajectories.jsonl one JSON line per run: full eval-by-eval trajectory
                       (for convergence curves / rate-of-convergence figures)
    progress.log       human-readable progress, appended to
"""

import os
import sys
import csv
import json
import time
import argparse
import multiprocessing as mp

import numpy as np

from benchmark import SUITE
from methods import run_method, METHOD_SPECS

N_RUNS = 30
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.csv")
TRAJ_PATH = os.path.join(RESULTS_DIR, "trajectories.jsonl")
PROGRESS_PATH = os.path.join(RESULTS_DIR, "progress.log")
SUMMARY_FIELDS = ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"]

METHOD_IDS = list(METHOD_SPECS.keys())


def _completed_triples():
    done = set()
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["method_id"], row["instance_id"], int(row["run_id"])))
    return done


def _worker(args):
    method_id, instance_id, run_id = args
    inst = SUITE[instance_id]
    t0 = time.time()
    try:
        traj, cpu_time = run_method(inst, method_id, seed=run_id)
        wall = time.time() - t0
        return dict(
            method_id=method_id, instance_id=instance_id, run_id=run_id,
            best=traj.best, feasible_found=traj.found_feasible, cpu_time_s=cpu_time,
            wall_s=wall, rows=traj.rows, error=None,
        )
    except Exception as e:
        return dict(
            method_id=method_id, instance_id=instance_id, run_id=run_id,
            best=np.nan, feasible_found=False, cpu_time_s=time.time() - t0,
            wall_s=time.time() - t0, rows=[], error=f"{type(e).__name__}: {e}",
        )


def build_worklist(method_ids, instance_ids, n_runs):
    done = _completed_triples()
    work = []
    for m in method_ids:
        for i in instance_ids:
            for r in range(n_runs):
                key = (m, i, r)
                if key not in done:
                    work.append(key)
    return work, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=METHOD_IDS)
    parser.add_argument("--instances", nargs="*", default=list(SUITE.keys()))
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(SUMMARY_PATH)

    work, done = build_worklist(args.methods, args.instances, args.runs)
    total_target = len(args.methods) * len(args.instances) * args.runs
    with open(PROGRESS_PATH, "a") as plog:
        plog.write(f"\n=== run_all.py start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        plog.write(f"target={total_target} already_done={len(done)} remaining={len(work)} workers={args.workers}\n")
        plog.flush()

    print(f"target={total_target} already_done={len(done)} remaining={len(work)} workers={args.workers}", flush=True)
    if not work:
        print("Nothing to do.", flush=True)
        return

    summary_fh = open(SUMMARY_PATH, "a", newline="")
    writer = csv.DictWriter(summary_fh, fieldnames=SUMMARY_FIELDS)
    if write_header:
        writer.writeheader()
        summary_fh.flush()
    traj_fh = open(TRAJ_PATH, "a")

    n_done_this_run = 0
    t_start = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_worker, work, chunksize=1):
            writer.writerow({k: result[k] for k in SUMMARY_FIELDS})
            summary_fh.flush()
            traj_fh.write(json.dumps(dict(
                method_id=result["method_id"], instance_id=result["instance_id"],
                run_id=result["run_id"], rows=result["rows"], error=result["error"],
            )) + "\n")
            traj_fh.flush()
            n_done_this_run += 1
            if result["error"]:
                with open(PROGRESS_PATH, "a") as plog:
                    plog.write(f"ERROR {result['method_id']} {result['instance_id']} "
                               f"run{result['run_id']}: {result['error']}\n")
            if n_done_this_run % 25 == 0 or n_done_this_run == len(work):
                elapsed = time.time() - t_start
                rate = n_done_this_run / elapsed if elapsed > 0 else 0
                remaining = len(work) - n_done_this_run
                eta_h = (remaining / rate / 3600) if rate > 0 else float("nan")
                msg = (f"[{time.strftime('%H:%M:%S')}] {n_done_this_run}/{len(work)} done "
                       f"this-session ({len(done) + n_done_this_run}/{total_target} overall), "
                       f"rate={rate:.3f}/s, ETA={eta_h:.2f}h")
                print(msg, flush=True)
                with open(PROGRESS_PATH, "a") as plog:
                    plog.write(msg + "\n")

    summary_fh.close()
    traj_fh.close()
    with open(PROGRESS_PATH, "a") as plog:
        plog.write(f"=== run_all.py finished {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
