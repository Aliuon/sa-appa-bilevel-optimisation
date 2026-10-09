


# ================================================================================================
# DRIVERS (run_all.py + run_m14.py, statistics, figures, Phase 7, self-validation)
# ================================================================================================
SWEEP_FIELDS = ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"]
ALL_CONFIGS = [f"M{i}" for i in range(1, 15)]


def _summary_paths(method_id):
    tag = "_m14" if method_id == "M14" else ""
    return (os.path.join(RESULTS_DIR, f"summary{tag}.csv"), os.path.join(RESULTS_DIR, f"trajectories{tag}.jsonl"))


def _sweep_worker(args):
    method_id, instance_id, run_id = args
    t0 = time.time()
    try:
        if method_id == "M14":
            traj, cpu = run_m14(SUITE[instance_id], seed=run_id)
        else:
            traj, cpu = run_method(SUITE[instance_id], method_id, seed=run_id)
        return dict(method_id=method_id, instance_id=instance_id, run_id=run_id, best=traj.best,
                    feasible_found=traj.found_feasible, cpu_time_s=cpu, rows=traj.rows, error=None)
    except Exception as e:  # recorded, never hidden
        return dict(method_id=method_id, instance_id=instance_id, run_id=run_id, best=np.nan,
                    feasible_found=False, cpu_time_s=time.time() - t0, rows=[], error=f"{type(e).__name__}: {e}")


def run_sweep(method_ids, instance_ids, n_runs, workers):
    """All (configuration, instance, run) triples, in parallel, resumable. Same file formats as
    run_all.py (M1-M13) and run_m14.py (M14)."""
    done = set()
    for m in set(method_ids):
        sp, _ = _summary_paths(m)
        if os.path.exists(sp):
            with open(sp, newline="") as fh:
                done |= {(r["method_id"], r["instance_id"], int(r["run_id"])) for r in csv.DictReader(fh)}
    work = [(m, i, r) for m in method_ids for i in instance_ids for r in range(n_runs) if (m, i, r) not in done]
    print(f"sweep: {len(work)} runs to do, {len(done)} already done, {workers} workers", flush=True)
    if not work:
        return
    handles = {}
    for m in set(method_ids):
        sp, tp = _summary_paths(m)
        if sp not in handles:
            new = not os.path.exists(sp)
            sf = open(sp, "a", newline="")
            w = csv.DictWriter(sf, fieldnames=SWEEP_FIELDS)
            if new:
                w.writeheader()
            handles[sp] = (sf, w, open(tp, "a"))
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_sweep_worker, work, chunksize=1), 1):
            sf, w, tf = handles[_summary_paths(res["method_id"])[0]]
            w.writerow({k: res[k] for k in SWEEP_FIELDS}); sf.flush()
            tf.write(json.dumps(dict(method_id=res["method_id"], instance_id=res["instance_id"],
                                     run_id=res["run_id"], rows=res["rows"], error=res["error"])) + "\n")
            tf.flush()
            if res["error"]:
                print("ERROR", res["method_id"], res["instance_id"], res["run_id"], res["error"], flush=True)
            if n % 25 == 0 or n == len(work):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(work)} runs, "
                      f"ETA {(len(work) - n) * el / n / 3600:.2f} h", flush=True)
    for sf, w, tf in handles.values():
        sf.close(); tf.close()


def bootstrap_stored_results():
    """Copy the stored results from the repository, if present, into RESULTS_DIR (read-only use)."""
    if all(os.path.exists(os.path.join(RESULTS_DIR, f)) for f in STORED_FILES):
        return True
    if not os.path.exists(os.path.join(REPO_RESULTS, "summary.csv")):
        return False
    import shutil
    for f in STORED_FILES:
        src = os.path.join(REPO_RESULTS, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(RESULTS_DIR, f)):
            shutil.copy2(src, os.path.join(RESULTS_DIR, f))
    ref = os.path.join(REPO_RESULTS, "resub_numbers.json")
    if os.path.exists(ref):
        shutil.copy2(ref, os.path.join(RESULTS_DIR, "resub_numbers_reference.json"))
    print("stored results copied from", REPO_RESULTS)
    return True


def run_statistics(quick=False):
    """Scenario tables, every test statistic of the paper, CD summaries, Appendix B LaTeX."""
    df = load_results(include_m14=True)
    for s in SCENARIOS:
        r = run_scenario(df, s["num"])
        r["table"].to_csv(os.path.join(RESULTS_DIR, "scenario_tables", f"scenario_{s['num']}.csv"), index=False)
        print_scenario_report(r)
    if quick:
        os.environ["RESUB_ALLOW_PARTIAL"] = "1"
    try:
        compute_resub_numbers()
        if not quick:
            run_llcheck()
    except Exception as e:
        print(f"[statistics] resub numbers skipped: {type(e).__name__}: {e}")
    export_cd_summaries()
    cwd = os.getcwd()
    try:
        os.chdir(RESULTS_DIR)
        build_appendix_b_main_tex()
    finally:
        os.chdir(cwd)


def build_appendix_b_main_tex():
    all_tex = ["\\section{Scenario Results (Appendix B)}", "\\label{app:scenarios}", ""]
    for s in SCENARIOS:
        tex = build_scenario_tex(s["num"])
        with open(f"appendix_b_scenario_{s['num']}.tex", "w") as fh:
            fh.write(tex)
        all_tex.append(tex)
    with open("appendix_b_all.tex", "w") as fh:
        fh.write("\n".join(all_tex))
    print("Appendix B LaTeX written to", RESULTS_DIR)


def run_figures():
    figures_resub_main()
    make_all_cd_diagrams(os.path.join(RESULTS_DIR, "scenario_cd_summaries.json"), FIG_DIR)


def run_phase7(workers):
    """MAKE_EXTRA_FIGS: Part A data (re-runs under locked seeds, checked before use), the
    instrumented runs 0-4 for B1, then every Phase 7 figure."""
    phase7_run_main(("A1", "A2", "A3", "A5"))
    smd6_check_main()
    instrument_ll_main(runs=5, workers=workers)
    figures_phase7_main()


# ------------------------------------------------------------------------------ self-validation
def self_validate(workers, report_path):
    lines = [f"SA_APPA_4nd.py self-validation, {time.strftime('%Y-%m-%d %H:%M')}",
             f"numpy {np.__version__}, scipy {__import__('scipy').__version__}, pandas {pd.__version__}", ""]
    ok_all = True
    # 1. benchmark sanity and a known optimum
    bad = []
    for iid, inst in SUITE.items():
        x = inst.x_bounds.mean(axis=1); y = inst.y_bounds.mean(axis=1)
        vals = np.concatenate([[inst.F(x, y), inst.f(x, y)], np.atleast_1d(inst.G(x, y)), np.atleast_1d(inst.g(x, y))])
        if not np.all(np.isfinite(vals)):
            bad.append(iid)
    lines.append(f"[1] {len(SUITE)} instances evaluate to finite values at the box centre: {'yes' if not bad else bad}")
    ok_all &= not bad and len(SUITE) == 44
    t = truth("TP3", np.array([15.0, 10.0]))
    tp3_ok = abs(t["theta"] - 325.0) < 1e-9 and np.allclose(t["y"], [10, 10]) and abs(t["fstar"] - 25.0) < 1e-9
    lines.append(f"[2] TP3 at x = (15, 10): theta = {t['theta']:.6g}, y* = {t['y'].tolist()}, f* = {t['fstar']:.6g} "
                 f"(expected 325, [10, 10], 25): {'ok' if tp3_ok else 'MISMATCH'}")
    ok_all &= tp3_ok
    # 2. locked runs re-run
    locked = _ll_load_locked()
    keys = [(m, i, r) for (i, r) in VALIDATION_RUNS for m in ALL_CONFIGS if (m, i, r) in locked]
    ltraj = _ll_load_locked_traj(keys)
    lines.append(f"[3] re-run of {len(keys)} locked runs under their seeds (tolerance {REPRO_TOL:g} relative):")
    with mp.Pool(processes=max(1, min(workers, len(keys)))) as pool:
        res = pool.map(_ll_worker, [(m, i, r, locked[(m, i, r)], ltraj.get((m, i, r))) for (m, i, r) in keys])
    n_ok = 0
    for row in res:
        good = bool(row["best_match"] and row["traj_match"])
        n_ok += good
        if not good:
            lines.append(f"    {row['method_id']} {row['instance_id']} run {row['run_id']}: best {row['best']} vs "
                         f"stored {row['locked_best']}, max trajectory deviation {row['traj_max_rel_dev']:.3g}")
    lines.append(f"    {n_ok} of {len(keys)} reproduce the stored best value and trajectory")
    ok_all &= n_ok == len(keys)
    # 3. statistics against the stored reference
    ref_p = os.path.join(RESULTS_DIR, "resub_numbers_reference.json")
    new_p = os.path.join(RESULTS_DIR, "resub_numbers.json")
    if os.path.exists(ref_p) and os.path.exists(new_p):
        ref, new = json.load(open(ref_p)), json.load(open(new_p))
        dmax = 0.0
        for n, s in ref["scenarios"].items():
            t = new["scenarios"][n]
            dmax = max(dmax, abs(s["T2"] - t["T2"]), abs(s["p"] - t["p"]),
                       max(abs(s["mean_ranks"][m] - t["mean_ranks"][m]) for m in s["mean_ranks"]))
        hmax = max(abs(a["p_holm"] - b["p_holm"]) for a, b in zip(
            sorted(ref["holm"], key=lambda h: h["label"]), sorted(new["holm"], key=lambda h: h["label"])))
        st_ok = dmax < 1e-9 and hmax < 1e-9
        lines.append(f"[4] eight scenario tests and the 16-test Holm correction recomputed from the stored CSVs: "
                     f"largest difference from the reference {max(dmax, hmax):.2g}: {'ok' if st_ok else 'MISMATCH'}")
        ok_all &= st_ok
    else:
        lines.append("[4] statistics check skipped (no reference resub_numbers.json, e.g. after a fresh sweep)")
    lines += ["", "OVERALL: " + ("PASS" if ok_all else "SEE DEVIATIONS ABOVE")]
    open(report_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok_all


# ------------------------------------------------------------------------------ command line
def main():
    global QUICK_TEST, RUN_SWEEP, MAKE_EXTRA_FIGS, SELF_VALIDATE, N_WORKERS
    ap = argparse.ArgumentParser(description="SA-APPA single-file experiment code")
    ap.add_argument("--quick", action="store_true", help="3 instances x 3 runs smoke test (own output folder)")
    ap.add_argument("--run-sweep", action="store_true", help="run the experiment instead of using stored results")
    ap.add_argument("--extra-figs", action="store_true", help="MAKE_EXTRA_FIGS: Phase 7 figures")
    ap.add_argument("--validate-only", action="store_true", help="only the self-validation report")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--output", default=None, help="output folder (default ./sa_appa_4nd_output)")
    a = ap.parse_args()
    QUICK_TEST |= a.quick; RUN_SWEEP |= a.run_sweep; MAKE_EXTRA_FIGS |= a.extra_figs
    SELF_VALIDATE = SELF_VALIDATE and not a.no_validate; N_WORKERS = a.workers
    out = a.output or os.path.join(HERE, "sa_appa_4nd_quick_output" if QUICK_TEST else "sa_appa_4nd_output")
    configure_paths(out)
    print(f"SA_APPA_4nd.py  quick={QUICK_TEST} run_sweep={RUN_SWEEP} extra_figs={MAKE_EXTRA_FIGS} "
          f"workers={N_WORKERS} output={OUTPUT_DIR}", flush=True)

    if a.validate_only:
        bootstrap_stored_results()
        if not os.path.exists(os.path.join(RESULTS_DIR, "resub_numbers.json")):
            compute_resub_numbers()
        self_validate(N_WORKERS, os.path.join(OUTPUT_DIR, "validation_report.txt"))
        return
    if QUICK_TEST:
        insts, runs = ["TP1", "TP3", "SMD1_p3q3"], 3
        run_sweep(ALL_CONFIGS, insts, runs, N_WORKERS)
    else:
        have = bootstrap_stored_results() if not RUN_SWEEP else False
        if RUN_SWEEP or not have:
            run_sweep(ALL_CONFIGS, list(SUITE.keys()), 30, N_WORKERS)
    run_statistics(quick=QUICK_TEST)
    try:
        run_figures()
    except Exception as e:
        print(f"[figures] {type(e).__name__}: {e}")
    if MAKE_EXTRA_FIGS:
        if QUICK_TEST:
            print("[Phase 7] needs the full 44-instance, 30-run results; skipped in quick mode.")
        else:
            run_phase7(N_WORKERS)
    if SELF_VALIDATE and not QUICK_TEST:
        self_validate(N_WORKERS, os.path.join(OUTPUT_DIR, "validation_report.txt"))
    print("done; outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
