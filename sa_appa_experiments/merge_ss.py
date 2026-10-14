"""
merge_ss.py
===========
Makes strong Stackelberg selection the main lower-level protocol of the result files.

The no-selection files (arbitrary member of the LL optimal set) are archived unchanged in
results/no_tie_selection/. In the canonical files the rows of the instances where the
selection changes results (run_strong_stackelberg.AFFECTED) are replaced by the strong
Stackelberg runs; all other rows are kept, which the scan of run_strong_stackelberg.py shows
to be the same results. Run once; refuses to run if the archive already exists.
"""
import csv, json, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); SS = os.path.join(RES, "strong_stackelberg")
ARC = os.path.join(RES, "no_tie_selection")
from run_strong_stackelberg import AFFECTED

FILES = ["summary.csv", "summary_m14.csv", "trajectories.jsonl", "trajectories_m14.jsonl",
         "scenario5_tuned_summary.csv", "scenario5_tuned_trajectories.jsonl",
         "summary_m14k.csv", "trajectories_m14k.jsonl", "ll_counts_subset.csv", "paper_numbers.json"]


def main():
    if os.path.exists(ARC):
        sys.exit("archive exists: already merged")
    full = pd.read_csv(os.path.join(SS, "summary_ss.csv")); full = full[full["mode"] == "full"]
    scan = pd.read_csv(os.path.join(SS, "summary_ss.csv")); scan = scan[scan["mode"] == "scan"]
    assert len(full) == len(AFFECTED) * 14 * 30, len(full)
    scan = scan[~scan.instance_id.isin(AFFECTED)]
    assert (scan.ss_accepted == 0).all(), "selection fired in a scan run of an unaffected instance"
    os.makedirs(ARC)
    for f in FILES:
        shutil.copy2(os.path.join(RES, f), os.path.join(ARC, f))
    cols = ["method_id", "instance_id", "run_id", "best", "feasible_found", "cpu_time_s"]
    # summaries
    for fn, keep in (("summary.csv", lambda d: d.method_id != "M14"), ("summary_m14.csv", lambda d: d.method_id == "M14")):
        old = pd.read_csv(os.path.join(ARC, fn))
        new = pd.concat([old[~old.instance_id.isin(AFFECTED)], full[keep(full)][cols]])
        new = new.sort_values(["instance_id", "method_id", "run_id"]).reset_index(drop=True)
        assert len(new) == len(old), (fn, len(new), len(old))
        new.to_csv(os.path.join(RES, fn), index=False)
    # trajectories
    ss_tr = {}
    for line in open(os.path.join(SS, "trajectories_ss.jsonl")):
        j = json.loads(line)
        if j["mode"] == "full":
            ss_tr[(j["method_id"], j["instance_id"], int(j["run_id"]))] = j
    for fn, isM14 in (("trajectories.jsonl", False), ("trajectories_m14.jsonl", True)):
        out, n = [], 0
        for line in open(os.path.join(ARC, fn)):
            j = json.loads(line)
            k = (j["method_id"], j["instance_id"], int(j["run_id"]))
            if j["instance_id"] in AFFECTED:
                s = ss_tr[k]; j = {kk: vv for kk, vv in j.items()}; j["rows"] = s["rows"]
                if "error" in j: j["error"] = s["error"]
                n += 1
            out.append(json.dumps(j, default=float))
        open(os.path.join(RES, fn), "w").write("\n".join(out) + "\n")
        print(fn, "replaced", n)
    # scenario 5 tuned
    old = pd.read_csv(os.path.join(ARC, "scenario5_tuned_summary.csv"))
    s5 = pd.read_csv(os.path.join(SS, "scenario5_tuned_ss.csv"))
    new = pd.concat([old[~old.instance_id.isin(AFFECTED)], s5[old.columns]])
    assert len(new) == len(old); new.to_csv(os.path.join(RES, "scenario5_tuned_summary.csv"), index=False)
    s5t = {(json.loads(l)["instance_id"], json.loads(l)["run_id"]): l for l in open(os.path.join(SS, "scenario5_tuned_trajectories_ss.jsonl"))}
    out = []
    for line in open(os.path.join(ARC, "scenario5_tuned_trajectories.jsonl")):
        j = json.loads(line)
        out.append(s5t[(j["instance_id"], j["run_id"])].rstrip("\n") if j["instance_id"] in AFFECTED else line.rstrip("\n"))
    open(os.path.join(RES, "scenario5_tuned_trajectories.jsonl"), "w").write("\n".join(out) + "\n")
    # M14K
    old = pd.read_csv(os.path.join(ARC, "summary_m14k.csv"))
    mk = pd.read_csv(os.path.join(SS, "summary_m14k_ss.csv"))
    aff = [i for i in AFFECTED if i in set(old.instance_id)]
    new = pd.concat([old[~old.instance_id.isin(aff)], mk[old.columns]])
    assert len(new) == len(old); new.to_csv(os.path.join(RES, "summary_m14k.csv"), index=False)
    mkt = {(json.loads(l)["instance_id"], json.loads(l)["run_id"]): l for l in open(os.path.join(SS, "trajectories_m14k_ss.jsonl"))}
    out = []
    for line in open(os.path.join(ARC, "trajectories_m14k.jsonl")):
        j = json.loads(line)
        out.append(mkt[(j["instance_id"], j["run_id"])].rstrip("\n") if j["instance_id"] in aff else line.rstrip("\n"))
    open(os.path.join(RES, "trajectories_m14k.jsonl"), "w").write("\n".join(out) + "\n")
    # B1 counts
    old = pd.read_csv(os.path.join(ARC, "ll_counts_subset.csv"))
    b1 = pd.read_csv(os.path.join(SS, "ll_counts_ss.csv"))
    b1 = b1.rename(columns={"ss_best": "locked_best"})
    b1["traj_match"] = b1["best_match"]; b1["traj_max_rel_dev"] = 0.0
    new = pd.concat([old[~old.instance_id.isin(AFFECTED)], b1[old.columns]])
    assert len(new) == len(old); new.to_csv(os.path.join(RES, "ll_counts_subset.csv"), index=False)
    print("merged")


if __name__ == "__main__":
    main()
