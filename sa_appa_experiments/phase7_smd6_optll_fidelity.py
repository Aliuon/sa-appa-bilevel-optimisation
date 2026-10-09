"""
phase7_smd6_optll_fidelity.py
=============================
Held-out fidelity of SA-APPA's GP on SMD6 at (3, 3) when the nested LL solve applies the
optimistic tie-break (optimistic_ll.py). Same rule as A5: the lower-median of SA-APPA's 30 runs
(here from results/optll/summary_optll.csv) is re-run under its seed with the patch, the
trajectory is checked against the stored optimistic run, and the final GP is scored on the A5
held-out points against the closed-form optimistic theta. Writes results/phase7/smd6_optll_fidelity.json.
"""
import json, os
import numpy as np
import pandas as pd

import optimistic_ll
optimistic_ll.patch()
from benchmark import SUITE
from methods import run_ablation_method, METHOD_SPECS
from phase7_smd6_check import theta_opt

HERE = os.path.dirname(os.path.abspath(__file__))
P7 = os.path.join(HERE, "results", "phase7")
IID = "SMD6_p3q3"


def main():
    o = pd.read_csv(os.path.join(HERE, "results", "optll", "summary_optll.csv"))
    m = o[(o.method_id == "M1") & (o.instance_id == IID)].copy()
    assert len(m) == 30, len(m)
    m["key"] = np.where(m.feasible_found.astype(bool), m.best, np.inf)
    m = m.sort_values(["key", "run_id"])
    run, best = int(m.iloc[14].run_id), float(m.iloc[14].best)
    stored = None
    with open(os.path.join(HERE, "results", "optll", "trajectories_optll.jsonl")) as fh:
        for line in fh:
            j = json.loads(line)
            if j["method_id"] == "M1" and j["instance_id"] == IID and j["run_id"] == run:
                stored = [r["best_feasible_obj"] for r in j["rows"]]
    snaps = {}

    def hook(stage, X, F, G, theta_gp, phi_gps):
        snaps[stage] = dict(X=X, F=np.asarray(F, float), theta_gp=theta_gp)
    sp = METHOD_SPECS["M1"]
    traj, _ = run_ablation_method(SUITE[IID], "M1", sp["engine"], sp["constraint_mode"], sp["adaptation"],
                                  sp["init"], seed=run, hook=hook)
    new = [r["best_feasible_obj"] for r in traj.rows]
    dev = max(abs(a - b) / max(1, abs(b)) for a, b in zip(new, stored)) if len(new) == len(stored) else np.inf
    d = np.load(os.path.join(P7, "a5_SMD6_p3q3.npz"))
    Xh = d["Xh"]; th = theta_opt(Xh)
    gp = snaps["final"]["theta_gp"]
    mu, sd = gp.predict(Xh)
    e = mu - th
    Xt, Ft = snaps["final"]["X"], snaps["final"]["F"]
    out = dict(run=run, best=best, reproduces=bool(dev <= 1e-6), max_rel_dev=float(dev), n=int(len(th)),
               rmse=float(np.sqrt(np.mean(e ** 2))), max_abs=float(np.max(np.abs(e))),
               coverage_2sd=float(np.mean(np.abs(e) <= 2 * sd)), mean_sd=float(np.mean(sd)),
               theta_range=float(np.ptp(th)), corr=float(np.corrcoef(mu, th)[0, 1]),
               lengthscale=float(gp.lengthscale),
               train_gap_max_abs=float(np.max(np.abs(Ft - theta_opt(Xt)))))
    np.savez(os.path.join(P7, "smd6_optll_fidelity.npz"), Xh=Xh, th=th, mu=mu, sd=sd, Xt=Xt, Ft=Ft)
    json.dump(out, open(os.path.join(P7, "smd6_optll_fidelity.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
