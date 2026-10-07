"""Compute and cache every quantity behind the Phase 7 Part A figures (see phase7_data.py)."""
import json, os, sys, time
import numpy as np
import phase7_data as P
from benchmark import SUITE

OUT = P.OUT
NG = 61                      # wireframe / map grid per axis
NH_TP3 = 2500                # held-out points for the TP3 fidelity table
R_NEAR = 1.5                 # neighbourhood of x* (10% of the TP3 box width)
X_STAR_TP3 = np.array([15.0, 10.0])
log = {"reruns": [], "notes": []}


def grid2(iid, n=NG):
    b = SUITE[iid].x_bounds
    g1, g2 = np.linspace(b[0, 0], b[0, 1], n), np.linspace(b[1, 0], b[1, 1], n)
    X1, X2 = np.meshgrid(g1, g2)
    return X1, X2, np.column_stack([X1.ravel(), X2.ravel()])


def part_A1_A4():
    X1, X2, XX = grid2("TP3")
    T = [P.truth("TP3", x) for x in XX]
    np.savez(os.path.join(OUT, "tp3_truth.npz"), X1=X1, X2=X2,
             theta=np.array([t["theta"] for t in T]).reshape(X1.shape),
             fstar=np.array([t["fstar"] for t in T]).reshape(X1.shape),
             y1=np.array([t["y"][0] for t in T]).reshape(X1.shape),
             y2=np.array([t["y"][1] for t in T]).reshape(X1.shape),
             phi=np.array([t["phi"] for t in T]).reshape(X1.shape + (2,)),
             xstar=X_STAR_TP3, theta_star=325.0)


def snap_pred(snap, XX):
    mu, sd = snap["theta_gp"].predict(XX)
    out = dict(mu=mu, sd=sd, X=snap["X"], F=snap["F"], G=snap["G"],
               ls=snap["theta_gp"].lengthscale)
    for j, g in enumerate(snap["phi_gps"]):
        m, s = g.predict(XX)
        out[f"phi{j}_mu"], out[f"phi{j}_sd"], out[f"phi{j}_ls"] = m, s, g.lengthscale
    return out


def part_A2():
    run, best = P.median_run("TP3")
    snaps, chk = P.rerun_with_snapshots("TP3", run)
    log["reruns"].append(chk)
    if not chk["reproduces"]:
        raise SystemExit(f"TP3 run {run} does not reproduce: {chk}")
    X1, X2, XX = grid2("TP3")
    # held-out sets, fixed seeds
    from scipy.stats import qmc
    Xh = qmc.scale(qmc.LatinHypercube(d=2, seed=P.HELDOUT_SEED).random(NH_TP3), [0, 0], [15, 15])
    th = np.array([P.truth("TP3", x)["theta"] for x in Xh])
    rng = np.random.default_rng(P.HELDOUT_SEED)
    near = []
    while len(near) < 500:
        z = X_STAR_TP3 + rng.uniform(-R_NEAR, R_NEAR, 2)
        if np.linalg.norm(z - X_STAR_TP3) <= R_NEAR and np.all((z >= 0) & (z <= 15)):
            near.append(z)
    Xn = np.array(near); tn = np.array([P.truth("TP3", x)["theta"] for x in Xn])
    feas = (Xh[:, 0] + 2 * Xh[:, 1] >= 30) & (Xh[:, 0] + Xh[:, 1] <= 25)
    res = {"run": run, "locked_best": best, "check": chk}
    for stage in ("init", "final"):
        s = snaps[stage]
        pr = snap_pred(s, XX)
        np.savez(os.path.join(OUT, f"tp3_gp_{stage}.npz"), X1=X1, X2=X2,
                 mu=pr["mu"].reshape(X1.shape), sd=pr["sd"].reshape(X1.shape),
                 phi0_mu=pr["phi0_mu"].reshape(X1.shape), phi1_mu=pr["phi1_mu"].reshape(X1.shape),
                 phi0_sd=pr["phi0_sd"].reshape(X1.shape), phi1_sd=pr["phi1_sd"].reshape(X1.shape),
                 Xtr=s["X"], Ftr=s["F"], Gtr=s["G"], lengthscale=pr["ls"])
        gp = s["theta_gp"]
        res[stage] = {"n_train": int(len(s["F"])), "lengthscale": float(gp.lengthscale),
                      "signal_var": gp.signal_var, "noise_var": gp.noise_var,
                      "whole_box": P.gp_metrics(gp, Xh, th),
                      "near_optimum": P.gp_metrics(gp, Xn, tn),
                      "feasible_region": P.gp_metrics(gp, Xh[feas], th[feas])}
        # SA-APPA's incumbent: best feasible truly evaluated point in the archive
        Ftr, Gtr = s["F"], s["G"]
        ok = np.all(Gtr <= 1e-6, axis=1)
        k = int(np.argmin(np.where(ok, Ftr, np.inf)))
        res[stage]["incumbent_x"] = s["X"][k].tolist(); res[stage]["incumbent_F"] = float(Ftr[k])
    np.savez(os.path.join(OUT, "tp3_heldout.npz"), Xh=Xh, th=th, Xn=Xn, tn=tn, feas=feas)
    return res


def part_A3():
    out = {}
    for iid in ["TP3", "TP4", "TP5", "TP7", "TP8"]:
        run, best = P.median_run(iid)
        snaps, chk = P.rerun_with_snapshots(iid, run)
        log["reruns"].append(chk)
        if not chk["reproduces"]:
            raise SystemExit(f"{iid} run {run} does not reproduce: {chk}")
        X1, X2, XX = grid2(iid)
        inst = SUITE[iid]
        if iid == "TP8":          # phi depends on y*: exact LL response
            PHI = np.array([P.truth(iid, x)["phi"] for x in XX])
        else:                     # phi depends on x only
            PHI = np.array([np.atleast_1d(inst.G(x, np.mean(inst.y_bounds, axis=1))) for x in XX])
        s = snaps["final"]
        pr = snap_pred(s, XX)
        m = PHI.shape[1]
        d = dict(X1=X1, X2=X2, Xtr=s["X"], Gtr=s["G"])
        for j in range(m):
            d[f"true{j}"] = PHI[:, j].reshape(X1.shape)
            d[f"mu{j}"] = pr[f"phi{j}_mu"].reshape(X1.shape)
            d[f"sd{j}"] = pr[f"phi{j}_sd"].reshape(X1.shape)
        Ftr, Gtr = s["F"], s["G"]
        ok = np.all(Gtr <= 1e-6, axis=1)
        k = int(np.argmin(np.where(ok, Ftr, np.inf))) if ok.any() else -1
        d["incumbent"] = s["X"][k] if k >= 0 else np.full(2, np.nan)
        np.savez(os.path.join(OUT, f"{iid.lower()}_phi_final.npz"), **d)
        met = {}
        for j in range(m):
            err = pr[f"phi{j}_mu"] - PHI[:, j]
            met[f"phi{j}"] = dict(rmse=float(np.sqrt(np.mean(err ** 2))),
                                  max_abs=float(np.max(np.abs(err))),
                                  coverage_2sd=float(np.mean(np.abs(err) <= 2 * pr[f"phi{j}_sd"])),
                                  sign_agreement=float(np.mean(np.sign(pr[f"phi{j}_mu"]) == np.sign(PHI[:, j]))))
        out[iid] = dict(run=run, locked_best=best, m_ul=m, metrics=met)
    return out


def part_A5():
    out = {}
    for iid in P.TP_LOWDIM + P.SMD_SMALL:
        t0 = time.time()
        run, best = P.median_run(iid)
        snaps, chk = P.rerun_with_snapshots(iid, run)
        log["reruns"].append(chk)
        if not chk["reproduces"]:
            out[iid] = dict(run=run, error="does not reproduce", check=chk); continue
        n = 400 if iid.startswith("TP") else 200
        Xh, th, phih, keep = P.heldout(iid, n)
        gp = snaps["final"]["theta_gp"]
        mu, sd = gp.predict(Xh[keep])
        np.savez(os.path.join(OUT, f"a5_{iid}.npz"), Xh=Xh[keep], th=th[keep], mu=mu, sd=sd)
        met = P.gp_metrics(gp, Xh[keep], th[keep])
        met.update(run=run, locked_best=best, n_dropped_ll_infeasible=int((~keep).sum()),
                   lengthscale=float(gp.lengthscale),
                   truth="exact QP" if iid in ("TP3", "TP8") else "reference DE (pop 30 n_ll, 300 gen, 2 seeds)")
        out[iid] = met
        print(iid, f"{time.time()-t0:.0f}s", met["rmse"], met["coverage_2sd"], flush=True)
    return out


if __name__ == "__main__":
    parts = sys.argv[1:] or ["A1", "A2", "A3", "A5"]
    path = os.path.join(OUT, "fidelity.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if "A1" in parts:
        part_A1_A4(); res["A1"] = {"grid": f"{NG}x{NG}", "solver": "scipy lsq_linear (bvls), checked against closed-form projection",
                                   "x_star": X_STAR_TP3.tolist(), "theta_star": 325.0, "y_star": [10.0, 10.0], "fstar_star": 25.0}
    if "A2" in parts:
        res["A2"] = part_A2()
    if "A3" in parts:
        res["A3"] = part_A3()
    if "A5" in parts:
        res["A5"] = part_A5()
    res.setdefault("reruns", [])
    res["reruns"] = [r for r in res["reruns"] if (r["instance"], r["run"]) not in
                     {(c["instance"], c["run"]) for c in log["reruns"]}] + log["reruns"]
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("done", parts)
