"""
phase7_smd6_check.py
====================
Tests why the A5 surrogate fit on SMD6 at (3, 3) was poor under the original lower-level protocol.

SMD6 has a continuum of LL optima: the LL objective contains sum((t[i+1] - t[i])^2) over the
s = 2 'tail' variables t = x_l1[q:q+s], so every t1 = t2 in [-5, 10] is LL-optimal, while the UL
objective contains +sum(t^2). Under the optimistic convention the response is t = 0, and
    theta_opt(x) = sum x_u1^2 + sum x_u2^2 - sum (x_u2 - clip(x_u2, -pi/2, pi/2))^2.
Any DE solve of the LL (the 60-solve nested DE inside SA-APPA, or the A5 reference DE) returns
some arbitrary point on that line, so theta = theta_opt + 2 t^2 with t effectively random.

Checks (nothing locked is modified):
  1. the closed form agrees with an optimistic numerical solve at a few points;
  2. at SA-APPA's own training points (run 1, the lower-median run) the targets it was given,
     F_train, differ from theta_opt by 2 t^2;
  3. the A5 'reference' truth differs from theta_opt in the same way;
  4. the A5 metrics for SMD6 recomputed against theta_opt;
  5. a counterfactual diagnostic: the same GP class and hyperparameter rule fitted to the same 60
     training inputs, but with the optimistic targets theta_opt(X_train), scored on the held-out
     points. This is not an SA-APPA run; it isolates how much of the error comes from the
     inconsistent LL responses rather than from the GP model.
Writes results/phase7/smd6_check.json.
"""
import json, os
import numpy as np
from scipy.optimize import minimize

# This check describes the ORIGINAL lower-level protocol (no strong Stackelberg selection), so it
# re-runs SA-APPA without the selection and against the archived no-selection results.
os.environ["SA_APPA_LL_PROTOCOL"] = "none"
from benchmark import SUITE
import phase7_data as P
P.RES = os.path.join(P.RES, "no_tie_selection")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phase7")
IID, P_, Q_, R_, S_ = "SMD6_p3q3", 3, 3, 2, 2


def theta_opt(X):
    X = np.atleast_2d(X)
    xu1, xu2 = X[:, :P_], X[:, P_:]
    c = np.clip(xu2, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
    return np.sum(xu1 ** 2, 1) + np.sum(xu2 ** 2, 1) - np.sum((xu2 - c) ** 2, 1)


def y_opt(x):
    xu2 = x[P_:]
    return np.concatenate([np.zeros(Q_ + S_), np.clip(xu2, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)])


def main():
    inst = SUITE[IID]
    out = {}
    # 1. closed form check: at y_opt the LL objective is at its minimum (f gradient ~ 0 on the
    #    free directions) and theta equals F(x, y_opt)
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(20):
        x = rng.uniform(inst.x_bounds[:, 0], inst.x_bounds[:, 1])
        y = y_opt(x)
        # LL optimality: no feasible y gives a lower f (local check from 5 random starts)
        fbest = min(minimize(lambda z: inst.f(x, z), rng.uniform(inst.y_bounds[:, 0], inst.y_bounds[:, 1]),
                             bounds=list(map(tuple, inst.y_bounds)), method="L-BFGS-B").fun for _ in range(5))
        errs.append(dict(f_at_yopt=float(inst.f(x, y)), f_best_local=float(fbest),
                         theta_closed=float(theta_opt(x)[0]), F_at_yopt=float(inst.F(x, y))))
    out["closed_form_check"] = dict(
        max_abs_theta_diff=max(abs(e["theta_closed"] - e["F_at_yopt"]) for e in errs),
        max_f_gap=max(e["f_at_yopt"] - e["f_best_local"] for e in errs))
    # 2. SA-APPA's training targets
    run, best = P.median_run(IID)
    snaps, chk = P.rerun_with_snapshots(IID, run)
    s = snaps["final"]
    Ft, Xt = np.asarray(s["F"], float), np.asarray(s["X"], float)
    gap_t = Ft - theta_opt(Xt)
    out["training"] = dict(run=run, reproduces=chk["reproduces"], max_rel_dev=chk["max_rel_dev"],
                           n=int(len(Ft)), gap_min=float(gap_t.min()), gap_median=float(np.median(gap_t)),
                           gap_max=float(gap_t.max()), gap_rms=float(np.sqrt(np.mean(gap_t ** 2))),
                           implied_abs_t_median=float(np.median(np.sqrt(np.maximum(gap_t, 0) / 2))))
    # 3. A5 reference truth vs theta_opt; 4. metrics against theta_opt
    d = np.load(os.path.join(OUT, "a5_SMD6_p3q3.npz"))
    Xh, th_ref, mu, sd = d["Xh"], d["th"], d["mu"], d["sd"]
    th_opt = theta_opt(Xh)
    gap_h = th_ref - th_opt
    out["reference_truth"] = dict(n=int(len(Xh)), gap_min=float(gap_h.min()), gap_median=float(np.median(gap_h)),
                                  gap_max=float(gap_h.max()), gap_rms=float(np.sqrt(np.mean(gap_h ** 2))))

    def met(truth):
        e = mu - truth
        return dict(rmse=float(np.sqrt(np.mean(e ** 2))), max_abs=float(np.max(np.abs(e))),
                    coverage_2sd=float(np.mean(np.abs(e) <= 2 * sd)), theta_range=float(np.ptp(truth)),
                    corr=float(np.corrcoef(mu, truth)[0, 1]))
    out["a5_vs_reference"] = met(th_ref)
    out["a5_vs_optimistic"] = met(th_opt)
    from gp import GPSurrogate
    gp_cf = GPSurrogate().fit(Xt, theta_opt(Xt))
    mu_cf, sd_cf = gp_cf.predict(Xh)
    e = mu_cf - th_opt
    out["counterfactual_consistent_targets"] = dict(
        rmse=float(np.sqrt(np.mean(e ** 2))), max_abs=float(np.max(np.abs(e))),
        coverage_2sd=float(np.mean(np.abs(e) <= 2 * sd_cf)), theta_range=float(np.ptp(th_opt)),
        corr=float(np.corrcoef(mu_cf, th_opt)[0, 1]), lengthscale=float(gp_cf.lengthscale))
    gp_same = s["theta_gp"]
    out["sa_appa_gp_lengthscale"] = float(gp_same.lengthscale)
    np.savez(os.path.join(OUT, "smd6_check.npz"), mu_cf=mu_cf, sd_cf=sd_cf, Xh=Xh, th_ref=th_ref, th_opt=th_opt, mu=mu, sd=sd,
             Xt=Xt, Ft=Ft, Ft_opt=theta_opt(Xt))
    json.dump(out, open(os.path.join(OUT, "smd6_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
