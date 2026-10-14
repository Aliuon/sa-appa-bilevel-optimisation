"""ss_diagnostic.py: where does strong Stackelberg selection change anything?
For each instance, draw N UL points (LHS, fixed seed), run the shared DE LL solve, apply the
selection, and record gate/accept outcomes and the change in (G-violation, F)."""
import sys, json, time
import numpy as np
from scipy.stats import qmc
sys.path.insert(0, '.')
from benchmark import SUITE
from ll_solver import ll_solve
import strong_stackelberg as ss

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ids = sys.argv[2].split(',') if len(sys.argv) > 2 else list(SUITE)
out = open('results/strong_stackelberg/diagnostic.jsonl', 'a')
for iid in ids:
    inst = SUITE[iid]
    lo, hi = inst.x_bounds[:, 0], inst.x_bounds[:, 1]
    X = qmc.scale(qmc.LatinHypercube(d=inst.n_ul, seed=12345).random(N), lo, hi)
    t = time.time(); rows = []
    for k, x in enumerate(X):
        y0, F0, G0, feas = ll_solve(inst, x, seed=1000 + k)
        ss.reset_counts()
        y1 = ss.strong_stackelberg_select(inst, x, y0)
        changed = not np.array_equal(y1, y0)
        rows.append(dict(k=k, gated=ss.COUNTS['gated'], changed=changed,
                         F0=float(F0), F1=float(inst.F(x, y1)),
                         v0=ss._viol(G0), v1=ss._viol(inst.G(x, y1)),
                         f0=float(inst.f(x, y0)), f1=float(inst.f(x, y1)),
                         dy=float(np.linalg.norm(np.asarray(y1) - np.asarray(y0)))))
    rec = dict(instance=iid, N=N, secs=round(time.time() - t, 1),
               n_changed=sum(r['changed'] for r in rows), n_gated=sum(r['gated'] for r in rows),
               rows=rows)
    out.write(json.dumps(rec) + '\n'); out.flush()
    ch = [r for r in rows if r['changed']]
    print(f"{iid:14s} changed {rec['n_changed']:2d}/{N} gated {rec['n_gated']:2d} "
          f"{rec['secs']:6.1f}s" + (f"  max|dF|={max(abs(r['F1']-r['F0']) for r in ch):.3g}"
          f" dv={sum(r['v0']-r['v1'] for r in ch):.3g}" if ch else ''), flush=True)
