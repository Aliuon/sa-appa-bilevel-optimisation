"""
real_data.py
============
Loader for the REAL experiment output (results/trajectories.jsonl,
results/summary.csv in ../sa_appa_experiments), replacing
synthetic_data.generate_synthetic_results() per this package's own
documented schema boundary (see synthetic_data.py's module docstring).

Produces the canonical long-format DataFrame:
    method_id, instance_id, run_id, eval_count, iteration,
    best_feasible_obj, is_feasible

directly from the per-run trajectory rows logged by
sa_appa_experiments/run_all.py (17,160 real runs, zero synthetic data).
"""

import json
import os
import pandas as pd

TRAJ_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sa_appa_experiments", "results", "trajectories.jsonl"
)


def load_real_results(traj_path=TRAJ_PATH, method_ids=None, instance_ids=None):
    """Flatten trajectories.jsonl (one JSON object per run, with a `rows`
    list of per-evaluation dicts) into the canonical long DataFrame."""
    records = []
    with open(traj_path) as f:
        for line in f:
            obj = json.loads(line)
            m, inst, run = obj["method_id"], obj["instance_id"], obj["run_id"]
            if method_ids is not None and m not in method_ids:
                continue
            if instance_ids is not None and inst not in instance_ids:
                continue
            for row in obj["rows"]:
                records.append(dict(
                    method_id=m, instance_id=inst, run_id=run,
                    eval_count=row["eval_count"], iteration=row["iteration"],
                    best_feasible_obj=row["best_feasible_obj"] if row["is_feasible"] else float("nan"),
                    is_feasible=row["is_feasible"],
                ))
    df = pd.DataFrame.from_records(records)
    df.attrs["SYNTHETIC"] = False
    return df


if __name__ == "__main__":
    df = load_real_results()
    print(df.shape)
    print(df.head())
    print(df.method_id.unique())
    print(df.instance_id.nunique(), "instances")
