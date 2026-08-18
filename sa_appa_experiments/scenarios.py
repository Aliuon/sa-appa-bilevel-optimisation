"""
scenarios.py
============
The 7 experimental scenarios exactly as specified by the user, each an
independent comparison (own Friedman test, not pooled into one omnibus
test across all 13 methods).

"Reconciled problem instances relevant to that scenario's benchmark
group" is read as: the full 44-instance reconciled suite for every
scenario, since Phase I / Phase II in the manuscript split the methods
into two research questions, not the instances into two benchmark
groups -- every method (M1-M13) is defined to run on the same suite.
Flagged explicitly here since it's an interpretive choice, not a
recovered fact.
"""

SCENARIOS = [
    dict(num=1, phase="I", name="Initialisation strategy",
         methods=["M1", "M2"], isolates="LHS vs uniform init"),
    dict(num=2, phase="I", name="Genetic operators",
         methods=["M1", "M3"], isolates="crossover/mutation contribution"),
    dict(num=3, phase="I", name="Hierarchical dynamics",
         methods=["M3", "M4", "M5", "M6"],
         isolates="PPA engine vs GA/PSO/DE (M3, bare PPA, used as the PPA "
                   "representative so Scenario 2's genetic-operator effect "
                   "is not confounded with the engine comparison)"),
    dict(num=4, phase="I", name="Constraint management",
         methods=["M1", "M7", "M8"], isolates="CEI vs dynamic penalty vs unconstrained EI"),
    dict(num=5, phase="I", name="Surrogate-informed adaptation",
         methods=["M1", "M9"], isolates="adaptive vs fixed parameter schedule"),
    dict(num=6, phase="II", name="Surrogate-assisted competitors",
         methods=["M1", "M10", "M11"], isolates="SA-APPA vs SADE-EI, SPSO-RBF"),
    dict(num=7, phase="II", name="Non-surrogate baselines",
         methods=["M1", "M12", "M13"], isolates="surrogate efficiency gain vs nested/co-evolutionary search"),
]

SCENARIO_BY_NUM = {s["num"]: s for s in SCENARIOS}
