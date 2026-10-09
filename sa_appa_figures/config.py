"""
config.py
=========
Single source of truth for (a) the thirteen algorithm variants defined in
Table 1 of the SA-APPA paper, and (b) the 44-instance benchmark suite
reconciled in Task 2 (9 TP problems + 12 SMD problems x 3 scales, minus the
SMD10 @ (p,q)=(30,30) instance excluded there for evaluation-budget
infeasibility).

Every figure-generation function in this package takes its method list and
problem list from here, rather than hard-coding names -- this is what lets
you add/remove/rename a method or instance in one place and have every
figure update consistently.

NOTHING in this file is experimental *data*. It is structural metadata only
(method names, citations, problem dimensions). The actual run results
(objective values, convergence trajectories) live in `synthetic_data.py`
(placeholder) or in whatever real-results loader you substitute for it --
see README.md.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Table 1: algorithm variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Method:
    id: str
    name: str
    ul_optimizer: str
    constraint_handling: str
    infill: str
    initialization: str
    surrogate: bool
    citation: str = ""   # empty for the paper's own methods (M1-M9)


METHODS = [
    Method("M1",  "SA-APPA",         "Adaptive PPA",           "GP-based CEI",           "CEI",                "LHS",     True),
    Method("M2",  "SA-APPA-wLHS",    "Adaptive PPA",           "GP-based CEI",           "CEI",                "Uniform", True),
    Method("M3",  "SA-PPA",          "Basic PPA (no GA ops)",  "GP-based CEI",           "CEI",                "LHS",     True),
    Method("M4",  "SA-GA",           "Genetic Algorithm",      "GP-based CEI",           "CEI",                "LHS",     True),
    Method("M5",  "SA-DE",           "Differential Evolution", "GP-based CEI",           "CEI",                "LHS",     True),
    Method("M6",  "SA-PSO",          "PSO",                    "GP-based CEI",           "CEI",                "LHS",     True),
    Method("M7",  "SA-APPA-Penalty", "Adaptive PPA",           "Dynamic penalty",        "EI (penalized)",     "LHS",     True),
    Method("M8",  "SA-APPA-EI",      "Adaptive PPA",           "None (checked post hoc)", "EI",                "LHS",     True),
    Method("M9",  "SA-APPA-Fixed",   "Adaptive PPA",           "GP-based CEI",           "CEI, fixed sched.",  "LHS",     True),
    # M10-M13 are in-house baselines (see sa_appa_experiments/methods.py METHOD_SPECS);
    # they are NOT re-implementations of published methods. M12 and M13 run the same
    # nested GA with different seed streams. M14 re-implements BL-SAEA (methods_m14.py).
    Method("M10", "SA-DE-Pen",       "Differential Evolution", "Penalised EI (GP)",      "EI (penalised)",     "LHS",     True),
    Method("M11", "SA-PSO-RBF",      "PSO",                    "Penalised EI (RBF)",     "EI (penalised)",     "LHS",     True),
    Method("M12", "Nest-GA (A)",     "GA, fully nested",       "Feasibility rules",      "--",                 "LHS",     False),
    Method("M13", "Nest-GA (B)",     "GA, fully nested",       "Feasibility rules",      "--",                 "LHS",     False),
    Method("M14", "BL-SAEA",         "GA (SBX+PM)",            "Feasibility penalty",    "ND screening",       "LHS",     True,  citation="Jiang et al., IEEE TCYB 54(7) 2024"),
]

METHOD_IDS = [m.id for m in METHODS]
METHOD_BY_ID = {m.id: m for m in METHODS}

# Phase I (internal ablation) vs Phase II (external comparison) groupings,
# per the paper's own Section 5 structure.
PHASE_I_METHODS = [m.id for m in METHODS if m.id in
                    {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"}]
PHASE_II_METHODS = [m.id for m in METHODS if m.id in
                     {"M1", "M10", "M11", "M12", "M13", "M14"}]


# ---------------------------------------------------------------------------
# Benchmark suite: 44 reconciled instances (Task 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    id: str
    family: str          # "TP" or "SMD"
    base_problem: str    # e.g. "TP1" or "SMD3"
    n_ul: int             # upper-level dimension at this instance's scale
    n_ll: int             # lower-level dimension at this instance's scale
    scale: str = ""       # "" for TP problems; "(3,3)" / "(10,10)" / "(30,30)" for SMD
    constrained: bool = True
    group: str = ""        # panel/grouping key used by Figure 1 (assigned below)


def _tp_instances():
    # (n_ul, n_ll) read directly from Appendix A's stated box dimensions.
    dims = {
        "TP1": (2, 1), "TP2": (2, 3), "TP3": (2, 2), "TP4": (2, 2),
        "TP5": (2, 2), "TP6": (1, 2), "TP7": (2, 2), "TP8": (2, 2),
        "TP9": (10, 10),
    }
    return [
        Instance(id=pid, family="TP", base_problem=pid,
                  n_ul=nul, n_ll=nll, constrained=True, group="TP1-TP9")
        for pid, (nul, nll) in dims.items()
    ]


def _smd_instances():
    """
    12 SMD problems x 3 scales (p,q) in {(3,3),(10,10),(30,30)}, r=2 fixed,
    minus SMD10 @ (30,30) -- excluded per Task 2 (constraint count scales as
    p+q+r, reaching 62 simultaneous cubic constraints at the largest scale,
    which the fixed evaluation budget could plausibly fail to satisfy even
    once; see claude/task2-appendix-a-vs-assumptions.md, item 5, for the
    [NEEDS AUTHOR CHECK] status of this reconstruction).
    """
    scales = [(3, 3), (10, 10), (30, 30)]
    r = 2
    instances = []
    for i in range(1, 13):
        base = f"SMD{i}"
        for (p, q) in scales:
            if base == "SMD10" and (p, q) == (30, 30):
                continue  # excluded instance, Task 2
            instances.append(Instance(
                id=f"{base}_p{p}q{q}",
                family="SMD",
                base_problem=base,
                n_ul=p + r,
                n_ll=q + r + (2 if base == "SMD6" else 0),  # SMD6 has s=2 extra LL vars
                scale=f"({p},{q})",
                constrained=base in {"SMD9", "SMD10", "SMD11", "SMD12"},
                group=f"SMD @ ({p},{q})",
            ))
    return instances


INSTANCES = _tp_instances() + _smd_instances()
INSTANCE_IDS = [inst.id for inst in INSTANCES]
INSTANCE_BY_ID = {inst.id: inst for inst in INSTANCES}

assert len(INSTANCES) == 44, (
    f"Expected 44 reconciled instances (Task 2), got {len(INSTANCES)}. "
    "If you've changed the exclusion list, update the count check here too."
)

# Panels for Figure 1 (one panel per benchmark group), in display order.
FIGURE1_GROUPS = ["TP1-TP9", "SMD @ (3,3)", "SMD @ (10,10)", "SMD @ (30,30)"]

N_RUNS = 5           # independent runs per (method, instance) cell -- Task 3
RANDOM_SEED = 20260711  # global seed for reproducibility (today's date, arbitrarily)
