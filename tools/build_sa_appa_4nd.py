"""
build_sa_appa_4nd.py
====================
Assembles SA_APPA_4nd.py, the single-file Python version of the SA-APPA codebase, from the
module files of the repository. Core modules (benchmark, gp, ll_solver, methods, methods_m14,
scenarios, friedman_test, analysis, analysis_extra) are copied verbatim apart from their
intra-package imports, so the merged file runs exactly the code that produced the locked results.
Driver, figure and Phase 7 modules are copied with the few renames listed in RENAMES, needed
because they share module-level names.

Usage (from the repository root):  python tools/build_sa_appa_4nd.py
"""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "sa_appa_experiments")
FIG = os.path.join(ROOT, "sa_appa_figures")
INTERNAL = r"(benchmark|gp|ll_solver|methods|methods_m14|scenarios|friedman_test|analysis|analysis_extra|" \
           r"phase7_data|config|figures(\.\w+)?)"


def read(path):
    return open(path, encoding="utf-8").read()


COLLECTED_IMPORTS = []


def collect_imports(src):
    """Remove every remaining top-level import statement (single or parenthesised multi-line) and
    remember it, so that all imports appear once at the top of the merged file."""
    pat = re.compile(r"^(?:from [\w.]+ import \([^)]*\)|from [\w.]+ import [^\n]*|import [^\n]*)\n", re.M)
    for m in pat.finditer(src):
        stmt = m.group(0).strip()
        if stmt not in COLLECTED_IMPORTS:
            COLLECTED_IMPORTS.append(stmt)
    src = pat.sub("", src)
    src = re.sub(r"^matplotlib\.use\([^\n]*\)\n", "", src, flags=re.M)
    return src


def strip_module(src, keep_doc_as_comment=True):
    """Drop intra-package imports, the __main__ block and turn the module docstring into comments."""
    # multi-line parenthesised internal imports
    src = re.sub(r"^from " + INTERNAL + r" import \([^)]*\)\s*\n", "", src, flags=re.M)
    src = re.sub(r"^from " + INTERNAL + r" import [^\n]*\n", "", src, flags=re.M)
    src = re.sub(r"^import " + INTERNAL + r"( as \w+)?[^\n]*\n", "", src, flags=re.M)
    src = re.sub(r"^sys\.path\.insert\([^\n]*\n", "", src, flags=re.M)
    # internal imports inside function bodies: the names already live in this file
    src = re.sub(r"^(\s+)(from " + INTERNAL + r" import [^\n]*|import " + INTERNAL + r"[^\n]*)\n",
                 r"\1pass  # (was: \2)\n", src, flags=re.M)
    i = src.find('\nif __name__ == "__main__":')
    if i >= 0:
        src = src[:i] + "\n"
    src = collect_imports(src)
    m = re.match(r'\s*"""(.*?)"""\s*\n', src, flags=re.S)
    if m and keep_doc_as_comment:
        doc = "\n".join("# " + ln if ln.strip() else "#" for ln in m.group(1).strip("\n").splitlines())
        src = doc + "\n" + src[m.end():]
    return src.strip("\n") + "\n"


def sub_words(src, mapping):
    for a, b in mapping.items():
        src = re.sub(r"(?<![\w.])" + re.escape(a) + r"(?!\w)", b, src)
    return src


def section(title, body):
    bar = "# " + "=" * 96
    return f"\n\n{bar}\n# {title}\n{bar}\n\n{body}"


def main():
    out = [read(os.path.join(ROOT, "tools", "sa_appa_4nd_head.py"))]

    for name, title in [("benchmark", "BENCHMARK SUITE (benchmark.py)"), ("gp", "SURROGATES (gp.py)"),
                        ("ll_solver", "LOWER-LEVEL SOLVER (ll_solver.py)"),
                        ("methods", "CONFIGURATIONS M1-M13 (methods.py)"),
                        ("methods_m14", "CONFIGURATION M14, BL-SAEA RE-IMPLEMENTATION (methods_m14.py)"),
                        ("scenarios", "THE EIGHT SCENARIOS (scenarios.py)"),
                        ("friedman_test", "FRIEDMAN / CONOVER / NEMENYI (friedman_test.py)")]:
        src = strip_module(read(os.path.join(EXP, name + ".py")))
        out.append(section(title, src))

    src = strip_module(read(os.path.join(EXP, "analysis.py")))
    src = src.replace('RESULTS_CSV = "results/summary.csv"', 'RESULTS_CSV = os.path.join(RESULTS_DIR, "summary.csv")')
    src = src.replace('RESULTS_CSV_M14 = "results/summary_m14.csv"', 'RESULTS_CSV_M14 = os.path.join(RESULTS_DIR, "summary_m14.csv")')
    src = src.replace("def load_results(include_m14=True):", "def load_results(include_m14=True):\n"
                      "    RESULTS_CSV = os.path.join(RESULTS_DIR, \"summary.csv\")\n"
                      "    RESULTS_CSV_M14 = os.path.join(RESULTS_DIR, \"summary_m14.csv\")")
    out.append(section("SCENARIO STATISTICS (analysis.py)", src))

    src = strip_module(read(os.path.join(EXP, "analysis_extra.py")))
    src = sub_words(src, {"main": "analysis_extra_main"})
    out.append(section("FEASIBILITY, CONSTRAINT-DENSITY SUBSET AND HOLM (analysis_extra.py)", src))

    # analysis_resub.main -> compute_resub_numbers(); export -> export_cd_summaries()
    src = strip_module(read(os.path.join(EXP, "analysis_resub.py")))
    src = sub_words(src, {"main": "compute_resub_numbers"})
    src = src.replace('"results/scenario_tables/scenario_8.csv"', 'os.path.join(RESULTS_DIR, "scenario_tables", "scenario_8.csv")')
    src = src.replace('open("results/resub_numbers.json", "w")', 'open(os.path.join(RESULTS_DIR, "resub_numbers.json"), "w")')
    src = src.replace("    import os\n", "")
    out.append(section("ALL TEST STATISTICS OF THE PAPER (analysis_resub.py)", src))

    src = strip_module(read(os.path.join(EXP, "export_scenario_summaries.py")))
    src = sub_words(src, {"main": "export_cd_summaries", "OUT_PATH": "CD_SUMMARY_PATH", "_to_native": "_cd_to_native"})
    src = src.replace('CD_SUMMARY_PATH = "results/scenario_cd_summaries.json"', "")
    src = src.replace("open(CD_SUMMARY_PATH, \"w\")", "open(os.path.join(RESULTS_DIR, \"scenario_cd_summaries.json\"), \"w\")")
    src = src.replace('print(f"Wrote {CD_SUMMARY_PATH}")', 'print("Wrote scenario_cd_summaries.json")')
    out.append(section("CD-DIAGRAM SUMMARIES (export_scenario_summaries.py)", src))

    # s8 LL-accuracy check: top-level script -> function
    src = strip_module(read(os.path.join(EXP, "analysis_s8_llcheck.py")))
    body = []
    for ln in src.splitlines():
        if ln.startswith("#") or ln.startswith("import ") or ln.startswith("from "):
            if ln.startswith("#"):
                body.append(ln)
            continue
        body.append(("    " + ln) if ln.strip() else "")
    fn = "def run_llcheck():\n" + "\n".join(b for b in body if not b.startswith("#")) + "\n"
    fn = fn.replace('pd.read_csv("results/summary.csv")', 'pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))')
    fn = fn.replace('pd.read_csv("results/summary_m14.csv")', 'pd.read_csv(os.path.join(RESULTS_DIR, "summary_m14.csv"))')
    fn = fn.replace('J = "results/resub_numbers.json"', 'J = os.path.join(RESULTS_DIR, "resub_numbers.json")')
    fn = sub_words(fn, {"TOL": "LLCHECK_TOL"})
    doc = "\n".join(b for b in body if b.startswith("#"))
    out.append(section("VALUES BELOW THE KNOWN OPTIMUM (analysis_s8_llcheck.py)", doc + "\n" + fn))

    src = strip_module(read(os.path.join(EXP, "build_appendix_b.py")))
    src = src.replace('f"results/scenario_tables/scenario_{scenario_num}.csv"',
                      'os.path.join(RESULTS_DIR, "scenario_tables", f"scenario_{scenario_num}.csv")')
    src = sub_words(src, {"main": "build_appendix_b_main"})
    out.append(section("APPENDIX B TABLES AS LATEX (build_appendix_b.py)", src))

    # figures: common.save_figure + scenario CD diagrams
    src = strip_module(read(os.path.join(FIG, "figures", "common.py")))
    out.append(section("FIGURE HELPERS (figures/common.py)", src))
    src = strip_module(read(os.path.join(FIG, "figures", "fig5_cd_diagram_scenarios.py")))
    out.append(section("CRITICAL-DIFFERENCE DIAGRAMS (figures/fig5_cd_diagram_scenarios.py)", src))

    src = strip_module(read(os.path.join(FIG, "figures_resub.py")))
    src = re.sub(r'^HERE = [^\n]*\nEXP = [^\n]*\n', "", src, flags=re.M)
    src = re.sub(r'^OUT = [^\n]*\n', "", src, flags=re.M)
    src = src.replace('os.path.join(EXP, "results", fn)', "os.path.join(RESULTS_DIR, fn)")
    src = sub_words(src, {"save": "rs_save", "lighten": "rs_lighten", "LABEL": "RS_LABEL", "main": "figures_resub_main",
                          "OUT": "FIG_DIR", "load": "rs_load"})
    out.append(section("PAPER FIGURES 1-4 (figures_resub.py)", src))

    # Phase 7
    src = strip_module(read(os.path.join(EXP, "phase7_data.py")))
    src = re.sub(r'^HERE = [^\n]*\nRES = [^\n]*\nOUT = [^\n]*\nos\.makedirs\(OUT, exist_ok=True\)\n', "", src, flags=re.M)
    src = sub_words(src, {"OUT": "P7_DIR", "RES": "RESULTS_DIR", "PEN": "P7_PEN"})
    out.append(section("PHASE 7 DATA: GROUND TRUTH, RUN SELECTION, GP SNAPSHOTS (phase7_data.py)", src))

    src = strip_module(read(os.path.join(EXP, "phase7_run.py")))
    src = re.sub(r"^OUT = P\.OUT\n", "", src, flags=re.M)
    src = src.replace("P.OUT", "P7_DIR")
    src = re.sub(r"\bP\.", "", src)
    src = sub_words(src, {"OUT": "P7_DIR", "log": "P7_LOG", "grid2": "p7_grid2"})
    main_block = read(os.path.join(EXP, "phase7_run.py"))
    mb = main_block[main_block.find('if __name__ == "__main__":'):]
    mb = mb.replace('if __name__ == "__main__":\n    parts = sys.argv[1:] or ["A1", "A2", "A3", "A5"]',
                    'def phase7_run_main(parts=("A1", "A2", "A3", "A5")):\n    parts = list(parts)')
    mb = mb.replace("OUT", "P7_DIR").replace("log[", "P7_LOG[")
    out.append(section("PHASE 7 RUNNER, PART A (phase7_run.py)", src + "\n\n" + mb))

    src = strip_module(read(os.path.join(EXP, "instrument_ll.py")))
    src = re.sub(r'^HERE = [^\n]*\nRES = [^\n]*\n', "", src, flags=re.M)
    src = src.replace('OUT = os.path.join(RES, "ll_counts_subset.csv")', "")
    src = sub_words(src, {"OUT": "LL_COUNTS_CSV", "RES": "RESULTS_DIR", "FIELDS": "LL_FIELDS", "METHODS": "ALL_METHODS",
                          "_worker": "_ll_worker", "main": "instrument_ll_main", "TOL": "REPRO_TOL",
                          "_Counter": "_LLCounter", "_load_locked": "_ll_load_locked",
                          "_load_locked_traj": "_ll_load_locked_traj"})
    src = src.replace("def instrument_ll_main():\n    ap = argparse.ArgumentParser()\n"
                      "    ap.add_argument(\"--runs\", type=int, default=5)\n"
                      "    ap.add_argument(\"--workers\", type=int, default=2)\n    a = ap.parse_args()",
                      "def instrument_ll_main(runs=5, workers=2):\n    import types\n"
                      "    a = types.SimpleNamespace(runs=runs, workers=workers)")
    out.append(section("PHASE 7 PART B1: INSTRUMENTED RE-RUN (instrument_ll.py)", src))

    src = strip_module(read(os.path.join(FIG, "figures_phase7.py")))
    src = re.sub(r'^HERE = [^\n]*\nEXP = [^\n]*\nRES = [^\n]*\nP7 = [^\n]*\nOUT = [^\n]*\n', "", src, flags=re.M)
    src = sub_words(src, {"OUT": "P7_FIG_DIR", "RES": "RESULTS_DIR", "P7": "P7_DIR", "save": "p7_save",
                          "lighten": "p7_lighten", "LABEL": "P7_LABEL", "MANIFEST": "P7_MANIFEST", "J": "p7_J"})
    mb = read(os.path.join(FIG, "figures_phase7.py"))
    mb = mb[mb.find('if __name__ == "__main__":'):]
    mb = mb.replace('if __name__ == "__main__":\n    which = sys.argv[1:] or ["A1", "A2", "A3", "A4", "A5", "B2", "B1"]',
                    'def figures_phase7_main(which=("A1", "A2", "A3", "A4", "A5", "B2", "B1")):')
    mb = mb.replace("MANIFEST", "P7_MANIFEST").replace("OUT", "P7_FIG_DIR")
    out.append(section("PHASE 7 FIGURES (figures_phase7.py)", src + "\n\n" + mb))

    out.append(read(os.path.join(ROOT, "tools", "sa_appa_4nd_tail.py")))
    head = out[0]
    import ast
    plain, froms = set(), {}
    for stmt in COLLECTED_IMPORTS:
        for node in ast.parse(stmt).body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("matplotlib"):
                        continue
                    plain.add((a.name, a.asname))
            else:
                froms.setdefault(node.module, set()).update((a.name, a.asname) for a in node.names)
    fmt = lambda n, a: f"{n} as {a}" if a else n
    lines = ["import matplotlib", 'matplotlib.use("Agg")', "import matplotlib.pyplot as plt"]
    lines += [f"import {fmt(n, a)}" for n, a in sorted(plain, key=lambda t: (t[0], t[1] or ""))]
    for mod in sorted(froms):
        names = ", ".join(fmt(n, a) for n, a in sorted(froms[mod], key=lambda t: (t[0], t[1] or "")))
        lines.append(f"from {mod} import {names}")
    imp_block = "\n".join(lines) + "\n"
    out[0] = head.replace("# @@IMPORTS@@\n", imp_block)
    text = "".join(out)
    dest = os.path.join(ROOT, "SA_APPA_4nd.py")
    open(dest, "w", encoding="utf-8").write(text)
    print("wrote", dest, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
