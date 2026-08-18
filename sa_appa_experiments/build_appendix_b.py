"""
build_appendix_b.py
====================
Generates the Appendix B LaTeX (one longtable per scenario, Best/Mean/Std/CPU
per method, one row per reconciled instance, with a Friedman-test summary
paragraph beneath each table) from the real results/scenario_tables/*.csv and
analysis.py's run_scenario().

Output: appendix_b_scenario_{1..7}.tex, appendix_b_all.tex (all 7, in order,
each starting a new \subsection, ready to \input into the appendix).
"""
import pandas as pd
from analysis import load_results, run_scenario
from scenarios import SCENARIOS

def esc(s):
    return str(s).replace("_", "\\_")

def fmt(v, nd=3):
    if pd.isna(v):
        return "--"
    return f"{v:.{nd}f}"

def method_label(m):
    # Table 1 short names, kept consistent with the manuscript's own naming
    names = {
        "M1": "SA-APPA", "M2": "M2 (unif.\\ init)", "M3": "M3 (bare PPA)",
        "M4": "M4 (GA engine)", "M5": "M5 (PSO engine)", "M6": "M6 (DE engine)",
        "M7": "M7 (dyn.\\ penalty)", "M8": "M8 (uncon.\\ EI)", "M9": "M9 (fixed sched.)",
        "M10": "M10 (SADE-EI)", "M11": "M11 (SPSO-RBF)",
        "M12": "M12 (Nested-NSGA-II)", "M13": "M13 (CoEA)",
    }
    return names.get(m, m)

def build_scenario_tex(scenario_num):
    s = next(x for x in SCENARIOS if x["num"] == scenario_num)
    methods = s["methods"]
    df = load_results()
    table_df = pd.read_csv(f"results/scenario_tables/scenario_{scenario_num}.csv")
    result = run_scenario(df, scenario_num)
    stat = result["stat"]

    k = len(methods)
    col_spec = "l" + "rrrr" * k
    header1 = " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{method_label(m)}}}" for m in methods) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2+4*i}-{5+4*i}}}" for i in range(k))
    header2 = "Instance & " + " & ".join("Best & Mean & Std & CPU (s)" for _ in methods) + " \\\\"

    lines = []
    lines.append(f"\\subsection{{Scenario {scenario_num} ({s['phase']}): {s['name']}}}")
    lines.append(f"\\label{{app:scenario{scenario_num}}}")
    lines.append(f"Methods compared: {', '.join(method_label(m) for m in methods)}. "
                 f"Isolates: {s['isolates']}. Each cell reports Best/Mean/Std (feasible runs "
                 f"only, out of 30 seeds) and mean CPU time per run; a dash indicates zero "
                 f"feasible runs on that instance.")
    lines.append("")
    lines.append("\\begingroup\\scriptsize")
    lines.append(f"\\begin{{longtable}}{{{col_spec}}}")
    lines.append(f"\\caption{{Scenario {scenario_num} raw results, all 44 reconciled instances.}}"
                 f"\\label{{tab:scenario{scenario_num}}}\\\\")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append(cmid)
    lines.append(header2)
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append(cmid)
    lines.append(header2)
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\midrule\\multicolumn{" + str(1+4*k) + "}{r}{\\emph{continued on next page}}\\\\\\midrule")
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for _, row in table_df.iterrows():
        cells = [esc(row["instance_id"])]
        for m in methods:
            cells.append(fmt(row[f"{m}_best"]))
            cells.append(fmt(row[f"{m}_mean"]))
            cells.append(fmt(row[f"{m}_std"]))
            cells.append(fmt(row[f"{m}_cpu"], nd=2))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    lines.append("")

    # Friedman summary
    decision05 = "reject" if stat["T2"] > result["f_crit_05"] else "fail to reject"
    decision01 = "reject" if stat["T2"] > result["f_crit_01"] else "fail to reject"
    mean_ranks_str = ", ".join(f"{method_label(m)}={r:.3f}" for m, r in result["mean_ranks"].items())
    lines.append(f"\\paragraph{{Friedman test summary (Scenario {scenario_num}).}}")
    lines.append(f"$n={stat['n']}$ instances, $k={stat['k']}$ methods. "
                 f"$A_2={stat['A2']:.3f}$, $B_2={stat['B2']:.3f}$, "
                 f"$T_2={stat['T2']:.4f}$, distributed $F({stat['df1']},{stat['df2']})$ under "
                 f"$H_0$, $p={stat['p_value']:.4g}$. Critical values: "
                 f"$F_{{0.05}}({stat['df1']},{stat['df2']})={result['f_crit_05']:.3f}$ "
                 f"({decision05} $H_0$ at $\\alpha=0.05$); "
                 f"$F_{{0.01}}({stat['df1']},{stat['df2']})={result['f_crit_01']:.3f}$ "
                 f"({decision01} $H_0$ at $\\alpha=0.01$). Mean ranks: {mean_ranks_str}. "
                 f"Conover post-hoc paired critical difference (rank-sum units): "
                 f"$CD_{{0.05}}={result['cd_conover_05']:.3f}$, $CD_{{0.01}}={result['cd_conover_01']:.3f}$. "
                 f"Nemenyi critical distance (mean-rank units, for the CD diagram, "
                 f"Figure~\\ref{{fig:cd{scenario_num}}}): $CD_{{0.05}}={result['cd_nemenyi_05']:.3f}$.")
    lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    all_tex = ["\\section{Scenario Results (Appendix B)}", "\\label{app:scenarios}", ""]
    for s in SCENARIOS:
        num = s["num"]
        tex = build_scenario_tex(num)
        with open(f"appendix_b_scenario_{num}.tex", "w") as fh:
            fh.write(tex)
        all_tex.append(tex)
        print(f"Scenario {num}: wrote appendix_b_scenario_{num}.tex ({len(tex)} chars)")
    with open("appendix_b_all.tex", "w") as fh:
        fh.write("\n".join(all_tex))
    print("Wrote appendix_b_all.tex")
