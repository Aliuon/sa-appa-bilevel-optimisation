"""
friedman_test.py
=================
Standalone implementation of the Friedman test with Conover's (1998)
F-approximation, and its associated post-hoc paired critical-difference
procedure, exactly as presented in the attached methodology note
(Villegas, based on Conover 1998, pp. 294-302) -- Task 5.

This module is deliberately independent of the rest of the figures
package (it only needs a "rank matrix": one row per block/instance, one
column per treatment/method, entries = within-block ranks). It is used
by two callers:

1. The Task-5 analysis in this file's own `__main__` / notes, applied to
   Phase I and Phase II separately (see task5-friedman-test.md).
2. `figures/fig5_cd_diagram.py`, which previously (Task 4) implemented
   its OWN, INCORRECT, version of this statistic -- see "Bug found and
   fixed" below. It now imports `friedman_f_statistic` from here instead
   of recomputing it.

Formulae (Conover 1998, via the Friedmantest.pdf / Villegas note)
-------------------------------------------------------------------
Let n = number of blocks (instances), k = number of treatments (methods),
R(X_ij) = rank of treatment j within block i (average ranks on ties,
1 = best), R_j = sum_i R(X_ij) = rank sum for treatment j.

    A2 = sum_i sum_j R(X_ij)^2            ("A_2" in the note)
    B2 = sum_j R_j^2                      ("B_2" in the note)
    C1 = n * k * (k+1)^2 / 4               (correction term)

    SST          = A2 - C1                 (total sum of squares of ranks)
    SS_treatments = B2 / n - C1            (between-treatments sum of squares)
    SS_error      = SST - SS_treatments = A2 - B2 / n

    T2 = (n - 1) * SS_treatments / SS_error
       = (n - 1) * (B2/n - C1) / (A2 - B2/n)

T2 is referred to the F distribution with df1 = k-1, df2 = (n-1)(k-1)
(Conover's F-approximation -- preferred over the older chi-square
approximation because it is more accurate for typical n, k; the note is
explicit about this preference).

Post-hoc paired comparison (same note): two treatments i, j are declared
significantly different at level alpha if

    |R_i - R_j| > t_{1-alpha/2, (n-1)(k-1)} * sqrt( 2*n*SS_error / ((n-1)(k-1)) )

i.e. CD_conover (in RANK-SUM units, matching the note's own worked
example, which reports and compares raw rank-sum differences like
|80-48|=32, not mean-rank differences).

IMPORTANT -- this is NOT the Nemenyi critical distance used by the CD
diagram in Figure 5 (Task 4). Nemenyi's CD is a different, more
conservative, procedure conventionally used to draw CD diagrams
(Demsar, 2006), operating on MEAN ranks rather than rank sums and using
the studentized range distribution rather than the t distribution. The
two are related (both test rank-sum/mean-rank differences under the
Friedman omnibus model) but are not the same number and must not be
reported as if interchangeable. Per the user's explicit Task 5
instruction: the omnibus test reported here uses Conover's
F-approximation; the CD diagram's post-hoc bars use Nemenyi. See
`nemenyi_cd()` below for the (unchanged, already-correct) Nemenyi
formula, included here only for the side-by-side comparison in
task5-friedman-test.md -- `figures/fig5_cd_diagram.py` keeps its own
copy for the diagram itself, unchanged.

Bug found and fixed
--------------------
Task 4's `figures/fig5_cd_diagram.py._friedman_stat()` computed:

    denom = A1 - C1                        # = SST, not SS_error
    T = (n_instances - 1) * ss_ranks / denom     # ss_ranks = B2 - n*C1*4/(k(k+1)^2)... actually
                                                   # ss_ranks as coded = sum_j (R_j - n(k+1)/2)^2
                                                   # which equals B2 - n(k+1)*sum(R_j) + n_k*(n(k+1)/2)^2
                                                   # and, since sum(R_j) = n*k*(k+1)/2 identically,
                                                   # reduces to B2 - n*k*(k+1)^2/4 = B2 - C1,
                                                   # i.e. n*SS_treatments, NOT SS_treatments itself.

So the old code's numerator was n times too large AND its denominator
was SST instead of SS_error -- on the worked example (n=21, k=4) this
produced T=236.38 against the correct T2=25.74, an ~9.2x inflation.
Validated below against the note's own worked example.
"""

import numpy as np
from scipy.stats import f as f_dist, t as t_dist, rankdata, studentized_range


def rank_matrix_from_wide(wide):
    """
    wide: 2D array-like, shape (n_blocks, k_treatments), raw scores (lower
    = better, i.e. minimisation -- matching SA-APPA's objective
    convention). Returns the within-block rank matrix (average ranks on
    ties, 1 = best), same shape.
    """
    wide = np.asarray(wide, dtype=float)
    return np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, wide)


def friedman_f_statistic(rank_matrix):
    """
    rank_matrix: 2D array, shape (n_blocks, k_treatments), already ranks
    (1 = best) within each row/block.

    Returns a dict with A2, B2, C1, SST, SS_treatments, SS_error, T2,
    df1, df2, p_value, rank_sums (length k), n, k -- everything needed to
    reproduce the note's worked-example presentation.
    """
    rank_matrix = np.asarray(rank_matrix, dtype=float)
    n, k = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)

    A2 = float((rank_matrix ** 2).sum())
    B2 = float((rank_sums ** 2).sum())
    C1 = n * k * (k + 1) ** 2 / 4.0

    SST = A2 - C1
    SS_treatments = B2 / n - C1
    SS_error = SST - SS_treatments  # == A2 - B2/n

    if SS_error <= 0:
        T2, p_value = np.nan, np.nan
    else:
        T2 = (n - 1) * SS_treatments / SS_error
        df1, df2 = k - 1, (n - 1) * (k - 1)
        p_value = 1 - f_dist.cdf(T2, df1, df2)

    df1, df2 = k - 1, (n - 1) * (k - 1)
    return {
        "n": n, "k": k, "rank_sums": rank_sums,
        "A2": A2, "B2": B2, "C1": C1,
        "SST": SST, "SS_treatments": SS_treatments, "SS_error": SS_error,
        "T2": T2, "df1": df1, "df2": df2, "p_value": p_value,
    }


def f_critical(df1, df2, alpha):
    return f_dist.ppf(1 - alpha, df1, df2)


def conover_posthoc_cd(stat, alpha=0.05):
    """
    Conover's post-hoc critical difference on |R_i - R_j| (RANK-SUM
    units, matching the note's worked example), using the SS_error from
    `friedman_f_statistic()`'s output dict (`stat`).
    """
    n, k = stat["n"], stat["k"]
    df2 = stat["df2"]
    t_crit = t_dist.ppf(1 - alpha / 2, df2)
    cd = t_crit * np.sqrt(2 * n * stat["SS_error"] / df2)
    return cd, t_crit


def nemenyi_cd(k, n, alpha=0.05):
    """
    Nemenyi critical distance on MEAN ranks (the convention used by
    figures/fig5_cd_diagram.py for the CD diagram) -- included here only
    for side-by-side documentation; NOT used to compute conover_posthoc_cd
    above, and not a substitute for it. q_alpha via the exact studentized
    range distribution (scipy), not a hand-typed table.
    """
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def pairwise_table(stat, alpha_list=(0.05, 0.01)):
    """
    Build the note's-style pairwise |R_i - R_j| summary: for every method
    pair, the rank-sum difference and whether it exceeds the Conover CD
    at each alpha in alpha_list. Returns a list of dicts.
    method_labels: optional; if omitted, treatments are 0..k-1.
    """
    k = stat["k"]
    rs = stat["rank_sums"]
    rows = []
    cds = {a: conover_posthoc_cd(stat, a)[0] for a in alpha_list}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(rs[i] - rs[j])
            row = {"i": i, "j": j, "abs_diff": diff}
            for a in alpha_list:
                row[f"significant_{a}"] = diff > cds[a]
            rows.append(row)
    rows.sort(key=lambda r: -r["abs_diff"])
    return rows, cds


# ---------------------------------------------------------------------------
# Validation against the Friedmantest.pdf worked example
# (n=21 instances/blocks, k=4 metaheuristics, rank sums R=[80,48,51,31])
# ---------------------------------------------------------------------------

def _validate_against_worked_example(verbose=True):
    n, k = 21, 4
    rank_sums = np.array([80.0, 48.0, 51.0, 31.0])
    A2_expected = 630.0
    B2_expected = 12266.0

    # We don't have the raw per-block rank matrix (not reproduced in the
    # extracted text), but A2 and B2 are given directly in the worked
    # example, so validate the formula algebraically from A2/B2 rather
    # than from a synthesised rank matrix.
    C1 = n * k * (k + 1) ** 2 / 4.0
    SST = A2_expected - C1
    SS_treatments = B2_expected / n - C1
    SS_error = SST - SS_treatments
    T2 = (n - 1) * SS_treatments / SS_error
    df1, df2 = k - 1, (n - 1) * (k - 1)
    p_value = 1 - f_dist.cdf(T2, df1, df2)

    f_crit_05 = f_critical(df1, df2, 0.05)
    f_crit_01 = f_critical(df1, df2, 0.01)

    stat = {"n": n, "k": k, "rank_sums": rank_sums, "SS_error": SS_error, "df2": df2}
    # NOTE: the worked example's post-hoc CD (~15.08, t_crit~2.660) turns
    # out to correspond to alpha=0.01 two-sided (t_{0.995,60}), not
    # alpha=0.05 (t_{0.975,60}=2.000) -- i.e. the note applies the
    # post-hoc comparison at the *stricter* of the two levels it tested
    # the omnibus statistic against, having rejected H0 at both 0.05 and
    # 0.01. Validate against alpha=0.01 accordingly.
    cd_01, t_crit_01 = conover_posthoc_cd(stat, 0.01)

    if verbose:
        print("=== Validation against Friedmantest.pdf worked example ===")
        print(f"C1 = {C1} (expect 525)")
        print(f"SST = {SST} (expect 105)")
        print(f"SS_treatments = {SS_treatments:.3f} (expect 59.095)")
        print(f"SS_error = {SS_error:.3f} (expect 45.905)")
        print(f"T2 = {T2:.3f} (expect approx 25.74)")
        print(f"df1={df1}, df2={df2}")
        print(f"F_crit(0.05) = {f_crit_05:.3f} (expect approx 2.76)")
        print(f"F_crit(0.01) = {f_crit_01:.3f} (expect approx 4.13)")
        print(f"p_value = {p_value:.6g}")
        print(f"t_crit(alpha=0.01 two-sided, df2={df2}) = {t_crit_01:.3f} (expect approx 2.660)")
        print(f"CD (rank-sum units, alpha=0.01) = {cd_01:.3f} (expect approx 15.08)")
        print("Worked-example pairwise |R_i-R_j|: 32,29,49,3,17,20 "
              "-> significant except |R2-R3|=3")

    assert abs(C1 - 525) < 1e-9
    assert abs(SST - 105) < 1e-6
    assert abs(SS_treatments - 59.095) < 0.01
    assert abs(SS_error - 45.905) < 0.01
    assert abs(T2 - 25.74) < 0.1
    assert abs(f_crit_05 - 2.76) < 0.05
    assert abs(f_crit_01 - 4.13) < 0.05
    assert abs(t_crit_01 - 2.660) < 0.01
    assert abs(cd_01 - 15.08) < 0.1
    return T2, cd_01


if __name__ == "__main__":
    _validate_against_worked_example()
    print("\nAll assertions passed: formula matches the worked example.")
