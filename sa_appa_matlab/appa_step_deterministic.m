function Xnew = appa_step_deterministic(X, best_idx, worst_idx, lam, U)
% APPA_STEP_DETERMINISTIC  One generation of the adaptive Prey-Predator
% update (Eqs. prey/pred of the manuscript), matching
% sa_appa_experiments/methods.py::_appa_step() exactly, EXCEPT that all
% random draws are passed in explicitly via the struct U rather than
% drawn from an internal RNG stream.
%
% This is deliberate: NumPy's and MATLAB/Octave's random number
% generators are different algorithms, so even with "the same seed" a
% Python run and a MATLAB run of the stochastic optimiser would not
% produce identical trajectories -- that is a property of comparing two
% different RNGs, not a translation error, and is not a target this
% validation tries to hit. What CAN be validated exactly is that, given
% IDENTICAL random draws (generated once in Python and shipped as
% reference_values.json), the two implementations of the update FORMULA
% produce identical output. That is what this function and
% validate_against_python.m's corresponding check test.
%
% X: (N x d) population. best_idx, worst_idx: 1-indexed row indices.
% lam: struct with fields explore, evade, chase, local, global_, p_f
% (MATLAB reserves the identifier "global", hence global_).
% U: struct with fields role (N x 1), r1 (N x 1), r2 (N x d, one row per
%    particle), r4 (N x 1), r5 (scalar), r2global (1 x d).
%
% See run_sa_appa.m for the production (internally-RNG-driven) version
% used for actual optimisation runs.

    [N, d] = size(X);
    Xnew = X;
    p_f = lam.p_f;

    for i = 1:N
        if i == worst_idx
            continue
        end
        if U.role(i) <= p_f
            r1 = U.r1(i);
            diff = X(best_idx, :) - X(i, :);
            n = norm(diff) + 1e-12;
            r2 = U.r2(i, :);
            Xnew(i, :) = X(i, :) + lam.explore * r1 * diff / n + lam.local * (r2 * 2 - 1);
        else
            r4 = U.r4(i);
            diff = X(i, :) - X(worst_idx, :);
            n = norm(diff) + 1e-12;
            Xnew(i, :) = X(i, :) + lam.evade * r4 * diff / n;
        end
    end

    r5 = U.r5;
    diff = X(best_idx, :) - X(worst_idx, :);
    Xnew(worst_idx, :) = X(worst_idx, :) + lam.chase * r5 * diff + lam.global_ * (U.r2global * 2 - 1);
end
