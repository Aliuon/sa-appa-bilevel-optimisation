function Xnew = appa_step(X, best_idx, worst_idx, lam)
% APPA_STEP  Production version of one generation of the adaptive
% Prey-Predator update, matching sa_appa_experiments/methods.py::
% _appa_step() structurally, but drawing random numbers from MATLAB's/
% Octave's own internal RNG stream (via rand()) rather than taking them
% as explicit arguments.
%
% Because NumPy's and MATLAB/Octave's random number generators are
% different algorithms, this will NOT reproduce a specific Python run's
% trajectory even with "the same seed" -- that is expected, not a
% translation defect (see appa_step_deterministic.m, which is what
% validate_against_python.m actually checks the update FORMULA against,
% using explicit pre-drawn random numbers shipped from the Python side).
% This production version is what run_sa_appa.m calls for an actual
% end-to-end optimisation run.
%
% X: (N x d) population. best_idx, worst_idx: 1-indexed row indices.
% lam: struct with fields explore, evade, chase, local, global_, p_f.
    [N, d] = size(X);
    Xnew = X;
    p_f = lam.p_f;

    for i = 1:N
        if i == worst_idx
            continue
        end
        if rand() <= p_f
            r1 = rand();
            diff = X(best_idx, :) - X(i, :);
            n = norm(diff) + 1e-12;
            r2 = rand(1, d);
            Xnew(i, :) = X(i, :) + lam.explore * r1 * diff / n + lam.local * (r2 * 2 - 1);
        else
            r4 = rand();
            diff = X(i, :) - X(worst_idx, :);
            n = norm(diff) + 1e-12;
            Xnew(i, :) = X(i, :) + lam.evade * r4 * diff / n;
        end
    end

    r5 = rand();
    diff = X(best_idx, :) - X(worst_idx, :);
    Xnew(worst_idx, :) = X(worst_idx, :) + lam.chase * r5 * diff + lam.global_ * (rand(1, d) * 2 - 1);
end
