function [y_star, F_val, G_val, feasible_ll] = ll_solve_simple(F_fn, f_fn, g_fn, x, y_bounds, varargin)
% LL_SOLVE_SIMPLE  Nested lower-level solve via a self-contained
% differential-evolution optimiser (best/1/bin, quadratic constraint
% penalty), matching the STRUCTURE of
% sa_appa_experiments/ll_solver.py::ll_solve() -- same penalty
% formulation (PENALTY * sum(max(g,0)^2)), same "one consistent solver
% for every method/instance" design intent -- but NOT the same optimiser
% implementation: ll_solver.py uses scipy.optimize.differential_evolution
% (a specific, independently-validated implementation with its own
% internal bookkeeping), and this file is a plain, dependency-free DE
% written directly in MATLAB/Octave so run_sa_appa.m does not require the
% Global Optimization Toolbox. Do not expect this to reproduce scipy's DE
% run-for-run; both are "a reasonable global optimiser for the LL
% subproblem," not the same code.
%
% F_fn, f_fn, g_fn: function handles (x, y) -> scalar / scalar / vector,
% e.g. as returned by wrapping benchmark_tp1.m's outputs.
% x: 1 x n_ul (fixed UL point). y_bounds: n_ll x 2 [lo, hi] per dimension.
% Name-value options: 'popsize' (default 20), 'maxiter' (default 60),
% 'seed' (default none -> current RNG state), 'penalty' (default 1e6).

    p = struct('popsize', 20, 'maxiter', 60, 'seed', [], 'penalty', 1e6, ...
               'mutation', 0.7, 'crossover', 0.7);
    for k = 1:2:numel(varargin)
        p.(varargin{k}) = varargin{k+1};
    end
    if ~isempty(p.seed)
        seed_rng(p.seed);
    end

    n_ll = size(y_bounds, 1);
    lo = y_bounds(:, 1)'; hi = y_bounds(:, 2)';

    obj = @(y) f_fn(x, y) + p.penalty * sum(max(g_fn(x, y), 0) .^ 2);

    Pop = lo + rand(p.popsize, n_ll) .* (hi - lo);
    fit = zeros(p.popsize, 1);
    for i = 1:p.popsize
        fit(i) = obj(Pop(i, :));
    end

    for gen = 1:p.maxiter
        for i = 1:p.popsize
            idxs = setdiff(1:p.popsize, i);
            picks = idxs(randperm(numel(idxs), 3));
            a = Pop(picks(1), :); b = Pop(picks(2), :); c = Pop(picks(3), :);
            mutant = a + p.mutation * (b - c);
            mutant = min(max(mutant, lo), hi);
            cross = rand(1, n_ll) < p.crossover;
            if ~any(cross)
                cross(randi(n_ll)) = true;
            end
            trial = Pop(i, :);
            trial(cross) = mutant(cross);
            trial_fit = obj(trial);
            if trial_fit <= fit(i)
                Pop(i, :) = trial;
                fit(i) = trial_fit;
            end
        end
    end

    [~, best_i] = min(fit);
    y_star = Pop(best_i, :);
    G_val = g_fn(x, y_star);
    feasible_ll = isempty(G_val) || all(G_val <= 1e-6);
    F_val = F_fn(x, y_star);
end
