function validate_against_python(tol)
% VALIDATE_AGAINST_PYTHON  Runs every translated deterministic function
% against fixed reference input/output pairs computed by the Python
% implementation (generate_reference_values.py, shipped as
% reference_values.json alongside this script) and reports PASS/FAIL per
% check at the given numerical tolerance (default 1e-6).
%
% THIS SCRIPT HAS NOT BEEN RUN: no MATLAB or Octave interpreter was
% available in the sandboxed environment this translation was produced
% in (Octave was attempted via apt-get and failed -- the package mirror
% required for it was not reachable from that sandbox). Everything in
% this package is a careful, line-by-line translation of the validated
% Python source, cross-checked by static reading against
% sa_appa_experiments/{gp.py,methods.py,benchmark.py}, but it has not
% been executed anywhere. Run this script yourself before relying on any
% of these files -- that is the entire point of shipping
% reference_values.json alongside the code, rather than asserting
% "validated" without having actually validated it.
%
% Usage: validate_against_python()          % tol = 1e-6
%        validate_against_python(1e-8)      % custom tolerance

    if nargin < 1
        tol = 1e-6;
    end

    ref = jsondecode(fileread('reference_values.json'));
    n_pass = 0;
    n_fail = 0;

    fprintf('=== SA-APPA MATLAB translation vs. Python reference values ===\n');
    fprintf('Tolerance: %.2e\n\n', tol);

    % --- GP surrogate ---
    try
        X = cell2mat(ref.gp.X')';
        if size(X, 2) ~= 2
            X = reshape([ref.gp.X{:}], 2, [])';
        end
        y = ref.gp.y(:);
        model = gp_surrogate(X, y);
        Xstar = reshape([ref.gp.Xstar{:}], 2, [])';
        [mu, sigma] = gp_predict(model, Xstar);
        [n_pass, n_fail] = check('gp.lengthscale', model.lengthscale, ref.gp.lengthscale, tol, n_pass, n_fail);
        [n_pass, n_fail] = check('gp.mu', mu, ref.gp.mu(:), tol, n_pass, n_fail);
        [n_pass, n_fail] = check('gp.sigma', sigma, ref.gp.sigma(:), tol, n_pass, n_fail);
    catch err
        fprintf('  [ERROR] GP surrogate check raised: %s\n', err.message);
        n_fail = n_fail + 3;
    end

    % --- EI ---
    try
        ei = expected_improvement(ref.ei.mu(:), ref.ei.sigma(:), ref.ei.f_best);
        [n_pass, n_fail] = check('expected_improvement', ei, ref.ei.ei(:), tol, n_pass, n_fail);
    catch err
        fprintf('  [ERROR] EI check raised: %s\n', err.message);
        n_fail = n_fail + 1;
    end

    % --- PoF ---
    try
        pof = probability_of_feasibility({ref.pof.mu_g(:)}, {ref.pof.sigma_g(:)});
        [n_pass, n_fail] = check('probability_of_feasibility', pof, ref.pof.pof(:), tol, n_pass, n_fail);
    catch err
        fprintf('  [ERROR] PoF check raised: %s\n', err.message);
        n_fail = n_fail + 1;
    end

    % --- APPA step (deterministic, explicit random draws) ---
    try
        a = ref.appa_step;
        X = reshape([a.X{:}], 2, [])';
        % lam.global_ (not "global", a MATLAB keyword) matches the JSON
        % key written by generate_reference_values.py -- see that
        % script's note on why the Python side also renames it.
        lam = struct('explore', a.lam.explore, 'evade', a.lam.evade, ...
                      'chase', a.lam.chase, 'local', a.lam.local, ...
                      'global_', a.lam.global_, 'p_f', a.lam.p_f);
        U = struct('role', a.U.role(:), 'r1', a.U.r1(:), 'r4', a.U.r4(:), ...
                    'r5', a.U.r5, 'r2global', a.U.r2global(:)');
        U.r2 = reshape([a.U.r2{:}], 2, [])';
        Xnew = appa_step_deterministic(X, a.best_idx, a.worst_idx, lam, U);
        Xnew_ref = reshape([a.Xnew{:}], 2, [])';
        [n_pass, n_fail] = check('appa_step_deterministic', Xnew, Xnew_ref, tol, n_pass, n_fail);
    catch err
        fprintf('  [ERROR] APPA step check raised: %s\n', err.message);
        n_fail = n_fail + 1;
    end

    % --- Benchmark problems (TP1, TP2) ---
    try
        for i = 1:numel(ref.benchmark)
            b = ref.benchmark(i);
            if strcmp(b.problem, 'TP1')
                [F, f, g] = benchmark_tp1(b.x(:)', b.y(:)');
            else
                [F, f, g] = benchmark_tp2(b.x(:)', b.y(:)');
            end
            label = sprintf('benchmark.%s[%d]', b.problem, i);
            [n_pass, n_fail] = check([label '.F'], F, b.F, tol, n_pass, n_fail);
            [n_pass, n_fail] = check([label '.f'], f, b.f, tol, n_pass, n_fail);
            [n_pass, n_fail] = check([label '.g'], g(:), b.g(:), tol, n_pass, n_fail);
        end
    catch err
        fprintf('  [ERROR] Benchmark check raised: %s\n', err.message);
        n_fail = n_fail + 1;
    end

    fprintf('\n=== %d passed, %d failed (tolerance %.2e) ===\n', n_pass, n_fail, tol);
    if n_fail > 0
        fprintf('One or more checks FAILED. Do not treat this translation as validated until every check passes.\n');
    else
        fprintf('All checks passed against the Python reference values.\n');
    end
end

function [n_pass, n_fail] = check(name, actual, expected, tol, n_pass, n_fail)
    actual = actual(:);
    expected = expected(:);
    if numel(actual) ~= numel(expected)
        fprintf('  [FAIL] %-32s size mismatch (got %d, expected %d)\n', name, numel(actual), numel(expected));
        n_fail = n_fail + 1;
        return
    end
    err = max(abs(actual - expected));
    if err <= tol
        fprintf('  [PASS] %-32s max abs error = %.3e\n', name, err);
        n_pass = n_pass + 1;
    else
        fprintf('  [FAIL] %-32s max abs error = %.3e (tol %.3e)\n', name, err, tol);
        n_fail = n_fail + 1;
    end
end
