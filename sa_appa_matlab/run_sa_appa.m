function traj = run_sa_appa(seed, varargin)
% RUN_SA_APPA  End-to-end MATLAB/Octave translation of the SA-APPA (M1)
% algorithm loop -- uniform-random initial design (NOT Latin Hypercube;
% see the note below), GP surrogates for the
% upper-level objective and each constraint, CEI infill, adaptive
% Prey-Predator step (Eqs. 8-10 of the manuscript) -- run on TP1
% (Appendix A.1), matching sa_appa_experiments/methods.py::
% run_ablation_method() with method_id='M1'
% (engine='appa', constraint_mode='cei', adaptation='adaptive', init='lhs')
% structurally.
%
% traj = run_sa_appa(seed)
% traj = run_sa_appa(seed, 'n0', 25, 'n_ul_pop', 15, 't_max', 35)
%
% Returns traj: a (t_max+1) x 3 matrix [eval_count, best_feasible_obj,
% is_feasible] tracking the running-best feasible objective, matching
% the Trajectory.record() convention in methods.py.
%
% IMPORTANT SCOPE AND VALIDATION NOTE: this driver is a faithful
% STRUCTURAL translation (same surrogate, same acquisition function, same
% adaptive control law, same infill loop), on ONE representative problem
% (TP1) rather than the full 44-instance suite -- translating and
% validating all 44 problems x 13 methods was out of scope for what could
% be produced and (see below) checked without a MATLAB/Octave interpreter
% in the environment this was written in. It has NOT been executed here.
% Even once you run it, do not expect it to reproduce a specific Python
% run's numbers: NumPy's and MATLAB/Octave's random number generators are
% different algorithms, so the same "seed" does not give the same
% trajectory across languages (see appa_step.m's header). What
% validate_against_python.m checks instead is that the DETERMINISTIC
% building blocks this loop is built from (the GP surrogate, the EI/PoF
% formulas, the PPA update formula given fixed inputs, the benchmark
% functions) match the Python implementation exactly. This driver
% composes those validated pieces into the same algorithm structure, but
% the emergent behaviour of a full stochastic run cannot be validated the
% same way, and is not claimed to be validated here.

    p = struct('n0', 25, 'n_ul_pop', 15, 't_max', 35, ...
               'beta', 0.2, 'gamma', 0.3, 'delta', 0.3, ...
               'll_popsize', 20, 'll_maxiter', 60);
    for k = 1:2:numel(varargin)
        p.(varargin{k}) = varargin{k+1};
    end
    seed_rng(seed);

    x_bounds = [-2 2; -2 2];
    y_bounds = [-2 2];
    F_fn = @(x, y) local_F(x, y);
    f_fn = @(x, y) local_f(x, y);
    g_fn = @(x, y) local_g(x, y);

    d_ul = size(x_bounds, 1);
    d_range_mean = mean(x_bounds(:, 2) - x_bounds(:, 1));

    % --- initial design (uniform-random in-box; a true Latin Hypercube
    % sampler is NOT included here, unlike methods.py's _lhs(), which
    % uses scipy.stats.qmc.LatinHypercube -- this is a scope reduction
    % from the Python implementation, not a hidden equivalence; flagged
    % here explicitly rather than silently substituted) ---
    X = x_bounds(:, 1)' + rand(p.n0, d_ul) .* (x_bounds(:, 2) - x_bounds(:, 1))';
    Ftr = zeros(p.n0, 1);
    Gtr = cell(p.n0, 1);
    feas = false(p.n0, 1);
    eval_count = 0;
    traj_rows = [];
    best_so_far = NaN;
    found_feasible = false;

    for i = 1:p.n0
        [~, F_i, G_i, ~] = ll_solve_simple(F_fn, f_fn, g_fn, X(i, :), y_bounds, ...
                                            'popsize', p.ll_popsize, 'maxiter', p.ll_maxiter);
        Ftr(i) = F_i;
        Gtr{i} = G_i;
        feas(i) = isempty(G_i) || all(G_i <= 1e-6);
        eval_count = eval_count + 1;
        if feas(i) && (~found_feasible || F_i < best_so_far)
            best_so_far = F_i;
            found_feasible = true;
        end
        traj_rows = [traj_rows; eval_count, iff(found_feasible, best_so_far, NaN), found_feasible]; %#ok<AGROW>
    end

    if any(feas)
        f_best = min(Ftr(feas));
    else
        f_best = Inf;
    end

    m_ul = numel(Gtr{1});
    theta_model = gp_surrogate(X, Ftr);
    phi_models = cell(1, m_ul);
    if m_ul > 0
        Garr = cell2mat(Gtr);
        for j = 1:m_ul
            phi_models{j} = gp_surrogate(X, Garr(:, j));
        end
    end

    Pop = x_bounds(:, 1)' + rand(p.n_ul_pop, d_ul) .* (x_bounds(:, 2) - x_bounds(:, 1))';
    lam = struct('explore', 0.3, 'local', 0.1, 'evade', 0.3, 'chase', 0.3, 'global_', 0.2, 'p_f', 0.6);
    sigma_init = std(Pop(:));
    tau_div = 0.3;

    for t = 0:(p.t_max - 1)
        [mu, sigma] = gp_predict(theta_model, Pop);
        ei = expected_improvement(mu, sigma, f_best);
        if m_ul > 0
            mu_g = cell(1, m_ul); sigma_g = cell(1, m_ul);
            for j = 1:m_ul
                [mu_g{j}, sigma_g{j}] = gp_predict(phi_models{j}, Pop);
            end
            pof = probability_of_feasibility(mu_g, sigma_g);
        else
            pof = ones(size(ei));
        end
        acq = ei .* pof;

        [~, best_idx] = max(acq);
        [~, worst_idx] = min(acq);

        % --- adaptive control law (Eqs. 8-10) ---
        sigma_pop = std(Pop(:));
        lam.explore = lam.explore * exp(p.beta * (tau_div - sigma_pop / (sigma_init + 1e-9)));
        delta_fit = (max(mu) - min(mu)) / (abs(max(mu)) + 1e-6);
        delta_fit = min(max(delta_fit, -1.0), 1.0);
        lam.evade = lam.evade * (1 - p.gamma * delta_fit);
        lam.chase = lam.chase * (1 + p.delta * delta_fit);
        lam.explore = min(max(lam.explore, 0.02), 1.0);
        lam.evade = min(max(lam.evade, 0.02), 1.0);
        lam.chase = min(max(lam.chase, 0.02), 1.0);

        lam_scaled = lam;
        lam_scaled.explore = lam.explore * d_range_mean;
        lam_scaled.local = lam.local * d_range_mean;
        lam_scaled.evade = lam.evade * d_range_mean;
        lam_scaled.chase = lam.chase * d_range_mean;
        lam_scaled.global_ = lam.global_ * d_range_mean;

        Pop = appa_step(Pop, best_idx, worst_idx, lam_scaled);
        Pop = min(max(Pop, x_bounds(:, 1)'), x_bounds(:, 2)');

        % --- infill ---
        [mu2, sigma2] = gp_predict(theta_model, Pop);
        ei2 = expected_improvement(mu2, sigma2, f_best);
        if m_ul > 0
            mu_g2 = cell(1, m_ul); sigma_g2 = cell(1, m_ul);
            for j = 1:m_ul
                [mu_g2{j}, sigma_g2{j}] = gp_predict(phi_models{j}, Pop);
            end
            pof2 = probability_of_feasibility(mu_g2, sigma_g2);
        else
            pof2 = ones(size(ei2));
        end
        acq2 = ei2 .* pof2;
        [~, infill_idx] = max(acq2);
        x_infill = Pop(infill_idx, :);

        [~, F_new, G_new, ~] = ll_solve_simple(F_fn, f_fn, g_fn, x_infill, y_bounds, ...
                                                'popsize', p.ll_popsize, 'maxiter', p.ll_maxiter);
        feas_new = isempty(G_new) || all(G_new <= 1e-6);
        eval_count = eval_count + 1;
        if feas_new && (~found_feasible || F_new < best_so_far)
            best_so_far = F_new;
            found_feasible = true;
        end
        traj_rows = [traj_rows; eval_count, iff(found_feasible, best_so_far, NaN), found_feasible]; %#ok<AGROW>

        X = [X; x_infill]; %#ok<AGROW>
        Ftr = [Ftr; F_new]; %#ok<AGROW>
        Gtr{end + 1} = G_new; %#ok<AGROW>
        theta_model = gp_surrogate(X, Ftr);
        if m_ul > 0
            Garr = cell2mat(Gtr);
            for j = 1:m_ul
                phi_models{j} = gp_surrogate(X, Garr(:, j));
            end
        end
    end

    traj = traj_rows;
end

function v = iff(cond, a, b)
    if cond
        v = a;
    else
        v = b;
    end
end

function F = local_F(x, y)
    [F, ~, ~] = benchmark_tp1(x, y);
end

function f = local_f(x, y)
    [~, f, ~] = benchmark_tp1(x, y);
end

function g = local_g(x, y)
    [~, ~, g] = benchmark_tp1(x, y);
end
