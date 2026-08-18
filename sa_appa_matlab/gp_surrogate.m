function model = gp_surrogate(X, y, varargin)
% GP_SURROGATE  Fit a Gaussian-process regressor (squared-exponential
% kernel, fixed lengthscale via the median heuristic, Cholesky-based
% training), a line-by-line MATLAB/Octave translation of
% sa_appa_experiments/gp.py::GPSurrogate.fit().
%
% model = gp_surrogate(X, y)
% model = gp_surrogate(X, y, 'lengthscale', ell, 'signal_var', sv, 'noise_var', nv)
%
% X: (n x d) training inputs. y: (n x 1) training targets.
% Returns a struct with fields X, y_mean, L, alpha, lengthscale,
% signal_var, noise_var, to be passed to gp_predict().
%
% Faithfulness note: this mirrors gp.py's fixed-hyperparameter branch
% (Assumption A4's "fixed" option) exactly, including the median-distance
% lengthscale heuristic and the same jitter constant. It has NOT been
% executed against an actual MATLAB/Octave interpreter in the environment
% this translation was produced in (none was available -- see
% README_MATLAB.md for what was and was not possible to verify here).
% validate_against_python.m checks this and the other translated
% functions against Python-computed reference values; run it once you
% have a MATLAB/Octave environment, before relying on this code.

    p = struct('lengthscale', [], 'signal_var', 1.0, 'noise_var', 1e-6);
    for k = 1:2:numel(varargin)
        p.(varargin{k}) = varargin{k+1};
    end

    JITTER = 1e-8;
    X = double(X);
    y = double(y(:));
    n = size(X, 1);

    if isempty(p.lengthscale)
        if n > 1
            D = pairwise_dist(X);
            iu = triu(true(n), 1);
            dvec = D(iu);
            if isempty(dvec)
                med = 1.0;
            else
                med = median(dvec);
            end
            ell = max(med, 1e-3);
        else
            ell = 1.0;
        end
    else
        ell = p.lengthscale;
    end

    K = se_kernel(X, X, ell, p.signal_var) + (p.noise_var + JITTER) * eye(n);
    L = chol(K, 'lower');
    y_mean = mean(y);
    yc = y - y_mean;
    alpha = L' \ (L \ yc);

    model = struct('X', X, 'y', y, 'y_mean', y_mean, 'L', L, 'alpha', alpha, ...
                    'lengthscale', ell, 'signal_var', p.signal_var, 'noise_var', p.noise_var);
end
