function ei = expected_improvement(mu, sigma, f_best)
% EXPECTED_IMPROVEMENT  Standard (minimisation) expected-improvement
% acquisition, matching methods.py::_ei() and Eq. (EI) of the manuscript:
%   z = (f_best - mu) / sigma
%   EI = (f_best - mu) * Phi(z) + sigma * phi(z)
% mu, sigma, f_best may be scalars or vectors (elementwise).
% Depends on std_normal_cdf.m / std_normal_pdf.m (no Statistics Toolbox
% dependency).
    sigma = max(sigma, 1e-12);
    z = (f_best - mu) ./ sigma;
    ei = (f_best - mu) .* std_normal_cdf(z) + sigma .* std_normal_pdf(z);
end
