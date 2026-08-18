function pof = probability_of_feasibility(mu_g_list, sigma_g_list)
% PROBABILITY_OF_FEASIBILITY  Product of per-constraint feasibility
% probabilities Phi(-mu_g/sigma_g), matching methods.py::_pof() and the
% PoF(x) term of Eq. (CEI) in the manuscript.
%
% mu_g_list, sigma_g_list: cell arrays, one (vector-valued) entry per
% upper-level constraint, each the GP-predicted mean/std of that
% constraint at a batch of candidate points. Returns a vector, one PoF
% value per candidate.
    if isempty(mu_g_list)
        pof = 1.0;
        return
    end
    n = numel(mu_g_list{1});
    pof = ones(n, 1);
    for j = 1:numel(mu_g_list)
        sigma_j = max(sigma_g_list{j}, 1e-12);
        pof = pof .* std_normal_cdf(-mu_g_list{j} ./ sigma_j);
    end
end
