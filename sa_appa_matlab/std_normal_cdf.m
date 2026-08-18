function p = std_normal_cdf(z)
% STD_NORMAL_CDF  Standard normal CDF via erf, avoiding a Statistics
% Toolbox dependency on normcdf.
    p = 0.5 * (1 + erf(z / sqrt(2)));
end
