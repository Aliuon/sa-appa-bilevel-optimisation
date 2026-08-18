function [mu, sigma] = gp_predict(model, Xstar)
% GP_PREDICT  Predictive mean/std at Xstar from a model fit by
% gp_surrogate(), matching gp.py::GPSurrogate.predict() (predictive mean
% Eq. 5, predictive variance Eq. 6 of the manuscript). See
% gp_surrogate.m's header for the translation-provenance and validation
% caveats (not executed by the environment that produced this file).
    Xstar = double(Xstar);
    Ks = se_kernel(Xstar, model.X, model.lengthscale, model.signal_var);
    mu = model.y_mean + Ks * model.alpha;
    v = model.L \ Ks';
    var_ = model.signal_var - sum(v .^ 2, 1)';
    var_ = max(var_, 1e-12);
    sigma = sqrt(var_);
end
