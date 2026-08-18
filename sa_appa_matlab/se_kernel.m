function K = se_kernel(A, B, lengthscale, signal_var)
% SE_KERNEL  Squared-exponential (RBF) kernel matrix between rows of A
% and rows of B, matching gp.py::GPSurrogate._kernel() exactly.
    na = size(A, 1);
    d2 = zeros(na, size(B, 1));
    for i = 1:na
        diff = B - A(i, :);
        d2(i, :) = sum(diff .^ 2, 2)';
    end
    K = signal_var * exp(-0.5 * d2 / (lengthscale ^ 2));
end
