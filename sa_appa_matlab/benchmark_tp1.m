function [F, f, g] = benchmark_tp1(x, y)
% BENCHMARK_TP1  TP1 (Appendix A.1 of the manuscript), matching
% sa_appa_experiments/benchmark.py's TP1 Instance definition exactly:
%   min_x  F(x,y) = x1^2 + x2^2,  x in [-2,2]^2
%   y in argmin_y { 2*x1 + y1 : x1^2 + y1^2 <= 4, x1 - y1 <= 0 }, y1 in [-2,2]
% x: 1x2, y: 1x1. Returns F (scalar), f (scalar), g (1x2, feasible iff g<=0).
    F = x(1)^2 + x(2)^2;
    f = 2 * x(1) + y(1);
    g = [x(1)^2 + y(1)^2 - 4, x(1) - y(1)];
end
