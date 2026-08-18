function [F, f, g] = benchmark_tp2(x, y)
% BENCHMARK_TP2  TP2 (Appendix A.1 of the manuscript), matching
% sa_appa_experiments/benchmark.py's TP2 Instance definition exactly:
%   min_x  F(x,y) = (1+x1-x2+2y2)(8-x1-2y1+y2+5y3),  x in [0,10]^2
%   y in argmin_y { 2y1 - y2 + y3 }, y in [0,10]^3, s.t.
%       -y1+y2+y3 <= 1
%       2x1-y1+2y2-0.5y3 <= 1
%       -(2x2+2y1-y2-0.5y3-1) <= 0   (i.e. 2x2+2y1-y2-0.5y3 >= 1)
% x: 1x2, y: 1x3. Returns F (scalar), f (scalar), g (1x3, feasible iff g<=0).
    F = (1 + x(1) - x(2) + 2*y(2)) * (8 - x(1) - 2*y(1) + y(2) + 5*y(3));
    f = 2*y(1) - y(2) + y(3);
    g = [ -y(1) + y(2) + y(3) - 1, ...
          2*x(1) - y(1) + 2*y(2) - 0.5*y(3) - 1, ...
          -(2*x(2) + 2*y(1) - y(2) - 0.5*y(3) - 1) ];
end
