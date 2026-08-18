function D = pairwise_dist(X)
% PAIRWISE_DIST  Pairwise Euclidean distance matrix (n x n) for rows of X.
% Dependency-free replacement for the Statistics Toolbox's pdist/
% squareform, so gp_surrogate.m runs in base Octave/MATLAB without any
% toolbox.
    n = size(X, 1);
    D = zeros(n, n);
    for i = 1:n
        diff = X - X(i, :);
        D(i, :) = sqrt(sum(diff .^ 2, 2))';
    end
end
