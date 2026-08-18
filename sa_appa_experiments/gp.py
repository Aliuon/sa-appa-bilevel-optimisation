"""
gp.py
=====
Lightweight Gaussian Process regressor, implemented directly with numpy
Cholesky factorisation (Definition 4 of the manuscript: predictive mean
Eq.(5), predictive variance Eq.(6)), rather than a heavier third-party
library, so that repeated retraining (once per generation, per Lemma 3)
is fast at the population/instance/run counts this project needs.

Kernel: squared-exponential (RBF) with automatic relevance determination
disabled (single lengthscale per surrogate, set once via the median
heuristic on the initial design and held fixed thereafter -- this keeps
retraining to "refactorise the Cholesky of the growing training set" per
Assumption A4 / Lemma 3's cost model, rather than re-running maximum
likelihood hyperparameter estimation at every one of the thousands of
retraining events this project performs; Assumption A4 explicitly
permits either "fixed or re-estimated by maximum likelihood" -- we use
the fixed-hyperparameter branch and say so, matching Lemma 3's own
"assumes hyperparameters are held fixed" cost-model note added in
Task 1's clarifying addition).

A simple RBF interpolant (RBFSurrogate) is also provided for M11
(SPSO-RBF, Table 1's only non-GP surrogate method).
"""

import numpy as np

JITTER = 1e-8


class GPSurrogate:
    def __init__(self, lengthscale=None, signal_var=1.0, noise_var=1e-6):
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.X = None
        self.y_mean = 0.0
        self.y = None
        self.L = None
        self.alpha = None

    def _kernel(self, A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return self.signal_var * np.exp(-0.5 * d2 / (self.lengthscale ** 2))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.lengthscale is None:
            # median-heuristic lengthscale, set once on first fit and held fixed
            if X.shape[0] > 1:
                d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
                iu = np.triu_indices_from(d, k=1)
                med = np.median(d[iu]) if len(iu[0]) else 1.0
                self.lengthscale = max(med, 1e-3)
            else:
                self.lengthscale = 1.0
        self.X = X
        self.y_mean = float(np.mean(y))
        yc = y - self.y_mean
        K = self._kernel(X, X) + (self.noise_var + JITTER) * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yc))
        self.y = y
        return self

    def predict(self, Xstar, return_std=True):
        Xstar = np.atleast_2d(Xstar)
        Ks = self._kernel(Xstar, self.X)
        mu = self.y_mean + Ks @ self.alpha
        if not return_std:
            return mu
        v = np.linalg.solve(self.L, Ks.T)
        var = self.signal_var - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-12)
        return mu, np.sqrt(var)


class RBFSurrogate:
    """Deterministic (multiquadric) RBF interpolant for M11 (SPSO-RBF), the
    only non-GP surrogate in Table 1. Has no native predictive variance;
    for the EI-style infill M11 uses per Table 1, we substitute a
    distance-to-nearest-training-point exploration term in place of GP
    variance, a standard proxy in the RBF-assisted global-optimisation
    literature (e.g. Regis & Shoemaker-style algorithms) -- documented
    here explicitly since the original SPSO-RBF paper's exact acquisition
    formula was not obtainable (paywalled, per Task 3's citation note)."""

    def __init__(self, epsilon=None):
        self.epsilon = epsilon
        self.X = None
        self.w = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X = X
        if self.epsilon is None:
            d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1))
            iu = np.triu_indices_from(d, k=1)
            self.epsilon = max(np.median(d[iu]) if len(iu[0]) else 1.0, 1e-3)
        Phi = np.sqrt(1.0 + (np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)) / self.epsilon ** 2)
        Phi += JITTER * np.eye(len(X))
        self.w = np.linalg.solve(Phi, y)
        return self

    def predict(self, Xstar, return_std=True):
        Xstar = np.atleast_2d(Xstar)
        d2 = np.sum((Xstar[:, None, :] - self.X[None, :, :]) ** 2, axis=-1)
        Phi = np.sqrt(1.0 + d2 / self.epsilon ** 2)
        mu = Phi @ self.w
        if not return_std:
            return mu
        dist = np.sqrt(np.min(d2, axis=1))
        pseudo_std = dist  # exploration proxy, see class docstring
        return mu, pseudo_std
