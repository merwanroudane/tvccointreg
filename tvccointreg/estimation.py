"""
Low-level estimation routines for the time-varying-coefficient (TVC) model.

The mapping between the paper (Hall, Swamy & Tavlas, 2015) and this module:

* Eq. (7)  y_t = gamma_0t + gamma_1t x_1t + ... + gamma_{K-1,t} x_{K-1,t}
* Eq. (8)  gamma_jt = pi_j0 + sum_d pi_jd z_dt + eps_jt

Substituting (8) into (7) gives the *concentrated* linear model

    y_t = w_t' pi + u_t ,      u_t = sum_j x_jt eps_jt ,

where ``w_t`` stacks, for every coefficient ``j``, the products ``x_jt * z_t``
of the regressor with the driver vector ``z_t = (1, z_1t, ..., z_{m,t})'``.
The composite error ``u_t`` is heteroskedastic with

    Var(u_t | x_t) = sum_j sigma_j^2 x_jt^2 ,

which is the Hildreth-Houck-Swamy random-coefficient structure.  We estimate
``pi`` by iteratively rescaled generalized least squares (the consistent,
asymptotically efficient estimator referenced in Section 3.3 of the paper),
recover the variance components ``sigma_j^2`` by a non-negative auxiliary
regression of squared residuals on squared regressors, and back out the
time-varying coefficients via the best linear predictor of ``eps_jt``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import nnls


@dataclass
class _GLSFit:
    pi: np.ndarray            # (P,)   stacked driver coefficients
    cov_pi: np.ndarray        # (P, P) GLS covariance of pi
    cov_pi_robust: np.ndarray  # (P, P) heteroskedasticity-robust covariance
    sigma2: np.ndarray        # (K,)   variance components of the coefficients
    h: np.ndarray             # (T,)   fitted variance of u_t
    resid: np.ndarray         # (T,)   composite residuals u_t
    fitted: np.ndarray        # (T,)   fitted y_t
    n_iter: int
    converged: bool
    loglik: float


def build_design(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Row-wise Khatri-Rao product giving the concentrated design ``W``.

    Parameters
    ----------
    X : (T, K) regressors *including* the leading column of ones (j = 0).
    Z : (T, m+1) driver matrix *including* the leading column of ones.

    Returns
    -------
    W : (T, K*(m+1)) with block ``j`` equal to ``X[:, j, None] * Z``.
    """
    T, K = X.shape
    _, q = Z.shape
    # (T, K, q) -> (T, K*q); block order is coefficient j, then driver d.
    return (X[:, :, None] * Z[:, None, :]).reshape(T, K * q)


def variance_components(resid: np.ndarray, X: np.ndarray,
                        floor: float = 1e-10) -> np.ndarray:
    """
    Hildreth-Houck-Swamy estimator of the coefficient variances ``sigma_j^2``.

    Regress the squared composite residuals on the squared regressors under a
    non-negativity constraint (variances cannot be negative).
    """
    b = resid ** 2
    A = X ** 2
    sigma2, _ = nnls(A, b)
    return np.maximum(sigma2, floor)


def _fitted_variance(sigma2: np.ndarray, X: np.ndarray,
                     floor: float = 1e-12) -> np.ndarray:
    h = (X ** 2) @ sigma2
    return np.maximum(h, floor)


def gls(y: np.ndarray, W: np.ndarray, h: np.ndarray):
    """Weighted least squares with diagonal weights ``1 / h``."""
    w = 1.0 / h
    Ww = W * w[:, None]
    XtX = W.T @ Ww
    Xty = Ww.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    pi = XtX_inv @ Xty
    return pi, XtX_inv, Ww


def irgls(y: np.ndarray, W: np.ndarray, X: np.ndarray,
          max_iter: int = 100, tol: float = 1e-8,
          verbose: bool = False) -> _GLSFit:
    """
    Iteratively rescaled GLS for the concentrated TVC model.

    Step 0 : OLS to start.
    Step k : (a) update variance components from current residuals,
             (b) rebuild the weights and re-estimate ``pi`` by GLS,
             until ``pi`` stabilises.
    """
    T, P = W.shape
    K = X.shape[1]

    # --- OLS start ---------------------------------------------------------
    pi = np.linalg.lstsq(W, y, rcond=None)[0]
    resid = y - W @ pi

    converged = False
    n_iter = 0
    h = np.ones(T)
    cov_pi = np.eye(P)
    for n_iter in range(1, max_iter + 1):
        sigma2 = variance_components(resid, X)
        h = _fitted_variance(sigma2, X)
        pi_new, XtX_inv, Ww = gls(y, W, h)
        delta = np.max(np.abs(pi_new - pi)) / (np.max(np.abs(pi)) + 1e-12)
        pi = pi_new
        resid = y - W @ pi
        cov_pi = XtX_inv  # (W' Omega^{-1} W)^{-1}
        if verbose:
            print(f"  iter {n_iter:3d}  rel.delta={delta:.3e}")
        if delta < tol:
            converged = True
            break

    # Heteroskedasticity-robust (sandwich) covariance ----------------------
    w = 1.0 / h
    Ww = W * w[:, None]
    bread = cov_pi
    meat = (Ww * (resid ** 2)[:, None]).T @ Ww
    cov_robust = bread @ meat @ bread

    fitted = W @ pi
    # Gaussian quasi-log-likelihood of the concentrated model.
    loglik = -0.5 * np.sum(np.log(2 * np.pi * h) + (resid ** 2) / h)

    return _GLSFit(
        pi=pi, cov_pi=cov_pi, cov_pi_robust=cov_robust, sigma2=sigma2,
        h=h, resid=resid, fitted=fitted, n_iter=n_iter, converged=converged,
        loglik=float(loglik),
    )


def predict_random_part(resid: np.ndarray, X: np.ndarray,
                        sigma2: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Best linear predictor of the coefficient errors ``eps_jt`` given the
    composite residual ``u_t`` (Swamy random-coefficient predictor):

        eps_hat_jt = sigma_j^2 x_jt / h_t * u_t .

    Returns an array of shape ``(T, K)``.
    """
    T, K = X.shape
    return (sigma2[None, :] * X) * (resid / h)[:, None]
