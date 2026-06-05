"""
High-level API: :class:`TVCModel` and :class:`TVCResults`.

This is the user-facing entry point of ``tvccointreg``.  It assembles the
concentrated design from the regressors and coefficient drivers, runs the
iteratively rescaled GLS estimator, recovers the time-varying coefficients and
their three-component decomposition, and exposes the generalized cointegration
tests, journal-style tables, and plots.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import estimation as _est
from . import cointegration as _coint
from .drivers import DriverSpec, default_spec

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, Sequence]


def _as_2d(obj, default_prefix: str):
    """Return (ndarray (T,k), names) from a DataFrame/array/Series."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy(dtype=float), list(obj.columns.astype(str))
    if isinstance(obj, pd.Series):
        return obj.to_numpy(dtype=float)[:, None], [str(obj.name or default_prefix)]
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    names = [f"{default_prefix}{i+1}" for i in range(arr.shape[1])]
    return arr, names


def _as_1d(obj, default_name: str):
    if isinstance(obj, pd.Series):
        return obj.to_numpy(dtype=float), str(obj.name or default_name)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError("y must be one column")
        return obj.iloc[:, 0].to_numpy(dtype=float), str(obj.columns[0])
    return np.asarray(obj, dtype=float).ravel(), default_name


class TVCModel:
    """
    Time-Varying-Coefficient regression with coefficient drivers.

    Parameters
    ----------
    y : array-like (T,)
        Dependent variable.
    X : array-like (T, K-1)
        Regressors (the constant is added automatically unless
        ``add_const=False``).
    drivers : array-like (T, m)
        Coefficient drivers ``z_dt``.  A constant driver is added automatically.
    driver_spec : DriverSpec, optional
        Three-set partition of the drivers.  If omitted, all drivers are treated
        as bias-free.  You may instead pass ``driver_sets`` as a dict.
    driver_sets : dict, optional
        Convenience alternative to ``driver_spec``, e.g.
        ``{"bias_free": [...], "omitted": [...], "measurement": [...]}``.
    add_const : bool, default True
        Add an intercept regressor (the time-varying intercept gamma_0t).
    index : array-like, optional
        Optional time index used for labelling plots/tables.
    """

    def __init__(
        self,
        y: ArrayLike,
        X: ArrayLike,
        drivers: ArrayLike,
        driver_spec: Optional[DriverSpec] = None,
        driver_sets: Optional[Dict[str, List[str]]] = None,
        add_const: bool = True,
        index: Optional[Sequence] = None,
    ):
        yv, yname = _as_1d(y, "y")
        Xv, xnames = _as_2d(X, "x")
        Zv, znames = _as_2d(drivers, "z")

        T = yv.shape[0]
        if Xv.shape[0] != T or Zv.shape[0] != T:
            raise ValueError("y, X and drivers must share the same length T")

        if add_const:
            Xv = np.column_stack([np.ones(T), Xv])
            xnames = ["const"] + xnames
        self.y = yv
        self.y_name = yname
        self.X = Xv
        self.coef_names = xnames           # length K
        self.K = Xv.shape[1]

        # driver design includes a leading constant
        self.Z_design = np.column_stack([np.ones(T), Zv])
        self.driver_names = znames

        if driver_spec is None:
            if driver_sets is None:
                driver_spec = default_spec(znames)
            else:
                driver_spec = DriverSpec(
                    names=list(znames),
                    bias_free=list(driver_sets.get("bias_free", [])),
                    omitted=list(driver_sets.get("omitted", [])),
                    measurement=list(driver_sets.get("measurement", [])),
                )
        if driver_spec.q != self.Z_design.shape[1]:
            raise ValueError("driver_spec does not match the number of drivers")
        self.spec = driver_spec
        self.q = driver_spec.q
        self.index = np.arange(T) if index is None else np.asarray(index)
        self.T = T

    # ------------------------------------------------------------------ fit
    def fit(self, method: str = "irgls", max_iter: int = 100,
            tol: float = 1e-8, cov_type: str = "gls",
            verbose: bool = False) -> "TVCResults":
        """
        Estimate the model.

        Parameters
        ----------
        method : {"irgls", "ols"}
            ``irgls`` is the iteratively rescaled GLS estimator (default and
            recommended).  ``ols`` runs a single OLS pass on the concentrated
            model (homoskedastic; mostly for teaching/diagnostics).
        cov_type : {"gls", "robust"}
            Covariance used for inference.  ``gls`` matches the standard-inference
            result of the paper; ``robust`` is the heteroskedasticity-robust
            sandwich.
        """
        W = _est.build_design(self.X, self.Z_design)
        if method == "ols":
            fit = _est.irgls(self.y, W, self.X, max_iter=1, tol=tol)
        elif method == "irgls":
            fit = _est.irgls(self.y, W, self.X, max_iter=max_iter, tol=tol,
                             verbose=verbose)
        else:
            raise ValueError("method must be 'irgls' or 'ols'")
        return TVCResults(self, W, fit, cov_type=cov_type)


class TVCResults:
    """Fitted-model container with coefficients, tests, tables and plots."""

    def __init__(self, model: TVCModel, W: np.ndarray, fit, cov_type: str):
        self.model = model
        self.W = W
        self._fit = fit
        self.cov_type = cov_type
        self.pi = fit.pi
        self.cov = fit.cov_pi_robust if cov_type == "robust" else fit.cov_pi
        self.sigma2 = fit.sigma2
        self.h = fit.h
        self.resid = fit.resid
        self.fitted = fit.fitted
        self.converged = fit.converged
        self.n_iter = fit.n_iter
        self.loglik = fit.loglik

        K, q = model.K, model.q
        self._blocks = [slice(j * q, (j + 1) * q) for j in range(K)]
        self._cache_coefficients()

    # -------------------------------------------------- coefficient recovery
    def _cache_coefficients(self):
        m = self.model
        Z = m.Z_design
        X = m.X
        K, q, T = m.K, m.q, m.T

        bf = m.spec.bias_free_mask()
        om = m.spec.omitted_mask()
        me = m.spec.measurement_mask()

        gamma_driver = np.zeros((T, K))
        bias_free = np.zeros((T, K))
        omitted = np.zeros((T, K))
        measurement = np.zeros((T, K))
        se_driver = np.zeros((T, K))

        for j in range(K):
            pij = self.pi[self._blocks[j]]
            covj = self.cov[self._blocks[j], self._blocks[j]]
            gamma_driver[:, j] = Z @ pij
            bias_free[:, j] = Z[:, bf] @ pij[bf]
            if om.any():
                omitted[:, j] = Z[:, om] @ pij[om]
            if me.any():
                measurement[:, j] = Z[:, me] @ pij[me]
            # pointwise SE of the driver-explained coefficient
            se_driver[:, j] = np.sqrt(np.einsum("td,de,te->t", Z, covj, Z,
                                                optimize=True).clip(min=0))

        eps = _est.predict_random_part(self.resid, X, self.sigma2, self.h)

        self._gamma_driver = gamma_driver
        self._bias_free = bias_free
        self._omitted = omitted
        self._measurement = measurement
        self._random = eps
        self._gamma = gamma_driver + eps
        self._se_driver = se_driver

    # ------------------------------------------------------------- accessors
    def _frame(self, arr) -> pd.DataFrame:
        return pd.DataFrame(arr, columns=self.model.coef_names,
                            index=pd.Index(self.model.index, name="t"))

    def tv_coefficients(self, include_random: bool = True) -> pd.DataFrame:
        """Time-varying coefficients gamma_jt as a (T x K) DataFrame."""
        return self._frame(self._gamma if include_random else self._gamma_driver)

    def bias_free_coefficients(self) -> pd.DataFrame:
        """Bias-free component of every coefficient (the structural derivative)."""
        return self._frame(self._bias_free)

    def coefficient_se(self) -> pd.DataFrame:
        """Pointwise standard errors of the driver-explained coefficients."""
        return self._frame(self._se_driver)

    def components(self, regressor: Union[str, int]) -> pd.DataFrame:
        """
        Three-component decomposition for one regressor:
        bias-free, omitted-variable bias, measurement-error bias, random,
        and total.
        """
        j = self._resolve(regressor)
        df = pd.DataFrame(
            {
                "bias_free": self._bias_free[:, j],
                "omitted_bias": self._omitted[:, j],
                "measurement_bias": self._measurement[:, j],
                "random": self._random[:, j],
                "total": self._gamma[:, j],
            },
            index=pd.Index(self.model.index, name="t"),
        )
        return df

    def _resolve(self, regressor: Union[str, int]) -> int:
        if isinstance(regressor, (int, np.integer)):
            return int(regressor)
        return self.model.coef_names.index(regressor)

    # -------------------------------------------------- cointegration testing
    def coint_test(self, alpha: float = 0.05,
                   skip_const: bool = True) -> pd.DataFrame:
        """
        Generalized cointegration test for every regressor.

        Returns a tidy DataFrame; the full :class:`CointResult` objects are kept
        in ``self.coint_results_``.
        """
        Zbar = self.model.Z_design.mean(axis=0)
        bf = self.model.spec.bias_free_mask()
        results = []
        for j, name in enumerate(self.model.coef_names):
            if skip_const and name == "const":
                continue
            pij = self.pi[self._blocks[j]]
            covj = self.cov[self._blocks[j], self._blocks[j]]
            res = _coint.test_coefficient(name, pij, covj, Zbar, bf, alpha=alpha)
            results.append(res)
        self.coint_results_ = results
        return pd.DataFrame([r.as_row() for r in results]).set_index("regressor")

    # ----------------------------------------------------------- diagnostics
    def r_squared(self) -> float:
        y = self.model.y
        ss_res = float(np.sum(self.resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def diagnostics(self) -> dict:
        """Fit and residual diagnostics, incl. ADF stationarity of u_t."""
        d = {
            "n_obs": self.model.T,
            "n_coefficients": self.model.K,
            "n_drivers": self.model.q - 1,
            "r_squared": self.r_squared(),
            "loglik": self.loglik,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "cov_type": self.cov_type,
        }
        adf = _coint.adf_pvalue(self.resid)
        if adf is not None:
            d["resid_adf_stat"] = adf["adf_stat"]
            d["resid_adf_pvalue"] = adf["p_value"]
            d["resid_stationary"] = adf["stationary"]
        return d

    # ---------------------------------------------------------------- tables
    def summary(self, fmt: str = "text") -> str:
        """Journal-style summary table.  ``fmt`` in {'text', 'latex', 'html'}."""
        from .tables import summary_table
        return summary_table(self, fmt=fmt)

    def coef_table(self) -> pd.DataFrame:
        """Average coefficients with bias-free component and significance."""
        from .tables import coefficient_frame
        return coefficient_frame(self)

    # ----------------------------------------------------------------- plots
    def plot_coefficients(self, **kwargs):
        from .plotting import plot_coefficients
        return plot_coefficients(self, **kwargs)

    def plot_decomposition(self, regressor, **kwargs):
        from .plotting import plot_decomposition
        return plot_decomposition(self, regressor, **kwargs)

    def plot_fit(self, **kwargs):
        from .plotting import plot_fit
        return plot_fit(self, **kwargs)

    def plot_coint_heatmap(self, **kwargs):
        from .plotting import plot_coint_heatmap
        return plot_coint_heatmap(self, **kwargs)

    def __repr__(self) -> str:
        return (f"<TVCResults: T={self.model.T}, K={self.model.K}, "
                f"drivers={self.model.q - 1}, converged={self.converged}>")
