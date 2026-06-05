"""
Generalized cointegration testing (Hall, Swamy & Tavlas, 2015).

Generalized cointegration between ``y`` and a regressor ``x_j`` holds iff the
*bias-free* component of the time-varying coefficient on ``x_j`` is nonzero
(eqs. 5-6).  In the operational TVC model the bias-free component is

    gamma_jt^BF = pi_j0 + sum_{d in bias_free} pi_jd z_dt ,

so the hypotheses are:

* **Joint (Wald) test**     H0 : pi_j0 = 0 and pi_jd = 0 for all bias-free d.
  Rejecting implies the bias-free coefficient is not identically zero -> the two
  variables are generalized-cointegrated.  Because inference rests on the
  stationary errors of the driver equations (Section 3.3), the statistic has a
  standard chi-square distribution -- no Dickey-Fuller critical values.

* **Average-effect test**   H0 : mean_t gamma_jt^BF = 0.
  A one-degree-of-freedom, easily interpreted statistic for "the average
  structural derivative is zero", using the delta method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class CointResult:
    name: str
    # joint Wald test of the bias-free block
    wald_stat: float
    wald_df: int
    wald_pvalue: float
    # average bias-free effect
    avg_effect: float
    avg_se: float
    avg_tstat: float
    avg_pvalue: float
    cointegrated: bool

    def as_row(self) -> dict:
        return {
            "regressor": self.name,
            "avg_bias_free": self.avg_effect,
            "std_err": self.avg_se,
            "t_stat": self.avg_tstat,
            "wald": self.wald_stat,
            "df": self.wald_df,
            "p_value": self.wald_pvalue,
            "cointegrated": self.cointegrated,
        }


def wald_test(pi_block: np.ndarray, cov_block: np.ndarray,
              mask: np.ndarray) -> tuple:
    """Chi-square Wald test that the masked subset of ``pi_block`` is zero."""
    sub = pi_block[mask]
    cov_sub = cov_block[np.ix_(mask, mask)]
    cov_inv = np.linalg.pinv(cov_sub)
    stat = float(sub @ cov_inv @ sub)
    df = int(mask.sum())
    pval = float(stats.chi2.sf(stat, df))
    return stat, df, pval


def average_effect(pi_block: np.ndarray, cov_block: np.ndarray,
                   Zbar: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Average bias-free effect and its delta-method standard error.

    ``Zbar`` is the mean driver design vector (length q).  The average bias-free
    coefficient is ``Zbar[mask] @ pi_block[mask]`` with variance
    ``Zbar[mask] @ cov_block[mask, mask] @ Zbar[mask]``.
    """
    m = Zbar[mask]
    sub = pi_block[mask]
    cov_sub = cov_block[np.ix_(mask, mask)]
    eff = float(m @ sub)
    var = float(m @ cov_sub @ m)
    se = float(np.sqrt(max(var, 0.0)))
    t = eff / se if se > 0 else np.nan
    pval = float(2 * stats.norm.sf(abs(t))) if se > 0 else np.nan
    return eff, se, t, pval


def test_coefficient(name: str, pi_block: np.ndarray, cov_block: np.ndarray,
                     Zbar: np.ndarray, bias_free_mask: np.ndarray,
                     alpha: float = 0.05) -> CointResult:
    """Run both the joint Wald and the average-effect tests for one regressor."""
    wstat, wdf, wp = wald_test(pi_block, cov_block, bias_free_mask)
    eff, se, t, ap = average_effect(pi_block, cov_block, Zbar, bias_free_mask)
    return CointResult(
        name=name, wald_stat=wstat, wald_df=wdf, wald_pvalue=wp,
        avg_effect=eff, avg_se=se, avg_tstat=t, avg_pvalue=ap,
        cointegrated=bool(wp < alpha),
    )


def adf_pvalue(series: np.ndarray) -> Optional[dict]:
    """
    Augmented Dickey-Fuller test on a residual series (used to check the paper's
    claim that the driver-equation errors are stationary).  Returns ``None`` if
    ``statsmodels`` is not installed.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return None
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 10:
        return None
    stat, pvalue, *_ = adfuller(series, autolag="AIC")
    return {"adf_stat": float(stat), "p_value": float(pvalue),
            "stationary": bool(pvalue < 0.05)}
