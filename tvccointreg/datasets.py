"""
Synthetic data generators for demonstrating and testing ``tvccointreg``.

These build *nonlinear* relationships among *nonstationary* variables -- exactly
the setting the generalized-cointegration definition targets (Hall, Swamy &
Tavlas, 2015).  Standard linear (Engle-Granger) cointegration tests fail here,
yet a structural relationship is present and is recovered by the TVC model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimulatedData:
    y: pd.Series
    X: pd.DataFrame
    drivers: pd.DataFrame
    true_beta: pd.DataFrame   # true time-varying structural coefficient(s)
    info: dict


def simulate_nonlinear_cointegration(
    T: int = 300,
    seed: Optional[int] = 0,
    nonlinearity: float = 0.15,
    omitted: bool = True,
    measurement_error: float = 0.0,
) -> SimulatedData:
    """
    Nonlinear, generalized-cointegrated DGP.

    The latent structural relationship is

        y_t = f(x_t, w_t) + e_t ,
        f(x_t, w_t) = a + b * x_t + c * x_t^2 / scale + d * w_t ,

    where ``x_t`` is a (near) random-walk nonstationary regressor, ``w_t`` is a
    relevant but possibly *omitted* preexisting condition (also nonstationary),
    and the quadratic term makes the true partial derivative
    ``dy/dx = b + 2c x_t / scale`` time-varying.  Good coefficient drivers
    therefore include ``x_t`` itself and ``w_t``.

    Parameters
    ----------
    nonlinearity : float
        Strength ``c`` of the quadratic term (0 => linear).
    omitted : bool
        If True, ``w`` is returned as a *driver* but withheld from ``X``,
        creating omitted-variable bias that the driver split must absorb.
    measurement_error : float
        Std. dev. of noise added to the observed ``x`` (measurement error).
    """
    rng = np.random.default_rng(seed)

    # nonstationary regressor and preexisting condition (random walks)
    x = np.cumsum(rng.normal(0, 1.0, T)) / np.sqrt(T) * 5.0
    w = np.cumsum(rng.normal(0, 1.0, T)) / np.sqrt(T) * 4.0

    a, b, c, d = 1.0, 0.8, nonlinearity, 0.5
    scale = np.std(x) * 3 + 1e-8
    true_dydx = b + 2.0 * c * x / scale  # time-varying structural derivative

    f = a + b * x + c * x ** 2 / scale + d * w
    e = rng.normal(0, 0.3, T)
    y = f + e

    # observed x may carry measurement error
    x_obs = x + (rng.normal(0, measurement_error, T)
                 if measurement_error > 0 else 0.0)

    index = pd.RangeIndex(T, name="t")
    y_s = pd.Series(y, index=index, name="y")

    X = pd.DataFrame({"x": x_obs}, index=index)

    # candidate drivers: lagged x, w, lagged y -- all observable
    drivers = pd.DataFrame({
        "x_lag": np.r_[x_obs[0], x_obs[:-1]],
        "w": w,
        "y_lag": np.r_[y[0], y[:-1]],
    }, index=index)

    true_beta = pd.DataFrame({"x": true_dydx}, index=index)

    info = {
        "model": "y = a + b x + c x^2/scale + d w + e",
        "a": a, "b": b, "c": c, "d": d, "scale": scale,
        "omitted_w": omitted,
        "measurement_error": measurement_error,
    }
    if not omitted:
        X["w"] = w
        drivers = drivers.drop(columns=["w"])
    return SimulatedData(y=y_s, X=X, drivers=drivers,
                         true_beta=true_beta, info=info)


def simulate_spurious(T: int = 300, seed: Optional[int] = 0) -> SimulatedData:
    """
    Two *independent* random walks (no structural relationship).

    A naive OLS regression of ``y`` on ``x`` finds a "significant" slope
    (spurious regression), but the generalized-cointegration test on the
    bias-free component should *not* reject the null of no cointegration.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1.0, T))
    y = np.cumsum(rng.normal(0, 1.0, T))
    index = pd.RangeIndex(T, name="t")
    drivers = pd.DataFrame({
        "x_lag": np.r_[x[0], x[:-1]],
        "y_lag": np.r_[y[0], y[:-1]],
    }, index=index)
    return SimulatedData(
        y=pd.Series(y, index=index, name="y"),
        X=pd.DataFrame({"x": x}, index=index),
        drivers=drivers,
        true_beta=pd.DataFrame({"x": np.zeros(T)}, index=index),
        info={"model": "independent random walks (no cointegration)"},
    )
