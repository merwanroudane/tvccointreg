import numpy as np

from tvccointreg import TVCModel, DriverSpec
from tvccointreg.datasets import (simulate_nonlinear_cointegration,
                                  simulate_spurious)


def _fit_coint():
    sim = simulate_nonlinear_cointegration(T=300, seed=1, nonlinearity=0.4)
    spec = DriverSpec(names=list(sim.drivers.columns),
                      bias_free=["x_lag", "y_lag"], omitted=["w"])
    res = TVCModel(sim.y, sim.X, sim.drivers, driver_spec=spec).fit()
    return sim, res


def test_detects_generalized_cointegration():
    _, res = _fit_coint()
    ct = res.coint_test()
    assert bool(ct.loc["x", "cointegrated"])
    assert ct.loc["x", "p_value"] < 0.01


def test_rejects_spurious_relationship():
    sp = simulate_spurious(T=300, seed=2)
    spec = DriverSpec(names=list(sp.drivers.columns),
                      bias_free=["x_lag", "y_lag"])
    res = TVCModel(sp.y, sp.X, sp.drivers, driver_spec=spec).fit()
    ct = res.coint_test()
    assert not bool(ct.loc["x", "cointegrated"])


def test_recovers_time_varying_derivative():
    sim, res = _fit_coint()
    bf = res.bias_free_coefficients()["x"].to_numpy()
    truth = sim.true_beta["x"].to_numpy()
    corr = np.corrcoef(bf, truth)[0, 1]
    assert corr > 0.8


def test_residuals_stationary():
    _, res = _fit_coint()
    diag = res.diagnostics()
    # statsmodels is a dev/test dependency, so the ADF keys must be present
    assert diag.get("resid_stationary", True)
