import numpy as np

from tvccointreg import TVCModel, DriverSpec, parula_colors, resolve_colorscale
from tvccointreg.datasets import simulate_nonlinear_cointegration


def test_components_sum_to_total():
    sim = simulate_nonlinear_cointegration(T=200, seed=3, nonlinearity=0.3)
    spec = DriverSpec(names=list(sim.drivers.columns),
                      bias_free=["x_lag"], omitted=["w"], measurement=["y_lag"])
    res = TVCModel(sim.y, sim.X, sim.drivers, driver_spec=spec).fit()
    comp = res.components("x")
    recon = (comp["bias_free"] + comp["omitted_bias"]
             + comp["measurement_bias"] + comp["random"])
    np.testing.assert_allclose(recon.to_numpy(), comp["total"].to_numpy(),
                               rtol=1e-8, atol=1e-8)


def test_driver_spec_defaults_unassigned_to_bias_free():
    spec = DriverSpec(names=["a", "b", "c"], omitted=["b"])
    assert set(spec.bias_free) == {"a", "c"}
    assert spec.omitted == ["b"]
    # masks have correct length (q = drivers + const)
    assert spec.bias_free_mask().shape[0] == 4


def test_parula_palette_and_colorscale():
    cols = parula_colors(8)
    assert len(cols) == 8
    assert all(c.startswith("#") for c in cols)
    scale = resolve_colorscale("Parula", 5)
    assert len(scale) == 5
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0


def test_summary_formats():
    sim = simulate_nonlinear_cointegration(T=150, seed=4)
    spec = DriverSpec(names=list(sim.drivers.columns),
                      bias_free=["x_lag"], omitted=["w"])
    res = TVCModel(sim.y, sim.X, sim.drivers, driver_spec=spec).fit()
    assert "Generalized Cointegration" in res.summary("text")
    assert "\\toprule" in res.summary("latex")
    assert "<table" in res.summary("html")
