"""
Quick-start example for tvccointreg.

Estimates a time-varying-coefficient model on a nonlinear, generalized-
cointegrated DGP, tests for generalized cointegration, prints a journal table,
and saves the figures into docs/img/.
"""
import os
import matplotlib
matplotlib.use("Agg")

from tvccointreg import TVCModel, DriverSpec
from tvccointreg.datasets import simulate_nonlinear_cointegration

HERE = os.path.dirname(__file__)
IMG = os.path.join(HERE, "..", "docs", "img")
os.makedirs(IMG, exist_ok=True)

# 1) Simulate a nonlinear relationship between nonstationary variables.
sim = simulate_nonlinear_cointegration(T=300, seed=1, nonlinearity=0.4)

# 2) Declare the three driver sets (Assumption 1 / eq. 8).
spec = DriverSpec(
    names=list(sim.drivers.columns),
    bias_free=["x_lag", "y_lag"],   # true coefficient variation
    omitted=["w"],                  # omitted-variable bias
    measurement=[],                 # measurement-error bias
)

# 3) Fit by iteratively rescaled GLS.
model = TVCModel(sim.y, sim.X, sim.drivers, driver_spec=spec)
res = model.fit()

# 4) Inspect results.
print(res.summary())
print("\nGeneralized cointegration test:")
print(res.coint_test())

# 5) Save publication-quality figures (Parula colormap).
res.plot_coefficients().savefig(os.path.join(IMG, "coefficients.png"),
                                bbox_inches="tight")
res.plot_decomposition("x").savefig(os.path.join(IMG, "decomposition.png"),
                                    bbox_inches="tight")
res.plot_fit().savefig(os.path.join(IMG, "fit.png"), bbox_inches="tight")
res.plot_coint_heatmap().savefig(os.path.join(IMG, "heatmap.png"),
                                 bbox_inches="tight")
print("\nFigures written to docs/img/")
