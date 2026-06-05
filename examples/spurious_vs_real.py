"""
Spurious regression vs. genuine generalized cointegration.

Demonstrates the central point of Hall, Swamy & Tavlas (2015): a naive OLS
regression of one random walk on another finds a "significant" slope, but the
generalized-cointegration test on the *bias-free* component does not reject the
null of no cointegration.
"""
import numpy as np
import statsmodels.api as sm

from tvccointreg import TVCModel, DriverSpec
from tvccointreg.datasets import simulate_spurious, simulate_nonlinear_cointegration


def naive_ols(y, x):
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()


print("================ SPURIOUS (independent random walks) ================")
sp = simulate_spurious(T=300, seed=2)
ols = naive_ols(sp.y.to_numpy(), sp.X["x"].to_numpy())
print(f"Naive OLS slope = {ols.params[1]:.3f}  (p = {ols.pvalues[1]:.3g}) "
      f"<- spuriously 'significant'")

res = TVCModel(sp.y, sp.X, sp.drivers,
               driver_spec=DriverSpec(names=list(sp.drivers.columns),
                                      bias_free=["x_lag", "y_lag"])).fit()
ct = res.coint_test()
print("Generalized cointegration:", "YES" if ct.loc["x", "cointegrated"]
      else "NO", f"(p = {ct.loc['x', 'p_value']:.3g})")

print("\n================ GENUINE generalized cointegration ================")
sim = simulate_nonlinear_cointegration(T=300, seed=1, nonlinearity=0.4)
res2 = TVCModel(sim.y, sim.X, sim.drivers,
                driver_spec=DriverSpec(names=list(sim.drivers.columns),
                                       bias_free=["x_lag", "y_lag"],
                                       omitted=["w"])).fit()
ct2 = res2.coint_test()
print("Generalized cointegration:", "YES" if ct2.loc["x", "cointegrated"]
      else "NO", f"(p = {ct2.loc['x', 'p_value']:.3g})")
