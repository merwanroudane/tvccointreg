"""
Empirical application: the US aggregate consumption function.

Data: ``statsmodels`` ``macrodata`` -- real US quarterly series, 1959Q1-2009Q3
(203 obs). We study the long-run relationship between real personal consumption
and real disposable income.

Both ``log(realcons)`` and ``log(realdpi)`` are strongly trending / nonstationary
(I(1)). The marginal propensity to consume (here the income *elasticity* of
consumption, since we work in logs) is widely believed to drift over the business
cycle and across policy regimes -- precisely a *generalized cointegration*
question: is the bias-free structural derivative dy/dx nonzero once we allow the
coefficient to vary and control for omitted business-cycle conditions?

Coefficient drivers (Assumption 1, eq. 8):
  * bias-free   : lagged log income, a time trend  -> true elasticity variation
  * omitted     : real interest rate, unemployment  -> omitted-variable bias
                  (interest-rate / labour-market channels left out of the bivariate
                   relation)
  * measurement : lagged log consumption            -> dynamics / measurement
"""
import numpy as np
import pandas as pd
from statsmodels.datasets import macrodata

from tvccointreg import TVCModel, DriverSpec


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def load():
    d = macrodata.load_pandas().data.copy()
    d.index = pd.period_range(start="1959Q1", periods=len(d), freq="Q")

    y = np.log(d["realcons"]).rename("log_cons")
    x = np.log(d["realdpi"]).rename("log_inc")

    # drivers (standardised for numerical conditioning; paths are invariant)
    drivers = pd.DataFrame({
        "inc_lag":  zscore(np.log(d["realdpi"]).shift(1)),
        "trend":    zscore(pd.Series(np.arange(len(d)), index=d.index)),
        "realint":  zscore(d["realint"]),
        "unemp":    zscore(d["unemp"]),
        "cons_lag": zscore(np.log(d["realcons"]).shift(1)),
    })

    df = pd.concat([y, x, drivers], axis=1).dropna()
    return df


def main():
    df = load()
    y = df["log_cons"]
    X = df[["log_inc"]]
    drivers = df[["inc_lag", "trend", "realint", "unemp", "cons_lag"]]

    spec = DriverSpec(
        names=list(drivers.columns),
        bias_free=["inc_lag", "trend"],
        omitted=["realint", "unemp"],
        measurement=["cons_lag"],
    )

    res = TVCModel(y, X, drivers, driver_spec=spec,
                   index=df.index.astype(str)).fit()

    print(res.summary())
    print("\nGeneralized cointegration test:")
    print(res.coint_test().round(4).to_string())
    print("\nDiagnostics:")
    for k, v in res.diagnostics().items():
        print(f"  {k}: {v}")

    bf = res.bias_free_coefficients()["log_inc"]
    print(f"\nIncome elasticity of consumption (bias-free):")
    print(f"  mean   = {bf.mean():.3f}")
    print(f"  min    = {bf.min():.3f}  ({bf.idxmin()})")
    print(f"  max    = {bf.max():.3f}  ({bf.idxmax()})")
    return res


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    res = main()
    img = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
    os.makedirs(img, exist_ok=True)
    res.plot_coefficients(regressors=["log_inc"]).savefig(
        os.path.join(img, "consumption_mpc.png"), bbox_inches="tight")
    res.plot_decomposition("log_inc").savefig(
        os.path.join(img, "consumption_decomp.png"), bbox_inches="tight")
    print("\nFigures written to docs/img/")
