"""
Full multiple-regressor application: a US consumption function.

Data: ``statsmodels`` ``macrodata`` (real US quarterly series, 1959Q1-2009Q3).

We estimate a behavioural consumption function with **two** structural
regressors -- income (a scale variable) and the real interest rate (the
intertemporal price of consumption):

    log(realCons)_t = gamma_0t + gamma_1t * log(realDPI)_t
                                + gamma_2t * realint_t + u_t .

All series are nonstationary; the income elasticity is widely believed to drift
across decades (consumption smoothing, financial deepening). This is a
*generalized cointegration* question with several regressors at once.

Coefficient drivers (Assumption 1, eq. 8), standardised for conditioning:
  * bias-free   : lagged income, a time trend       -> true coefficient drift
  * omitted     : unemployment                       -> omitted-variable bias
  * measurement : lagged real interest rate          -> measurement / dynamics

Outputs: prints the journal table and the per-regressor cointegration test,
writes HTML table fragments and all figures (light theme) into docs/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.datasets import macrodata

from tvccointreg import TVCModel, DriverSpec

# Force a clean, light theme everywhere (white backgrounds, dark text).
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")
IMG = os.path.join(DOCS, "img")
os.makedirs(IMG, exist_ok=True)


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def build():
    d = macrodata.load_pandas().data.copy()
    d.index = pd.period_range("1959Q1", periods=len(d), freq="Q")

    y = np.log(d["realcons"]).rename("log_realCons")
    X = pd.concat([
        np.log(d["realdpi"]).rename("log_realDPI"),
        d["realint"].rename("realint"),
    ], axis=1)
    drivers = pd.DataFrame({
        "inc_lag":  zscore(np.log(d["realdpi"]).shift(1)),
        "trend":    zscore(pd.Series(np.arange(len(d)), index=d.index)),
        "unemp":    zscore(d["unemp"]),
        "rate_lag": zscore(d["realint"].shift(1)),
    })
    df = pd.concat([y, X, drivers], axis=1).dropna()
    return df


def main():
    df = build()
    y = df["log_realCons"]
    X = df[["log_realDPI", "realint"]]
    drivers = df[["inc_lag", "trend", "unemp", "rate_lag"]]

    spec = DriverSpec(
        names=list(drivers.columns),
        bias_free=["inc_lag", "trend"],
        omitted=["unemp"],
        measurement=["rate_lag"],
    )

    res = TVCModel(y, X, drivers, driver_spec=spec,
                   index=df.index.astype(str)).fit()

    print(res.summary())
    print("\nGeneralized cointegration test (one row per regressor):")
    print(res.coint_test().round(4).to_string())
    print("\nDiagnostics:")
    for k, v in res.diagnostics().items():
        print(f"  {k}: {v}")
    for name in ["log_realDPI", "realint"]:
        bf = res.bias_free_coefficients()[name]
        print(f"\n{name}: bias-free coefficient  mean={bf.mean():.4f}  "
              f"range=[{bf.min():.4f}, {bf.max():.4f}]")

    # ---- HTML fragments for the GitHub Pages site ----
    with open(os.path.join(DOCS, "_summary_table.html"), "w",
              encoding="utf-8") as f:
        f.write(res.summary(fmt="html"))
    ct = res.coint_test().round(4)
    with open(os.path.join(DOCS, "_coint_table.html"), "w",
              encoding="utf-8") as f:
        f.write(ct.to_html(classes="restab", border=0))

    # ---- figures (light theme) ----
    res.plot_coefficients(regressors=["log_realDPI", "realint"]).savefig(
        os.path.join(IMG, "cm_coefficients.png"), bbox_inches="tight")
    res.plot_decomposition("log_realDPI").savefig(
        os.path.join(IMG, "cm_decomp_income.png"), bbox_inches="tight")
    res.plot_decomposition("realint").savefig(
        os.path.join(IMG, "cm_decomp_rate.png"), bbox_inches="tight")
    res.plot_coint_heatmap().savefig(
        os.path.join(IMG, "cm_heatmap.png"), bbox_inches="tight")
    res.plot_fit().savefig(
        os.path.join(IMG, "cm_fit.png"), bbox_inches="tight")
    print("\nTables and figures written to docs/")
    return res


if __name__ == "__main__":
    main()
