# tvccointreg

[![PyPI version](https://img.shields.io/pypi/v/tvccointreg.svg)](https://pypi.org/project/tvccointreg/)
[![Python versions](https://img.shields.io/pypi/pyversions/tvccointreg.svg)](https://pypi.org/project/tvccointreg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/merwanroudane/tvccointreg/blob/main/LICENSE)
[![CI](https://github.com/merwanroudane/tvccointreg/actions/workflows/ci.yml/badge.svg)](https://github.com/merwanroudane/tvccointreg/actions/workflows/ci.yml)

**Time-Varying-Coefficient regression and Generalized Cointegration in Python.**

📦 **PyPI:** https://pypi.org/project/tvccointreg/ — install with `pip install tvccointreg`
🌐 **Documentation site:** https://merwanroudane.github.io/tvccointreg/

`tvccointreg` implements the *generalized cointegration* framework of

> **Hall, S. G., Swamy, P. A. V. B., & Tavlas, G. S. (2015).**
> *A Note on Generalizing the Concept of Cointegration.*

together with the surrounding Swamy time-varying-coefficient (TVC) literature
(Swamy & Mehta 1975; Swamy, Tavlas, Hall & co-authors 2003–2014; Granger 2008).

Conventional cointegration (Engle–Granger 1987) is an *inherently linear*
concept: it can only recover a structural relationship if that relationship
happens to be linear in unit-root variables. Most economic theory, however,
implies **nonlinear** relationships among variables that are **nonstationary but
not necessarily unit-root**. `tvccointreg` lets you:

- estimate a model that is **linear in variables but with time-varying
  coefficients** — the Swamy–Mehta / Granger representation of *any* nonlinear
  relationship (eq. 7);
- **decompose** each coefficient into a **bias-free structural component** plus
  **omitted-variable bias** and **measurement-error bias** using *coefficient
  drivers* (eq. 8);
- **test for generalized cointegration** — i.e. whether the bias-free structural
  partial derivative is nonzero — with **standard** χ²/normal inference (no
  Dickey–Fuller tables);
- produce **journal-quality tables** (text / LaTeX booktabs / HTML) and
  **publication-quality plots** with the MATLAB **Parula** colormap by default.

---

## Table of contents

- [Installation](#installation)
- [The 60-second tour](#the-60-second-tour)
- [Concepts: how the paper maps to the code](#concepts-how-the-paper-maps-to-the-code)
- [The estimator](#the-estimator)
- [API reference](#api-reference)
- [Detailed syntax](#detailed-syntax)
- [Visualizations](#visualizations)
- [Worked example: spurious vs. real](#worked-example-spurious-vs-real)
- [Testing](#testing)
- [Citing](#citing)
- [References](#references)

---

## Installation

From [PyPI](https://pypi.org/project/tvccointreg/):

```bash
pip install tvccointreg

# with the optional ADF stationarity diagnostic
pip install "tvccointreg[adf]"
```

Or from source:

```bash
git clone https://github.com/merwanroudane/tvccointreg.git
cd tvccointreg
pip install -e ".[dev]"      # editable install with test extras
```

Requirements: `numpy`, `scipy`, `pandas`, `matplotlib`. `statsmodels` is optional
(used only for the ADF stationarity diagnostic and some examples).

---

## The 60-second tour

```python
from tvccointreg import TVCModel, DriverSpec
from tvccointreg.datasets import simulate_nonlinear_cointegration

# 1) A nonlinear relationship between nonstationary variables.
sim = simulate_nonlinear_cointegration(T=300, seed=1, nonlinearity=0.4)

# 2) Partition the drivers into the three sets (Assumption 1 / eq. 8).
spec = DriverSpec(
    names=list(sim.drivers.columns),
    bias_free=["x_lag", "y_lag"],   # true coefficient variation
    omitted=["w"],                  # omitted-variable bias
    measurement=[],                 # measurement-error bias
)

# 3) Fit by iteratively rescaled GLS.
res = TVCModel(sim.y, sim.X, sim.drivers, driver_spec=spec).fit()

# 4) Journal table + generalized cointegration test.
print(res.summary())
res.coint_test()
```

```
==============================================================================
      Time-Varying-Coefficient Regression  /  Generalized Cointegration
==============================================================================
Dep. variable: y                       No. observations: 300
No. coefficients: 2                    No. drivers: 3
Estimator: Iteratively rescaled GLS    Covariance: GLS
R-squared: 0.8311                      Log-likelihood: -78.07
Converged: True                        Resid ADF p-value: 0.0000
==============================================================================
Variable    Coef (mean)  Bias-free    Std.Err          t    p-value    G-Coint
------------------------------------------------------------------------------
x                0.3283     0.3392     0.0698     4.8614     0.0000 ***     Yes
==============================================================================
                   Signif.: *** p<0.01  ** p<0.05  * p<0.10
       Bias-free = structural derivative; G-Coint via Wald test on it.
==============================================================================
```

On this DGP the recovered bias-free coefficient correlates **0.99** with the
true time-varying derivative, and the residuals are stationary (ADF p ≈ 0) —
exactly the standard-inference result the paper proves.

---

## Concepts: how the paper maps to the code

| Paper (Hall–Swamy–Tavlas 2015) | Symbol | In `tvccointreg` |
|---|---|---|
| TVC model, eq. (7) | `y_t = γ_0t + γ_1t x_1t + … + γ_{K-1,t} x_{K-1,t}` | `TVCModel(y, X, drivers)` |
| Coefficient-driver eq., eq. (8) | `γ_jt = π_j0 + Σ_d π_jd z_dt + ε_jt` | `DriverSpec(...)` + the drivers matrix |
| Three components of a coefficient | bias-free / omitted-variable / measurement-error | `DriverSpec(bias_free=, omitted=, measurement=)` |
| Concentrated model | `y_t = w_t'π + u_t`, `u_t = Σ_j x_jt ε_jt` | `estimation.build_design`, `estimation.irgls` |
| Variance structure | `Var(u_t) = Σ_j σ_j² x_jt²` | `estimation.variance_components` (Hildreth–Houck–Swamy) |
| Consistent estimator (§3.3) | iteratively rescaled GLS | `TVCModel.fit(method="irgls")` |
| Bias-free component | `γ_jt^BF = π_j0 + Σ_{d∈BF} π_jd z_dt` | `res.bias_free_coefficients()` |
| Generalized cointegration, eqs. (5)–(6) | `∂y/∂x ≠ 0` ⇔ bias-free component ≠ 0 | `res.coint_test()` |
| Standard inference (§3.3) | χ² Wald / normal, **not** Dickey–Fuller | Wald + delta-method t in `cointegration.py` |
| Drivers vs. instruments (Table 1) | drivers *should* be correlated with the misspecification | the `omitted` / `measurement` sets |

### The definition, in one line

> *y* and *x* are **generalized-cointegrated** if, holding all other relevant
> preexisting conditions *w* constant, the **bias-free** part of `∂y/∂x` is
> nonzero.

Cointegration is thus a property of the **real-world structural relationship**,
not of any particular statistical model — and it survives nonlinearity *and*
omitted regressors.

---

## The estimator

Substituting the driver equation (8) into the TVC model (7) gives a model that
is linear in the stacked parameter vector `π`:

```
y_t = w_t' π + u_t ,   w_t = ( x_0t·z_t , x_1t·z_t , … ) ,   u_t = Σ_j x_jt ε_jt
Var(u_t | x_t) = Σ_j σ_j² x_jt²          (Hildreth–Houck–Swamy structure)
```

`tvccointreg` then:

1. **OLS** start on the concentrated design `W`.
2. **Variance components** `σ_j²` from a non-negative regression of squared
   residuals on squared regressors (`scipy.optimize.nnls`).
3. **GLS** with weights `1/h_t`, `h_t = Σ_j σ_j² x_jt²`; iterate to convergence
   (*iteratively rescaled GLS*).
4. **Recover** the time-varying coefficients
   `γ_jt = z_t'π_j + ε̂_jt`, where the random part is the best linear predictor
   `ε̂_jt = σ_j² x_jt u_t / h_t`.
5. **Decompose** `γ_jt` into bias-free / omitted / measurement parts using the
   driver-set masks, and **test** the bias-free block.

Inference uses the GLS covariance `(W'Ω⁻¹W)⁻¹` (the paper's standard-inference
result) or a heteroskedasticity-robust sandwich (`cov_type="robust"`).

> **Note / honest caveat.** As in the paper, validity hinges on *Assumption 1* —
> that the chosen drivers genuinely span the bias components. This is an
> identifying assumption, not something the data can confirm; choose drivers that
> are plausibly correlated with the suspected misspecification (Table 1).

---

## API reference

### `TVCModel(y, X, drivers, driver_spec=None, driver_sets=None, add_const=True, index=None)`
Build a model. `y` (T,), `X` (T, K−1), `drivers` (T, m). A constant regressor and
a constant driver are added automatically. Accepts NumPy arrays or pandas
objects (column names are picked up automatically).

- **`.fit(method="irgls", max_iter=100, tol=1e-8, cov_type="gls", verbose=False)`**
  → `TVCResults`.

### `DriverSpec(names, bias_free=[], omitted=[], measurement=[])`
Three-set partition of the drivers. Any driver not listed defaults to
`bias_free`. `.describe()` prints the partition.

### `TVCResults`
| Method | Returns |
|---|---|
| `.summary(fmt="text"|"latex"|"html")` | journal-style table |
| `.coef_table()` | per-regressor average coefficient frame |
| `.coint_test(alpha=0.05, skip_const=True)` | tidy DataFrame of test results (and `.coint_results_`) |
| `.tv_coefficients(include_random=True)` | T×K time-varying coefficients |
| `.bias_free_coefficients()` | T×K bias-free (structural) coefficients |
| `.components(regressor)` | bias-free / omitted / measurement / random / total |
| `.coefficient_se()` | pointwise standard errors |
| `.diagnostics()` | R², log-lik, convergence, residual ADF |
| `.plot_coefficients(...)` | TVC paths with CI bands |
| `.plot_decomposition(regressor)` | the three-component decomposition |
| `.plot_fit()` | actual vs fitted + residuals |
| `.plot_coint_heatmap()` | Parula heatmap of bias-free paths |

### Datasets
- `simulate_nonlinear_cointegration(T, seed, nonlinearity, omitted, measurement_error)`
- `simulate_spurious(T, seed)`

### Colormaps
`parula_colors(n)`, `matlab_jet_colors(n)`, `turbo_colors(n)`, `bluered_colors(n)`,
`sinha_colors(n)`, `resolve_colorscale("Parula", n)`, `get_cmap("parula")`.

---

## Detailed syntax

**Using raw NumPy arrays and reading off a specific test:**

```python
import numpy as np
from tvccointreg import TVCModel, DriverSpec

res = TVCModel(y, X, Z,                       # y:(T,), X:(T,K-1), Z:(T,m)
               driver_sets={"bias_free": ["z1"],
                            "omitted":   ["z2"],
                            "measurement": ["z3"]}).fit()

ct = res.coint_test()
print(ct.loc["x1", "cointegrated"], ct.loc["x1", "p_value"])
```

**Robust covariance and OLS (homoskedastic) comparison:**

```python
res_robust = TVCModel(y, X, Z, driver_spec=spec).fit(cov_type="robust")
res_ols    = TVCModel(y, X, Z, driver_spec=spec).fit(method="ols")
```

**Exporting a LaTeX table for a paper:**

```python
with open("table1.tex", "w") as f:
    f.write(res.summary(fmt="latex"))
```

**Pulling the time-varying coefficient path of one regressor:**

```python
gamma_x = res.tv_coefficients()["x"]          # pandas Series indexed by t
bf_x    = res.bias_free_coefficients()["x"]
```

---

## Visualizations

All plots default to the MATLAB **Parula** colormap and return a matplotlib
`Figure`.

| | |
|---|---|
| `plot_coefficients()` | `plot_decomposition("x")` |
| ![coefficients](https://raw.githubusercontent.com/merwanroudane/tvccointreg/main/docs/img/coefficients.png) | ![decomposition](https://raw.githubusercontent.com/merwanroudane/tvccointreg/main/docs/img/decomposition.png) |
| `plot_fit()` | `plot_coint_heatmap()` |
| ![fit](https://raw.githubusercontent.com/merwanroudane/tvccointreg/main/docs/img/fit.png) | ![heatmap](https://raw.githubusercontent.com/merwanroudane/tvccointreg/main/docs/img/heatmap.png) |

---

## Worked example: spurious vs. real

`examples/spurious_vs_real.py` reproduces the central point of the paper:

```
================ SPURIOUS (independent random walks) ================
Naive OLS slope = 1.125  (p = 5.97e-46)   <- spuriously 'significant'
Generalized cointegration: NO (p = 0.708)

================ GENUINE generalized cointegration ================
Generalized cointegration: YES (p = 1.34e-35)
```

A naive OLS regression of one random walk on another is wildly "significant",
yet the generalized-cointegration test on the bias-free component correctly
finds **no** structural relationship.

---

## Empirical application: the US consumption function (real data)

`examples/real_data_consumption.py` applies the method to **real US quarterly
macro data** (`statsmodels` `macrodata`, 1959Q1–2009Q3, 202 obs after lagging).
We model the long-run relationship between **real personal consumption** and
**real disposable income** (both in logs, both strongly trending / I(1)):

```python
import numpy as np, pandas as pd
from statsmodels.datasets import macrodata
from tvccointreg import TVCModel, DriverSpec

d = macrodata.load_pandas().data
d.index = pd.period_range("1959Q1", periods=len(d), freq="Q")
zscore = lambda s: (s - s.mean()) / s.std(ddof=0)

y = np.log(d["realcons"]).rename("log_cons")
X = np.log(d[["realdpi"]]).rename(columns={"realdpi": "log_inc"})
drivers = pd.DataFrame({
    "inc_lag":  zscore(np.log(d["realdpi"]).shift(1)),
    "trend":    zscore(pd.Series(np.arange(len(d)), index=d.index)),
    "realint":  zscore(d["realint"]),
    "unemp":    zscore(d["unemp"]),
    "cons_lag": zscore(np.log(d["realcons"]).shift(1)),
})
df = pd.concat([y, X, drivers], axis=1).dropna()

spec = DriverSpec(
    names=["inc_lag", "trend", "realint", "unemp", "cons_lag"],
    bias_free=["inc_lag", "trend"],   # true elasticity variation
    omitted=["realint", "unemp"],     # interest-rate / labour-market channels
    measurement=["cons_lag"],         # dynamics / measurement
)
res = TVCModel(df["log_cons"], df[["log_inc"]], df[spec.names],
               driver_spec=spec, index=df.index.astype(str)).fit()
print(res.summary())
print(res.coint_test())
```

**Result (verbatim output):**

```
==============================================================================
      Time-Varying-Coefficient Regression  /  Generalized Cointegration
==============================================================================
Dep. variable: log_cons                No. observations: 202
No. coefficients: 2                    No. drivers: 5
Estimator: Iteratively rescaled GLS    Covariance: GLS
R-squared: 0.9999                      Log-likelihood: 752.32
Converged: True                        Resid ADF p-value: 0.0000
==============================================================================
Variable    Coef (mean)  Bias-free    Std.Err          t    p-value    G-Coint
------------------------------------------------------------------------------
log_inc          0.3886     0.3886     0.0528     7.3626     0.0000 ***     Yes
==============================================================================
```

| regressor | avg_bias_free | std_err | t_stat | wald | df | p_value | cointegrated |
|---|---|---|---|---|---|---|---|
| log_inc | 0.3886 | 0.0528 | 7.3626 | 56.9504 | 3 | 0.0000 | **True** |

**Reading the result.**

- **Generalized cointegration is confirmed** between consumption and income
  (Wald = 56.95, 3 df, *p* ≈ 0): the bias-free structural elasticity is firmly
  nonzero, so the long-run relationship is genuine, not spurious.
- **Standard inference is valid here**: the composite residuals are stationary
  (ADF *p* = 1.9 × 10⁻⁷), exactly the condition under which Hall–Swamy–Tavlas
  show the TVC test uses ordinary χ²/normal critical values rather than
  Dickey–Fuller ones.
- The **bias-free income elasticity drifts down over the sample**, from **0.43
  (1959Q2)** to **0.34 (2009Q3)** — the *direct* income channel after the
  dynamics (`cons_lag`) and omitted business-cycle conditions (`realint`,
  `unemp`) are stripped out into the bias terms. A declining direct sensitivity
  of consumption to current income over five decades is consistent with greater
  consumption smoothing / financial deepening.

| Bias-free income elasticity over time | Coefficient decomposition |
|---|---|
| ![mpc](https://raw.githubusercontent.com/merwanroudane/tvccointreg/main/docs/img/consumption_mpc.png) | ![decomp](https://raw.githubusercontent.com/merwanroudane/tvccointreg/main/docs/img/consumption_decomp.png) |

Run it yourself:

```bash
python examples/real_data_consumption.py
```

### Multiple regressors

The framework handles **several regressors at once** — each gets its own
time-varying coefficient and its own cointegration verdict.
`examples/consumption_multivariate.py` estimates a two-regressor consumption
function (disposable income **and** the real interest rate) on the same data:

| Regressor | Bias-free coef | t | Wald | p-value | Cointegrated |
|---|---|---|---|---|---|
| log_realDPI (income) | 0.6666 *** | 10.13 | 111.40 | <0.0001 | **Yes** |
| realint (real rate) | −0.0007 ** | −2.26 | 8.37 | 0.039 | **Yes** |

Positive income elasticity (~0.67) and a negative real-rate effect
(intertemporal substitution); R² = 0.9998, residuals stationary (ADF
p = 3.5×10⁻⁹). See the rendered results on the
[documentation site](https://merwanroudane.github.io/tvccointreg/#example).

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

The suite checks coefficient recovery, that the three components sum exactly to
the total coefficient, detection of genuine generalized cointegration, rejection
of spurious relationships, and the Parula palette/colorscale helpers.

---

## Publishing (maintainer notes)

The package is PyPI-ready (`python -m build` + `twine check` both pass).

**First release — manual upload with an API token**

1. Create accounts on [TestPyPI](https://test.pypi.org/) and
   [PyPI](https://pypi.org/), and generate an API token for each
   (Account settings → API tokens).
2. Build and check:
   ```bash
   python -m build
   python -m twine check dist/*
   ```
3. Dry-run on TestPyPI first, then install from there to confirm:
   ```bash
   python -m twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ tvccointreg
   ```
4. Upload to the real PyPI:
   ```bash
   python -m twine upload dist/*
   ```
   When prompted, use `__token__` as the username and the API token (starting
   with `pypi-`) as the password.

**Subsequent releases — automated via Trusted Publishing (recommended)**

`.github/workflows/publish.yml` publishes automatically when you publish a
GitHub Release. One-time setup: on PyPI add a *Trusted Publisher*
(Project → Publishing) for repository `merwanroudane/tvccointreg`, workflow
`publish.yml`, environment `pypi`. After that, no tokens or secrets are needed —
just bump `version` in `pyproject.toml` and `__version__`, tag, and publish a
Release.

> **Remember:** a version number can only be uploaded to PyPI **once**. Bump the
> version for every release.

---

## Citing

If you use this package, please cite both the package and the underlying paper.

```bibtex
@software{roudane_tvccointreg,
  author  = {Merwan Roudane},
  title   = {tvccointreg: Time-Varying-Coefficient Regression and
             Generalized Cointegration in Python},
  year    = {2026},
  url     = {https://github.com/merwanroudane/tvccointreg}
}

@incollection{hall_swamy_tavlas_2015,
  author  = {Hall, Stephen G. and Swamy, P. A. V. B. and Tavlas, George S.},
  title   = {A Note on Generalizing the Concept of Cointegration},
  year    = {2015}
}
```

---

## References

- Hall, S. G., Swamy, P. A. V. B., & Tavlas, G. S. (2015). *A Note on
  Generalizing the Concept of Cointegration.*
- Hall, S. G., Swamy, P. A. V. B., & Tavlas, G. S. (2014). *Time Varying
  Coefficient Models; A Proposal for Selecting the Coefficient Driver Sets.*
  University of Leicester WP 14/18.
- Swamy, P. A. V. B., & Mehta, J. S. (1975). *Bayesian and non-Bayesian Analysis
  of Switching Regressions and of Random Coefficient Regression Models.* JASA.
- Swamy, P. A. V. B., Tavlas, G. S., Hall, S. G., et al. (2010). *Nonparametric
  Nonstationary Regression.*
- Granger, C. W. J. (2008). *Non-linear Models: Where Do We Go Next —
  Time-Varying Parameter Models?* Studies in Nonlinear Dynamics & Econometrics.
- Hildreth, C., & Houck, J. P. (1968). *Some Estimators for a Linear Model with
  Random Coefficients.* JASA.
- Engle, R. F., & Granger, C. W. J. (1987). *Co-integration and Error
  Correction.* Econometrica.
- Cramér, H. (1946). *Mathematical Methods of Statistics.*

---

## Author

**Dr Merwan Roudane** · merwanroudane920@gmail.com ·
[github.com/merwanroudane](https://github.com/merwanroudane)

Released under the MIT License.
