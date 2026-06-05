"""
Journal-quality result tables (plain text, LaTeX booktabs, and HTML).

The layout follows the conventions of top econometrics journals: a header block
with the estimator and sample, a coefficient block reporting the average
time-varying coefficient, the *bias-free* component and its standard error,
t-statistic, p-value and significance stars, the generalized-cointegration
verdict, and a footnote block with fit diagnostics and the significance legend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def coefficient_frame(res) -> pd.DataFrame:
    """
    One row per regressor (excluding the constant) with the average total and
    bias-free coefficients, delta-method SE, t-stat, p-value and stars.
    """
    coint = res.coint_test(skip_const=True)
    gamma = res.tv_coefficients()
    rows = []
    for name in coint.index:
        r = coint.loc[name]
        rows.append({
            "Variable": name,
            "Coef (mean)": gamma[name].mean(),
            "Bias-free": r["avg_bias_free"],
            "Std.Err": r["std_err"],
            "t": r["t_stat"],
            "p-value": r["p_value"],
            "": _stars(r["p_value"]),
            "G-Coint": "Yes" if r["cointegrated"] else "No",
        })
    return pd.DataFrame(rows).set_index("Variable")


def _fmt(x, nd=4):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "."
    return f"{x:.{nd}f}"


def summary_table(res, fmt: str = "text") -> str:
    df = coefficient_frame(res)
    diag = res.diagnostics()
    model = res.model

    title = "Time-Varying-Coefficient Regression  /  Generalized Cointegration"
    info = [
        ("Dep. variable", model.y_name),
        ("No. observations", diag["n_obs"]),
        ("No. coefficients", diag["n_coefficients"]),
        ("No. drivers", diag["n_drivers"]),
        ("Estimator", "Iteratively rescaled GLS"),
        ("Covariance", res.cov_type.upper()),
        ("R-squared", _fmt(diag["r_squared"])),
        ("Log-likelihood", _fmt(diag["loglik"], 2)),
        ("Converged", diag["converged"]),
    ]
    if "resid_adf_pvalue" in diag:
        info.append(("Resid ADF p-value", _fmt(diag["resid_adf_pvalue"])))

    if fmt == "text":
        return _text(title, info, df)
    if fmt == "latex":
        return _latex(title, info, df, diag)
    if fmt == "html":
        return _html(title, info, df, diag)
    raise ValueError("fmt must be 'text', 'latex' or 'html'")


# --------------------------------------------------------------------- text
def _text(title, info, df) -> str:
    width = 78
    line = "=" * width
    thin = "-" * width
    out = [line, title.center(width), line]

    # two-column info block
    items = [f"{k}: {v}" for k, v in info]
    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i + 1] if i + 1 < len(items) else ""
        out.append(f"{left:<39}{right:<39}")
    out.append(line)

    cols = ["Coef (mean)", "Bias-free", "Std.Err", "t", "p-value", "", "G-Coint"]
    header = f"{'Variable':<12}" + "".join(
        f"{c:>11}" for c in cols)
    out.append(header)
    out.append(thin)
    for name, row in df.iterrows():
        cells = (f"{_fmt(row['Coef (mean)'])  :>11}"
                 f"{_fmt(row['Bias-free'])   :>11}"
                 f"{_fmt(row['Std.Err'])     :>11}"
                 f"{_fmt(row['t'])           :>11}"
                 f"{_fmt(row['p-value'])     :>11}"
                 f"{row['']                  :>11}"
                 f"{row['G-Coint']           :>11}")
        out.append(f"{str(name):<12}{cells}")
    out.append(line)
    out.append("Signif.: *** p<0.01  ** p<0.05  * p<0.10".center(width))
    out.append("Bias-free = structural derivative; G-Coint via Wald test on it."
               .center(width))
    out.append(line)
    return "\n".join(out)


# -------------------------------------------------------------------- latex
def _latex(title, info, df, diag) -> str:
    note = ("Bias-free is the structural partial derivative; the "
            "generalized-cointegration verdict comes from a standard "
            "$\\chi^2$ Wald test on the bias-free block. "
            "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.")
    lines = [
        "\\begin{table}[!htbp]\\centering",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variable & Coef (mean) & Bias-free & Std.\\ Err. & $t$ & $p$ \\\\",
        "\\midrule",
    ]
    for name, row in df.iterrows():
        lines.append(
            f"{name} & {_fmt(row['Coef (mean)'])} & "
            f"{_fmt(row['Bias-free'])}{_stars_tex(row[''])} & "
            f"{_fmt(row['Std.Err'])} & {_fmt(row['t'])} & "
            f"{_fmt(row['p-value'])} \\\\")
    lines += [
        "\\midrule",
        f"Observations & \\multicolumn{{5}}{{r}}{{{diag['n_obs']}}} \\\\",
        f"$R^2$ & \\multicolumn{{5}}{{r}}{{{_fmt(diag['r_squared'])}}} \\\\",
        f"Log-likelihood & \\multicolumn{{5}}{{r}}{{{_fmt(diag['loglik'], 2)}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        f"\\begin{{tablenotes}}\\footnotesize\\item {note}\\end{{tablenotes}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _stars_tex(stars: str) -> str:
    return ("$^{" + stars + "}$") if stars else ""


# --------------------------------------------------------------------- html
def _html(title, info, df, diag) -> str:
    info_html = "".join(
        f"<tr><th style='text-align:left'>{k}</th><td>{v}</td></tr>"
        for k, v in info)
    body = []
    for name, row in df.iterrows():
        body.append(
            "<tr>"
            f"<td style='text-align:left'>{name}</td>"
            f"<td>{_fmt(row['Coef (mean)'])}</td>"
            f"<td>{_fmt(row['Bias-free'])}{row['']}</td>"
            f"<td>{_fmt(row['Std.Err'])}</td>"
            f"<td>{_fmt(row['t'])}</td>"
            f"<td>{_fmt(row['p-value'])}</td>"
            f"<td>{row['G-Coint']}</td>"
            "</tr>")
    style = (
        "<style>"
        ".tvc{border-collapse:collapse;font-family:Georgia,serif;font-size:13px;}"
        ".tvc th,.tvc td{padding:4px 10px;text-align:right;}"
        ".tvc thead tr{border-top:2px solid #222;border-bottom:1px solid #222;}"
        ".tvc tbody tr:last-child{border-bottom:2px solid #222;}"
        "</style>")
    return (
        style +
        f"<h3 style='font-family:Georgia,serif'>{title}</h3>"
        f"<table class='tvc'><tbody>{info_html}</tbody></table><br>"
        "<table class='tvc'><thead><tr>"
        "<th style='text-align:left'>Variable</th><th>Coef (mean)</th>"
        "<th>Bias-free</th><th>Std.Err</th><th>t</th><th>p-value</th>"
        "<th>G-Coint</th></tr></thead><tbody>"
        + "".join(body) +
        "</tbody></table>"
        "<p style='font-family:Georgia,serif;font-size:11px'>"
        "*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.10. Bias-free = structural "
        "derivative; generalized cointegration via a standard &chi;&sup2; "
        "Wald test on the bias-free block.</p>")
