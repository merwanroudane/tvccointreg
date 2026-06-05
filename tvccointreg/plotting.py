"""
Publication-quality visualizations for ``tvccointreg``.

All plots default to the MATLAB *Parula* colormap.  Each function returns the
matplotlib ``Figure`` so callers can further tweak or save it.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .colormaps import get_cmap, parula_colors

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
}


def _style():
    import matplotlib.pyplot as plt
    return plt.rc_context(_RC)


def _line_colors(n: int):
    # sample away from the extreme light yellow end for line legibility
    cols = parula_colors(max(n + 1, 2))
    return cols[:n]


def _positions(idx):
    """Return integer x positions and (is_numeric, idx) for tick labelling."""
    arr = np.asarray(idx)
    is_numeric = np.issubdtype(arr.dtype, np.number)
    return np.arange(len(arr)), is_numeric, arr


def _set_time_axis(ax, idx, n_ticks: int = 8):
    """
    Put a clean, sparse time axis on ``ax``.

    Works whether the model index is numeric or string/date labels (e.g.
    '1959Q2'); avoids cramming hundreds of tick labels onto the axis.
    """
    pos, is_numeric, arr = _positions(idx)
    T = len(arr)
    if is_numeric:
        return  # numeric x already plotted against real values
    locs = np.linspace(0, T - 1, min(n_ticks, T)).astype(int)
    ax.set_xticks(locs)
    ax.set_xticklabels([str(arr[i]) for i in locs], rotation=0)


def plot_coefficients(res, regressors: Optional[Sequence] = None,
                      ci: float = 0.95, include_random: bool = False,
                      ncols: int = 2, figsize=None, cmap: str = "parula"):
    """
    Small-multiples plot of each time-varying coefficient with a confidence band.

    Parameters
    ----------
    regressors : list, optional
        Subset of regressor names to plot (default: all except the constant if
        there are several).
    ci : float
        Confidence level for the band around the driver-explained coefficient.
    include_random : bool
        Overlay the coefficient including the random component.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    names = list(res.model.coef_names)
    if regressors is None:
        regressors = [n for n in names if n != "const"] or names
    gamma = res.tv_coefficients(include_random=False)
    gamma_full = res.tv_coefficients(include_random=True)
    se = res.coefficient_se()
    bf = res.bias_free_coefficients()
    idx = res.model.index
    pos, is_numeric, _ = _positions(idx)
    x = idx if is_numeric else pos
    z = stats.norm.ppf(0.5 + ci / 2)

    n = len(regressors)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (6.2 * ncols, 3.4 * nrows)
    colors = _line_colors(max(n, 3))

    with _style():
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for k, name in enumerate(regressors):
            ax = axes[k // ncols][k % ncols]
            c = colors[k % len(colors)]
            lo = gamma[name] - z * se[name]
            hi = gamma[name] + z * se[name]
            ax.fill_between(x, lo, hi, color=c, alpha=0.18,
                            label=f"{int(ci*100)}% CI")
            ax.plot(x, gamma[name], color=c, lw=1.8, label=r"$\gamma_{jt}$")
            ax.plot(x, bf[name], color="#222222", lw=1.1, ls="--",
                    label="bias-free")
            if include_random:
                ax.plot(x, gamma_full[name], color=c, lw=0.8, alpha=0.5,
                        label=r"$\gamma_{jt}+\hat\varepsilon$")
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"Coefficient on {name}")
            ax.set_xlabel("t")
            _set_time_axis(ax, idx)
            ax.legend(fontsize=8, framealpha=0.9, loc="best")
        # hide any empty panels
        for k in range(n, nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")
        fig.suptitle("Time-varying coefficients", fontsize=13, y=1.0)
        fig.tight_layout()
    return fig


def plot_decomposition(res, regressor: Union[str, int], figsize=(8, 4.6),
                       cmap: str = "parula"):
    """Plot the bias-free / omitted / measurement / random decomposition."""
    import matplotlib.pyplot as plt

    comp = res.components(regressor)
    name = regressor if isinstance(regressor, str) else res.model.coef_names[regressor]
    idx = res.model.index
    pos, is_numeric, _ = _positions(idx)
    x = idx if is_numeric else pos
    cols = parula_colors(6)

    with _style():
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, comp["total"], color="#111111", lw=2.0, label="total")
        ax.plot(x, comp["bias_free"], color=cols[0], lw=1.6,
                label="bias-free (structural)")
        if np.any(comp["omitted_bias"] != 0):
            ax.plot(x, comp["omitted_bias"], color=cols[2], lw=1.3, ls="--",
                    label="omitted-variable bias")
        if np.any(comp["measurement_bias"] != 0):
            ax.plot(x, comp["measurement_bias"], color=cols[4], lw=1.3,
                    ls=":", label="measurement-error bias")
        ax.plot(x, comp["random"], color="grey", lw=0.8, alpha=0.6,
                label="random part")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"Coefficient decomposition: {name}")
        ax.set_xlabel("t")
        ax.set_ylabel("coefficient")
        _set_time_axis(ax, idx)
        ax.legend(fontsize=8, ncol=2, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_fit(res, figsize=(9, 5.2)):
    """Actual vs fitted (top) and residuals (bottom)."""
    import matplotlib.pyplot as plt

    idx = res.model.index
    pos, is_numeric, _ = _positions(idx)
    x = idx if is_numeric else pos
    y = res.model.y
    cols = parula_colors(4)
    with _style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(x, y, color="#111111", lw=1.4, label="actual")
        ax1.plot(x, res.fitted, color=cols[1], lw=1.6, ls="--",
                 label="fitted")
        ax1.set_title("Model fit")
        ax1.set_ylabel(res.model.y_name)
        ax1.legend(fontsize=9)
        ax2.plot(x, res.resid, color=cols[2], lw=1.0)
        ax2.axhline(0, color="grey", lw=0.6)
        ax2.set_ylabel("residual")
        ax2.set_xlabel("t")
        _set_time_axis(ax2, idx)
        fig.tight_layout()
    return fig


def plot_coint_heatmap(res, figsize=None, cmap: str = "parula"):
    """
    Heatmap of the bias-free coefficient paths across regressors and time.

    Rows are regressors, columns are time; color encodes the magnitude of the
    bias-free (structural) coefficient with the Parula colormap.
    """
    import matplotlib.pyplot as plt

    bf = res.bias_free_coefficients()
    names = [n for n in bf.columns if n != "const"] or list(bf.columns)
    M = bf[names].to_numpy().T
    idx = res.model.index
    figsize = figsize or (9, 0.7 * len(names) + 2)
    vmax = np.nanmax(np.abs(M)) or 1.0

    with _style():
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M, aspect="auto", cmap=get_cmap(cmap),
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        # sparse time ticks (handles numeric or string/date indices)
        arr = np.asarray(idx)
        locs = np.linspace(0, len(arr) - 1, min(8, len(arr))).astype(int)
        ax.set_xticks(locs)
        ax.set_xticklabels([str(arr[i]) for i in locs])
        ax.set_xlabel("t")
        ax.set_title("Bias-free (structural) coefficient over time")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$\gamma_{jt}^{\,BF}$")
        fig.tight_layout()
    return fig
