"""
Color maps for ``tvccointreg``.

Reproduces MATLAB's R2014b *Parula* colormap (the package default) together with
``jet``, ``turbo``, ``bluered`` and a ``sinha`` diverging map.  All helpers return
either a list of hex strings (for matplotlib) or a list of ``(value, hex)`` stops
(for plotly-style colorscales), built with ``matplotlib``'s
``LinearSegmentedColormap`` for smooth interpolation between the anchor stops.

The Parula anchors are the canonical 64-stop RGB control points of the original
MATLAB Parula colormap (sub-sampled to the well-known 17 anchors that reproduce
it to the eye).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex

# ---------------------------------------------------------------------------
# Anchor RGB stops (0-1 scale)
# ---------------------------------------------------------------------------
_PARULA_ANCHORS = [
    (0.2081, 0.1663, 0.5292),
    (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2138, 0.6270),
    (0.2081, 0.2386, 0.6771),
    (0.1959, 0.2645, 0.7279),
    (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303),
    (0.0591, 0.3598, 0.8683),
    (0.0117, 0.3875, 0.8820),
    (0.0060, 0.4086, 0.8828),
    (0.0165, 0.4266, 0.8786),
    (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641),
    (0.0629, 0.4737, 0.8554),
    (0.0723, 0.4887, 0.8467),
    (0.0779, 0.5040, 0.8384),
    (0.0793, 0.5200, 0.8312),
    (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240),
    (0.0488, 0.5772, 0.8228),
    (0.0343, 0.5966, 0.8199),
    (0.0265, 0.6137, 0.8135),
    (0.0239, 0.6287, 0.8038),
    (0.0231, 0.6418, 0.7913),
    (0.0228, 0.6535, 0.7768),
    (0.0267, 0.6642, 0.7607),
    (0.0384, 0.6743, 0.7436),
    (0.0590, 0.6838, 0.7254),
    (0.0843, 0.6928, 0.7062),
    (0.1133, 0.7015, 0.6859),
    (0.1453, 0.7098, 0.6646),
    (0.1801, 0.7177, 0.6424),
    (0.2178, 0.7250, 0.6193),
    (0.2586, 0.7317, 0.5954),
    (0.3022, 0.7376, 0.5712),
    (0.3482, 0.7424, 0.5473),
    (0.3953, 0.7459, 0.5244),
    (0.4420, 0.7481, 0.5033),
    (0.4871, 0.7491, 0.4840),
    (0.5300, 0.7491, 0.4661),
    (0.5709, 0.7485, 0.4494),
    (0.6100, 0.7473, 0.4337),
    (0.6473, 0.7456, 0.4188),
    (0.6834, 0.7435, 0.4044),
    (0.7184, 0.7411, 0.3905),
    (0.7525, 0.7384, 0.3768),
    (0.7858, 0.7356, 0.3633),
    (0.8185, 0.7327, 0.3498),
    (0.8507, 0.7299, 0.3360),
    (0.8824, 0.7274, 0.3217),
    (0.9139, 0.7258, 0.3063),
    (0.9450, 0.7261, 0.2886),
    (0.9739, 0.7314, 0.2666),
    (0.9938, 0.7455, 0.2403),
    (0.9990, 0.7653, 0.2164),
    (0.9955, 0.7861, 0.1967),
    (0.9880, 0.8066, 0.1794),
    (0.9789, 0.8271, 0.1633),
    (0.9697, 0.8481, 0.1475),
    (0.9626, 0.8705, 0.1309),
    (0.9589, 0.8949, 0.1132),
    (0.9598, 0.9218, 0.0948),
    (0.9661, 0.9514, 0.0755),
    (0.9763, 0.9831, 0.0538),
]

_JET_ANCHORS = [
    (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.5, 1.0), (0.0, 1.0, 1.0),
    (0.5, 1.0, 0.5), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
]

_TURBO_ANCHORS = [
    (0.18995, 0.07176, 0.23217), (0.25107, 0.25237, 0.63374),
    (0.27628, 0.42118, 0.89123), (0.25862, 0.57958, 0.99876),
    (0.15844, 0.73551, 0.92305), (0.09267, 0.86554, 0.71774),
    (0.19659, 0.94901, 0.47860), (0.42778, 0.99419, 0.20760),
    (0.64362, 0.98999, 0.23356), (0.80473, 0.89180, 0.17813),
    (0.93301, 0.74785, 0.16275), (0.99314, 0.55582, 0.13455),
    (0.93590, 0.33310, 0.05878), (0.79747, 0.16523, 0.01756),
    (0.61088, 0.05475, 0.00787), (0.47960, 0.01583, 0.01055),
]

_BLUERED_ANCHORS = [
    (0.0196, 0.1882, 0.3804), (0.1294, 0.4000, 0.6745),
    (0.4039, 0.6627, 0.8118), (0.8431, 0.8980, 0.9412),
    (0.9686, 0.9686, 0.9686), (0.9922, 0.8588, 0.7804),
    (0.9569, 0.6471, 0.5098), (0.8392, 0.3765, 0.3020),
    (0.6980, 0.0941, 0.1686),
]

# A warm sequential map in the spirit of Sinha et al. (2023) quantile heatmaps.
_SINHA_ANCHORS = [
    (0.031, 0.188, 0.420), (0.129, 0.443, 0.710), (0.420, 0.682, 0.839),
    (0.780, 0.863, 0.937), (0.996, 0.878, 0.565), (0.992, 0.682, 0.380),
    (0.957, 0.427, 0.263), (0.835, 0.243, 0.310), (0.620, 0.004, 0.259),
]

_REGISTRY = {
    "parula": _PARULA_ANCHORS,
    "jet": _JET_ANCHORS,
    "turbo": _TURBO_ANCHORS,
    "bluered": _BLUERED_ANCHORS,
    "sinha": _SINHA_ANCHORS,
}


def _make_cmap(name: str) -> LinearSegmentedColormap:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown colormap '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return LinearSegmentedColormap.from_list(key, _REGISTRY[key], N=256)


def get_cmap(name: str = "parula") -> LinearSegmentedColormap:
    """Return a matplotlib ``LinearSegmentedColormap`` for ``name``."""
    return _make_cmap(name)


def _palette(anchors, n: int) -> List[str]:
    cmap = LinearSegmentedColormap.from_list("tmp", anchors, N=256)
    xs = np.linspace(0.0, 1.0, max(int(n), 1))
    return [to_hex(cmap(x)) for x in xs]


def parula_colors(n: int = 64) -> List[str]:
    """``n`` evenly spaced hex colors from the MATLAB Parula colormap."""
    return _palette(_PARULA_ANCHORS, n)


def matlab_jet_colors(n: int = 64) -> List[str]:
    """``n`` evenly spaced hex colors from the MATLAB jet colormap."""
    return _palette(_JET_ANCHORS, n)


def turbo_colors(n: int = 64) -> List[str]:
    """``n`` evenly spaced hex colors from Google's turbo colormap."""
    return _palette(_TURBO_ANCHORS, n)


def bluered_colors(n: int = 64) -> List[str]:
    """``n`` evenly spaced hex colors from a blue-white-red diverging map."""
    return _palette(_BLUERED_ANCHORS, n)


def sinha_colors(n: int = 64) -> List[str]:
    """``n`` evenly spaced hex colors from a Sinha-style diverging map."""
    return _palette(_SINHA_ANCHORS, n)


def resolve_colorscale(name: str = "Parula", n: int = 64) -> List[Tuple[float, str]]:
    """
    Return a plotly-style colorscale: a list of ``(value, hex)`` stops with
    ``value`` running over ``[0, 1]``.  Useful for 3D / heatmap / contour plots.
    """
    colors = _palette(_REGISTRY[name.lower()], n)
    values = np.linspace(0.0, 1.0, len(colors))
    return [(float(v), c) for v, c in zip(values, colors)]


__all__ = [
    "get_cmap",
    "parula_colors",
    "matlab_jet_colors",
    "turbo_colors",
    "bluered_colors",
    "sinha_colors",
    "resolve_colorscale",
]
