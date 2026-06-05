"""
tvccointreg
===========

Time-Varying-Coefficient regression and **Generalized Cointegration**, after

    Hall, S. G., Swamy, P. A. V. B., & Tavlas, G. S. (2015).
    *A Note on Generalizing the Concept of Cointegration.*

and the surrounding Swamy time-varying-coefficient literature.

The package estimates a model that is linear in variables but with time-varying
coefficients (the Swamy-Mehta / Granger representation of any nonlinear
relationship), decomposes each coefficient into a **bias-free** structural part
plus omitted-variable and measurement-error biases via *coefficient drivers*,
and tests for generalized cointegration with **standard** (chi-square / normal)
inference.

Quick start
-----------
>>> from tvccointreg import TVCModel, DriverSpec
>>> from tvccointreg.datasets import simulate_nonlinear_cointegration
>>> sim = simulate_nonlinear_cointegration(T=300, seed=1)
>>> spec = DriverSpec(names=list(sim.drivers.columns),
...                   bias_free=["x_lag"], omitted=["w"], measurement=[])
>>> model = TVCModel(sim.y, sim.X, sim.drivers, driver_spec=spec)
>>> res = model.fit()
>>> print(res.summary())
>>> res.coint_test()
"""
from .core import TVCModel, TVCResults
from .drivers import DriverSpec, default_spec
from .cointegration import CointResult
from .datasets import (
    simulate_nonlinear_cointegration,
    simulate_spurious,
    SimulatedData,
)
from . import colormaps
from .colormaps import (
    parula_colors,
    matlab_jet_colors,
    turbo_colors,
    bluered_colors,
    sinha_colors,
    resolve_colorscale,
    get_cmap,
)

__version__ = "0.1.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

__all__ = [
    "TVCModel",
    "TVCResults",
    "DriverSpec",
    "default_spec",
    "CointResult",
    "simulate_nonlinear_cointegration",
    "simulate_spurious",
    "SimulatedData",
    "colormaps",
    "parula_colors",
    "matlab_jet_colors",
    "turbo_colors",
    "bluered_colors",
    "sinha_colors",
    "resolve_colorscale",
    "get_cmap",
    "__version__",
]
