"""
Coefficient-driver specification (Assumption 1 and the three-set split).

In Hall, Swamy & Tavlas (2015) each time-varying coefficient is written as a
linear function of *coefficient drivers* plus a random error (eq. 8).  The
drivers are partitioned into **three sets** so that each coefficient can be
decomposed into three economically meaningful components:

* ``bias_free``        -- drivers associated with the time variation in the
                          *true* coefficient (the nonlinear functional form).
                          The constant is always part of this set: it carries
                          the "true bias-free" level of the partial derivative.
* ``omitted``          -- drivers correlated with the omitted-variable bias.
* ``measurement``      -- drivers correlated with the measurement-error bias.

The bias-free component is the object of interest: generalized cointegration
between ``y`` and a regressor holds iff that regressor's bias-free component is
nonzero (eqs. 5-6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class DriverSpec:
    """
    Container describing how driver columns map onto the three component sets.

    Parameters
    ----------
    names : list of str
        Names of the driver columns (excluding the always-present constant).
    bias_free : list of str
        Driver names whose movement reflects the *true* coefficient variation.
    omitted : list of str
        Driver names correlated with omitted-variable bias.
    measurement : list of str
        Driver names correlated with measurement-error bias.
    """

    names: List[str]
    bias_free: List[str] = field(default_factory=list)
    omitted: List[str] = field(default_factory=list)
    measurement: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        assigned = self.bias_free + self.omitted + self.measurement
        unknown = [d for d in assigned if d not in self.names]
        if unknown:
            raise ValueError(f"Unknown driver(s) in sets: {unknown}")
        dupes = [d for d in assigned if assigned.count(d) > 1]
        if dupes:
            raise ValueError(f"Driver(s) assigned to multiple sets: {set(dupes)}")
        # Any driver not explicitly assigned defaults to the bias-free set.
        unassigned = [d for d in self.names if d not in assigned]
        self.bias_free = list(self.bias_free) + unassigned

    # -- layout helpers -----------------------------------------------------
    @property
    def design_names(self) -> List[str]:
        """Driver design names including the leading constant."""
        return ["const"] + list(self.names)

    @property
    def q(self) -> int:
        """Number of driver design columns per coefficient (incl. constant)."""
        return len(self.names) + 1

    def _idx(self, names: Sequence[str]) -> List[int]:
        design = self.design_names
        return [design.index(n) for n in names]

    def bias_free_mask(self) -> np.ndarray:
        """Boolean mask (length q) selecting constant + bias-free drivers."""
        mask = np.zeros(self.q, dtype=bool)
        mask[self._idx(["const"] + self.bias_free)] = True
        return mask

    def omitted_mask(self) -> np.ndarray:
        mask = np.zeros(self.q, dtype=bool)
        if self.omitted:
            mask[self._idx(self.omitted)] = True
        return mask

    def measurement_mask(self) -> np.ndarray:
        mask = np.zeros(self.q, dtype=bool)
        if self.measurement:
            mask[self._idx(self.measurement)] = True
        return mask

    def describe(self) -> str:
        lines = ["Coefficient-driver specification (Assumption 1)"]
        lines.append(f"  bias-free   : const + {self.bias_free or '[]'}")
        lines.append(f"  omitted     : {self.omitted or '[]'}")
        lines.append(f"  measurement : {self.measurement or '[]'}")
        return "\n".join(lines)


def default_spec(driver_names: Sequence[str]) -> DriverSpec:
    """All drivers treated as bias-free (no explicit bias modelling)."""
    return DriverSpec(names=list(driver_names))
