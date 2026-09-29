"""Search space: bounds, log scaling and the mapping u in [0,1]^n <-> physical parameters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str            # must match a CellConfig field, e.g. "D_n"
    lower: float
    upper: float
    log: bool = True     # search on log10 scale


class ParameterSpace:
    """Maps normalized vectors u (each entry 0..1) to parameter dicts and back."""

    def __init__(self, specs: list[ParameterSpec]):
        self.specs = list(specs)
        self.names = [s.name for s in self.specs]
        for s in self.specs:
            if s.upper <= s.lower:
                raise ValueError(f"{s.name}: upper bound must be > lower bound")
            if s.log and s.lower <= 0:
                raise ValueError(f"{s.name}: log scale needs a lower bound > 0")
        self._lo = np.array([np.log10(s.lower) if s.log else s.lower for s in self.specs])
        self._hi = np.array([np.log10(s.upper) if s.log else s.upper for s in self.specs])

    @property
    def n(self) -> int:
        return len(self.specs)

    def to_params(self, u) -> dict:
        z = self._lo + np.clip(np.asarray(u, float), 0, 1) * (self._hi - self._lo)
        return {s.name: float(10 ** z[i]) if s.log else float(z[i]) for i, s in enumerate(self.specs)}

    def to_unit(self, params: dict) -> np.ndarray:
        z = np.array([np.log10(params[s.name]) if s.log else params[s.name] for s in self.specs])
        return (z - self._lo) / (self._hi - self._lo)

    def without(self, *names: str) -> "ParameterSpace":
        return ParameterSpace([s for s in self.specs if s.name not in names])

    def bounds_dict(self) -> dict:
        return {s.name: [s.lower, s.upper] for s in self.specs}