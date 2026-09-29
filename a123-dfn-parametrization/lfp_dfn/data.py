"""Loading of the measured test data (.mat files)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


@dataclass
class TestData:
    """One measured test. Current uses the PyBaMM sign convention (+ = discharge)."""

    name: str
    t: np.ndarray        # time from test start [s]
    I: np.ndarray        # current [A], + = discharge
    V: np.ndarray        # terminal voltage [V]
    T_C: float           # mean cell temperature [°C]

    @property
    def duration_h(self) -> float:
        return float(self.t[-1] / 3600)


def load_test(folder: str | Path, name: str, T_default_C: float = 23.0) -> TestData:
    """Load `<folder>/<name>.mat`.

    Works for files that store a struct (e.g. `meas`) as well as files that store
    the signals as separate variables. Field names are matched case-insensitively.
    """
    path = Path(folder) / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    first = mat[keys[0]]
    if hasattr(first, "_fieldnames"):
        lookup = {f.lower(): getattr(first, f) for f in first._fieldnames}
    else:
        lookup = {k.lower(): mat[k] for k in keys}

    for field in ("time", "current", "voltage"):
        if field not in lookup:
            raise KeyError(f"{path.name}: field '{field}' not found (available: {sorted(lookup)})")

    t_raw = np.asarray(lookup["time"], float).ravel()
    t, idx = np.unique(t_raw - t_raw[0], return_index=True)       # sorted, no duplicate times
    I = -np.asarray(lookup["current"], float).ravel()[idx]        # cycler sign -> PyBaMM sign
    V = np.asarray(lookup["voltage"], float).ravel()[idx]
    T_C = (float(np.mean(lookup["battery_temp_degc"]))
           if "battery_temp_degc" in lookup else float(T_default_C))
    return TestData(name=name, t=t, I=I, V=V, T_C=T_C)