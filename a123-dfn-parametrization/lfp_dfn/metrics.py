"""Comparison of simulated and measured data, and error metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProtocolConfig
from .data import TestData
from .simulate import SimResult


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def mean_or_nan(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


# ---------------------------------------------------------------------------
# Charge tests (CC-CV)
# ---------------------------------------------------------------------------
@dataclass
class ChargeComparison:
    """Measured and simulated signals on the measured time grid."""

    t: np.ndarray
    V_meas: np.ndarray
    V_model: np.ndarray
    I_meas: np.ndarray
    I_model: np.ndarray
    cc: np.ndarray       # mask: CC phase (voltage compared)
    cv: np.ndarray       # mask: CV phase (current compared)


def compare_cccv(data: TestData, sim: SimResult, protocol: ProtocolConfig) -> ChargeComparison:
    tc = data.t[data.t <= min(data.t[-1], sim.t[-1])]
    V_meas, I_meas = np.interp(tc, data.t, data.V), np.interp(tc, data.t, data.I)
    cv = V_meas >= protocol.v_cv - protocol.v_tol
    cc = (~cv) & (I_meas < -protocol.i_threshold)
    return ChargeComparison(t=tc, V_meas=V_meas, V_model=np.interp(tc, sim.t, sim.V),
                            I_meas=I_meas, I_model=np.interp(tc, sim.t, sim.I), cc=cc, cv=cv)


def cccv_metrics(name: str, cmp: ChargeComparison, protocol: ProtocolConfig) -> dict:
    dV_cc = cmp.V_model[cmp.cc] - cmp.V_meas[cmp.cc]
    dI_cv = cmp.I_model[cmp.cv] - cmp.I_meas[cmp.cv]
    reached = cmp.V_model >= protocol.v_cv - protocol.v_tol
    return {
        "Test": name,
        "RMSE V CC [mV]":       rmse(dV_cc) * 1000,
        "Mean offset CC [mV]":  mean_or_nan(dV_cc) * 1000,
        "RMSE I CV [A]":        rmse(dI_cv),
        "CV start model [min]": cmp.t[np.argmax(reached)] / 60 if reached.any() else float("nan"),
        "CV start meas [min]":  cmp.t[np.argmax(cmp.cv)] / 60 if cmp.cv.any() else float("nan"),
        "Charge model [Ah]":    float(np.trapezoid(-cmp.I_model, cmp.t) / 3600),
        "Charge meas [Ah]":     float(np.trapezoid(-cmp.I_meas, cmp.t) / 3600),
    }


# ---------------------------------------------------------------------------
# Current-driven tests (discharge, HPPC)
# ---------------------------------------------------------------------------
@dataclass
class CurrentComparison:
    """Voltage error on the part of the measured time grid the model covered."""

    t: np.ndarray
    V_meas: np.ndarray
    V_model: np.ndarray
    error: np.ndarray    # model - measured [V]
    rest: np.ndarray     # mask: |I| below threshold


def compare_current(data: TestData, sim: SimResult, protocol: ProtocolConfig) -> CurrentComparison:
    mask = data.t <= sim.t[-1]
    t = data.t[mask]
    V_model = np.interp(t, sim.t, sim.V)
    return CurrentComparison(t=t, V_meas=data.V[mask], V_model=V_model,
                             error=V_model - data.V[mask],
                             rest=np.abs(data.I[mask]) < protocol.i_threshold)


def current_metrics(name: str, data: TestData, sim: SimResult, cmp: CurrentComparison) -> dict:
    e = cmp.error
    return {
        "Test": name,
        "Simulated [h]":        sim.t[-1] / 3600,
        "Test length [h]":      data.t[-1] / 3600,
        "Stop reason":          sim.termination,
        "RMSE V [mV]":          rmse(e) * 1000,
        "MAE V [mV]":           float(np.mean(np.abs(e))) * 1000,
        "Max |error| [mV]":     float(np.max(np.abs(e))) * 1000,
        "RMSE rest [mV]":       rmse(e[cmp.rest]) * 1000,
        "RMSE current on [mV]": rmse(e[~cmp.rest]) * 1000,
    }