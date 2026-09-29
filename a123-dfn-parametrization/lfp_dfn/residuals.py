"""Residual analysis on HPPC: where and why the model deviates from the measurement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cell import CellModel
from .config import ProtocolConfig
from .data import TestData
from .metrics import rmse
from .simulate import simulate_current_driven

SOC_RANGES = {"high SOC 100-70 %": (0.7, 1.01),
              "mid SOC 70-30 %":   (0.3, 0.7),
              "low SOC 30-0 %":    (-0.01, 0.3)}
COLORS = ["#eb6834", "#2a78d6", "#1baf7a"]


@dataclass
class Residuals:
    t: np.ndarray
    I: np.ndarray
    V_model: np.ndarray
    error_mV: np.ndarray       # model - measured
    soc: np.ndarray            # from measured charge throughput
    coverage: float            # % of the profile the model simulated
    termination: str


def measured_soc(data: TestData) -> np.ndarray:
    """SOC from the measured current, assuming the test starts full."""
    Ah = np.concatenate([[0], np.cumsum(0.5 * (data.I[1:] + data.I[:-1]) * np.diff(data.t))]) / 3600
    return 1 - Ah / Ah.max()


def hppc_residuals(cell: CellModel, data: TestData) -> Residuals:
    sim = simulate_current_driven(cell, data)
    ok = data.t <= sim.t[-1]
    V_model = np.interp(data.t[ok], sim.t, sim.V)
    return Residuals(t=data.t[ok], I=data.I[ok], V_model=V_model,
                     error_mV=(V_model - data.V[ok]) * 1000, soc=measured_soc(data)[ok],
                     coverage=100 * sim.t[-1] / data.t[-1], termination=sim.termination)


def residual_table(results: dict[str, Residuals], protocol: ProtocolConfig = ProtocolConfig()) -> pd.DataFrame:
    """RMSE overall, per operating state (rest / pulses) and per SOC range."""
    thr = protocol.i_threshold
    rows = {}
    for label, r in results.items():
        e = r.error_mV
        row = {"Coverage [%]": r.coverage,
               "RMSE all [mV]": rmse(e),
               "Mean error [mV]": float(np.mean(e)),
               "RMSE rest [mV]": rmse(e[np.abs(r.I) < thr]),
               "RMSE discharge pulses [mV]": rmse(e[r.I > thr]),
               "RMSE charge pulses [mV]": rmse(e[r.I < -thr])}
        for name, (lo, hi) in SOC_RANGES.items():
            row[f"RMSE {name} [mV]"] = rmse(e[(r.soc >= lo) & (r.soc < hi)])
        rows[label] = row
    return pd.DataFrame(rows).T


def plot_residuals(data: TestData, results: dict[str, Residuals], save_path: str | Path | None = None,
                   show: bool = True, low_soc: float = 0.2, high_soc: float = 0.9, start_s: float = 60) -> None:
    """Voltage, residual over time (with the problem regions marked) and residual vs SOC."""
    soc = measured_soc(data)
    th = data.t / 3600
    t_high_end = th[np.argmax(soc < high_soc)]
    t_low_start = th[np.argmax(soc < low_soc)]

    fig, ax = plt.subplots(3, 1, figsize=(12, 11))
    ax[1].sharex(ax[0])

    ax[0].plot(th, data.V, color="#222222", lw=1.0, label="Measured")
    for (label, r), c in zip(results.items(), COLORS):
        ax[0].plot(r.t / 3600, r.V_model, color=c, lw=0.9, label=label)
        ax[1].plot(r.t / 3600, r.error_mV, color=c, lw=0.7, label=label)
        ax[2].plot(r.soc * 100, r.error_mV, ".", ms=1.5, color=c, label=label)

    for a in ax[:2]:
        a.axvspan(0, start_s / 3600, color="#999999", alpha=0.25)
        a.axvspan(0, t_high_end, color="#999999", alpha=0.10)
        a.axvspan(t_low_start, th[-1], color="#999999", alpha=0.15)
    ax[1].text(t_high_end / 2, 0.95, "near full", transform=ax[1].get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax[1].text((t_low_start + th[-1]) / 2, 0.95, f"SOC < {low_soc:.0%}", transform=ax[1].get_xaxis_transform(),
               ha="center", va="top", fontsize=8)

    ax[0].set_ylabel("Voltage [V]"); ax[0].set_title("HPPC: measured vs model"); ax[0].legend(fontsize=8, frameon=False)
    ax[1].axhline(0, color="#222222", lw=0.8)
    ax[1].set_ylabel("Residual [mV]"); ax[1].set_xlabel("Time [h]")
    ax[1].set_title("Residual over time (shaded: first minute, near full, low SOC)")
    ax[2].axhline(0, color="#222222", lw=0.8)
    ax[2].set_xlim(100, 0); ax[2].set_xlabel("SOC [%]"); ax[2].set_ylabel("Residual [mV]")
    ax[2].set_title("Residual vs SOC"); ax[2].legend(fontsize=8, frameon=False, markerscale=6)
    for a in ax:
        a.grid(True, color="#dddddd", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)