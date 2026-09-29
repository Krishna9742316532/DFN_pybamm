"""Figures for model vs measurement."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .data import TestData
from .metrics import ChargeComparison, CurrentComparison
from .simulate import SimResult


def _finish(fig, save_path: str | Path | None, show: bool) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_charge(name: str, cmp: ChargeComparison, metrics: dict,
                save_path: str | Path | None = None, show: bool = True) -> None:
    """Voltage, CC voltage error and charge current of a CC-CV test."""
    tm = cmp.t / 60
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax[0].plot(tm, cmp.V_meas, label="Measured")
    ax[0].plot(tm, cmp.V_model, "--", label="Model")
    ax[0].set_ylabel("Voltage [V]"); ax[0].legend()
    ax[0].set_title(f"{name} | CC RMSE {metrics['RMSE V CC [mV]']:.1f} mV | "
                    f"CV current RMSE {metrics['RMSE I CV [A]']:.3f} A")

    ax[1].plot(tm[cmp.cc], (cmp.V_model[cmp.cc] - cmp.V_meas[cmp.cc]) * 1000, color="red")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("CC error [mV]")

    ax[2].plot(tm, -cmp.I_meas, label="Measured")
    ax[2].plot(tm, -cmp.I_model, "--", label="Model")
    ax[2].set_ylabel("Charge current [A]"); ax[2].set_xlabel("Time [min]"); ax[2].legend()

    for a in ax:
        a.grid(True)
    _finish(fig, save_path, show)


def plot_current(name: str, data: TestData, sim: SimResult, cmp: CurrentComparison, metrics: dict,
                 save_path: str | Path | None = None, show: bool = True) -> None:
    """Voltage and voltage error of a current-driven test."""
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax[0].plot(data.t / 3600, data.V, label="Measured")
    ax[0].plot(sim.t / 3600, sim.V, "--", label="Model")
    ax[0].set_ylabel("Voltage [V]"); ax[0].legend()
    ax[0].set_title(f"{name} | RMSE {metrics['RMSE V [mV]']:.1f} mV")

    ax[1].plot(cmp.t / 3600, cmp.error * 1000, color="red")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("Error [mV]"); ax[1].set_xlabel("Time [h]")

    for a in ax:
        a.grid(True)
    _finish(fig, save_path, show)