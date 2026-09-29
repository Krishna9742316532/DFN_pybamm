"""Internal model variables (anode potential vs Li/Li+, stoichiometries, overpotentials, electrolyte)."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

from .cell import CellModel
from .config import ProtocolConfig
from .data import TestData
from .simulate import extract_cccv_protocol

STATE_VARIABLES = {
    "V":         "Voltage [V]",
    "I":         "Current [A]",
    "phi_n_sep": "Negative electrode surface potential difference at separator interface [V]",
    "phi_n_avg": "X-averaged negative electrode surface potential difference [V]",
    "x_surf":    "X-averaged negative particle surface stoichiometry",
    "x_avg":     "Average negative particle stoichiometry",
    "y_surf":    "X-averaged positive particle surface stoichiometry",
    "y_avg":     "Average positive particle stoichiometry",
    "eta_n":     "X-averaged negative electrode reaction overpotential [V]",
    "eta_p":     "X-averaged positive electrode reaction overpotential [V]",
}

C_NEG, C_POS, C_DARK, C_PLATING = "#2a78d6", "#eb6834", "#222222", "#d03b3b"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TIME_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#104281"]


@dataclass
class InternalStates:
    name: str
    t: np.ndarray
    values: dict[str, np.ndarray]
    x_e: np.ndarray | None = None               # electrolyte grid [m]
    c_e: np.ndarray | None = None               # electrolyte concentration (n_x, n_t) [mol/m3]
    x_boundaries: tuple[float, float] | None = None   # (end of negative, end of separator) [m]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.values[key]


def _electrode_boundaries(cell: CellModel) -> tuple[float, float]:
    p = pybamm.ParameterValues(cell.cfg.base_set)
    L_n, L_s = p["Negative electrode thickness [m]"], p["Separator thickness [m]"]
    return L_n, L_n + L_s


def charge_internal_states(cell: CellModel, data: TestData, protocol: ProtocolConfig = ProtocolConfig(),
                           with_electrolyte: bool = True) -> InternalStates:
    """CC-CV charge like the cycler, keeping the internal variables."""
    proto = extract_cccv_protocol(data, protocol)
    x0, y0 = cell.state_before_charge(proto.Q_in_Ah)
    steps = ([f"Rest for {proto.t_rest:.0f} seconds"] if proto.t_rest > 1 else []) + [
        f"Charge at {proto.I_cc:.4f} A until {protocol.v_cv} V",
        f"Hold at {protocol.v_cv} V for {proto.t_end - proto.t_rest:.0f} seconds",
    ]
    sim = pybamm.Simulation(cell.build_model(),
                            parameter_values=cell.parameter_values(x0, y0, data.T_C, charge_branch=True),
                            experiment=pybamm.Experiment(steps, period=f"{protocol.period_s:g} seconds"))
    sol = sim.solve()
    ts, k = np.unique(sol["Time [s]"].entries, return_index=True)
    values = {key: np.asarray(sol[var].entries)[k] for key, var in STATE_VARIABLES.items()}

    x_e = c_e = None
    if with_electrolyte:
        c_e = np.asarray(sol["Electrolyte concentration [mol.m-3]"].entries)[:, k]
        try:
            x_e = np.asarray(sol["x [m]"].entries)[:, 0]
        except Exception:
            x_e = np.linspace(0, _electrode_boundaries(cell)[1] * 2, c_e.shape[0])   # fallback grid
    del sim, sol
    gc.collect()
    return InternalStates(data.name, ts, values, x_e, c_e, _electrode_boundaries(cell))


def current_internal_states(cell: CellModel, data: TestData) -> InternalStates:
    """Current-driven run (discharge, HPPC), starting full; only scalar variables to save memory."""
    x0, y0 = cell.state_full()
    p = cell.parameter_values(x0, y0, data.T_C, charge_branch=False)
    p["Current function [A]"] = pybamm.Interpolant(data.t, data.I, pybamm.t)
    solver = pybamm.IDAKLUSolver(output_variables=list(STATE_VARIABLES.values()))
    sim = pybamm.Simulation(cell.build_model(), parameter_values=p, solver=solver)
    sol = sim.solve(t_eval=data.t)
    values = {key: np.asarray(sol[var].entries) for key, var in STATE_VARIABLES.items()}
    ts = np.asarray(sol.t)
    del sim, sol, solver
    gc.collect()
    return InternalStates(data.name, ts, values)


def plating_summary(states: dict[str, InternalStates]) -> pd.DataFrame:
    """Key numbers per run: plating margin, diffusion limits, kinetic losses."""
    rows = {}
    for label, st in states.items():
        phi = st["phi_n_sep"]
        below = phi[:-1] < 0
        rows[label] = {
            "Min anode potential (separator side) [mV]": float(phi.min() * 1000),
            "Time of minimum [min]":                     float(st.t[np.argmin(phi)] / 60),
            "Time below 0 V [s]":                        float(np.sum(np.diff(st.t)[below])),
            "Min anode potential (average) [mV]":        float(st["phi_n_avg"].min() * 1000),
            "Min graphite surface stoichiometry":        float(st["x_surf"].min()),
            "Max graphite surface stoichiometry":        float(st["x_surf"].max()),
            "Min LFP surface stoichiometry":             float(st["y_surf"].min()),
            "Max |eta_n| [mV]":                          float(np.abs(st["eta_n"]).max() * 1000),
            "Max |eta_p| [mV]":                          float(np.abs(st["eta_p"]).max() * 1000),
        }
    return pd.DataFrame(rows).T


def _style(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_anode_potential(states: dict[str, InternalStates], save_path: str | Path | None = None,
                         show: bool = True, title: str = "Anode potential vs Li/Li+ (separator side)") -> None:
    """Separator-side anode potential of several runs on one axis, with the 0 V plating limit."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, st), c in zip(states.items(), CATEGORICAL):
        ax.plot(st.t / 60, st["phi_n_sep"] * 1000, color=c, lw=1.5, label=label)
    ax.axhline(0, color=C_PLATING, lw=1.0, ls="--", label="0 V: plating possible below")
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Anode potential vs Li/Li+ [mV]")
    ax.set_title(title); ax.legend(fontsize=8, frameon=False)
    _style(ax)
    _finish(fig, save_path, show)


def plot_internal_states(st: InternalStates, title: str | None = None, save_path: str | Path | None = None,
                         show: bool = True, time_unit: str = "min") -> None:
    """Current, voltage, anode potential, stoichiometries and overpotentials of one run."""
    scale = 60 if time_unit == "min" else 3600
    tt = st.t / scale
    fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    ax[0].plot(tt, st["I"], color=C_DARK, lw=1.0)
    ax[0].set_ylabel("Current [A]\n(+ = discharge)")

    ax[1].plot(tt, st["V"], color=C_DARK, lw=1.2)
    ax[1].set_ylabel("Voltage [V]")

    phi = st["phi_n_sep"] * 1000
    ax[2].plot(tt, phi, color=C_NEG, lw=1.3, label="Separator side")
    ax[2].plot(tt, st["phi_n_avg"] * 1000, color=C_NEG, lw=1.0, ls="--", label="Electrode average")
    ax[2].axhline(0, color=C_PLATING, lw=1.0, ls="--", label="0 V (Li/Li+)")
    ax[2].fill_between(tt, phi, 0, where=phi < 0, color=C_PLATING, alpha=0.2, label="Plating possible")
    ax[2].set_ylabel("Anode potential\nvs Li/Li+ [mV]"); ax[2].legend(fontsize=8, frameon=False, ncol=2)

    ax[3].plot(tt, st["x_surf"], color=C_NEG, lw=1.3, label="Graphite surface")
    ax[3].plot(tt, st["x_avg"], color=C_NEG, lw=1.0, ls="--", label="Graphite average")
    ax[3].plot(tt, st["y_surf"], color=C_POS, lw=1.3, label="LFP surface")
    ax[3].plot(tt, st["y_avg"], color=C_POS, lw=1.0, ls="--", label="LFP average")
    ax[3].set_ylabel("Stoichiometry [-]"); ax[3].set_ylim(0, 1); ax[3].legend(fontsize=8, frameon=False, ncol=2)

    ax[4].plot(tt, st["eta_n"] * 1000, color=C_NEG, lw=1.2, label="Graphite")
    ax[4].plot(tt, st["eta_p"] * 1000, color=C_POS, lw=1.2, label="LFP")
    ax[4].axhline(0, color=C_DARK, lw=0.8)
    ax[4].set_ylabel("Reaction\noverpotential [mV]"); ax[4].legend(fontsize=8, frameon=False)
    ax[4].set_xlabel(f"Time [{time_unit}]")

    ax[0].set_title(title or f"{st.name}: internal variables")
    for a in ax:
        _style(a)
    _finish(fig, save_path, show)


def plot_electrolyte_profiles(st: InternalStates, n_snapshots: int = 5, save_path: str | Path | None = None,
                              show: bool = True, i_threshold: float = 0.05) -> None:
    """Electrolyte concentration across the cell at several times during charging."""
    if st.c_e is None:
        raise ValueError("No electrolyte data: use charge_internal_states(..., with_electrolyte=True)")
    charging = np.where(st["I"] < -i_threshold)[0]
    idx = np.linspace(charging[0], charging[-1], n_snapshots).astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    ramp = TIME_RAMP if n_snapshots <= len(TIME_RAMP) else plt.cm.Blues(np.linspace(0.3, 0.95, n_snapshots))
    for i, c in zip(idx, ramp):
        ax.plot(st.x_e * 1e6, st.c_e[:, i], color=c, lw=1.6, label=f"t = {st.t[i] / 60:.0f} min")
    if st.x_boundaries is not None:
        for xb in st.x_boundaries:
            ax.axvline(xb * 1e6, color="#999999", lw=0.8, ls=":")
        x_end = st.x_e[-1] * 1e6
        for x_mid, name in [(st.x_boundaries[0] / 2, "graphite"),
                            ((st.x_boundaries[0] + st.x_boundaries[1]) / 2, "sep."),
                            ((st.x_boundaries[1] * 1e6 + x_end) / 2 / 1e6, "LFP")]:
            ax.text(x_mid * 1e6, 0.97, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.set_xlabel("Position across the cell [µm]"); ax.set_ylabel("Electrolyte concentration [mol/m³]")
    ax.set_title(f"{st.name}: electrolyte concentration during charging")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    _finish(fig, save_path, show)