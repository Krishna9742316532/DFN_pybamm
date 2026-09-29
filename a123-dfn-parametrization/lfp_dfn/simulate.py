"""PyBaMM simulations for the two test types (CC-CV charge, current-driven)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybamm

from .cell import CellModel
from .config import ProtocolConfig
from .data import TestData


@dataclass
class SimResult:
    """Simulated time series (current: + = discharge)."""

    t: np.ndarray
    V: np.ndarray
    I: np.ndarray
    termination: str = ""


@dataclass(frozen=True)
class CCCVProtocol:
    """CC-CV settings extracted from a measured charge test."""

    t_rest: float        # rest before charging [s]
    I_cc: float          # constant charge current [A] (positive number)
    t_end: float         # test length [s]
    Q_in_Ah: float       # charge put in during the test [Ah]


def extract_cccv_protocol(data: TestData, protocol: ProtocolConfig) -> CCCVProtocol:
    """Read rest time, CC current and charge throughput from the measured test."""
    charging = data.I < -protocol.i_threshold
    if not charging.any():
        raise ValueError(f"{data.name}: no charging current found")
    cc_points = charging & (data.V < protocol.v_cv - protocol.v_tol)
    return CCCVProtocol(
        t_rest=float(data.t[np.argmax(charging)]),
        I_cc=float(np.median(-data.I[cc_points])),
        t_end=float(data.t[-1]),
        Q_in_Ah=float(np.trapezoid(-data.I, data.t) / 3600),
    )


def simulate_cccv(cell: CellModel, data: TestData, protocol: ProtocolConfig,
                  hold_s: float | None = None) -> tuple[SimResult, CCCVProtocol]:
    """Charge test like the cycler: rest -> CC until V_cv -> CV hold (voltage-controlled).

    hold_s: CV hold duration [s]. Default: the full test length.
    """
    proto = extract_cccv_protocol(data, protocol)
    x0, y0 = cell.state_before_charge(proto.Q_in_Ah)          # test ends full
    hold = proto.t_end if hold_s is None else hold_s

    steps = ([f"Rest for {proto.t_rest:.0f} seconds"] if proto.t_rest > 1 else []) + [
        f"Charge at {proto.I_cc:.4f} A until {protocol.v_cv} V",
        f"Hold at {protocol.v_cv} V for {hold:.0f} seconds",
    ]
    sim = pybamm.Simulation(
        cell.build_model(),
        parameter_values=cell.parameter_values(x0, y0, data.T_C, charge_branch=True),
        experiment=pybamm.Experiment(steps, period=f"{protocol.period_s:g} seconds"),
    )
    sol = sim.solve()
    ts, k = np.unique(sol["Time [s]"].entries, return_index=True)
    result = SimResult(t=ts,
                       V=sol["Voltage [V]"].entries[k],
                       I=sol["Current [A]"].entries[k],
                       termination=str(getattr(sol, "termination", "")))
    del sim, sol
    return result, proto


def simulate_current_driven(cell: CellModel, data: TestData,
                            x0: float | None = None, y0: float | None = None,
                            charge_branch: bool = False) -> SimResult:
    """Follow the measured current exactly (discharge tests, HPPC). Starts full by default."""
    if x0 is None or y0 is None:
        x0, y0 = cell.state_full()
    p = cell.parameter_values(x0, y0, data.T_C, charge_branch=charge_branch)
    p["Current function [A]"] = pybamm.Interpolant(data.t, data.I, pybamm.t)

    solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]"])      # low memory
    sim = pybamm.Simulation(cell.build_model(), parameter_values=p, solver=solver)
    sol = sim.solve(t_eval=data.t)
    result = SimResult(t=sol.t,
                       V=sol["Voltage [V]"].entries,
                       I=np.interp(sol.t, data.t, data.I),               # input current
                       termination=str(sol.termination))
    del sim, sol, solver
    return result