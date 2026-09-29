"""Cell model: electrode capacities, OCP curves and PyBaMM parameter values."""
from __future__ import annotations

import numpy as np
import pybamm
from scipy.optimize import brentq

from .config import CellConfig

FARADAY = 96485.33212          # [C/mol]
GRAPHITE_OCP_OPTIONS = ("Chen2020", "Ecker2015")


# ---------------------------------------------------------------------------
# Parameter functions (closures, so each model build uses its own values)
# ---------------------------------------------------------------------------
def soft_wall_lfp_ocp(k_wall: float):
    """LFP OCP (Afshar2017 form) with an adjustable wall steepness near y -> 0."""
    def U_p(sto):
        return (3.4077 - 0.020269 * sto
                + 0.5 * np.exp(-k_wall * sto)
                - 0.9 * np.exp(-30 * (1 - sto)))
    return U_p


def lfp_diffusivity(D_p: float, beta: float, delta: float):
    """D_p(y) = D_p * [1 - (1 - beta) * exp(-y / delta)]: slower diffusion near full charge."""
    def D(sto, T):
        return D_p * (1 - (1 - beta) * np.exp(-sto / delta))
    return D


def scaled_exchange_current(j0, k: float):
    """Multiply a PyBaMM exchange-current density function by a constant k."""
    def j0_scaled(c_e, c_s_surf, c_s_max, T):
        return k * j0(c_e, c_s_surf, c_s_max, T)
    return j0_scaled


def shifted_ocp(U, dU: float):
    """OCP shifted by a constant dU (used for the charge branch)."""
    def U_shifted(sto):
        return U(sto) + dU
    return U_shifted


# ---------------------------------------------------------------------------
# Cell model
# ---------------------------------------------------------------------------
class CellModel:
    """Derived cell quantities plus a factory for PyBaMM parameter values and models."""

    def __init__(self, cfg: CellConfig):
        if cfg.graphite_ocp not in GRAPHITE_OCP_OPTIONS:
            raise ValueError(f"graphite_ocp must be one of {GRAPHITE_OCP_OPTIONS}, got '{cfg.graphite_ocp}'")
        self.cfg = cfg

        base = pybamm.ParameterValues(cfg.base_set)
        area = base["Electrode height [m]"] * base["Electrode width [m]"]
        self.cn_max = base["Maximum concentration in negative electrode [mol.m-3]"]
        self.cp_max = base["Maximum concentration in positive electrode [mol.m-3]"]

        # Electrode capacities Q = F * A * L * eps * c_max  [Ah]
        self.Q_n = FARADAY * area * base["Negative electrode thickness [m]"] * cfg.eps_n * self.cn_max / 3600
        self.Q_p = FARADAY * area * base["Positive electrode thickness [m]"] * cfg.eps_p * self.cp_max / 3600

        # OCP curves
        if cfg.graphite_ocp == "Ecker2015":
            self.U_n = pybamm.ParameterValues("Ecker2015")["Negative electrode OCP [V]"]
        else:
            self.U_n = base["Negative electrode OCP [V]"]

        if cfg.use_soft_wall:
            self.U_p_base = soft_wall_lfp_ocp(cfg.k_wall)
            # y100 chosen so the full-charge rest OCV equals the measured value
            self.y100 = float(brentq(
                lambda y: self.U_p_base(y) - self.U_n_value(cfg.x100) - cfg.v_full_rest, 1e-4, 0.3))
        else:
            self.U_p_base = base["Positive electrode OCP [V]"]
            self.y100 = cfg.y100

    # ---- helpers -----------------------------------------------------------
    def U_n_value(self, x: float) -> float:
        """Graphite OCP as a plain float (works for numpy- and PyBaMM-based functions)."""
        out = self.U_n(x)
        return float(out.evaluate()) if isinstance(out, pybamm.Symbol) else float(out)

    def state_full(self) -> tuple[float, float]:
        """Stoichiometries (x0, y0) of a fully charged cell."""
        return self.cfg.x100, self.y100

    def state_before_charge(self, Q_in_Ah: float) -> tuple[float, float]:
        """Start state of a test that ends full after charging Q_in_Ah."""
        return self.cfg.x100 - Q_in_Ah / self.Q_n, self.y100 + Q_in_Ah / self.Q_p

    # ---- PyBaMM objects ----------------------------------------------------
    def build_model(self) -> pybamm.BaseModel:
        """DFN model; contact resistance is switched on only if R_contact > 0."""
        options = {"contact resistance": "true"} if self.cfg.R_contact > 0 else {}
        return pybamm.lithium_ion.DFN(options=options)

    def parameter_values(self, x0: float, y0: float, T_C: float,
                         charge_branch: bool = False) -> pybamm.ParameterValues:
        """Full PyBaMM parameter set for one simulation."""
        cfg = self.cfg
        p = pybamm.ParameterValues(cfg.base_set)

        p["Negative electrode OCP [V]"] = self.U_n
        p["Positive electrode OCP [V]"] = shifted_ocp(self.U_p_base, cfg.du_charge if charge_branch else 0.0)

        p["Nominal cell capacity [A.h]"] = cfg.nominal_capacity_Ah
        p["Upper voltage cut-off [V]"] = cfg.v_max
        p["Lower voltage cut-off [V]"] = cfg.v_min
        p["Ambient temperature [K]"] = 273.15 + T_C
        p["Initial temperature [K]"] = 273.15 + T_C

        p["Initial concentration in negative electrode [mol.m-3]"] = x0 * self.cn_max
        p["Initial concentration in positive electrode [mol.m-3]"] = y0 * self.cp_max
        p["Negative electrode active material volume fraction"] = cfg.eps_n
        p["Positive electrode active material volume fraction"] = cfg.eps_p

        p["Negative particle diffusivity [m2.s-1]"] = cfg.D_n
        p["Positive particle diffusivity [m2.s-1]"] = lfp_diffusivity(cfg.D_p, cfg.beta_p, cfg.delta_p)

        p["Negative electrode exchange-current density [A.m-2]"] = scaled_exchange_current(
            p["Negative electrode exchange-current density [A.m-2]"], cfg.k_j0_n)
        p["Positive electrode exchange-current density [A.m-2]"] = scaled_exchange_current(
            p["Positive electrode exchange-current density [A.m-2]"], cfg.k_j0_p)

        p["Contact resistance [Ohm]"] = cfg.R_contact
        return p

    def summary(self) -> str:
        cfg = self.cfg
        return (f"Graphite OCP: {cfg.graphite_ocp} | soft wall: {cfg.use_soft_wall} (k={cfg.k_wall:g}) | "
                f"y100 = {self.y100:.4f} | dU_charge = {cfg.du_charge * 1000:.0f} mV | "
                f"Q_n = {self.Q_n:.3f} Ah, Q_p = {self.Q_p:.3f} Ah | "
                f"R_contact = {cfg.R_contact * 1000:.2f} mOhm")