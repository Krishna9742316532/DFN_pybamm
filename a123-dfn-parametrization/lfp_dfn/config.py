"""Configuration objects for the A123 ANR26650M1B (LFP/graphite) DFN model.

All model settings live here, so a notebook only has to create a config,
optionally change a few fields, and pass it on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class CellConfig:
    """Physical and tuning parameters of the cell model (SI units unless noted)."""

    # Base parameter set and solver safety limits
    base_set: str = "Prada2013"
    nominal_capacity_Ah: float = 2.5
    v_max: float = 4.2                 # upper voltage cut-off [V] (solver safety only)
    v_min: float = 1.0                 # lower voltage cut-off [V] (solver safety only)

    # Stoichiometry window at 100 % SOC
    x100: float = 0.9082               # graphite stoichiometry at full charge
    y100: float = 0.01                 # LFP stoichiometry at full charge (only used without soft wall)

    # Open-circuit potentials
    graphite_ocp: str = "Chen2020"     # "Chen2020" (Prada2013 default) or "Ecker2015"
    use_soft_wall: bool = True         # softer LFP OCP wall near full charge
    k_wall: float = 70.0               # wall steepness (Afshar original = 150)
    v_full_rest: float = 3.447         # measured HPPC rest voltage at full charge [V]
    du_charge: float = 0.0             # LFP charge-branch offset [V] (charge tests only)

    # Solid-phase transport
    D_n: float = 1.5e-14               # graphite diffusivity [m2/s]
    D_p: float = 5.8e-18               # LFP diffusivity [m2/s]
    beta_p: float = 0.2                # LFP diffusivity ratio near full charge (1 = off)
    delta_p: float = 0.15              # width of the slow-diffusion region in stoichiometry

    # Active material volume fractions
    eps_n: float = 0.58
    eps_p: float = 0.374

    # Kinetics and resistance
    k_j0_n: float = 1.0                # multiplier on graphite exchange-current density
    k_j0_p: float = 1.0                # multiplier on LFP exchange-current density
    R_contact: float = 0.0             # contact resistance [Ohm]; > 0 enables it in the DFN

    # Thermal
    T_default_C: float = 23.0          # used if a data file has no temperature field

    def updated(self, **changes) -> "CellConfig":
        """Return a copy with some fields changed, e.g. cfg.updated(D_n=2e-14)."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolConfig:
    """Test-protocol settings used to split and compare the measured data."""

    v_cv: float = 3.6                  # CV hold voltage used by the cycler [V]
    v_tol: float = 0.005               # voltage tolerance for detecting the CV phase [V]
    i_threshold: float = 0.05          # |I| below this counts as rest [A]
    period_s: int = 5                  # output period of the CC-CV experiment [s]

    def to_dict(self) -> dict:
        return asdict(self)