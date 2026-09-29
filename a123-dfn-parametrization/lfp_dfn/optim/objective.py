"""Training data, residual vector and objective J (with cache, evaluation budget and log)."""
from __future__ import annotations

import gc
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from ..cell import CellModel
from ..config import CellConfig, ProtocolConfig
from ..data import TestData, load_test
from ..simulate import CCCVProtocol, SimResult, extract_cccv_protocol, simulate_cccv, simulate_current_driven
from ..workflow import is_charge_test
from .settings import OptimizationSettings
from .space import ParameterSpace


class BudgetExceeded(Exception):
    """Raised when a method asks for more model evaluations than its budget allows."""


@dataclass
class TrainingCase:
    data: TestData
    kind: str                           # "charge" (CC-CV) or "current" (current-driven)
    cccv: CCCVProtocol | None = None
    cc: np.ndarray | None = None        # mask: CC phase -> compare voltage
    cv: np.ndarray | None = None        # mask: CV phase -> compare current

    @property
    def name(self) -> str:
        return self.data.name


def build_training_set(names: list[str], data_dir: str | Path, protocol: ProtocolConfig,
                       T_default_C: float = 23.0) -> dict[str, TrainingCase]:
    """Load the training tests and prepare CC/CV masks for the charge tests."""
    cases = {}
    for name in names:
        data = load_test(data_dir, name, T_default_C)
        if is_charge_test(name):
            charging = data.I < -protocol.i_threshold
            cv = data.V >= protocol.v_cv - protocol.v_tol
            cases[name] = TrainingCase(data, "charge", cccv=extract_cccv_protocol(data, protocol),
                                       cc=(~cv) & charging, cv=cv)
        else:
            cases[name] = TrainingCase(data, "current")
    return cases


class CalibrationObjective:
    """J(u) = sum of squared normalized residuals over all training tests.

    Charge tests : CC voltage error / v_scale  and  CV current error / i_scale
    Discharge    : voltage error / v_scale
    Each block is divided by sqrt(N) and each test by sqrt(n_tests), so every
    block contributes its mean squared error and every test counts equally.
    """

    def __init__(self, base_cfg: CellConfig, space: ParameterSpace, cases: dict[str, TrainingCase],
                 protocol: ProtocolConfig, settings: OptimizationSettings, verbose: bool = True):
        cfg_fields = {f.name for f in fields(CellConfig)}
        unknown = [n for n in space.names if n not in cfg_fields]
        if unknown:
            raise ValueError(f"Parameters not in CellConfig: {unknown}")
        self.base_cfg, self.space, self.cases = base_cfg, space, cases
        self.protocol, self.settings, self.verbose = protocol, settings, verbose
        self.log: list[dict] = []
        self.method = ""
        self.max_evals = 10**9
        self._cache: dict = {}

    # ---- model -------------------------------------------------------------
    def cell_for(self, params: dict) -> CellModel:
        return CellModel(self.base_cfg.updated(**params))

    def _simulate(self, cell: CellModel, case: TrainingCase) -> SimResult:
        if case.kind == "charge":
            hold = case.data.t[-1] - case.cccv.t_rest          # long enough to cover the test
            sim, _ = simulate_cccv(cell, case.data, self.protocol, hold_s=hold)
            return sim
        return simulate_current_driven(cell, case.data)

    def simulate(self, case: TrainingCase, params: dict) -> SimResult:
        return self._simulate(self.cell_for(params), case)

    # ---- residuals ---------------------------------------------------------
    def residuals_and_metrics(self, params: dict) -> tuple[np.ndarray, dict]:
        s = self.settings
        cell = self.cell_for(params)
        res_all, metrics = [], {}
        for name, case in self.cases.items():
            t = case.data.t
            try:
                sim = self._simulate(cell, case)
                ok = t <= sim.t[-1]
                Vm, Im = np.interp(t, sim.t, sim.V), np.interp(t, sim.t, sim.I)
            except Exception:
                ok = np.zeros(len(t), bool)
                Vm, Im = np.zeros(len(t)), np.zeros(len(t))

            if case.kind == "charge":
                cc, cv = case.cc, case.cv
                eV = np.where(ok[cc], (Vm[cc] - case.data.V[cc]) / s.v_scale, s.penalty)
                eI = np.where(ok[cv], (Im[cv] - case.data.I[cv]) / s.i_scale, s.penalty)
                r = np.concatenate([eV / np.sqrt(max(len(eV), 1)), eI / np.sqrt(max(len(eI), 1))])
                metrics[name] = {
                    "RMSE V CC [mV]": float(np.sqrt(np.mean(eV**2)) * s.v_scale * 1000),
                    "RMSE I CV [A]": float(np.sqrt(np.mean(eI**2)) * s.i_scale) if len(eI) else np.nan}
            else:
                e = np.where(ok, (Vm - case.data.V) / s.v_scale, s.penalty)
                r = e / np.sqrt(len(e))
                metrics[name] = {"RMSE V [mV]": float(np.sqrt(np.mean(e**2)) * s.v_scale * 1000)}
            res_all.append(r / np.sqrt(len(self.cases)))
        return np.concatenate(res_all), metrics

    # ---- bookkeeping -------------------------------------------------------
    def start_method(self, method: str, budget: int) -> None:
        """Fresh cache and a new evaluation budget for one method (fair comparison)."""
        self.method = method
        self._cache = {}
        self.max_evals = len(self.log) + budget

    def reset_budget(self) -> None:
        self.max_evals = 10**9

    @property
    def n_evals(self) -> int:
        return len(self.log)

    # ---- functions handed to the optimizers --------------------------------
    def residuals(self, u) -> np.ndarray:
        """Residual vector r(u) (used by TRF)."""
        u = np.clip(np.asarray(u, float), 0, 1)
        key = tuple(np.round(u, 6))
        if key in self._cache:                               # repeated point: no new simulation
            return self._cache[key]
        if len(self.log) >= self.max_evals:
            raise BudgetExceeded()

        params = self.space.to_params(u)
        r, met = self.residuals_and_metrics(params)
        J = float(np.sum(r**2))
        self._cache[key] = r
        self.log.append({"method": self.method, "J": J, **params,
                         **{f"{t} {k}": v for t, mm in met.items() for k, v in mm.items()}})
        if self.verbose:
            own = [e["J"] for e in self.log if e["method"] == self.method]
            print(f"  [{self.method}] eval {len(own):3d}: J = {J:8.4f} (best {min(own):8.4f}) | "
                  + "  ".join(f"{n}={v:.3g}" for n, v in params.items()), flush=True)
        gc.collect()
        return r

    def __call__(self, u) -> float:
        """Scalar objective J(u) (used by L-BFGS-B, Nelder-Mead, GA, DE)."""
        return float(np.sum(self.residuals(u) ** 2))