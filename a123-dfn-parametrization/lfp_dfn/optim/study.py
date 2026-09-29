"""Run all methods on the same problem, build the comparison table, save the results."""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..cell import CellModel
from .methods import FAMILY, METHODS
from .objective import BudgetExceeded, CalibrationObjective
from .settings import OptimizationSettings


@dataclass
class MethodRun:
    method: str
    family: str
    best_params: dict
    J: float
    evals: int
    time_min: float
    best_row: dict              # log entry of the best evaluation (includes per-test RMSE)


@dataclass
class StudyResult:
    runs: dict[str, MethodRun]
    history: pd.DataFrame
    comparison: pd.DataFrame
    J_start: float
    metrics_start: dict
    start_params: dict

    @property
    def best_method(self) -> str:
        return min(self.runs, key=lambda m: self.runs[m].J)

    @property
    def best_params(self) -> dict:
        return dict(self.runs[self.best_method].best_params)


def _comparison_table(runs: dict[str, MethodRun], J0: float, met0: dict, start: dict) -> pd.DataFrame:
    rows = []
    for m, r in runs.items():
        rows.append({"Method": m, "Family": r.family, "Best J": r.J, "J / J_start": r.J / J0,
                     "Evaluations": r.evals, "Time [min]": r.time_min, **r.best_params,
                     **{k: v for k, v in r.best_row.items() if "RMSE" in k}})
    rows.append({"Method": "Start (manual)", "Family": "-", "Best J": J0, "J / J_start": 1.0,
                 "Evaluations": 0, "Time [min]": 0.0, **start,
                 **{f"{t} {k}": v for t, mm in met0.items() for k, v in mm.items()}})
    return pd.DataFrame(rows).set_index("Method").sort_values("Best J")


def run_comparison(obj: CalibrationObjective, start_params: dict, settings: OptimizationSettings,
                   verbose: bool = True) -> StudyResult:
    """Run every method in settings.methods from the same start with the same budget."""
    unknown = [m for m in settings.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHODS)}")
    start = {n: start_params[n] for n in obj.space.names}
    u0 = obj.space.to_unit(start)

    t0 = time.time()
    r0, met0 = obj.residuals_and_metrics(start)
    J0 = float((r0**2).sum())
    t_eval = time.time() - t0
    if verbose:
        print(f"Start J = {J0:.4f}  ({t_eval:.1f} s per evaluation)")
        print(f"Estimated total time: {t_eval * settings.budget * len(settings.methods) / 3600:.1f} h "
              f"({settings.budget} evals x {len(settings.methods)} methods)")

    runs: dict[str, MethodRun] = {}
    for method in settings.methods:
        if verbose:
            print(f"\n===== {method} =====")
        obj.start_method(method, settings.budget)
        n_before = obj.n_evals
        t0 = time.time()
        try:
            METHODS[method](obj, u0.copy(), settings.budget, settings)
        except BudgetExceeded:
            if verbose:
                print("  budget reached")
        except Exception as ex:
            print(f"  {method} stopped with error: {ex}")
        entries = obj.log[n_before:]
        if not entries:
            continue
        best = min(entries, key=lambda e: e["J"])
        runs[method] = MethodRun(method=method, family=FAMILY[method],
                                 best_params={n: best[n] for n in obj.space.names},
                                 J=best["J"], evals=len(entries),
                                 time_min=(time.time() - t0) / 60, best_row=best)
        if verbose:
            print(f"{method}: best J = {best['J']:.4f} ({best['J'] / J0:.2f} x start) after "
                  f"{len(entries)} evals, {runs[method].time_min:.1f} min")
        gc.collect()

    obj.reset_budget()
    return StudyResult(runs=runs, history=pd.DataFrame(obj.log),
                       comparison=_comparison_table(runs, J0, met0, start),
                       J_start=J0, metrics_start=met0, start_params=start)


def print_summary(result: StudyResult, names: list[str]) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    comp = result.comparison
    print("=== Optimizer comparison (sorted by objective) ===")
    print(comp[["Family", "Best J", "J / J_start", "Evaluations", "Time [min]"]].round(4))
    print("\n=== Parameters found ===")
    print(comp[names].to_string(float_format=lambda v: f"{v:.3e}"))
    print("\n=== Per-test RMSE ===")
    print(comp[[c for c in comp.columns if "RMSE" in c]].round(3))
    print(f"\nBest method: {result.best_method}  "
          f"(J = {result.runs[result.best_method].J:.4f}, start J = {result.J_start:.4f})")


def save_study(result: StudyResult, obj: CalibrationObjective, settings: OptimizationSettings,
               out_dir: str | Path, hppc: dict | None = None) -> Path:
    """Save history CSV, comparison CSV and optimized_parameters.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(out / "optimizer_history.csv", index=False)
    result.comparison.to_csv(out / "optimizer_comparison.csv")

    fixed = {k: v for k, v in obj.base_cfg.to_dict().items() if k not in obj.space.names}
    fixed["y100 (derived)"] = CellModel(obj.base_cfg).y100
    payload = {
        "best_method": result.best_method,
        "objective_J_best": result.runs[result.best_method].J,
        "objective_J_start": result.J_start,
        "optimized_parameters": result.best_params,
        "start_parameters": result.start_params,
        "bounds": obj.space.bounds_dict(),
        "fixed_parameters": fixed,
        "protocol": obj.protocol.to_dict(),
        "settings": {**settings.to_dict(), "train_tests": list(obj.cases)},
        "all_methods": {m: {"J": r.J, "evaluations": r.evals, "time_min": r.time_min,
                            "parameters": r.best_params,
                            "metrics": {k: v for k, v in r.best_row.items() if "RMSE" in k}}
                        for m, r in result.runs.items()},
        "hppc_check": hppc or {},
    }
    path = out / "optimized_parameters.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path