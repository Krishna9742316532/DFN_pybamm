"""Task 5: load the optimized parameter set from Task 4 and validate it against measured data."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cell import CellModel
from .config import CellConfig, ProtocolConfig
from .data import TestData
from .metrics import rmse
from .residuals import COLORS, Residuals, hppc_residuals, plot_residuals, residual_table


def load_study(json_path: str | Path) -> dict:
    """Read optimized_parameters.json (Task 4) and rebuild the start and optimized configs."""
    with open(json_path) as fh:
        payload = json.load(fh)
    cfg_fields = {f.name for f in fields(CellConfig)}
    fixed_cfg = CellConfig(**{k: v for k, v in payload["fixed_parameters"].items() if k in cfg_fields})
    return {
        "best_method": payload.get("best_method", ""),
        "optimized_params": payload["optimized_parameters"],
        "start_params": payload["start_parameters"],
        "optimized_cfg": fixed_cfg.updated(**payload["optimized_parameters"]),
        "start_cfg": fixed_cfg.updated(**payload["start_parameters"]),
    }


def prada_baseline_config(T_default_C: float = 23.0) -> CellConfig:
    """Original Prada2013 values, no tuning (reference for the validation)."""
    return CellConfig(x100=0.81, y100=0.0038, use_soft_wall=False, D_n=3e-15, D_p=5.9e-18,
                      beta_p=1.0, k_j0_n=1.0, k_j0_p=1.0, R_contact=0.0, T_default_C=T_default_C)


def plot_pulse_zoom(data: TestData, results: dict[str, Residuals], save_path: str | Path | None = None,
                    show: bool = True, at_fraction: float = 0.5, before_s: float = 120,
                    after_s: float = 900, i_threshold: float = 0.05) -> None:
    """Zoom on the first discharge pulse after `at_fraction` of the test."""
    i_p = int(np.argmax((data.I > i_threshold) & (data.t > at_fraction * data.t[-1])))
    t0 = data.t[i_p]
    win = (data.t >= t0 - before_s) & (data.t <= t0 + after_s)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(data.t[win] - t0, data.I[win], color="#222222", lw=1.2)
    ax[0].set_ylabel("Current [A]\n(+ = discharge)")
    ax[1].plot(data.t[win] - t0, data.V[win], color="#222222", lw=1.5, label="Measured")
    for (label, r), c in zip(results.items(), COLORS):
        w2 = (r.t >= t0 - before_s) & (r.t <= t0 + after_s)
        ax[1].plot(r.t[w2] - t0, r.V_model[w2], color=c, lw=1.3, label=label)
    ax[1].set_ylabel("Voltage [V]"); ax[1].set_xlabel("Time from pulse start [s]")
    ax[1].legend(fontsize=8, frameon=False)
    ax[0].set_title(f"{data.name} pulse zoom at t = {t0 / 3600:.2f} h")
    for a in ax:
        a.grid(True, color="#dddddd", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def validate_on_hppc(configs: dict[str, CellConfig], data: TestData,
                     protocol: ProtocolConfig = ProtocolConfig(), out_dir: str | Path | None = None,
                     show: bool = True) -> tuple[pd.DataFrame, dict[str, Residuals]]:
    """Simulate the HPPC profile for each parameter set and compare with the measurement."""
    results = {}
    for label, cfg in configs.items():
        print(f"  simulating {data.name}: {label} ...", flush=True)
        try:
            results[label] = hppc_residuals(CellModel(cfg), data)
        except Exception as ex:
            print(f"    failed: {ex}")
            continue
        r = results[label]
        print(f"    RMSE {rmse(r.error_mV):.1f} mV | coverage {r.coverage:.1f} %")

    table = residual_table(results, protocol)
    table.insert(2, "MAE [mV]", [float(np.mean(np.abs(r.error_mV))) for r in results.values()])
    table.insert(3, "Max |error| [mV]", [float(np.max(np.abs(r.error_mV))) for r in results.values()])
    ref = table["RMSE all [mV]"].iloc[0]
    table["RMSE change vs first [%]"] = 100 * (table["RMSE all [mV]"] - ref) / ref

    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "validation_hppc_metrics.csv")
    plot_residuals(data, results, out / "validation_hppc_residuals.png" if out else None, show)
    plot_pulse_zoom(data, results, out / "validation_hppc_pulse_zoom.png" if out else None, show)
    return table, results