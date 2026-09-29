"""HPPC check of parameter sets (validation profile, not used in training)."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cell import CellModel
from ..config import CellConfig, ProtocolConfig
from ..data import load_test
from ..metrics import rmse
from ..simulate import simulate_current_driven

_STYLES = [":", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))]


def run_hppc_check(base_cfg: CellConfig, cases: dict[str, dict], data_dir: str | Path,
                   protocol: ProtocolConfig = ProtocolConfig(), out_dir: str | Path | None = None,
                   show: bool = True, test_name: str = "HPPC") -> pd.DataFrame:
    """Simulate HPPC for each parameter set in `cases` ({label: params}) and compare.

    The first two cases are treated as reference and optimized for the improvement printout.
    """
    data = load_test(data_dir, test_name, base_cfg.T_default_C)
    t, I, V = data.t, data.I, data.V
    rest, dis, chg = np.abs(I) < protocol.i_threshold, I > protocol.i_threshold, I < -protocol.i_threshold
    print(f"{test_name}: {data.duration_h:.2f} h, {len(t)} points, T = {data.T_C:.1f} °C")

    metrics, traces = {}, {}
    for label, params in cases.items():
        print(f"  running {test_name}: {label} ...", flush=True)
        t0 = time.time()
        cell = CellModel(base_cfg.updated(**params))
        try:
            sim = simulate_current_driven(cell, data)
        except Exception as ex:
            print(f"    failed: {ex}")
            continue
        ok = t <= sim.t[-1]
        Vm = np.interp(t[ok], sim.t, sim.V)
        err = (Vm - V[ok]) * 1000
        traces[label] = (t[ok], Vm, err)
        metrics[label] = {
            "Coverage [%]":               100 * sim.t[-1] / t[-1],
            "RMSE [mV]":                  rmse(err),
            "MAE [mV]":                   float(np.mean(np.abs(err))),
            "Max |err| [mV]":             float(np.max(np.abs(err))),
            "Mean offset [mV]":           float(np.mean(err)),
            "RMSE rest [mV]":             rmse(err[rest[ok]]),
            "RMSE discharge pulses [mV]": rmse(err[dis[ok]]),
            "RMSE charge pulses [mV]":    rmse(err[chg[ok]]),
            "Sim time [s]":               time.time() - t0,
        }
        print(f"    done in {time.time() - t0:.0f} s | RMSE {metrics[label]['RMSE [mV]']:.1f} mV | "
              f"coverage {metrics[label]['Coverage [%]']:.1f} %")
        del sim
        gc.collect()

    table = pd.DataFrame(metrics).T
    if table.empty:
        return table
    print(f"\n=== {test_name} results (validation) ===")
    print(table.round(2).to_string())

    labels = list(metrics)
    if len(labels) >= 2:
        ref, opt = labels[0], labels[1]
        print(f"\n{ref}  ->  {opt}:")
        for k in ["RMSE [mV]", "MAE [mV]", "Max |err| [mV]", "RMSE rest [mV]",
                  "RMSE discharge pulses [mV]", "RMSE charge pulses [mV]"]:
            a, b = metrics[ref][k], metrics[opt][k]
            print(f"  {k:28s}: {a:7.2f} -> {b:7.2f} mV  ({100 * (b - a) / a:+.1f} %)")
        if metrics[opt]["Coverage [%]"] < 99.9:
            print(f"  WARNING: {opt} stopped early ({metrics[opt]['Coverage [%]']:.1f} % simulated)")

    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "hppc_metrics.csv")

    # full profile
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t / 3600, I, "k", lw=0.8); ax[0].set_ylabel("Current [A]\n(+ = discharge)")
    ax[1].plot(t / 3600, V, "k", lw=1.2, label="Measured")
    for (label, (tt, Vm, err)), st in zip(traces.items(), _STYLES):
        ax[1].plot(tt / 3600, Vm, linestyle=st, lw=1.2, label=f"{label} ({metrics[label]['RMSE [mV]']:.1f} mV)")
        ax[2].plot(tt / 3600, err, linestyle=st, lw=0.8, label=label)
    ax[1].set_ylabel("Voltage [V]"); ax[1].legend(fontsize=8)
    ax[2].axhline(0, color="black", lw=0.8); ax[2].set_ylabel("Error [mV]"); ax[2].set_xlabel("Time [h]")
    ax[2].legend(fontsize=8)
    ax[0].set_title(f"{test_name} validation: measured vs model")
    for a in ax:
        a.grid(True)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out / "hppc_check.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # zoom on one pulse in the middle of the test
    i_p = int(np.argmax(dis & (t > 0.5 * t[-1])))
    win = (t >= t[i_p] - 120) & (t <= t[i_p] + 900)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t[win] - t[i_p], I[win], "k"); ax[0].set_ylabel("Current [A]")
    ax[1].plot(t[win] - t[i_p], V[win], "k", lw=1.5, label="Measured")
    for (label, (tt, Vm, err)), st in zip(traces.items(), _STYLES):
        w2 = (tt >= t[i_p] - 120) & (tt <= t[i_p] + 900)
        ax[1].plot(tt[w2] - t[i_p], Vm[w2], linestyle=st, lw=1.5, label=label)
    ax[1].set_ylabel("Voltage [V]"); ax[1].set_xlabel("Time from pulse start [s]"); ax[1].legend(fontsize=8)
    ax[0].set_title(f"{test_name} pulse zoom (t = {t[i_p] / 3600:.2f} h)")
    for a in ax:
        a.grid(True)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out / "hppc_zoom.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    del traces
    gc.collect()
    return table