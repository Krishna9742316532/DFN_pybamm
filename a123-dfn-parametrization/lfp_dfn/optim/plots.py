"""Figures for the optimization study."""
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .objective import CalibrationObjective
from .study import StudyResult


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_convergence(result: StudyResult, save_path: str | Path | None = None, show: bool = True) -> None:
    """Best objective so far vs. number of model evaluations, per method."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in result.runs:
        J = result.history.loc[result.history["method"] == m, "J"].values
        ax.plot(np.arange(1, len(J) + 1), np.minimum.accumulate(J), marker=".", label=m)
    ax.axhline(result.J_start, color="grey", ls=":", label="Start (manual)")
    ax.set_yscale("log")
    ax.set_xlabel("Model evaluations"); ax.set_ylabel("Best objective J so far")
    ax.set_title("Convergence of the optimization methods")
    ax.grid(True, which="both", alpha=0.4); ax.legend()
    _finish(fig, save_path, show)


def plot_parameter_comparison(result: StudyResult, obj: CalibrationObjective,
                              save_path: str | Path | None = None, show: bool = True) -> None:
    """Normalized parameter values (0 = lower bound, 1 = upper bound) per method."""
    space = obj.space
    labels = ["Start (manual)"] + list(result.runs)
    U = np.array([space.to_unit(result.start_params)] +
                 [space.to_unit(result.runs[m].best_params) for m in result.runs])
    x, w = np.arange(space.n), 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, lab in enumerate(labels):
        ax.bar(x - 0.4 + w / 2 + i * w, U[i], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(space.names); ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value (0 = lower, 1 = upper bound)")
    ax.set_title("Parameters found by each method (log scale for D, k_j0, beta)")
    ax.grid(True, axis="y", alpha=0.4); ax.legend(fontsize=8, ncol=4)
    _finish(fig, save_path, show)


def plot_training_fit(obj: CalibrationObjective, name: str, params_opt: dict, params_ref: dict,
                      label_opt: str = "Optimized", save_path: str | Path | None = None,
                      show: bool = True) -> None:
    """Measured vs. manual start vs. optimized for one training test."""
    case = obj.cases[name]
    t, tm = case.data.t, case.data.t / 60
    sims = {}
    for label, params in [("Manual start", params_ref), (label_opt, params_opt)]:
        try:
            sims[label] = obj.simulate(case, params)
        except Exception as ex:
            print(f"  {name} / {label} failed: {ex}")

    n_ax = 3 if case.kind == "charge" else 2
    fig, ax = plt.subplots(n_ax, 1, figsize=(11, 3 * n_ax), sharex=True)
    ax[0].plot(tm, case.data.V, "k", lw=1.5, label="Measured")
    if case.kind == "charge":
        ax[2].plot(tm, -case.data.I, "k", lw=1.5, label="Measured")

    title = []
    for (label, sim), st in zip(sims.items(), [":", "--"]):
        ok = t <= sim.t[-1]
        Vm, Im = np.interp(t[ok], sim.t, sim.V), np.interp(t[ok], sim.t, sim.I)
        ax[0].plot(tm[ok], Vm, st, lw=1.5, label=label)
        short = label.split(" (")[0]
        if case.kind == "charge":
            cc, cv = case.cc[ok], case.cv[ok]
            eV = (Vm[cc] - case.data.V[ok][cc]) * 1000
            eI = Im[cv] - case.data.I[ok][cv]
            ax[1].plot(tm[ok][cc], eV, st, label=label)
            ax[2].plot(tm[ok], -Im, st, lw=1.5, label=label)
            cv_rmse = np.sqrt(np.mean(eI**2)) if cv.any() else np.nan
            title.append(f"{short}: CC {np.sqrt(np.mean(eV**2)):.1f} mV, CV {cv_rmse:.3f} A")
        else:
            e = (Vm - case.data.V[ok]) * 1000
            ax[1].plot(tm[ok], e, st, label=label)
            title.append(f"{short}: {np.sqrt(np.mean(e**2)):.1f} mV")

    ax[0].set_ylabel("Voltage [V]"); ax[0].set_title(f"{name}  |  " + "  |  ".join(title))
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("CC voltage error [mV]" if case.kind == "charge" else "Voltage error [mV]")
    if case.kind == "charge":
        ax[2].set_ylabel("Charge current [A]")
    ax[-1].set_xlabel("Time [min]")
    for a in ax:
        a.grid(True); a.legend(fontsize=8)
    _finish(fig, save_path, show)
    del sims
    gc.collect()