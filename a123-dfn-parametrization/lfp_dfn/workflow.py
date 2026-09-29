"""High-level workflow: run one test or a list of tests and save the results."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import pandas as pd

from .cell import CellModel
from .config import ProtocolConfig
from .data import load_test
from .metrics import cccv_metrics, compare_cccv, compare_current, current_metrics
from .plots import plot_charge, plot_current
from .simulate import simulate_cccv, simulate_current_driven


def is_charge_test(name: str) -> bool:
    """Charge tests are simulated as CC-CV; everything else follows the measured current."""
    return name.lower().startswith("charge")


def run_test(cell: CellModel, name: str, data_dir: str | Path,
             protocol: ProtocolConfig = ProtocolConfig(),
             out_dir: str | Path = "results", show_plots: bool = True) -> tuple[str, dict]:
    """Simulate one test, compute its metrics and save its figure.

    Returns ("charge" | "current", metrics dict).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_test(data_dir, name, cell.cfg.T_default_C)
    fig_path = out / f"all_{name}.png"

    if is_charge_test(name):
        sim, _ = simulate_cccv(cell, data, protocol)
        cmp = compare_cccv(data, sim, protocol)
        res = cccv_metrics(name, cmp, protocol)
        plot_charge(name, cmp, res, fig_path, show_plots)
        return "charge", res

    sim = simulate_current_driven(cell, data)
    cmp = compare_current(data, sim, protocol)
    res = current_metrics(name, data, sim, cmp)
    plot_current(name, data, sim, cmp, res, fig_path, show_plots)
    return "current", res


def run_all(cell: CellModel, tests: list[str], data_dir: str | Path,
            protocol: ProtocolConfig = ProtocolConfig(),
            out_dir: str | Path = "results", show_plots: bool = True,
            verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all tests, save summary CSVs and the configuration used.

    Returns (charge_table, current_table).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = {"charge": [], "current": []}
    failed = {}

    for name in tests:
        if verbose:
            print(f"Running {name} ...", flush=True)
        t0 = time.time()
        try:
            kind, res = run_test(cell, name, data_dir, protocol, out, show_plots)
            rows[kind].append(res)
            if verbose:
                print(f"  done in {time.time() - t0:.0f} s")
        except Exception as ex:                        # keep going with the other tests
            failed[name] = str(ex)
            if verbose:
                print(f"  {name} failed: {ex}")
        gc.collect()

    charge_tbl = pd.DataFrame(rows["charge"]).set_index("Test") if rows["charge"] else pd.DataFrame()
    current_tbl = pd.DataFrame(rows["current"]).set_index("Test") if rows["current"] else pd.DataFrame()
    if not charge_tbl.empty:
        charge_tbl.to_csv(out / "all_tests_charge_summary.csv")
    if not current_tbl.empty:
        current_tbl.to_csv(out / "all_tests_current_summary.csv")

    run_info = {
        "cell_config": cell.cfg.to_dict(),
        "protocol": protocol.to_dict(),
        "derived": {"y100": cell.y100, "Q_n_Ah": cell.Q_n, "Q_p_Ah": cell.Q_p},
        "tests": list(tests),
        "failed": failed,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(run_info, f, indent=2)

    if verbose:
        print(f"\nSaved to {out}: all_<test>.png, summary CSVs, run_config.json")
        if failed:
            print(f"Failed tests: {list(failed)}")
    return charge_tbl, current_tbl