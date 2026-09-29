# A123 ANR26650M1B – DFN model parametrization with PyBaMM

Physics-based modelling of the **A123 ANR26650M1B** lithium-ion cell (LFP cathode / graphite anode, 2.5 Ah nominal) with the **Doyle–Fuller–Newman (DFN)** model in [PyBaMM](https://pybamm.org).

The repository takes the model from a literature parameter set to a cell-specific, validated parameter set:

| Step | Task | What happens |
|---|---|---|
| 1 | Model and parameter set | DFN model with the Prada2013 parameter set as starting point |
| 2 | Baseline simulation | HPPC test simulated with the untouched Prada2013 values; voltage and error plotted; RMSE and other metrics calculated |
| 3 | Analysis and manual tuning | Residuals analysed, parameters grouped, capacity window, OCV and diffusion tuned by hand |
| 4 | Optimization | Six dynamic parameters fitted to Charge 1C–4C and Discharge 1C; six optimization methods compared |
| 5 | Validation | Optimized set checked on HPPC (not used for fitting); internal variables such as the anode potential vs Li/Li⁺ analysed |

All model logic lives in the `lfp_dfn` Python package. The notebooks only configure, run and display. This keeps the notebooks short and makes every step reproducible.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Data](#4-data)
5. [Conventions](#5-conventions)
6. [How to run the notebooks](#6-how-to-run-the-notebooks)
7. [Methodology in short](#7-methodology-in-short)
8. [Parameters](#8-parameters)
9. [Configuration](#9-configuration)
10. [Package reference](#10-package-reference)
11. [Outputs and file formats](#11-outputs-and-file-formats)
12. [Results summary](#12-results-summary)
13. [Extending the code](#13-extending-the-code)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations and assumptions](#15-limitations-and-assumptions)
16. [References](#16-references)

---

## 1. Quick start

```bash
git clone <repository-url>
cd a123-dfn-parametrization
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
# copy the .mat files into data/
jupyter lab
```

Then open the notebooks in `notebooks/` and run them in order: Task 2 → 3 → 4 → 5.

For a quick check before the long optimization, set `budget=10` in the Task 4 settings cell.

---

## 2. Repository structure

```
a123-dfn-parametrization/
├── README.md
├── requirements.txt
├── .gitignore
│
├── lfp_dfn/                           # the model package – no hardcoded paths
│   ├── __init__.py                   # public API
│   ├── config.py                     # CellConfig, ProtocolConfig
│   ├── data.py                       # TestData, load_test()
│   ├── cell.py                       # CellModel: capacities, OCPs, PyBaMM parameters, DFN model
│   ├── simulate.py                   # simulate_cccv(), simulate_current_driven()
│   ├── metrics.py                    # model vs measurement comparison, error metrics
│   ├── plots.py                      # standard comparison figures
│   ├── workflow.py                   # run_test(), run_all()
│   ├── residuals.py                  # residual analysis on HPPC (Task 3)
│   ├── validation.py                 # load Task 4 result, validate on HPPC (Task 5)
│   ├── internal_states.py            # anode potential, stoichiometries, overpotentials, electrolyte (Task 5)
│   └── optim/                        # parameter optimization (Task 4)
│       ├── __init__.py
│       ├── settings.py               # OptimizationSettings
│       ├── space.py                  # ParameterSpec, ParameterSpace (bounds, log scale, u <-> θ)
│       ├── objective.py              # build_training_set(), CalibrationObjective
│       ├── methods.py                # TRF, L-BFGS-B, Nelder-Mead, GA, DE, DE->TRF
│       ├── study.py                  # run_comparison(), print_summary(), save_study()
│       ├── plots.py                  # convergence, parameter comparison, training fits
│       └── validation.py             # quick HPPC check after optimization
│    ├── Data_dir
│       ├── Charge_1c.mat               
│       ├── Charge_2c.mat                  
│       ├── Charge_3c.mat              
│       ├── Charge_4c.mat                
│       ├── disCharge_1c.mat                  
│       ├── Hppc.mat                  
│       
|
├── notebooks/
│   ├── DFN_task_1 and_2.ipynb
│   ├── DFN_task_3.ipynb
│   ├── DFN_task_4.ipynb
│   └── DFN_task_5.ipynb
│
└── A123 ANR26650M1B – DFN Model Parametrization and Validation(report)/                           
```

**Design idea.** Every function sits in `lfp_dfn/`. The notebooks contain only settings, function calls and short explanations. Changing a parameter or a setting never requires editing the package.

---

## 3. Installation

**Requirements**

- Python 3.10 or newer
- About 8 GB RAM recommended (the HPPC profile is 25 h long)
- Windows, Linux or macOS

**Install**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt`**

```
pybamm==26.8
numpy>=2.0          # np.trapezoid is used
scipy
pandas
matplotlib
deap                # genetic algorithm (Task 4)
jupyter
```

**Check the installation**

```python
import pybamm
print(pybamm.__version__)                       # 26.8
print("Prada2013" in pybamm.parameter_sets)     # True
```

---

## 4. Data

### 4.1 Files

Copy the measured files into `data/`:

| File | Test | Length | Used in |
|---|---|---|---|
| `HPPC.mat` | Hybrid pulse power characterization, from full to empty | ~25 h | Task 2 (baseline), Task 3 (rest voltages for OCV, residuals), Task 5 (validation) |
| `Charge_1C.mat` | CC-CV charge, 1C to 3.6 V | ~1 h | Task 4 (training) |
| `Charge_2C.mat` | CC-CV charge, 2C to 3.6 V | | Task 4 (training) |
| `Charge_3C.mat` | CC-CV charge, 3C to 3.6 V | | Task 4 (training) |
| `Charge_4C.mat` | CC-CV charge, 4C to 3.6 V | | Task 4 (training), Task 5 (plating check) |
| `Discharge_1C.mat` | CC discharge, 1C | ~1 h | Task 4 (training) |

> The data files belong to the company and are **not** part of the repository. `data/` is listed in `.gitignore`.

### 4.2 Expected content

`load_test()` accepts two layouts:

- one struct (for example `meas`) with fields `Time`, `Current`, `Voltage` and optionally `Battery_Temp_degC`;
- separate variables with the same names.

Field names are matched case-insensitively. Duplicate time stamps are removed, and time is shifted to start at 0 s. If there's no temperature field, `CellConfig.T_default_C` (23 °C) is used.

### 4.3 Test assumptions

- **Charge tests** end at full charge. Their start state is calculated backwards from the measured charge throughput.
- **Discharge and HPPC** start at full charge.
- **The CV phase** is detected as V ≥ 3.595 V (3.6 V minus a 5 mV tolerance).
- **Rest** is |I| < 0.05 A.

---

## 5. Conventions

| Item | Convention |
|---|---|
| Current sign | PyBaMM convention: **+ = discharge, − = charge**. The cycler sign is flipped in `load_test()` |
| Units | SI inside the package (V, A, s, m, mol/m³). Plots show mV, min or h where easier to read |
| Graphite stoichiometry | `x` (0 = empty, 1 = full of Li). `x100` = value at 100 % SOC |
| LFP stoichiometry | `y` (0 = delithiated, i.e. cell full; 1 = lithiated). `y100` = value at 100 % SOC |
| Residual / error | model − measured (positive = model voltage too high) |
| Normalized parameters | `u` ∈ [0, 1] per parameter (0 = lower bound, 1 = upper bound) |
| Anode potential vs Li/Li⁺ | φ_s − φ_e in the negative electrode; below 0 V lithium plating becomes possible |

---

## 6. How to run the notebooks

Run them **in order**. Each notebook only depends on the package and, for Task 5, on the Task 4 result file.

### Paths

The first two cells of every notebook set the paths:

```python
PROJECT_DIR = Path.cwd().parent               # repository root (notebooks/ is one level down)
DATA_DIR    = PROJECT_DIR / "data"
OUT_DIR     = PROJECT_DIR / "results" / "task4"   # task2 / task3 / task4 / task5
```

If you run the notebooks from another folder, set `PROJECT_DIR` to the repository root.

### Overview

| # | Notebook | Purpose | Input | Output folder | Runtime* |
|---|---|---|---|---|---|
| 1 | `DFN_task_1 and_2.ipynb` | Baseline Prada2013 on HPPC | `HPPC.mat` | `results/task2/` | few min |
| 2 | `DFN_task_3.ipynb` | Manual set on all tests, residual analysis | all `.mat` | `results/task3/` | ~30 min |
| 3 | `DFN_task_4.ipynb` | Six optimizers, best parameter set | training `.mat` | `results/task4/` | 4-6 h |
| 4 | `DFN_task_5.ipynb` | HPPC validation, internal variables | Task 4 JSON + all `.mat` | `results/task5/` | 30–60 min |

\* On a normal laptop. One optimization evaluation (five DFN simulations) takes about 30–60 s.

### 6.1 Task 2 – Baseline simulation

**Goal:** see how well the unmodified Prada2013 parameters describe this cell.

1. Load HPPC and apply its measured current profile to the DFN.
2. Plot measured vs simulated voltage and the voltage error.
3. Calculate the metrics: RMSE (main), MAE, max error, bias, RMSE during rest and during pulses, and coverage (how much of the profile the model simulated before stopping).

**Outputs:** voltage comparison plot, error plot, metric table.

### 6.2 Task 3 – Analysis and manual tuning

| Cell | Content |
|---|---|
| 1 | Setup and imports |
| 2 | Configuration: paths, test list, protocol, manually tuned `CellConfig` |
| 3 | `run_all()` on all six tests (charge tests as CC-CV, discharge and HPPC current-driven) |
| 4 | Result tables for charge and current-driven tests |
| 5 | Residual analysis: baseline vs manual on HPPC, residual over time and vs SOC, RMSE per SOC range |

**Outputs:** `all_<test>.png`, `all_tests_charge_summary.csv`, `all_tests_current_summary.csv`, `run_config.json`, `hppc_residuals.png`, `hppc_residual_table.csv`.

### 6.3 Task 4 – Optimization

| Cell | Content |
|---|---|
| 1 | Setup and imports |
| 2 | Settings, fixed parameters, search space, start point |
| 3 | Load the training tests and build the objective |
| 4 | Run all six methods (**long cell**) |
| 5 | Comparison table, convergence plot, parameter plot |
| 6 | Training fits: measured vs manual vs optimized |
| 7 | Quick HPPC check |
| 8 | Save results |

**Recommended procedure:**

1. Set `budget=10` and run everything once. This takes about an hour and checks the whole chain.
2. Set `budget=60–100` and run again, ideally overnight. After the first evaluation the notebook prints an estimated total time.
3. Cells 5–8 only use the stored result, so re-running them does not repeat the optimization.

**Outputs:** `optimized_parameters.json`, `optimizer_comparison.csv`, `optimizer_history.csv`, `convergence.png`, `parameter_comparison.png`, `fit_<test>.png`, `hppc_metrics.csv`, `hppc_check.png`, `hppc_zoom.png`.

### 6.4 Task 5 – Validation

| Cell | Content |
|---|---|
| 1 | Setup and imports |
| 2 | Load `optimized_parameters.json` and rebuild the baseline, manual and optimized sets |
| 3 | HPPC validation of all three sets: metrics, residual plots, pulse zoom |
| 4 | Training tests re-run with the optimized set |
| 5 | Internal variables for the four charge tests: anode potential, plating summary, 4C details, electrolyte profile |
| 6 | Internal variables during HPPC |
| 7 | Save the summary tables |

**Outputs:** `validation_hppc_metrics.csv`, `validation_hppc_residuals.png`, `validation_hppc_pulse_zoom.png`, `training/`, `anode_potential_charge.png`, `internal_states_charge_4C.png`, `electrolyte_charge_4C.png`, `internal_states_hppc.png`, `internal_states_summary.csv`.

---

## 7. Methodology in short

The full reasoning is in the report. This section is a summary.

### 7.1 Model

- **Model:** PyBaMM DFN (pseudo-2D), isothermal at the measured test temperature.
- **Starting parameters:** Prada2013, an LFP/graphite parameter set for an A123 26650 cell.
- **Graphite OCP:** Chen2020, the Prada2013 default (Ecker2015 available as an option).
- **LFP OCP:** Afshar2017 form with an adjustable "soft wall" near full charge.
- **Initial state:** set from the stoichiometry window (x100, y100) and the charge throughput of each test.

### 7.2 Simulation modes

| Test type | How it is simulated | Why |
|---|---|---|
| Charge 1C–4C | PyBaMM `Experiment`: rest → CC until 3.6 V → CV hold at 3.6 V | same control as the cycler; the CV phase is voltage-controlled, so the current is a model output |
| Discharge 1C, HPPC | measured current as input (`Interpolant`) | follows the real profile exactly |

### 7.3 Error metrics

- **RMSE (main metric):**
  - it is in mV, so it is easy to read;
  - it weights large errors more, and the large deviations during pulses and at low SOC are what matter for a BMS;
  - it matches the least-squares objective used in the optimization.
- **Supporting metrics:**
  - **MAE:** typical error;
  - **max error:** worst case;
  - **mean error:** systematic offset;
  - **RMSE rest vs pulses:** OCV errors vs dynamic errors;
  - **RMSE per SOC range:** where the model is weakest.
- **Not used:** MAPE and R². The high voltage level makes them look good even for poor fits.

### 7.4 Manual tuning (Task 3)

| Problem seen in the residuals | Cause | Fix |
|---|---|---|
| Model stops early, wrong capacity | stoichiometry window | x100 0.81 → 0.9082 |
| Voltage collapse at low SOC | graphite surface runs empty (D_n too low) | D_n 3·10⁻¹⁵ → 1.5·10⁻¹⁴ m²/s |
| Offset during rests | OCV mismatch | OCV fitted to HPPC rest voltages; soft LFP wall (k_wall 150 → 70); y100 from V_rest,full = 3.447 V |
| High voltage near full charge / long CV tail | slow LFP diffusion near full charge | concentration-dependent D_p with β_p = 0.2 |
| Offset in the first minute | cell not relaxed at test start | measurement effect, excluded from conclusions |

### 7.5 Optimization (Task 4)

**Objective.** For each charge test, the CC-phase voltage error (divided by 10 mV) and the CV-phase current error (divided by 0.1 A). For the discharge, the voltage error (divided by 10 mV). Each block is divided by √N and each test by √5:

```
J = (1/5) · Σ_blocks (RMSE_block / scale)²
```

Every test counts equally, whatever its length. Current is used in the CV phase because the voltage is fixed at 3.6 V there, in both the test and the model.

**Search space.** Six parameters in a normalized [0, 1]⁶ space, on a log10 scale except R_contact. Bounds are physical ranges from the literature.

**Methods compared:**

| Method | Family | Library |
|---|---|---|
| TRF (Trust Region Reflective) | gradient-based, least squares | `scipy.optimize.least_squares` |
| L-BFGS-B | gradient-based, quasi-Newton | `scipy.optimize.minimize` |
| Nelder-Mead | gradient-free, local | `scipy.optimize.minimize` |
| GA (Genetic Algorithm) | gradient-free, global | `deap` |
| DE (Differential Evolution) | gradient-free, global | `scipy.optimize.differential_evolution` |
| DE → TRF | hybrid (70 % DE, 30 % TRF) | both |

**Fair comparison:** every method gets the same objective, start point, bounds, seed and evaluation budget. Gradients are estimated by finite differences with a step of 0.02 in normalized space. Repeated points are cached and not counted against the budget.

### 7.6 Validation (Task 5)

- **HPPC comparison.** HPPC was not used to fit the dynamic parameters (only its rest voltages were used for the OCV). The baseline, manual and optimized sets are compared on it.
- **Internal variables:**
  - anode potential vs Li/Li⁺ at the separator side (the plating indicator);
  - surface vs average stoichiometry of both electrodes (diffusion limits);
  - reaction overpotentials;
  - electrolyte concentration across the cell.

---

## 8. Parameters

### 8.1 Fixed parameters (from Task 3)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Graphite stoichiometry at 100 % SOC | x100 | 0.9082 | fitted to capacity |
| LFP stoichiometry at 100 % SOC | y100 | ≈ 0.019 (calculated) | from V_rest,full = 3.447 V |
| LFP OCP wall steepness | k_wall | 70 | fitted to HPPC rests (Afshar original: 150) |
| Slow-diffusion width (LFP) | δ_p | 0.15 | manual |
| Active material fraction, negative | ε_n | 0.58 | Prada2013 |
| Active material fraction, positive | ε_p | 0.374 | Prada2013 |
| LFP charge-branch offset | dU_charge | 0 V | tested, not used |
| Geometry, particle radii, electrolyte | – | Prada2013 | literature |

### 8.2 Optimized parameters (Task 4)

| Parameter | Meaning | Start (manual) | Bounds | Scale | Optimized |
|---|---|---|---|---|---|
| D_n | graphite diffusivity [m²/s] | 1.5·10⁻¹⁴ | 10⁻¹⁵ – 10⁻¹² | log | … |
| D_p | LFP diffusivity [m²/s] | 5.8·10⁻¹⁸ | 10⁻¹⁹ – 10⁻¹⁵ | log | … |
| k_j0_n | multiplier on graphite exchange-current density | 1.11 | 0.1 – 10 | log | … |
| k_j0_p | multiplier on LFP exchange-current density | 1.14 | 0.1 – 10 | log | … |
| R_contact | contact resistance [Ω] | 0 | 0 – 0.015 | linear | … |
| β_p | LFP diffusivity ratio near full charge | 0.2 | 0.01 – 1 | log | … |

The LFP diffusivity is concentration-dependent:

```
D_p(y) = D_p · [ 1 − (1 − β_p) · exp(−y / δ_p) ]
```

The exchange-current densities are the Prada2013 functions multiplied by k_j0_n and k_j0_p.

---

## 9. Configuration

### 9.1 `CellConfig` – cell parameters

```python
from lfp_dfn import CellConfig

cfg = CellConfig(
    x100=0.9082, graphite_ocp="Chen2020", use_soft_wall=True, k_wall=70,
    v_full_rest=3.447, du_charge=0.0,
    D_n=1.5e-14, D_p=5.8e-18, beta_p=0.2, delta_p=0.15,
    eps_n=0.58, eps_p=0.374,
    k_j0_n=1.0, k_j0_p=1.0, R_contact=0.0,
    T_default_C=23.0,
)
cfg2 = cfg.updated(D_n=2e-14, R_contact=0.002)      # copy with changes (cfg itself is not modified)
```

| Field | Default | Meaning |
|---|---|---|
| `base_set` | `"Prada2013"` | PyBaMM parameter set used as base |
| `nominal_capacity_Ah` | 2.5 | nominal capacity |
| `v_max`, `v_min` | 4.2, 1.0 | solver safety cut-offs (wide on purpose, so the model only stops if it really fails) |
| `x100`, `y100` | 0.9082, 0.01 | stoichiometry at 100 % SOC (`y100` is only used without the soft wall) |
| `graphite_ocp` | `"Chen2020"` | or `"Ecker2015"` |
| `use_soft_wall`, `k_wall` | True, 70 | soft LFP OCP wall and its steepness |
| `v_full_rest` | 3.447 | rest voltage at full charge, used to calculate y100 |
| `du_charge` | 0.0 | LFP OCP offset on the charge branch |
| `D_n`, `D_p`, `beta_p`, `delta_p` | … | solid diffusion |
| `eps_n`, `eps_p` | 0.58, 0.374 | active material volume fractions |
| `k_j0_n`, `k_j0_p` | 1.0 | exchange-current multipliers |
| `R_contact` | 0.0 | contact resistance; switched on in the DFN only when > 0 |
| `T_default_C` | 23.0 | temperature if the file has none |

### 9.2 `ProtocolConfig` – test protocol

```python
from lfp_dfn import ProtocolConfig
protocol = ProtocolConfig(v_cv=3.6, v_tol=0.005, i_threshold=0.05, period_s=5)
```

| Field | Meaning |
|---|---|
| `v_cv` | CV hold voltage of the cycler [V] |
| `v_tol` | tolerance to detect the CV phase [V] |
| `i_threshold` | \|I\| below this counts as rest [A] |
| `period_s` | output period of the CC-CV experiment [s] |

### 9.3 `OptimizationSettings` – Task 4

```python
from lfp_dfn.optim import OptimizationSettings
settings = OptimizationSettings(
    methods=("TRF", "L-BFGS-B", "Nelder-Mead", "GA", "DE", "DE->TRF"),
    budget=100, fd_step=0.02, seed=0,
    v_scale=0.010, i_scale=0.10, penalty=10.0,
)
```

| Field | Meaning |
|---|---|
| `methods` | which methods to run, in order |
| `budget` | model evaluations per method (1 evaluation = all training tests) |
| `fd_step` | finite-difference step in normalized space |
| `seed` | random seed (GA, DE) |
| `v_scale`, `i_scale` | error scales: 10 mV and 0.1 A count as "1 unit" |
| `penalty` | residual given to points the model could not simulate |

### 9.4 Search space

```python
from lfp_dfn.optim import ParameterSpec, ParameterSpace
space = ParameterSpace([
    ParameterSpec("D_n", 1e-15, 1e-12),                    # log scale by default
    ParameterSpec("R_contact", 0.0, 0.015, log=False),
    ...
])
space = space.without("beta_p")                            # fix a parameter
```

Parameter names must match `CellConfig` field names.

---

## 10. Package reference

### Core (`lfp_dfn`)

| Function / class | Module | Description |
|---|---|---|
| `CellConfig`, `ProtocolConfig` | `config` | settings (frozen dataclasses, `.updated()`, `.to_dict()`) |
| `load_test(folder, name, T_default_C)` | `data` | load a `.mat` file → `TestData(name, t, I, V, T_C)` |
| `CellModel(cfg)` | `cell` | capacities Q_n/Q_p, OCPs, y100; `.parameter_values(x0, y0, T_C, charge_branch)`, `.build_model()`, `.state_full()`, `.state_before_charge(Q)`, `.summary()` |
| `simulate_cccv(cell, data, protocol, hold_s=None)` | `simulate` | CC-CV charge like the cycler → `(SimResult, CCCVProtocol)` |
| `simulate_current_driven(cell, data)` | `simulate` | follow the measured current → `SimResult(t, V, I, termination)` |
| `compare_cccv`, `cccv_metrics` | `metrics` | CC voltage RMSE, CV current RMSE, CV start time, charge accepted |
| `compare_current`, `current_metrics` | `metrics` | RMSE, MAE, max error, rest vs current-on RMSE |
| `run_test`, `run_all` | `workflow` | run one test or a list of tests, save figures, CSVs and `run_config.json` |
| `hppc_residuals`, `residual_table`, `plot_residuals` | `residuals` | residual analysis over time, state and SOC |
| `load_study`, `prada_baseline_config`, `validate_on_hppc` | `validation` | Task 5 validation |
| `charge_internal_states`, `current_internal_states`, `plating_summary`, `plot_anode_potential`, `plot_internal_states`, `plot_electrolyte_profiles` | `internal_states` | internal variables and plating margin |

### Optimization (`lfp_dfn.optim`)

| Function / class | Description |
|---|---|
| `build_training_set(names, data_dir, protocol)` | loads the training tests, prepares CC/CV masks |
| `CalibrationObjective(base_cfg, space, cases, protocol, settings)` | residual vector `r(u)` (`.residuals`), scalar `J(u)` (call), cache, budget and log |
| `METHODS`, `FAMILY` | the six method runners and their families |
| `run_comparison(objective, start, settings)` | runs all methods → `StudyResult` (`.best_method`, `.best_params`, `.comparison`, `.history`) |
| `print_summary`, `save_study` | tables and `optimized_parameters.json` |
| `plot_convergence`, `plot_parameter_comparison`, `plot_training_fit` | figures |
| `run_hppc_check` | quick HPPC comparison after the optimization |

### Internal variables used (PyBaMM names)

| Key | PyBaMM variable |
|---|---|
| `phi_n_sep` | `Negative electrode surface potential difference at separator interface [V]` |
| `phi_n_avg` | `X-averaged negative electrode surface potential difference [V]` |
| `x_surf`, `x_avg` | `X-averaged negative particle surface stoichiometry`, `Average negative particle stoichiometry` |
| `y_surf`, `y_avg` | `X-averaged positive particle surface stoichiometry`, `Average positive particle stoichiometry` |
| `eta_n`, `eta_p` | `X-averaged negative/positive electrode reaction overpotential [V]` |
| electrolyte | `Electrolyte concentration [mol.m-3]` |

---

## 11. Outputs and file formats

### 11.1 Result folders

| Folder | Files |
|---|---|
| `results/task2/` | voltage comparison, voltage error, metrics |
| `results/task3/` | `all_<test>.png`, `all_tests_charge_summary.csv`, `all_tests_current_summary.csv`, `run_config.json`, `hppc_residuals.png`, `hppc_residual_table.csv` |
| `results/task4/` | `optimized_parameters.json`, `optimizer_comparison.csv`, `optimizer_history.csv`, `convergence.png`, `parameter_comparison.png`, `fit_<test>.png`, `hppc_metrics.csv`, `hppc_check.png`, `hppc_zoom.png` |
| `results/task5/` | `validation_hppc_metrics.csv`, `validation_hppc_residuals.png`, `validation_hppc_pulse_zoom.png`, `training/…`, `anode_potential_charge.png`, `internal_states_charge_4C.png`, `electrolyte_charge_4C.png`, `internal_states_hppc.png`, `internal_states_summary.csv` |

### 11.2 `optimized_parameters.json`

This is the main deliverable of Task 4. Task 5 rebuilds the model from this file, so the validation always uses exactly the optimized parameters.

```json
{
  "best_method": "TRF",
  "objective_J_best": 0.0,
  "objective_J_start": 0.0,
  "optimized_parameters": { "D_n": 0.0, "D_p": 0.0, "k_j0_n": 0.0, "k_j0_p": 0.0, "R_contact": 0.0, "beta_p": 0.0 },
  "start_parameters":     { "...": "..." },
  "bounds":               { "D_n": [1e-15, 1e-12], "...": "..." },
  "fixed_parameters":     { "x100": 0.9082, "k_wall": 70.0, "...": "...", "y100 (derived)": 0.019 },
  "protocol":             { "v_cv": 3.6, "v_tol": 0.005, "i_threshold": 0.05, "period_s": 5 },
  "settings":             { "methods": ["TRF", "..."], "budget": 100, "fd_step": 0.02, "seed": 0,
                            "v_scale": 0.01, "i_scale": 0.1, "penalty": 10.0,
                            "train_tests": ["Charge_1C", "..."] },
  "all_methods":          { "TRF": { "J": 0.0, "evaluations": 0, "time_min": 0.0,
                                     "parameters": { "...": "..." }, "metrics": { "...": "..." } } },
  "hppc_check":           { "...": "..." }
}
```

### 11.3 CSV files

| File | One row per | Main columns |
|---|---|---|
| `optimizer_history.csv` | model evaluation | method, J, the six parameters, per-test RMSE |
| `optimizer_comparison.csv` | method (+ start) | family, best J, J/J_start, evaluations, runtime, parameters, per-test RMSE |
| `all_tests_charge_summary.csv` | charge test | RMSE V CC, mean CC offset, RMSE I CV, CV start (model/measured), charge accepted (model/measured) |
| `all_tests_current_summary.csv` | discharge/HPPC | simulated hours, stop reason, RMSE, MAE, max error, RMSE rest, RMSE current-on |
| `validation_hppc_metrics.csv` | parameter set | coverage, RMSE, MAE, max error, bias, RMSE rest/pulses, RMSE per SOC range, change vs baseline |
| `internal_states_summary.csv` | test | minimum anode potential (separator side and average), time below 0 V, min/max surface stoichiometries, max overpotentials |

---

## 12. Results summary

*Fill in after the final runs.*

| Parameter set | HPPC RMSE [mV] | RMSE rest [mV] | RMSE pulses [mV] | Coverage [%] |
|---|---|---|---|---|
| Prada2013 baseline | … | … | … | … |
| Manual tuning | … | … | … | … |
| Optimized (…) | … | … | … | … |

- **Best optimization method:** …, with J reduced from … to … (… %).
- **Minimum anode potential during the 4C charge:** … mV vs Li/Li⁺, i.e. [no plating risk / plating possible for … s].

---

## 13. Extending the code

**Optimize an additional parameter**

1. Make sure it is a `CellConfig` field and is used in `CellModel.parameter_values()`.
2. Add a `ParameterSpec` for it in the Task 4 settings cell, with physical bounds.
3. Add a start value to `start`.

**Add an optimization method**

1. Write a function `run_mymethod(obj, u0, budget, settings)` in `lfp_dfn/optim/methods.py`. It should call `obj.residuals(u)` (vector) or `obj(u)` (scalar J).
2. Register it in `METHODS` and `FAMILY`.
3. Add its name to `OptimizationSettings.methods`.

The budget, cache and log work automatically.

**Change the weighting between voltage and current**

Change `v_scale` / `i_scale` in `OptimizationSettings`. Only their ratio matters: a smaller `v_scale` makes the voltage fit more important.

**Use another training set**

Change `TRAIN_TESTS`. Files starting with `Charge` are simulated as CC-CV; all others follow the measured current.

---

## 14. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `module 'pybamm' has no attribute 'parameter_sets'` | an earlier `import pybamm` was interrupted, or packages were installed while the kernel was running | restart the kernel |
| `FileNotFoundError: Data file not found` | wrong `DATA_DIR` or file name | check the paths in Cell 1/2 |
| `KeyError: field 'time' not found` | the `.mat` file uses other field names | the error lists the available fields; rename them or extend `load_test()` |
| `ModuleNotFoundError: deap` | GA library missing | `pip install deap`, then restart the kernel |
| `NameError` in a notebook | cells run out of order | run the notebook from the top |
| Simulation stops early ("Minimum/Maximum voltage") | parameters outside a sensible range, or wrong start state | check `coverage` / `Stop reason`; check x100, y100 and the diffusivities |
| Very slow optimization | each evaluation solves five DFN simulations | test with `budget=10` first; run the full study overnight; reduce `methods` if needed |
| Memory problems during HPPC | long profile | the code already keeps only the needed output variables; close other programs |
| `R_contact` seems to have no effect | contact resistance is only enabled when `R_contact > 0` | normal at 0; any value > 0 switches the DFN option on |

---

## 15. Limitations and assumptions

- **Base parameter set.** Prada2013 describes an A123 26650 LFP/graphite cell with 2.3 Ah nominal capacity. The ANR26650M1B is the 2.5 Ah version. Chemistry and geometry are assumed to be the same; the capacity difference is handled by the stoichiometry window.
- **OCP curves.** The graphite curve (Chen2020) was measured on an LG M50 cell. Graphite OCP depends mainly on the material, so the shape is assumed to carry over. The LFP curve (Afshar2017) was adapted near full charge only.
- **Isothermal model.** The mean measured temperature is used for each test; self-heating at 3C–4C is not modelled.
- **No LFP hysteresis model.** A constant charge-branch offset was tested and not used in the final set.
- **Identifiability.** Terminal voltage alone cannot fully separate all parameters (for example k_j0_n vs k_j0_p). Similar objective values with different parameter combinations are possible.
- **Runtime.** A full optimization takes several hours, which limits the evaluation budget, especially for the global methods (GA, DE).

---

## 16. References

- Doyle, M., Fuller, T. F., Newman, J. (1993). Modeling of galvanostatic charge and discharge of the lithium/polymer/insertion cell. *J. Electrochem. Soc.*
- Prada, E. et al. (2013). A simplified electrochemical and thermal aging model of LiFePO₄–graphite Li-ion batteries. *J. Electrochem. Soc.*
- Chen, C.-H. et al. (2020). Development of experimental techniques for parameterization of multi-scale lithium-ion battery models. *J. Electrochem. Soc.*
- Sulzer, V. et al. (2021). Python Battery Mathematical Modelling (PyBaMM). *Journal of Open Research Software.*
- Forman, J. C. et al. (2012). Genetic identification and Fisher identifiability analysis of the Doyle–Fuller–Newman model from experimental cycling of a LiFePO₄ cell. *J. Power Sources.*
- Branch, M. A., Coleman, T. F., Li, Y. (1999). A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems. *SIAM J. Sci. Comput.* (TRF)
- Byrd, R. H. et al. (1995). A limited memory algorithm for bound constrained optimization. *SIAM J. Sci. Comput.* (L-BFGS-B)
- Nelder, J. A., Mead, R. (1965). A simplex method for function minimization. *The Computer Journal.*
- Storn, R., Price, K. (1997). Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces. *J. Global Optimization.*
- Fortin, F.-A. et al. (2012). DEAP: Evolutionary algorithms made easy. *J. Machine Learning Research.*
