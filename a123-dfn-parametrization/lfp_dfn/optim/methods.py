"""The six optimization methods. All share: objective, start u0, bounds [0,1]^n, budget, seed."""
from __future__ import annotations

import random

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

from .objective import BudgetExceeded, CalibrationObjective
from .settings import OptimizationSettings


def run_trf(obj: CalibrationObjective, u0: np.ndarray, budget: int, s: OptimizationSettings) -> None:
    """Trust Region Reflective (gradient-based, least squares)."""
    least_squares(obj.residuals, u0, bounds=(0, 1), method="trf",
                  diff_step=s.fd_step, max_nfev=max(1, budget))


def run_lbfgsb(obj: CalibrationObjective, u0: np.ndarray, budget: int, s: OptimizationSettings) -> None:
    """L-BFGS-B (gradient-based quasi-Newton, scalar objective)."""
    minimize(obj, u0, method="L-BFGS-B", bounds=[(0, 1)] * len(u0),
             options={"maxfun": budget, "maxiter": budget, "eps": s.fd_step})


def run_nelder_mead(obj: CalibrationObjective, u0: np.ndarray, budget: int, s: OptimizationSettings,
                    step: float = 0.1) -> None:
    """Nelder-Mead simplex (gradient-free, local)."""
    n = len(u0)
    d = np.where(u0 + step <= 1, step, -step)
    simplex = np.vstack([u0] + [u0 + d[i] * np.eye(n)[i] for i in range(n)])
    minimize(obj, u0, method="Nelder-Mead", bounds=[(0, 1)] * n,
             options={"maxfev": budget, "initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-6})


def run_de(obj: CalibrationObjective, u0: np.ndarray, budget: int, s: OptimizationSettings,
           popsize: int = 4) -> None:
    """Differential Evolution (gradient-free, global). Stopped by the evaluation budget."""
    differential_evolution(obj, bounds=[(0, 1)] * len(u0), x0=u0, popsize=popsize, maxiter=1000,
                           tol=0, atol=0, polish=False, init="latinhypercube",
                           mutation=(0.5, 1.0), recombination=0.7, updating="immediate", seed=s.seed)


def run_ga(obj: CalibrationObjective, u0: np.ndarray, budget: int, s: OptimizationSettings,
           pop_size: int = 12, cxpb: float = 0.7, mutpb: float = 0.3,
           sigma: float = 0.1, indpb: float = 0.3) -> None:
    """Genetic Algorithm with DEAP (gradient-free, global). Stopped by the evaluation budget."""
    from deap import base, creator, tools                      # pip install deap

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    random.seed(s.seed)
    tb = base.Toolbox()
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=sigma, indpb=indpb)
    tb.register("select", tools.selTournament, tournsize=3)

    n = len(u0)

    def evaluate(ind):
        ind[:] = [min(max(v, 0.0), 1.0) for v in ind]
        return (obj(np.array(ind)),)

    popu = [creator.Individual(list(u0))] + \
           [creator.Individual([random.random() for _ in range(n)]) for _ in range(pop_size - 1)]
    for ind in popu:
        ind.fitness.values = evaluate(ind)
    hof = tools.HallOfFame(1)
    hof.update(popu)

    for _ in range(10 * budget):                              # in practice stopped by the budget
        offspring = list(map(tb.clone, tb.select(popu, len(popu))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < mutpb:
                tb.mutate(mut)
                del mut.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)
        offspring[0] = tb.clone(hof[0])                       # elitism: keep the best one
        popu[:] = offspring
        hof.update(popu)


def run_de_trf(obj: CalibrationObjective, u0: np.ndarray, budget: int, s: OptimizationSettings,
               de_fraction: float = 0.7) -> None:
    """Hybrid: DE explores (70 % of budget), then TRF refines from the best DE point."""
    n_start = obj.n_evals
    n_de = int(de_fraction * budget)
    obj.max_evals = n_start + n_de
    try:
        run_de(obj, u0, n_de, s)
    except BudgetExceeded:
        pass
    obj.max_evals = n_start + budget
    best = min(obj.log[n_start:], key=lambda e: e["J"])
    if obj.verbose:
        print("  -> switching to TRF from the best DE point")
    run_trf(obj, obj.space.to_unit(best), budget - (obj.n_evals - n_start), s)


METHODS = {"TRF": run_trf, "L-BFGS-B": run_lbfgsb, "Nelder-Mead": run_nelder_mead,
           "GA": run_ga, "DE": run_de, "DE->TRF": run_de_trf}

FAMILY = {"TRF": "gradient-based (local)", "L-BFGS-B": "gradient-based (local)",
          "Nelder-Mead": "gradient-free (local)", "GA": "gradient-free (global)",
          "DE": "gradient-free (global)", "DE->TRF": "hybrid (global + gradient)"}