"""Parameter optimization for the LFP DFN model (Task 4)."""
from .settings import OptimizationSettings
from .space import ParameterSpec, ParameterSpace
from .objective import BudgetExceeded, TrainingCase, build_training_set, CalibrationObjective
from .methods import METHODS, FAMILY
from .study import MethodRun, StudyResult, run_comparison, print_summary, save_study
from .plots import plot_convergence, plot_parameter_comparison, plot_training_fit
from .validation import run_hppc_check

__all__ = [
    "OptimizationSettings", "ParameterSpec", "ParameterSpace",
    "BudgetExceeded", "TrainingCase", "build_training_set", "CalibrationObjective",
    "METHODS", "FAMILY", "MethodRun", "StudyResult", "run_comparison", "print_summary", "save_study",
    "plot_convergence", "plot_parameter_comparison", "plot_training_fit", "run_hppc_check",
]