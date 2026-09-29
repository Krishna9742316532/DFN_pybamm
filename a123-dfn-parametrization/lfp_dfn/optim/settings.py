"""Settings of the optimization study."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OptimizationSettings:
    methods: tuple[str, ...] = ("TRF", "L-BFGS-B", "Nelder-Mead", "GA", "DE", "DE->TRF")
    budget: int = 100          # model evaluations per method (1 evaluation = all training tests)
    fd_step: float = 0.02      # finite-difference step in normalized [0,1] space
    seed: int = 0
    v_scale: float = 0.010     # 10 mV  -> 1 unit of residual
    i_scale: float = 0.10      # 0.1 A  -> 1 unit of residual
    penalty: float = 10.0      # residual used where the model did not run

    def to_dict(self) -> dict:
        d = asdict(self)
        d["methods"] = list(self.methods)
        return d