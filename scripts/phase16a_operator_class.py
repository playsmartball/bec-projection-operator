from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Union

Form = Literal["soft_cutoff", "exponential"]


@dataclass(frozen=True)
class KOperator:
    """
    Projection/boundary operator in k-space:
        P_obs(k) = W(k) * P(k)

    Constraints (by construction):
    - W(k -> 0) = 1
    - W'(k) <= 0 for k >= 0 (monotonic suppression)
    - Single characteristic scale kc
    - No redshift/time dependence

    Supported forms:
    - soft_cutoff:     W(k) = 1 / (1 + (k/kc)**n)
    - exponential:     W(k) = exp(-(k/kc)**n)

    with n in {1, 2}.
    """

    form: Form
    kc: float
    n: int

    def __post_init__(self) -> None:
        if self.form not in ("soft_cutoff", "exponential"):
            raise ValueError("form must be 'soft_cutoff' or 'exponential'")
        if not np.isfinite(self.kc) or self.kc <= 0:
            raise ValueError("kc must be a positive finite number")
        if self.n not in (1, 2):
            raise ValueError("n must be 1 or 2")

    def W(self, k: Union[float, np.ndarray]) -> np.ndarray:
        k_arr = np.asarray(k, dtype=float)
        x = np.clip(k_arr / self.kc, a_min=0.0, a_max=None)
        if self.form == "soft_cutoff":
            w = 1.0 / (1.0 + np.power(x, self.n))
        else:  # exponential
            w = np.exp(-np.power(x, self.n))
        # Enforce exact W(0)=1 for any tiny numerical residual
        if np.isscalar(k) or k_arr.shape == ():
            if k_arr == 0.0:
                return np.array(1.0)
        else:
            w = w.copy()
            w[k_arr == 0.0] = 1.0
        return w

    def apply(self, k: Union[float, np.ndarray], Pk: Union[float, np.ndarray]) -> np.ndarray:
        """Apply P_obs(k) = W(k) * P(k)."""
        return self.W(k) * np.asarray(Pk, dtype=float)

    def is_monotone_nonincreasing(self, kmin: float = 0.0, kmax: float = 10.0, num: int = 1001) -> bool:
        """Numerical check that W'(k) <= 0 over [kmin, kmax]."""
        if kmin < 0 or kmax <= kmin:
            raise ValueError("Require 0 <= kmin < kmax")
        ks = np.linspace(kmin, kmax, num=num)
        ws = self.W(ks)
        return np.all(np.diff(ws) <= 1e-12)  # allow tiny numerical noise


__all__ = ["KOperator"]
