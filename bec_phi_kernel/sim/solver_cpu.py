from __future__ import annotations

import numpy as np

from bec_phi_kernel.math.phi_state import phi_state
from bec_phi_kernel.math.operators import L_function


def solve(phi_start: float, phi_end: float, n: int):
    """Deterministic CPU solver for FMI v0.1.

    Returns a dict with numpy arrays for phi, state components, and L(φ), and
    a list of (phi_lo, phi_hi) intervals where L < 0.
    """
    phi = np.linspace(phi_start, phi_end, int(n), dtype=np.float64)
    st = phi_state(phi, xp=np)
    L = L_function(st["phi"], st["rho"], st["P"], use_gpu=False)

    # Identify contiguous intervals where L < 0
    neg = L < 0
    intervals = []
    if np.any(neg):
        idx = np.where(neg)[0]
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            intervals.append((float(phi[start]), float(phi[prev])))
            start = k
            prev = k
        intervals.append((float(phi[start]), float(phi[prev])))

    st_out = {
        "phi": np.asarray(st["phi"], dtype=np.float64),
        "rho": np.asarray(st["rho"], dtype=np.float64),
        "P": np.asarray(st["P"], dtype=np.float64),
        "c_s": np.asarray(st["c_s"], dtype=np.float64),
        "lambda_c": np.asarray(st["lambda_c"], dtype=np.float64),
        "epsilon": np.asarray(st["epsilon"], dtype=np.float64),
        "L": np.asarray(L, dtype=np.float64),
    }
    return st_out, intervals
