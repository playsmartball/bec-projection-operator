from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # optional GPU
except Exception:  # pragma: no cover
    cp = None

from bec_phi_kernel.math.phi_state import phi_state
from bec_phi_kernel.math.operators import L_function


def solve(phi_start: float, phi_end: float, n: int):
    """Deterministic GPU solver using CuPy (falls back to CPU if CuPy missing).

    Returns numpy arrays and negative L intervals, same contract as solver_cpu.
    """
    use_gpu = False
    if cp is not None:
        try:  # Ensure CUDA device and NVRTC are available before using CuPy
            from cupy.cuda import runtime as _runtime  # type: ignore
            from cupy_backends.cuda.libs import nvrtc as _nvrtc  # type: ignore
            _ = _nvrtc.getVersion()
            if _runtime.getDeviceCount() > 0:
                use_gpu = True
        except Exception:
            use_gpu = False
    xp = cp if use_gpu else np

    phi = xp.linspace(phi_start, phi_end, int(n), dtype=xp.float64)
    st = phi_state(phi, xp=xp)
    L = L_function(st["phi"], st["rho"], st["P"], use_gpu=use_gpu)

    # Move to numpy for output
    if use_gpu:
        phi_np = cp.asnumpy(st["phi"])  # type: ignore[attr-defined]
        rho_np = cp.asnumpy(st["rho"])  # type: ignore[attr-defined]
        P_np = cp.asnumpy(st["P"])      # type: ignore[attr-defined]
        cs_np = cp.asnumpy(st["c_s"])   # type: ignore[attr-defined]
        lc_np = cp.asnumpy(st["lambda_c"])  # type: ignore[attr-defined]
        eps_np = cp.asnumpy(st["epsilon"])  # type: ignore[attr-defined]
        L_np = cp.asnumpy(L)  # type: ignore[attr-defined]
    else:
        phi_np = np.asarray(st["phi"], dtype=np.float64)
        rho_np = np.asarray(st["rho"], dtype=np.float64)
        P_np = np.asarray(st["P"], dtype=np.float64)
        cs_np = np.asarray(st["c_s"], dtype=np.float64)
        lc_np = np.asarray(st["lambda_c"], dtype=np.float64)
        eps_np = np.asarray(st["epsilon"], dtype=np.float64)
        L_np = np.asarray(L, dtype=np.float64)

    neg = L_np < 0
    intervals = []
    if np.any(neg):
        idx = np.where(neg)[0]
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            intervals.append((float(phi_np[start]), float(phi_np[prev])))
            start = k
            prev = k
        intervals.append((float(phi_np[start]), float(phi_np[prev])))

    st_out = {
        "phi": phi_np,
        "rho": rho_np,
        "P": P_np,
        "c_s": cs_np,
        "lambda_c": lc_np,
        "epsilon": eps_np,
        "L": L_np,
    }
    return st_out, intervals
