from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # optional GPU
except Exception:  # pragma: no cover
    cp = None


def _ensure_backend(xp):
    if xp is None:
        return np
    return xp


def phi_state(phi, xp=None):
    """Compute the Phi-driven state vector S(Φ) deterministically.

    S(Φ) = [rho(Φ), P(Φ), c_s(Φ), lambda_c(Φ), epsilon(Φ)]

    Mapping (dimensionless, Planck-normalized):
      - rho(Φ)     = 1 / (1 + Φ)
      - P(Φ)       = 1 / (1 + Φ)
      - c_s(Φ)     = 1 / (1 + Φ)
      - lambda_c   = 1 + Φ
      - epsilon(Φ) = Φ / (1 + Φ)

    xp may be numpy or cupy; defaults to numpy.
    """
    xp = _ensure_backend(xp)
    phi = xp.asarray(phi, dtype=xp.float64)

    one = xp.array(1.0, dtype=xp.float64)
    denom = one + phi
    inv = one / denom

    rho = inv
    P = inv
    c_s = inv
    lambda_c = denom
    epsilon = phi * inv

    return {
        "phi": phi,
        "rho": rho,
        "P": P,
        "c_s": c_s,
        "lambda_c": lambda_c,
        "epsilon": epsilon,
    }
