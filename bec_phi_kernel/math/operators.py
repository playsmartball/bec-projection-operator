from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # optional GPU
except Exception:  # pragma: no cover
    cp = None


def _xp(use_gpu: bool):
    if use_gpu and cp is not None:
        return cp
    return np


def dphi(phi, y, use_gpu: bool = False):
    """First derivative wrt phi using second-order finite differences.

    y' = d/dphi y
    """
    xp = _xp(use_gpu)
    phi = xp.asarray(phi, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    n = y.shape[0]
    dyd = xp.empty_like(y)
    # central differences interior
    dphi_c = phi[2:] - phi[:-2]
    dyd[1:-1] = (y[2:] - y[:-2]) / dphi_c
    # forward/backward at ends (second-order)
    dyd[0] = ( -3*y[0] + 4*y[1] - y[2] ) / (phi[2] - phi[0])
    dyd[-1] = ( 3*y[-1] - 4*y[-2] + y[-3] ) / (phi[-1] - phi[-3])
    return dyd


def d2phi(phi, y, use_gpu: bool = False):
    """Second derivative wrt phi using second-order finite differences.

    y'' = d^2/dphi^2 y
    """
    xp = _xp(use_gpu)
    phi = xp.asarray(phi, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    n = y.shape[0]
    ydd = xp.empty_like(y)
    # For nonuniform spacing, use three-point stencil with local spacings
    # interior points
    h_prev = phi[1:-1] - phi[:-2]
    h_next = phi[2:] - phi[1:-1]
    denom = 0.5 * h_prev * h_next * (h_prev + h_next)
    ydd[1:-1] = 2 * (
        (y[2:] - y[1:-1]) / (h_next * (h_prev + h_next))
        - (y[1:-1] - y[:-2]) / (h_prev * (h_prev + h_next))
    )
    # ends: use three-point one-sided approximations
    h0 = phi[1] - phi[0]
    h1 = phi[2] - phi[1]
    ydd[0] = 2 * (
        (y[2] - y[1]) / (h1 * (h0 + h1))
        - (y[1] - y[0]) / (h0 * (h0 + h1))
    )
    hm1 = phi[-1] - phi[-2]
    hm2 = phi[-2] - phi[-3]
    ydd[-1] = 2 * (
        (y[-1] - y[-2]) / (hm1 * (hm1 + hm2))
        - (y[-2] - y[-3]) / (hm2 * (hm1 + hm2))
    )
    return ydd


def L_function(phi, rho, P, use_gpu: bool = False):
    """Stability functional L(Φ) = ρ(Φ) − D_Φ^2 P(Φ) with α=β=1 by normalization."""
    xp = _xp(use_gpu)
    d2P = d2phi(phi, P, use_gpu=use_gpu)
    return xp.asarray(rho) - d2P
