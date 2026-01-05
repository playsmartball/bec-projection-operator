#!/usr/bin/env python3
"""
Dedalus 1D linear Alfvén solver with explicit effective inertia.

Matches the minimal reference solver and validates the dispersion:
    omega = k * B0 / sqrt(1 + rho_eff)

Usage example:
  python examples/dedalus_alfven_1d.py --rho 20 --k 1 --N 128 --L 128.0 --B0 1.0 --tmax 1200.0 --dt 0.01

Note: Requires Dedalus (https://dedalus-project.org/). Install with pip or conda.
"""
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core import timesteppers as ts
except Exception as e:
    sys.stderr.write("Dedalus is required to run this script. Install via pip/conda.\n")
    raise


def dz_project_series(field_g: np.ndarray, z: np.ndarray, k_phys: float):
    """Return scalar projection of field onto sin/cos(k z)."""
    sin_b = np.sin(k_phys * z)
    cos_b = np.cos(k_phys * z)
    s_sin = np.mean(field_g * sin_b)
    s_cos = np.mean(field_g * cos_b)
    return s_sin, s_cos


def hilbert_phase_slope(times: np.ndarray, series: np.ndarray) -> float:
    """Compute frequency from unwrapped phase slope of analytic signal."""
    N = len(series)
    S = np.fft.fft(series)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[1:N//2] = 2.0
        H[N//2] = 1.0
    else:
        H[0] = 1.0
        H[1:(N+1)//2] = 2.0
    a = np.fft.ifft(S * H)
    phi = np.unwrap(np.angle(a))
    m, b = np.polyfit(times, phi, 1)
    return float(abs(m))


def run_dedalus(L=2*np.pi, N=128, B0=1.0, rho_eff=20.0, k=1, tmax=400.0, dt_step=0.01, amp0=1e-6, tau=1.0):
    # Coordinates, distributor, basis (Dedalus v3 API)
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.float64)
    zbasis = de.RealFourier(coords['z'], size=N, bounds=(0.0, L))

    # Fields
    v = dist.Field(name='v', bases=zbasis)
    b = dist.Field(name='b', bases=zbasis)

    # Problem
    def d_z(f):
        return de.Differentiate(f, coords['z'])

    alpha = B0 / (1.0 + rho_eff)
    problem = de.IVP([v, b], namespace=locals())
    problem.add_equation("dt(v) - alpha*d_z(b) = 0")
    problem.add_equation("dt(b) - tau*B0*d_z(v) = 0")

    solver = problem.build_solver(ts.RK443)
    solver.stop_sim_time = tmax

    # Initial conditions
    z = dist.local_grid(zbasis)
    k_phys = 2.0 * np.pi * k / L
    v['g'] = amp0 * np.sin(k_phys * z)
    b['g'] = 0.0

    # Time stepping & diagnostics
    times = []
    s_sin_list = []
    s_cos_list = []

    while solver.proceed:
        solver.step(dt_step)
        ss, sc = dz_project_series(b['g'], z, k_phys)
        times.append(solver.sim_time)
        s_sin_list.append(ss)
        s_cos_list.append(sc)

    times = np.asarray(times)
    s_sin = np.asarray(s_sin_list)
    s_cos = np.asarray(s_cos_list)

    # Choose stronger projection
    pw_sin = float(np.mean(s_sin * s_sin))
    pw_cos = float(np.mean(s_cos * s_cos))
    series = s_sin if pw_sin >= pw_cos else s_cos

    omega_num = hilbert_phase_slope(times, series)
    omega_th = k_phys * B0 * np.sqrt(tau) / np.sqrt(1.0 + rho_eff)
    return omega_num, omega_th


def main():
    p = argparse.ArgumentParser(description="Dedalus 1D linear Alfvén solver with effective inertia")
    p.add_argument("--L", type=float, default=2*np.pi, help="Domain length")
    p.add_argument("--N", type=int, default=128, help="Grid points")
    p.add_argument("--B0", type=float, default=1.0, help="Background B0 magnitude")
    p.add_argument("--rho", type=float, default=20.0, help="Effective inertia rho_eff (dimensionless)")
    p.add_argument("--k", type=int, default=1, help="Mode index (integer Fourier mode)")
    p.add_argument("--tmax", type=float, default=400.0, help="Final time")
    p.add_argument("--dt", type=float, default=0.01, help="Time step")
    p.add_argument("--amp0", type=float, default=1e-6, help="Initial velocity amplitude")
    p.add_argument("--tau", type=float, default=1.0, help="Tension coefficient multiplier (affects induction)")
    args = p.parse_args()

    omega_num, omega_th = run_dedalus(L=args.L, N=args.N, B0=args.B0, rho_eff=args.rho,
                                      k=args.k, tmax=args.tmax, dt_step=args.dt, amp0=args.amp0, tau=args.tau)

    ratio = omega_th / omega_num if omega_num != 0 else np.nan
    sqrt_factor = np.sqrt(1.0 + args.rho)

    print(f"Numerical omega = {omega_num:.6e}")
    print(f"Theory    omega = {omega_th:.6e}")
    print(f"sqrt(1+rho_eff) = {sqrt_factor:.6f}")
    print(f"Theory/Numerical omega ratio = {ratio:.6f}")


if __name__ == "__main__":
    main()
