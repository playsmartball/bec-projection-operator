#!/usr/bin/env python3
"""
Minimal 1D linear Alfvén solver with effective inertia.

Validates dispersion: omega = k * B0 / sqrt(1 + rho_eff)

Usage:
  python examples/alfven_1d.py --rho 20 --k 1 --N 128 --L 6.283185307179586 --B0 1.0 --tmax 400.0 --dt 0.01

All parameters have sensible defaults; run without args to use rho_eff=20, etc.
"""
import argparse
import numpy as np


def dz_central(f: np.ndarray, dz: float) -> np.ndarray:
    """Periodic central difference in 1D."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dz)


def run_solver(L=2*np.pi, N=128, B0=1.0, rho_eff=20.0, k=1, tmax=400.0, dt=0.01, amp0=1e-6):
    z = np.linspace(0.0, L, N, endpoint=False)
    dz = L / N
    k_phys = 2.0 * np.pi * k / L

    # Initial condition: transverse velocity; magnetic perturbation zero
    vy = amp0 * np.sin(k_phys * z)
    By = np.zeros_like(vy)

    # Time integration (RK4)
    t = 0.0
    times = []
    s_sin_series = []  # scalar projection time series using sin basis
    s_cos_series = []  # scalar projection time series using cos basis

    def rhs(vy_arr: np.ndarray, By_arr: np.ndarray):
        dvy = (B0 / (1.0 + rho_eff)) * dz_central(By_arr, dz)
        dBy = B0 * dz_central(vy_arr, dz)
        return dvy, dBy

    # Precompute mode bases for projection
    sin_b = np.sin(k_phys * z)
    cos_b = np.cos(k_phys * z)

    while t < tmax:
        k1_v, k1_B = rhs(vy, By)
        k2_v, k2_B = rhs(vy + 0.5 * dt * k1_v, By + 0.5 * dt * k1_B)
        k3_v, k3_B = rhs(vy + 0.5 * dt * k2_v, By + 0.5 * dt * k2_B)
        k4_v, k4_B = rhs(vy + dt * k3_v, By + dt * k3_B)

        vy = vy + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        By = By + (dt / 6.0) * (k1_B + 2.0 * k2_B + 2.0 * k3_B + k4_B)

        # Scalar projections of By on sin/cos for robust Hilbert-phase extraction
        s_sin = np.sum(By * sin_b) / N
        s_cos = np.sum(By * cos_b) / N
        times.append(t)
        s_sin_series.append(s_sin)
        s_cos_series.append(s_cos)

        t += dt

    times = np.asarray(times)
    s_sin_series = np.asarray(s_sin_series)
    s_cos_series = np.asarray(s_cos_series)

    # Choose the projection with higher power
    pw_sin = float(np.mean(s_sin_series * s_sin_series))
    pw_cos = float(np.mean(s_cos_series * s_cos_series))
    s = s_sin_series if pw_sin >= pw_cos else s_cos_series

    # Hilbert transform via FFT to get analytic signal and instantaneous phase
    Nt = len(s)
    S = np.fft.fft(s)
    H = np.zeros(Nt)
    if Nt % 2 == 0:
        H[0] = 1.0
        H[1:Nt//2] = 2.0
        H[Nt//2] = 1.0
    else:
        H[0] = 1.0
        H[1:(Nt+1)//2] = 2.0
    a = np.fft.ifft(S * H)
    phi = np.unwrap(np.angle(a))

    m, b = np.polyfit(times, phi, 1)
    omega_num = float(abs(m))

    omega_th = k_phys * B0 / np.sqrt(1.0 + rho_eff)

    return omega_num, omega_th


def main():
    p = argparse.ArgumentParser(description="Minimal 1D linear Alfvén solver with effective inertia")
    p.add_argument("--L", type=float, default=2*np.pi, help="Domain length")
    p.add_argument("--N", type=int, default=128, help="Grid points")
    p.add_argument("--B0", type=float, default=1.0, help="Background B0 magnitude")
    p.add_argument("--rho", type=float, default=20.0, help="Effective inertia rho_eff (dimensionless)")
    p.add_argument("--k", type=int, default=1, help="Mode index (integer Fourier mode)")
    p.add_argument("--tmax", type=float, default=400.0, help="Final time")
    p.add_argument("--dt", type=float, default=0.01, help="Time step")
    p.add_argument("--amp0", type=float, default=1e-6, help="Initial velocity amplitude")
    args = p.parse_args()

    omega_num, omega_th = run_solver(L=args.L, N=args.N, B0=args.B0, rho_eff=args.rho,
                                     k=args.k, tmax=args.tmax, dt=args.dt, amp0=args.amp0)

    ratio = omega_th / omega_num if omega_num != 0 else np.nan
    sqrt_factor = np.sqrt(1.0 + args.rho)

    print(f"Numerical omega = {omega_num:.6e}")
    print(f"Theory    omega = {omega_th:.6e}")
    print(f"sqrt(1+rho_eff) = {sqrt_factor:.6f}")
    print(f"Theory/Numerical omega ratio = {ratio:.6f}")


if __name__ == "__main__":
    main()
