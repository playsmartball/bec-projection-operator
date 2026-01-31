#!/usr/bin/env python3
"""
Dedalus 1D linear Alfven solver with line-tied boundaries (Chebyshev basis).

Implements Dirichlet BCs: v(z=0)=0, v(z=L)=0.
Equations:
    dt(v) = (B0/(1+rho_eff)) * dz(b)
    dt(b) = tau*B0*dz(v) + eta*dzz(b)

Discrete eigenmodes under line-tying: v_n(z) ~ sin(n*pi*z/L)
Dispersion: omega_n = (n*pi/L) * B0 * sqrt(tau) / sqrt(1+rho_eff)
Damping:    gamma_n = -(eta/2) * (n*pi/L)^2

Usage example:
  python examples/dedalus_alfven_1d_linetied.py --rho 0 --n 1 --N 129 --L 128.0 --B0 1.0 --tmax 2000 --dt 0.01
"""
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core import timesteppers as ts
    from dedalus.core.operators import Lift as LiftOp
except Exception as e:
    sys.stderr.write("Dedalus is required to run this script. Install via pip/conda.\n")
    raise


def dz_project_series(field_g: np.ndarray, z: np.ndarray, k_phys: float):
    sin_b = np.sin(k_phys * z)
    cos_b = np.cos(k_phys * z)
    s_sin = np.mean(field_g * sin_b)
    s_cos = np.mean(field_g * cos_b)
    return s_sin, s_cos


def hilbert_phase_slope(times: np.ndarray, series: np.ndarray) -> float:
    times = np.asarray(times)
    x = series - np.mean(series)
    N = len(x)
    X = np.fft.fft(x)
    H = np.zeros(N)
    if N % 2 == 0:
        H[1:N//2] = 2.0
        H[N//2] = 1.0
    else:
        H[0] = 1.0
        H[1:(N+1)//2] = 2.0
    a = np.fft.ifft(X * H)
    phi = np.unwrap(np.angle(a))
    m, _ = np.polyfit(times, phi, 1)
    return float(abs(m))


def run_dedalus_linetied(L=2*np.pi, N=129, B0=1.0, rho_eff=0.0, n=1, tmax=400.0, dt_step=0.01, amp0=1e-6, tau=1.0, eta=0.0):
    # Coordinates, distributor, basis (Dedalus v3 API)
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.float64)
    zbasis = de.Chebyshev(coords['z'], size=N, bounds=(0.0, L))

    # Fields
    v = dist.Field(name='v', bases=zbasis)  # velocity
    b = dist.Field(name='b', bases=zbasis)  # magnetic perturbation
    w = dist.Field(name='w', bases=zbasis)  # aux: w = dz(v)
    # Tau fields for boundary conditions (scalars, no bases)
    tau_v1 = dist.Field(name='tau_v1')
    tau_v2 = dist.Field(name='tau_v2')

    # Operators
    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])
    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    # Parameters
    alpha = B0 / (1.0 + rho_eff)

    # Problem and BCs (line-tied: v=0 at both ends). First-order reduction with w = dz(v)
    problem = de.IVP([v, b, w, tau_v1, tau_v2], namespace=locals())
    problem.add_equation("dt(v) - alpha*d_z(b) + lift(tau_v1, -1) + lift(tau_v2, -2) = 0")
    problem.add_equation("dt(b) - tau*B0*w - eta*d_zz(b) = 0")
    problem.add_equation("w - d_z(v) = 0")
    problem.add_equation("v(z='left') = 0")
    problem.add_equation("v(z='right') = 0")

    solver = problem.build_solver(ts.RK443)
    solver.stop_sim_time = tmax

    # Initial condition: excite n-th line-tied eigenmode
    z = dist.local_grid(zbasis)
    k_phys = n * np.pi / L
    v['g'] = amp0 * np.sin(k_phys * z)
    b['g'] = 0.0
    w['g'] = amp0 * k_phys * np.cos(k_phys * z)

    # Time stepping and diagnostics
    times, s_sin_list, s_cos_list, energy_list = [], [], [], []
    while solver.proceed:
        solver.step(dt_step)
        ss, sc = dz_project_series(b['g'], z, k_phys)
        times.append(solver.sim_time)
        s_sin_list.append(ss)
        s_cos_list.append(sc)
        e_inst = 0.5 * (np.mean(v['g']*v['g']) + np.mean(b['g']*b['g']))
        energy_list.append(float(e_inst))

    times = np.asarray(times)
    s_sin = np.asarray(s_sin_list)
    s_cos = np.asarray(s_cos_list)

    # Choose stronger projection
    pw_sin = float(np.mean(s_sin*s_sin))
    pw_cos = float(np.mean(s_cos*s_cos))
    series = s_sin if pw_sin >= pw_cos else s_cos

    omega_num = hilbert_phase_slope(times, series)
    omega_th = k_phys * B0 * np.sqrt(tau) / np.sqrt(1.0 + rho_eff)

    # Damping via energy envelope (skip transients)
    E = np.asarray(energy_list)
    eps = 1e-300
    lnE = np.log(np.maximum(E, eps))
    if omega_num > 0:
        T_est = 2.0 * np.pi / omega_num
        mask = times > (5.0 * T_est)
        if np.count_nonzero(mask) < 10:
            mask = slice(None)
    else:
        mask = slice(None)
    m, _ = np.polyfit(times[mask], lnE[mask], 1)
    gamma_num = float(0.5 * m)
    gamma_th = -0.5 * eta * (k_phys**2)

    return omega_num, omega_th, gamma_num, gamma_th


def main():
    p = argparse.ArgumentParser(description="Dedalus 1D line-tied Alfven solver (Chebyshev)")
    p.add_argument("--L", type=float, default=2*np.pi, help="Domain length")
    p.add_argument("--N", type=int, default=129, help="Chebyshev points (odd preferred)")
    p.add_argument("--B0", type=float, default=1.0, help="Background B0 magnitude")
    p.add_argument("--rho", type=float, default=0.0, help="Effective inertia rho_eff")
    p.add_argument("--n", type=int, default=1, help="Line-tied mode index n (k_n = n*pi/L)")
    p.add_argument("--tmax", type=float, default=400.0, help="Final time")
    p.add_argument("--dt", type=float, default=0.01, help="Time step")
    p.add_argument("--amp0", type=float, default=1e-6, help="Initial velocity amplitude")
    p.add_argument("--tau", type=float, default=1.0, help="Tension coefficient multiplier")
    p.add_argument("--eta", type=float, default=0.0, help="Resistivity")
    args = p.parse_args()

    omega_num, omega_th, gamma_num, gamma_th = run_dedalus_linetied(
        L=args.L, N=args.N, B0=args.B0, rho_eff=args.rho, n=args.n,
        tmax=args.tmax, dt_step=args.dt, amp0=args.amp0, tau=args.tau, eta=args.eta
    )

    ratio_w = omega_th / omega_num if omega_num != 0 else np.nan
    print(f"Numerical omega = {omega_num:.6e}")
    print(f"Theory    omega = {omega_th:.6e}")
    print(f"Numerical gamma = {gamma_num:.6e}")
    print(f"Theory    gamma = {gamma_th:.6e}")
    print(f"Theory/Numerical omega ratio = {ratio_w:.6f}")


if __name__ == "__main__":
    main()
