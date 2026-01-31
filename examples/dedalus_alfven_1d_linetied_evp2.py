#!/usr/bin/env python3
"""
Dedalus 1D linear Alfven EVP with line-tied boundaries (Chebyshev basis).

Eigenproblem for fields (v, b):
    lam * v - alpha * dz(b) + lift(tau1, -1) + lift(tau2, -2) = 0
    lam * b - tau*B0 * dz(v) - eta * dzz(b) = 0
BCs:
    v(z='left') = 0
    v(z='right') = 0

For eta=0, analytic discrete spectrum:
    omega_n = (n*pi/L) * B0 * sqrt(tau) / sqrt(1 + rho_eff)
    gamma_n = 0
Small eta > 0:
    gamma_n ≈ -0.5 * eta * (n*pi/L)^2

Usage:
  python examples/dedalus_alfven_1d_linetied_evp2.py --rho 0 --n 1 --N 129 --L 128.0 --B0 1.0 --tau 1.0 --eta 0.0
"""
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core.operators import Lift as LiftOp
except Exception as e:
    sys.stderr.write("Dedalus is required to run this script. Install via pip/conda.\n")
    raise


def run_dedalus_linetied_evp(L=2*np.pi, N=129, B0=1.0, rho_eff=0.0, n=1, tau=1.0, eta=0.0):
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.complex128)
    zbasis = de.Chebyshev(coords['z'], size=N, bounds=(0.0, L))

    v = dist.Field(name='v', bases=zbasis)
    b = dist.Field(name='b', bases=zbasis)
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    lam = dist.Field(name='lam')  # eigenvalue placeholder (scalar, no basis)

    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])
    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    alpha = B0 / (1.0 + rho_eff)

    problem = de.EVP([v, b, tau1, tau2], eigenvalue=lam, namespace=locals())
    problem.add_equation("lam*v - alpha*d_z(b) + lift(tau1, -1) + lift(tau2, -2) = 0")
    problem.add_equation("lam*b - tau*B0*d_z(v) - eta*d_zz(b) = 0")
    problem.add_equation("v(z='left') = 0")
    problem.add_equation("v(z='right') = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    evals = solver.eigenvalues

    k_th = n * np.pi / L
    lam_th = (-0.5 * eta * (k_th**2)) + 1j * (k_th * B0 * np.sqrt(tau) / np.sqrt(1.0 + rho_eff))

    idx = int(np.argmin(np.abs(evals - lam_th)))
    lam = evals[idx]

    omega_num = float(np.imag(lam))
    gamma_num = float(np.real(lam))
    omega_th = float(np.imag(lam_th))
    gamma_th = float(np.real(lam_th))

    return omega_num, omega_th, gamma_num, gamma_th


def main():
    p = argparse.ArgumentParser(description="Dedalus 1D line-tied Alfven EVP (Chebyshev)")
    p.add_argument("--L", type=float, default=2*np.pi, help="Domain length")
    p.add_argument("--N", type=int, default=129, help="Chebyshev points (odd preferred)")
    p.add_argument("--B0", type=float, default=1.0, help="Background B0 magnitude")
    p.add_argument("--rho", type=float, default=0.0, help="Effective inertia rho_eff")
    p.add_argument("--n", type=int, default=1, help="Mode index n (k_n = n*pi/L)")
    p.add_argument("--tau", type=float, default=1.0, help="Tension coefficient multiplier")
    p.add_argument("--eta", type=float, default=0.0, help="Resistivity")
    args = p.parse_args()

    omega_num, omega_th, gamma_num, gamma_th = run_dedalus_linetied_evp(
        L=args.L, N=args.N, B0=args.B0, rho_eff=args.rho, n=args.n, tau=args.tau, eta=args.eta
    )

    ratio_w = omega_th / omega_num if omega_num != 0 else np.nan
    print(f"Numerical omega = {omega_num:.6e}")
    print(f"Theory    omega = {omega_th:.6e}")
    print(f"Numerical gamma = {gamma_num:.6e}")
    print(f"Theory    gamma = {gamma_th:.6e}")
    print(f"Theory/Numerical omega ratio = {ratio_w:.6f}")

if __name__ == "__main__":
    main()
