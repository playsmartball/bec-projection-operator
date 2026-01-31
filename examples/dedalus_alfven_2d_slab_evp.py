#!/usr/bin/env python3
"""
Dedalus 2D slab linear Alfven EVP (parameterized in kx): Chebyshev in z with tau BCs, kx treated as a constant parameter.

Equations (complex EVP):
  lam * v - alpha * dz(b) + lift(tau1, -1) + lift(tau2, -2) = 0
  lam * b - tau*B0 * dz(v) - eta * (dzz(b) - kx**2 * b) = 0   # lap(b) = dzz - kx^2
BCs:
  v(z='left') = 0
  v(z='right') = 0

Theory (shear-Alfvén in uniform slab):
  kx = 2*pi*m/Lx,   kz = n*pi/Lz
  omega = sqrt(tau) * B0/sqrt(1+rho_eff) * |kz|
  gamma = -0.5 * eta * (kx^2 + kz^2)
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


def run_evp_2d(Lx=2*np.pi, Lz=2*np.pi, Nx=64, Nz=129, B0=1.0, rho_eff=0.0, tau=1.0, eta=0.0, m=0, n=1):
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.complex128)
    zbasis = de.Chebyshev(coords['z'], size=Nz, bounds=(0.0, Lz))

    v = dist.Field(name='v', bases=(zbasis,))
    b = dist.Field(name='b', bases=(zbasis,))
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    lam = dist.Field(name='lam')

    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])
    kx = 2.0 * np.pi * m / Lx
    def lap_param(f):
        return d_zz(f) - (kx**2) * f

    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    alpha = B0 / (1.0 + rho_eff)

    problem = de.EVP([v, b, tau1, tau2], eigenvalue=lam, namespace=locals())
    problem.add_equation("lam*v - alpha*d_z(b) + lift(tau1, -1) + lift(tau2, -2) = 0")
    problem.add_equation("lam*b - tau*B0*d_z(v) - eta*lap_param(b) = 0")
    problem.add_equation("v(z='left') = 0")
    problem.add_equation("v(z='right') = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    evals = solver.eigenvalues

    kz = np.pi * n / Lz
    # Shear Alfvén in this reduced model depends only on k_parallel = |kz| for frequency,
    # and resistive damping uses total k^2 = kx^2 + kz^2.
    lam_th = (-0.5 * eta * (kx**2 + kz**2)) + 1j * (np.sqrt(tau) * B0 / np.sqrt(1.0 + rho_eff) * np.abs(kz))

    idx = int(np.argmin(np.abs(evals - lam_th)))
    lam_sel = evals[idx]

    omega_num = float(np.imag(lam_sel))
    gamma_num = float(np.real(lam_sel))
    omega_th = float(np.imag(lam_th))
    gamma_th = float(np.real(lam_th))

    return {
        'm': m,
        'n': n,
        'kx': float(kx),
        'kz': float(kz),
        'omega_num': omega_num,
        'omega_th': omega_th,
        'gamma_num': gamma_num,
        'gamma_th': gamma_th,
    }


def parse_int_list(s):
    return [int(x) for x in s.split(',') if x.strip() != '']


def main():
    p = argparse.ArgumentParser(description="Dedalus 2D slab Alfven EVP (Fourier x, Chebyshev z)")
    p.add_argument("--Lx", type=float, default=128.0)
    p.add_argument("--Lz", type=float, default=128.0)
    p.add_argument("--Nx", type=int, default=64)
    p.add_argument("--Nz", type=int, default=129)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--scan", action='store_true', help="Scan over m_list,n_list instead of single m,n")
    p.add_argument("--m_list", type=str, default="0,1,2")
    p.add_argument("--n_list", type=str, default="1")
    args = p.parse_args()

    if not args.scan:
        res = run_evp_2d(Lx=args.Lx, Lz=args.Lz, Nx=args.Nx, Nz=args.Nz, B0=args.B0, rho_eff=args.rho,
                         tau=args.tau, eta=args.eta, m=args.m, n=args.n)
        werr = abs(res['omega_num'] - res['omega_th'])/res['omega_th'] if res['omega_th'] != 0 else 0.0
        gden = abs(res['gamma_th']) if res['gamma_th'] != 0 else 1.0
        gerr = abs(res['gamma_num'] - res['gamma_th'])/gden
        print(f"m={res['m']} n={res['n']} kx={res['kx']:.6e} kz={res['kz']:.6e}")
        print(f"Numerical omega = {res['omega_num']:.6e}")
        print(f"Theory    omega = {res['omega_th']:.6e}")
        print(f"Numerical gamma = {res['gamma_num']:.6e}")
        print(f"Theory    gamma = {res['gamma_th']:.6e}")
        print(f"omega %% error = {100*werr:.3f}%")
        print(f"gamma %% error = {100*gerr:.3f}%")
    else:
        ms = parse_int_list(args.m_list)
        ns = parse_int_list(args.n_list)
        print("| m | n | kx        | kz        | omega_num   | omega_th    | %err_w | gamma_num   | gamma_th    | %err_g |")
        print("|---|---|-----------|-----------|-------------|-------------|--------|-------------|-------------|--------|")
        for m in ms:
            for n in ns:
                r = run_evp_2d(Lx=args.Lx, Lz=args.Lz, Nx=args.Nx, Nz=args.Nz, B0=args.B0, rho_eff=args.rho,
                                tau=args.tau, eta=args.eta, m=m, n=n)
                werr = abs(r['omega_num'] - r['omega_th'])/r['omega_th'] if r['omega_th'] != 0 else 0.0
                gden = abs(r['gamma_th']) if r['gamma_th'] != 0 else 1.0
                gerr = abs(r['gamma_num'] - r['gamma_th'])/gden
                print(f"| {m} | {n} | {r['kx']:.6e} | {r['kz']:.6e} | {r['omega_num']:.6e} | {r['omega_th']:.6e} | {100*werr:6.3f}% | {r['gamma_num']:.6e} | {r['gamma_th']:.6e} | {100*gerr:6.3f}% |")


if __name__ == "__main__":
    main()
