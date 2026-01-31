#!/usr/bin/env python3
"""
Dedalus 2D slab anisotropic inertia Alfven EVP (parameterized in kx): Chebyshev in z with tau BCs.

Equations (complex EVP):
  lam * v - alpha * dz(b) + lift(tau1, -1) + lift(tau2, -2) = 0
  lam * b - tau*B0 * dz(v) - eta * (dzz(b) - kx**2 * b) = 0
BCs:
  v(z='left') = 0
  v(z='right') = 0

Anisotropy:
  alpha = B0 / (1 + rho_p), with p ∈ {perp, para} selected by --pol
  pol = 'perp' uses rho_perp; pol = 'para' uses rho_par

Theory (shear-Alfvén in uniform slab):
  kx = 2*pi*m/Lx,   kz = n*pi/Lz
  omega = sqrt(tau) * B0/sqrt(1+rho_p) * |kz|
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


def run_aniso_evp_2d(Lx=128.0, Lz=128.0, Nz=129, B0=1.0, tau=1.0, eta=0.0,
                      rho_perp=0.0, rho_par=0.0, pol='perp', m=0, n=1):
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

    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    if pol not in ('perp', 'para'):
        raise ValueError("pol must be 'perp' or 'para'")
    rho_p = rho_perp if pol == 'perp' else rho_par
    alpha = B0 / (1.0 + rho_p)

    # Build EVP
    problem = de.EVP([v, b, tau1, tau2], eigenvalue=lam, namespace=locals())
    problem.add_equation("lam*v - alpha*d_z(b) + lift(tau1, -1) + lift(tau2, -2) = 0")
    problem.add_equation("lam*b - tau*B0*d_z(v) - eta*(d_zz(b) - (kx**2)*b) = 0")
    problem.add_equation("v(z='left') = 0")
    problem.add_equation("v(z='right') = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    evals = solver.eigenvalues

    kz = np.pi * n / Lz
    lam_th = (-0.5 * eta * (kx**2 + kz**2)) + 1j * (np.sqrt(tau) * B0 / np.sqrt(1.0 + rho_p) * np.abs(kz))

    idx = int(np.argmin(np.abs(evals - lam_th)))
    lam_sel = evals[idx]

    return {
        'm': m,
        'n': n,
        'kx': float(kx),
        'kz': float(kz),
        'omega_num': float(np.imag(lam_sel)),
        'omega_th': float(np.imag(lam_th)),
        'gamma_num': float(np.real(lam_sel)),
        'gamma_th': float(np.real(lam_th)),
    }


def parse_int_list(s):
    return [int(x) for x in s.split(',') if x.strip() != '']


def main():
    p = argparse.ArgumentParser(description="Dedalus 2D slab anisotropic Alfven EVP (param kx, Chebyshev z)")
    p.add_argument("--Lx", type=float, default=128.0)
    p.add_argument("--Lz", type=float, default=128.0)
    p.add_argument("--Nz", type=int, default=129)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--rho_perp", type=float, default=0.0)
    p.add_argument("--rho_par", type=float, default=0.0)
    p.add_argument("--pol", type=str, default='perp', choices=['perp','para'])
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--scan", action='store_true')
    p.add_argument("--m_list", type=str, default="0,1,2")
    p.add_argument("--n_list", type=str, default="1")
    args = p.parse_args()

    if not args.scan:
        res = run_aniso_evp_2d(Lx=args.Lx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, tau=args.tau, eta=args.eta,
                               rho_perp=args.rho_perp, rho_par=args.rho_par, pol=args.pol,
                               m=args.m, n=args.n)
        werr = abs(res['omega_num'] - res['omega_th'])/res['omega_th'] if res['omega_th'] != 0 else 0.0
        gden = abs(res['gamma_th']) if res['gamma_th'] != 0 else 1.0
        gerr = abs(res['gamma_num'] - res['gamma_th'])/gden
        print(f"pol={args.pol} m={res['m']} n={res['n']} kx={res['kx']:.6e} kz={res['kz']:.6e}")
        print(f"Numerical omega = {res['omega_num']:.6e}")
        print(f"Theory    omega = {res['omega_th']:.6e}")
        print(f"Numerical gamma = {res['gamma_num']:.6e}")
        print(f"Theory    gamma = {res['gamma_th']:.6e}")
        print(f"omega % error = {100*werr:.3f}%")
        print(f"gamma % error = {100*gerr:.3f}%")
    else:
        ms = parse_int_list(args.m_list)
        ns = parse_int_list(args.n_list)
        print("| pol | m | n | kx        | kz        | omega_num   | omega_th    | %err_w | gamma_num   | gamma_th    | %err_g |")
        print("|-----|---|---|-----------|-----------|-------------|-------------|--------|-------------|-------------|--------|")
        for m in ms:
            for n in ns:
                r = run_aniso_evp_2d(Lx=args.Lx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, tau=args.tau, eta=args.eta,
                                      rho_perp=args.rho_perp, rho_par=args.rho_par, pol=args.pol,
                                      m=m, n=n)
                werr = abs(r['omega_num'] - r['omega_th'])/r['omega_th'] if r['omega_th'] != 0 else 0.0
                gden = abs(r['gamma_th']) if r['gamma_th'] != 0 else 1.0
                gerr = abs(r['gamma_num'] - r['gamma_th'])/gden
                print(f"| {args.pol} | {m} | {n} | {r['kx']:.6e} | {r['kz']:.6e} | {r['omega_num']:.6e} | {r['omega_th']:.6e} | {100*werr:6.3f}% | {r['gamma_num']:.6e} | {r['gamma_th']:.6e} | {100*gerr:6.3f}% |")


if __name__ == "__main__":
    main()
