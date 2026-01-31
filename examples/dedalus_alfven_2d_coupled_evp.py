#!/usr/bin/env python3
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core.operators import Lift as LiftOp
except Exception:
    sys.stderr.write("Dedalus is required to run this script.\n")
    raise


def run_coupled_evp_2d(Lx=128.0, Lz=128.0, Nz=129, B0=1.0, tau=1.0, eta=0.0,
                        rho_perp=20.0, rho_par=5.0, eps=0.0, m=1, n=1):
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.complex128)
    zbasis = de.Chebyshev(coords['z'], size=Nz, bounds=(0.0, Lz))

    vx = dist.Field(name='vx', bases=(zbasis,))
    vy = dist.Field(name='vy', bases=(zbasis,))
    bx = dist.Field(name='bx', bases=(zbasis,))
    by = dist.Field(name='by', bases=(zbasis,))

    tau1x = dist.Field(name='tau1x')
    tau2x = dist.Field(name='tau2x')
    tau1y = dist.Field(name='tau1y')
    tau2y = dist.Field(name='tau2y')

    lam = dist.Field(name='lam')

    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])

    kx = 2.0 * np.pi * m / Lx
    kz = np.pi * n / Lz

    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    alpha_perp = B0 / (1.0 + rho_perp)
    alpha_par = B0 / (1.0 + rho_par)

    problem = de.EVP([vx, vy, bx, by, tau1x, tau2x, tau1y, tau2y], eigenvalue=lam, namespace=locals())
    # Energy-preserving coupling normalized by inertia: a=eps/(1+rho_perp), b=eps/(1+rho_par)
    problem.add_equation("lam*vx - alpha_perp*d_z(bx) - (eps/(1+rho_perp))*vy + lift(tau1x, -1) + lift(tau2x, -2) = 0")
    problem.add_equation("lam*vy - alpha_par*d_z(by) + (eps/(1+rho_par))*vx + lift(tau1y, -1) + lift(tau2y, -2) = 0")
    problem.add_equation("lam*bx - tau*B0*d_z(vx) - eta*(d_zz(bx) - (kx**2)*bx) = 0")
    problem.add_equation("lam*by - tau*B0*d_z(vy) - eta*(d_zz(by) - (kx**2)*by) = 0")
    problem.add_equation("vx(z='left') = 0")
    problem.add_equation("vx(z='right') = 0")
    problem.add_equation("vy(z='left') = 0")
    problem.add_equation("vy(z='right') = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    evals = solver.eigenvalues

    # Theoretical uncoupled eigenvalues
    lam_perp_th = (-0.5 * eta * (kx**2 + kz**2)) + 1j * (np.sqrt(tau) * B0 / np.sqrt(1.0 + rho_perp) * abs(kz))
    lam_par_th = (-0.5 * eta * (kx**2 + kz**2)) + 1j * (np.sqrt(tau) * B0 / np.sqrt(1.0 + rho_par) * abs(kz))

    # Identify target pair via imag-band window then select by real-part proximity to re_th
    re = np.real(evals)
    im = np.imag(evals)
    re_th = float(np.real(lam_perp_th))
    w_lo = float(min(np.imag(lam_perp_th), np.imag(lam_par_th)))
    w_hi = float(max(np.imag(lam_perp_th), np.imag(lam_par_th)))
    band = w_hi - w_lo
    margin = max(0.5*band, 1e-3)
    cand = np.where((im >= w_lo - margin) & (im <= w_hi + margin))[0]
    if cand.size < 2:
        margin = max(2.0*band, 5e-3)
        cand = np.where((im >= w_lo - margin) & (im <= w_hi + margin))[0]
    # Choose two with smallest |re - re_th|
    if cand.size >= 2:
        order = cand[np.argsort(np.abs(re[cand] - re_th))]
        idx1, idx2 = int(order[0]), int(order[1])
    else:
        # Fallback: nearest two overall in imag to targets
        order_im = np.argsort(np.minimum(np.abs(im - w_lo), np.abs(im - w_hi)))
        idx1, idx2 = int(order_im[0]), int(order_im[1])
    lam1 = evals[idx1]
    lam2 = evals[idx2]
    # Labeling to keep continuity with perp/para proximity
    if abs(lam1 - lam_perp_th) + abs(lam2 - lam_par_th) <= abs(lam1 - lam_par_th) + abs(lam2 - lam_perp_th):
        lam_perp, lam_para = lam1, lam2
    else:
        lam_perp, lam_para = lam2, lam1

    # Sort by frequency for reporting (omega+ >= omega-)
    omegas = np.array([np.imag(lam_perp), np.imag(lam_para)])
    gammas = np.array([np.real(lam_perp), np.real(lam_para)])
    order = np.argsort(omegas)
    omega_minus = float(omegas[order[0]])
    omega_plus = float(omegas[order[1]])
    gamma_minus = float(gammas[order[0]])
    gamma_plus = float(gammas[order[1]])

    delta0 = float(abs(np.imag(lam_par_th) - np.imag(lam_perp_th)))
    delta = float(abs(omega_plus - omega_minus))
    delta_star = float(np.sqrt(max(delta*delta - delta0*delta0, 0.0)))

    return {
        'eps': eps,
        'eta': eta,
        'kx': float(kx),
        'kz': float(kz),
        'omega_plus': omega_plus,
        'omega_minus': omega_minus,
        'delta_omega': delta,
        'delta0': delta0,
        'delta_star': delta_star,
        'gamma_plus': gamma_plus,
        'gamma_minus': gamma_minus,
        'lam_perp_th_im': float(np.imag(lam_perp_th)),
        'lam_par_th_im': float(np.imag(lam_par_th)),
        'lam_th_re': float(np.real(lam_perp_th)),
    }


def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip() != '']


def main():
    p = argparse.ArgumentParser(description="Dedalus 2D slab coupled (epsilon) anisotropic EVP")
    p.add_argument("--Lx", type=float, default=128.0)
    p.add_argument("--Lz", type=float, default=128.0)
    p.add_argument("--Nz", type=int, default=129)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--rho_perp", type=float, default=20.0)
    p.add_argument("--rho_par", type=float, default=5.0)
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--scan", action='store_true')
    p.add_argument("--eps_list", type=str, default="0,1e-3,3e-3,1e-2")
    args = p.parse_args()

    if not args.scan:
        res = run_coupled_evp_2d(Lx=args.Lx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, tau=args.tau, eta=args.eta,
                                 rho_perp=args.rho_perp, rho_par=args.rho_par, eps=args.eps, m=args.m, n=args.n)
        print(f"eps={res['eps']} eta={res['eta']} kx={res['kx']:.6e} kz={res['kz']:.6e}")
        print(f"omega+ = {res['omega_plus']:.6e}  omega- = {res['omega_minus']:.6e}  Δω = {res['delta_omega']:.6e}")
        print(f"gamma+ = {res['gamma_plus']:.6e}  gamma- = {res['gamma_minus']:.6e}")
        print(f"uncoupled: omega_perp={res['lam_perp_th_im']:.6e}  omega_para={res['lam_par_th_im']:.6e}  gamma_th={res['lam_th_re']:.6e}")
    else:
        eps_list = parse_float_list(args.eps_list)
        print("| eps    | eta   | omega+     | omega-     | Δω         | Δω*/eps    | gamma+     | gamma-     |")
        print("|--------|-------|------------|------------|------------|------------|------------|------------|")
        for e in eps_list:
            r = run_coupled_evp_2d(Lx=args.Lx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, tau=args.tau, eta=args.eta,
                                   rho_perp=args.rho_perp, rho_par=args.rho_par, eps=e, m=args.m, n=args.n)
            ratio_star = r['delta_star']/e if e != 0 else 0.0
            print(f"| {e:6.3e} | {args.eta:5.1e} | {r['omega_plus']:.6e} | {r['omega_minus']:.6e} | {r['delta_omega']:.6e} | {ratio_star:.6e} | {r['gamma_plus']:.6e} | {r['gamma_minus']:.6e} |")


if __name__ == "__main__":
    main()
