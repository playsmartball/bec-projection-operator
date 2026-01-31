#!/usr/bin/env python3
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core.operators import Lift as LiftOp
    from dedalus.core import timesteppers as ts
except Exception:
    sys.stderr.write("Dedalus is required to run this script.\n")
    raise


def run_coupled_ivp_2d(Lx=128.0, Lz=128.0, Nz=129, B0=1.0, tau=1.0, eta=0.0,
                        rho_perp=20.0, rho_par=5.0, eps=0.0, m=1, n=1,
                        tmax=512.0, tstep=0.05, amp=1e-6):
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.complex128)
    zbasis = de.Chebyshev(coords['z'], size=Nz, bounds=(0.0, Lz))

    vx = dist.Field(name='vx', bases=(zbasis,))
    vy = dist.Field(name='vy', bases=(zbasis,))
    bx = dist.Field(name='bx', bases=(zbasis,))
    by = dist.Field(name='by', bases=(zbasis,))
    wx = dist.Field(name='wx', bases=(zbasis,))
    wy = dist.Field(name='wy', bases=(zbasis,))

    tau1x = dist.Field(name='tau1x')
    tau2x = dist.Field(name='tau2x')
    tau1y = dist.Field(name='tau1y')
    tau2y = dist.Field(name='tau2y')

    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])

    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    kx = 2.0 * np.pi * m / Lx
    kx2 = float(kx**2)

    alpha_perp = B0 / (1.0 + rho_perp)
    alpha_par = B0 / (1.0 + rho_par)

    problem = de.IVP([vx, vy, bx, by, wx, wy, tau1x, tau2x, tau1y, tau2y], namespace=locals())
    # Metric-consistent coupling to preserve energy at eta=0
    problem.add_equation("dt(vx) - alpha_perp*d_z(bx) - eps*(1+rho_par)*vy + lift(tau1x, -1) + lift(tau2x, -2) = 0")
    problem.add_equation("dt(vy) - alpha_par*d_z(by) + eps*(1+rho_perp)*vx + lift(tau1y, -1) + lift(tau2y, -2) = 0")
    problem.add_equation("dt(bx) - tau*B0*wx - eta*(d_zz(bx) - kx2*bx) = 0")
    problem.add_equation("dt(by) - tau*B0*wy - eta*(d_zz(by) - kx2*by) = 0")
    problem.add_equation("wx - d_z(vx) = 0")
    problem.add_equation("wy - d_z(vy) = 0")
    problem.add_equation("vx(z='left') = 0")
    problem.add_equation("vx(z='right') = 0")
    problem.add_equation("vy(z='left') = 0")
    problem.add_equation("vy(z='right') = 0")

    solver = problem.build_solver(ts.RK443)

    z = dist.local_grid(zbasis)
    s = np.sin(np.pi * n * z / Lz)
    c = np.cos(np.pi * n * z / Lz)

    # Initialize: excite x polarization only
    vx['g'] = amp * s
    vy['g'] = 0.0
    bx['g'] = 0.0
    by['g'] = 0.0
    wx['g'] = (np.pi * n / Lz) * amp * c
    wy['g'] = 0.0

    times = []
    E_tot = []
    E_x = []
    E_y = []

    solver.stop_sim_time = tmax
    step = 0
    while solver.proceed:
        solver.step(tstep)
        step += 1
        if np.isnan(np.nanmax(np.abs(vx['g'])) + np.nanmax(np.abs(vy['g'])) + np.nanmax(np.abs(bx['g'])) + np.nanmax(np.abs(by['g']))):
            raise RuntimeError("NaN encountered in fields")
        if step % 10 == 0:
            t = solver.sim_time
            times.append(t)
            # Use Dedalus quadrature for proper energy integrals in Chebyshev
            density_x = 0.5*(1.0 + rho_perp)*vx*vx + 0.5*(1.0/tau)*bx*bx
            density_y = 0.5*(1.0 + rho_par)*vy*vy + 0.5*(1.0/tau)*by*by
            Ex = de.Integrate(density_x, coords['z']).evaluate()['g'][0]
            Ey = de.Integrate(density_y, coords['z']).evaluate()['g'][0]
            E_x.append(float(np.real(Ex)))
            E_y.append(float(np.real(Ey)))
            E_tot.append(E_x[-1] + E_y[-1])

    times = np.array(times)
    E_tot = np.array(E_tot)
    E_x = np.array(E_x)
    E_y = np.array(E_y)

    # Diagnostics
    E0 = E_tot[0] if len(E_tot) else 1.0
    drift = float((np.max(E_tot) - np.min(E_tot)) / (abs(E0) + 1e-300)) if len(E_tot) else np.nan
    leak_eps0 = None
    if eps == 0.0 and len(E_y):
        leak_eps0 = float(np.max(E_y) / (np.max(E_x) + 1e-300))

    # Exchange symmetry metric for eps>0, eta=0
    exch_sym = None
    if eps > 0.0 and eta == 0.0 and len(E_x) and len(E_y):
        amp_x = float(np.max(E_x) - np.min(E_x))
        amp_y = float(np.max(E_y) - np.min(E_y))
        exch_sym = float(abs(amp_x - amp_y) / (0.5*(amp_x + amp_y) + 1e-300))

    return {
        'times': times,
        'E_tot': E_tot,
        'E_x': E_x,
        'E_y': E_y,
        'energy_drift_rel': drift,
        'leak_eps0_ratio': leak_eps0,
        'exchange_sym_rel': exch_sym,
    }


def main():
    p = argparse.ArgumentParser(description="Dedalus 2D slab coupled (epsilon) anisotropic IVP")
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
    p.add_argument("--tmax", type=float, default=512.0)
    p.add_argument("--dt", type=float, default=0.05)
    args = p.parse_args()

    res = run_coupled_ivp_2d(Lx=args.Lx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, tau=args.tau, eta=args.eta,
                              rho_perp=args.rho_perp, rho_par=args.rho_par, eps=args.eps, m=args.m, n=args.n,
                              tmax=args.tmax, tstep=args.dt)
    print(f"energy drift (rel) = {res['energy_drift_rel']:.3e}")
    if res['leak_eps0_ratio'] is not None:
        print(f"eps=0 leakage ratio Ey/Ex = {res['leak_eps0_ratio']:.3e}")
    if res['exchange_sym_rel'] is not None:
        print(f"exchange symmetry (rel diff) = {res['exchange_sym_rel']:.3e}")


if __name__ == "__main__":
    main()
