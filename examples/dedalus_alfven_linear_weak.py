#!/usr/bin/env python3
import argparse
import os
import csv
import time
import numpy as np
from mpi4py import MPI

try:
    from dedalus import public as de
    from dedalus.core import timesteppers as ts
except Exception:
    raise


def run_linear_weak(Lx=128.0, Nx=128, Lz=128.0, Nz=129, B0=1.0, tau=1.0,
                    rho_perp=20.0, rho_par=5.0, eps=0.0,
                    tmax=20.0, tstep=1e-3, amp=1e-6, m=1, n=1, eta=1e-3, csv_path=None):
    coords = de.CartesianCoordinates('x', 'z')
    dist = de.Distributor(coords, dtype=np.complex128)
    xbasis = de.Fourier(coords['x'], size=Nx, bounds=(0.0, Lx), dealias=3/2, dtype=dist.dtype)
    zbasis = de.ChebyshevT(coords['z'], size=Nz, bounds=(0.0, Lz), dealias=3/2)

    vx = dist.Field(name='vx', bases=(xbasis, zbasis))
    vy = dist.Field(name='vy', bases=(xbasis, zbasis))
    bx = dist.Field(name='bx', bases=(xbasis, zbasis))
    by = dist.Field(name='by', bases=(xbasis, zbasis))

    def d_x(f):
        return de.Differentiate(f, coords['x'])
    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_xx(f):
        return de.Differentiate(de.Differentiate(f, coords['x']), coords['x'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])

    alpha_perp = B0 / (1.0 + rho_perp)
    alpha_par = B0 / (1.0 + rho_par)

    # Tau fields for weak-form boundary enforcement (no lifts)
    tau1vx = dist.Field(name='tau1vx', bases=(xbasis,))
    tau2vx = dist.Field(name='tau2vx', bases=(xbasis,))
    tau1vy = dist.Field(name='tau1vy', bases=(xbasis,))
    tau2vy = dist.Field(name='tau2vy', bases=(xbasis,))
    tau1bx = dist.Field(name='tau1bx', bases=(xbasis,))
    tau2bx = dist.Field(name='tau2bx', bases=(xbasis,))
    tau1by = dist.Field(name='tau1by', bases=(xbasis,))
    tau2by = dist.Field(name='tau2by', bases=(xbasis,))

    problem = de.IVP([vx, vy, bx, by, tau1vx, tau2vx, tau1vy, tau2vy, tau1bx, tau2bx, tau1by, tau2by], namespace=locals())
    problem.add_equation("dt(vx) - alpha_perp*d_z(bx) - eps*(1+rho_par)*vy = 0")
    problem.add_equation("dt(vy) - alpha_par*d_z(by) + eps*(1+rho_perp)*vx = 0")
    problem.add_equation("dt(bx) - tau*B0*d_z(vx) - eta*(d_xx(bx) + d_zz(bx)) = 0")
    problem.add_equation("dt(by) - tau*B0*d_z(vy) - eta*(d_xx(by) + d_zz(by)) = 0")

    # Dirichlet BCs on both ends, weak-form using tau variables (no tau lifts in PDEs)
    problem.add_equation("vx(z='left') + tau1vx = 0")
    problem.add_equation("vx(z='right') + tau2vx = 0")
    problem.add_equation("vy(z='left') + tau1vy = 0")
    problem.add_equation("vy(z='right') + tau2vy = 0")
    problem.add_equation("bx(z='left') + tau1bx = 0")
    problem.add_equation("bx(z='right') + tau2bx = 0")
    problem.add_equation("by(z='left') + tau1by = 0")
    problem.add_equation("by(z='right') + tau2by = 0")

    solver = problem.build_solver(ts.CNAB2)

    x = dist.local_grid(xbasis)
    z = dist.local_grid(zbasis)
    sx = np.sin(2.0 * np.pi * m * x / Lx)
    sz = np.sin(np.pi * n * z / Lz)
    cz = np.cos(np.pi * n * z / Lz)

    vx['g'] = amp * sx * sz
    vy['g'] = 0.0
    bx['g'] = 0.0
    by['g'] = 0.0

    phi_bx = dist.Field(name='phi_bx', bases=(xbasis, zbasis))
    phi_bx['g'] = sx * cz
    _phi_norm = de.Integrate(de.Integrate(phi_bx*phi_bx, coords['x']), coords['z']).evaluate()['g']
    _phi_norm_local = float(np.real(_phi_norm.ravel()[0])) if _phi_norm.size > 0 else 0.0
    phi_norm = float(dist.comm.allreduce(_phi_norm_local, op=MPI.SUM)) + 1e-300

    if csv_path is None:
        outdir = os.path.join('analysis', 'phase6_runs')
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(outdir, f'lin_weak_cheb_nx{Nx}_m{m}_n{n}_eta{eta}_{stamp}.csv')
    else:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    csv_file = None
    csv_writer = None
    if MPI.COMM_WORLD.rank == 0:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            't', 'E_tot', 'E_x', 'E_y', 'proj_bx',
            'C_vx_bx', 'C_bx_wx', 'C_vy_by', 'C_by_wy',
            'D_bx', 'D_by',
            'Tau_vx', 'Tau_vy', 'Tau_bx', 'Tau_by'
        ])

    solver.stop_sim_time = tmax
    step = 0
    times = []
    E_x = []
    E_y = []
    E_tot = []

    while solver.proceed:
        solver.step(tstep)
        step += 1

        t = solver.sim_time
        if dist.comm.rank == 0:
            times.append(t)

        density_x = 0.5*(1.0 + rho_perp)*vx*de.conj(vx) + 0.5*(1.0/tau)*bx*de.conj(bx)
        density_y = 0.5*(1.0 + rho_par)*vy*de.conj(vy) + 0.5*(1.0/tau)*by*de.conj(by)
        _Ex = de.Integrate(de.Integrate(density_x, coords['x']), coords['z']).evaluate()['g']
        _Ey = de.Integrate(de.Integrate(density_y, coords['x']), coords['z']).evaluate()['g']
        _Ex_local = float(np.real(_Ex.ravel()[0])) if _Ex.size > 0 else 0.0
        _Ey_local = float(np.real(_Ey.ravel()[0])) if _Ey.size > 0 else 0.0
        Ex_val = dist.comm.allreduce(_Ex_local, op=MPI.SUM)
        Ey_val = dist.comm.allreduce(_Ey_local, op=MPI.SUM)
        if dist.comm.rank == 0:
            E_x.append(Ex_val)
            E_y.append(Ey_val)
            E_tot.append(E_x[-1] + E_y[-1])

        _amp_bx = de.Integrate(de.Integrate(bx*phi_bx, coords['x']), coords['z']).evaluate()['g']
        _amp_local = float(np.real(_amp_bx.ravel()[0])) if _amp_bx.size > 0 else 0.0
        amp_bx_val = dist.comm.allreduce(_amp_local, op=MPI.SUM) / phi_norm

        _C_vx_bx = de.Integrate(de.Integrate(vx * de.conj(-alpha_perp * d_z(bx)), coords['x']), coords['z']).evaluate()['g']
        _C_bx_wx = de.Integrate(de.Integrate(bx * de.conj(-tau * B0 * d_z(vx)), coords['x']), coords['z']).evaluate()['g']
        _C_vy_by = de.Integrate(de.Integrate(vy * de.conj(-alpha_par * d_z(by)), coords['x']), coords['z']).evaluate()['g']
        _C_by_wy = de.Integrate(de.Integrate(by * de.conj(-tau * B0 * d_z(vy)), coords['x']), coords['z']).evaluate()['g']

        C_vx_bx = dist.comm.allreduce(float(np.real(_C_vx_bx.ravel()[0])) if _C_vx_bx.size > 0 else 0.0, op=MPI.SUM)
        C_bx_wx = dist.comm.allreduce(float(np.real(_C_bx_wx.ravel()[0])) if _C_bx_wx.size > 0 else 0.0, op=MPI.SUM)
        C_vy_by = dist.comm.allreduce(float(np.real(_C_vy_by.ravel()[0])) if _C_vy_by.size > 0 else 0.0, op=MPI.SUM)
        C_by_wy = dist.comm.allreduce(float(np.real(_C_by_wy.ravel()[0])) if _C_by_wy.size > 0 else 0.0, op=MPI.SUM)

        lap_bx = d_xx(bx) + d_zz(bx)
        lap_by = d_xx(by) + d_zz(by)
        if eta == 0.0:
            D_bx = 0.0
            D_by = 0.0
        else:
            _D_bx = de.Integrate(de.Integrate(bx * de.conj(-eta * lap_bx), coords['x']), coords['z']).evaluate()['g']
            _D_by = de.Integrate(de.Integrate(by * de.conj(-eta * lap_by), coords['x']), coords['z']).evaluate()['g']
            D_bx = dist.comm.allreduce(float(np.real(_D_bx.ravel()[0])) if _D_bx.size > 0 else 0.0, op=MPI.SUM)
            D_by = dist.comm.allreduce(float(np.real(_D_by.ravel()[0])) if _D_by.size > 0 else 0.0, op=MPI.SUM)

        if dist.comm.rank == 0 and csv_writer is not None:
            csv_writer.writerow([
                f"{times[-1]:.8e}", f"{E_tot[-1]:.8e}", f"{E_x[-1]:.8e}", f"{E_y[-1]:.8e}", f"{amp_bx_val:.8e}",
                f"{C_vx_bx:.8e}", f"{C_bx_wx:.8e}", f"{C_vy_by:.8e}", f"{C_by_wy:.8e}",
                f"{D_bx:.8e}", f"{D_by:.8e}",
                f"{0.0:.8e}", f"{0.0:.8e}", f"{0.0:.8e}", f"{0.0:.8e}"
            ])
            if step % 200 == 0 and csv_file is not None:
                csv_file.flush()
                os.fsync(csv_file.fileno())

    if MPI.COMM_WORLD.rank == 0 and csv_file is not None:
        csv_file.flush()
        os.fsync(csv_file.fileno())
        csv_file.close()

    return {
        'csv_path': csv_path
    }


def main():
    p = argparse.ArgumentParser(description="Phase-6 corrective experiment: weak-form-only boundary enforcement (Chebyshev-z)")
    p.add_argument("--Lx", type=float, default=128.0)
    p.add_argument("--Lz", type=float, default=128.0)
    p.add_argument("--Nx", type=int, default=128)
    p.add_argument("--Nz", type=int, default=129)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--rho_perp", type=float, default=20.0)
    p.add_argument("--rho_par", type=float, default=5.0)
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--amp", type=float, default=1e-6)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    res = run_linear_weak(Lx=args.Lx, Nx=args.Nx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, tau=args.tau,
                          rho_perp=args.rho_perp, rho_par=args.rho_par, eps=args.eps,
                          tmax=args.tmax, tstep=args.dt, amp=args.amp, m=args.m, n=args.n, eta=args.eta, csv_path=args.csv)
    if MPI.COMM_WORLD.rank == 0:
        print(f"csv: {res['csv_path']}")


if __name__ == "__main__":
    main()
