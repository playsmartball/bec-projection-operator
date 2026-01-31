#!/usr/bin/env python3
import argparse
import numpy as np
from mpi4py import MPI

try:
    from dedalus import public as de
except Exception:
    raise


def main():
    p = argparse.ArgumentParser(description="Phase-6 Task 6.2: Linear operator skew-symmetry test (periodic z)")
    p.add_argument("--Lx", type=float, default=128.0)
    p.add_argument("--Lz", type=float, default=128.0)
    p.add_argument("--Nx", type=int, default=64)
    p.add_argument("--Nz", type=int, default=64)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--rho_perp", type=float, default=20.0)
    p.add_argument("--rho_par", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    coords = de.CartesianCoordinates('x', 'z')
    dist = de.Distributor(coords, dtype=np.complex128)
    xbasis = de.Fourier(coords['x'], size=args.Nx, bounds=(0.0, args.Lx), dealias=3/2, dtype=dist.dtype)
    zbasis = de.Fourier(coords['z'], size=args.Nz, bounds=(0.0, args.Lz), dealias=3/2, dtype=dist.dtype)

    def d_z(f):
        return de.Differentiate(f, coords['z'])

    vx_u = dist.Field(name='vx_u', bases=(xbasis, zbasis))
    vy_u = dist.Field(name='vy_u', bases=(xbasis, zbasis))
    bx_u = dist.Field(name='bx_u', bases=(xbasis, zbasis))
    by_u = dist.Field(name='by_u', bases=(xbasis, zbasis))

    vx_v = dist.Field(name='vx_v', bases=(xbasis, zbasis))
    vy_v = dist.Field(name='vy_v', bases=(xbasis, zbasis))
    bx_v = dist.Field(name='bx_v', bases=(xbasis, zbasis))
    by_v = dist.Field(name='by_v', bases=(xbasis, zbasis))

    rng = np.random.default_rng(args.seed + MPI.COMM_WORLD.rank)
    for f in (vx_u, vy_u, bx_u, by_u, vx_v, vy_v, bx_v, by_v):
        real = rng.standard_normal(f['g'].shape)
        imag = rng.standard_normal(f['g'].shape)
        f['g'] = (real + 1j*imag) * 1e-3

    alpha_perp = args.B0 / (1.0 + args.rho_perp)
    alpha_par = args.B0 / (1.0 + args.rho_par)

    def L_of(vx, vy, bx, by):
        Lvx = -alpha_perp * d_z(bx)
        Lvy = -alpha_par * d_z(by)
        Lbx = -args.tau * args.B0 * d_z(vx)
        Lby = -args.tau * args.B0 * d_z(vy)
        return Lvx, Lvy, Lbx, Lby

    def inner(U, V):
        vxU, vyU, bxU, byU = U
        vxV, vyV, bxV, byV = V
        term_x = (1.0 + args.rho_perp) * de.Integrate(de.Integrate(vxU * de.conj(vxV), coords['x']), coords['z'])
        term_y = (1.0 + args.rho_par) * de.Integrate(de.Integrate(vyU * de.conj(vyV), coords['x']), coords['z'])
        term_bx = (1.0/args.tau) * de.Integrate(de.Integrate(bxU * de.conj(bxV), coords['x']), coords['z'])
        term_by = (1.0/args.tau) * de.Integrate(de.Integrate(byU * de.conj(byV), coords['x']), coords['z'])
        val = (term_x + term_y + term_bx + term_by).evaluate()['g']
        local = complex(val.ravel()[0]) if val.size > 0 else 0.0 + 0.0j
        real = MPI.COMM_WORLD.allreduce(np.real(local), op=MPI.SUM)
        imag = MPI.COMM_WORLD.allreduce(np.imag(local), op=MPI.SUM)
        return real + 1j*imag

    Lu = L_of(vx_u, vy_u, bx_u, by_u)
    Lv = L_of(vx_v, vy_v, bx_v, by_v)

    s1 = inner((vx_u, vy_u, bx_u, by_u), Lv)
    s2 = inner(Lu, (vx_v, vy_v, bx_v, by_v))
    ssum = s1 + s2

    if MPI.COMM_WORLD.rank == 0:
        print(f"<u,Lv> = {s1.real:.6e} + {s1.imag:.6e}i")
        print(f"<Lu,v> = {s2.real:.6e} + {s2.imag:.6e}i")
        print(f"<u,Lv> + <Lu,v> = {ssum.real:.6e} + {ssum.imag:.6e}i")


if __name__ == "__main__":
    main()
