"""
Dedalus 1D anisotropic-inertia Alfven benchmark (two perpendicular polarizations).

Equations (two decoupled polarizations x and y):
  dt(vx) - alpha_x * dz(bx) = 0,   dt(bx) - B0 * dz(vx) = 0
  dt(vy) - alpha_y * dz(by) = 0,   dt(by) - B0 * dz(vy) = 0
where alpha_{x,y} = B0 / (1 + rho_{x,y}).

Expected dispersion per polarization p in {x,y}:
  omega_p = k * B0 / sqrt(1 + rho_p)

Usage examples:
  python examples/dedalus_anisotropic_alfven_1d.py --rho_x 0 --rho_y 20 --pol x --N 128 --L 128.0 --k 1 --tmax 600 --dt 0.01
  python examples/dedalus_anisotropic_alfven_1d.py --rho_x 0 --rho_y 20 --pol y --N 128 --L 128.0 --k 1 --tmax 1200 --dt 0.01
"""
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core import timesteppers as ts
except Exception:
    sys.stderr.write("Dedalus is required to run this script. Install via pip/conda.\n")
    raise


def dz_project_series(field_g: np.ndarray, z: np.ndarray, k_phys: float):
    sin_b = np.sin(k_phys * z)
    cos_b = np.cos(k_phys * z)
    s_sin = np.mean(field_g * sin_b)
    s_cos = np.mean(field_g * cos_b)
    return float(s_sin), float(s_cos)


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


def run_dedalus_aniso(L=2*np.pi, N=128, B0=1.0, rho_x=0.0, rho_y=20.0, k=1, tmax=400.0, dt_step=0.01, amp0=1e-6, pol='x'):
    # Coordinates, distributor, basis (Dedalus v3 API)
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.float64)
    zbasis = de.RealFourier(coords['z'], size=N, bounds=(0.0, L))

    # Fields for both polarizations
    vx = dist.Field(name='vx', bases=zbasis)
    vy = dist.Field(name='vy', bases=zbasis)
    bx = dist.Field(name='bx', bases=zbasis)
    by = dist.Field(name='by', bases=zbasis)

    # Operators
    def d_z(f):
        return de.Differentiate(f, coords['z'])

    # Parameters
    alpha_x = B0 / (1.0 + rho_x)
    alpha_y = B0 / (1.0 + rho_y)

    # Two decoupled IVPs in one system
    problem = de.IVP([vx, vy, bx, by], namespace=locals())
    problem.add_equation("dt(vx) - alpha_x*d_z(bx) = 0")
    problem.add_equation("dt(bx) - B0*d_z(vx) = 0")
    problem.add_equation("dt(vy) - alpha_y*d_z(by) = 0")
    problem.add_equation("dt(by) - B0*d_z(vy) = 0")

    solver = problem.build_solver(ts.RK443)
    solver.stop_sim_time = tmax

    # Initial conditions: excite selected polarization only
    z = dist.local_grid(zbasis)
    k_phys = 2.0 * np.pi * k / L
    vx['g'] = amp0 * np.sin(k_phys * z) if pol == 'x' else 0.0
    vy['g'] = amp0 * np.sin(k_phys * z) if pol == 'y' else 0.0
    bx['g'] = 0.0
    by['g'] = 0.0

    # Time stepping & diagnostics (project corresponding b)
    times, s_sin_list, s_cos_list = [], [], []
    while solver.proceed:
        solver.step(dt_step)
        if pol == 'x':
            ss, sc = dz_project_series(bx['g'], z, k_phys)
        else:
            ss, sc = dz_project_series(by['g'], z, k_phys)
        times.append(solver.sim_time)
        s_sin_list.append(ss)
        s_cos_list.append(sc)

    times = np.asarray(times)
    s_sin = np.asarray(s_sin_list)
    s_cos = np.asarray(s_cos_list)
    pw_sin = float(np.mean(s_sin * s_sin))
    pw_cos = float(np.mean(s_cos * s_cos))
    series = s_sin if pw_sin >= pw_cos else s_cos

    omega_num = hilbert_phase_slope(times, series)
    rho_sel = rho_x if pol == 'x' else rho_y
    omega_th = k_phys * B0 / np.sqrt(1.0 + rho_sel)
    return omega_num, omega_th


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=128.0)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--rho_x", type=float, default=0.0)
    p.add_argument("--rho_y", type=float, default=20.0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--tmax", type=float, default=600.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--amp0", type=float, default=1e-6)
    p.add_argument("--pol", type=str, choices=["x","y"], default="x")
    args = p.parse_args()

    omega_num, omega_th = run_dedalus_aniso(L=args.L, N=args.N, B0=args.B0, rho_x=args.rho_x, rho_y=args.rho_y,
                                            k=args.k, tmax=args.tmax, dt_step=args.dt, amp0=args.amp0, pol=args.pol)
    ratio = omega_th / omega_num if omega_num != 0 else np.nan
    print(f"Polarization = {args.pol}")
    print(f"Numerical omega = {omega_num:.6e}")
    print(f"Theory    omega = {omega_th:.6e}")
    print(f"Theory/Numerical omega ratio = {ratio:.6f}")


if __name__ == "__main__":
    main()
