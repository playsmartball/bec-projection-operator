#!/usr/bin/env python3
import argparse
import sys
import numpy as np

try:
    from dedalus import public as de
    from dedalus.core.operators import Lift as LiftOp
    from dedalus.core import timesteppers as ts
except Exception:
    sys.stderr.write("Dedalus is required to run this script. Install via pip/conda.\n")
    raise


def hilbert_phase_slope(times: np.ndarray, series: np.ndarray) -> float:
    N = len(series)
    if N < 4:
        return np.nan
    S = np.fft.fft(series)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[1:N//2] = 2.0
        H[N//2] = 1.0
    else:
        H[0] = 1.0
        H[1:(N+1)//2] = 2.0
    a = np.fft.ifft(S * H)
    phi = np.unwrap(np.angle(a))
    m, _ = np.polyfit(times, phi, 1)
    return float(abs(m))

def run_ivp_2d(Lx=128.0, Lz=128.0, Nz=129, B0=1.0, rho_eff=0.0, tau=1.0, eta=0.0,
               m=0, n=1, tmax=512.0, tstep=0.05, amp=1e-6):
    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, dtype=np.complex128)
    zbasis = de.Chebyshev(coords['z'], size=Nz, bounds=(0.0, Lz))

    v = dist.Field(name='v', bases=(zbasis,))
    b = dist.Field(name='b', bases=(zbasis,))
    w = dist.Field(name='w', bases=(zbasis,))
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')

    def d_z(f):
        return de.Differentiate(f, coords['z'])
    def d_zz(f):
        return de.Differentiate(de.Differentiate(f, coords['z']), coords['z'])

    lift_basis = zbasis.derivative_basis(1)
    def lift(A, n):
        return LiftOp(A, lift_basis, n)

    kx = 2.0 * np.pi * m / Lx
    kx2 = float(kx**2)
    alpha = B0 / (1.0 + rho_eff)

    problem = de.IVP([v, b, w, tau1, tau2], namespace=locals())

    problem.add_equation("dt(v) - alpha*d_z(b) + lift(tau1, -1) + lift(tau2, -2) = 0")
    problem.add_equation("dt(b) - tau*B0*w - eta*(d_zz(b) - kx2*b) = 0")
    problem.add_equation("w - d_z(v) = 0")
    problem.add_equation("v(z='left') = 0")
    problem.add_equation("v(z='right') = 0")

    solver = problem.build_solver(ts.RK443)

    z = dist.local_grid(zbasis)
    v['g'] = amp * np.sin(np.pi * n * z / Lz)
    b['g'] = 0.0
    w['g'] = (np.pi * n / Lz) * amp * np.cos(np.pi * n * z / Lz)

    times = []
    energies = []
    gamma_inst_list = []
    proj_sin = []
    proj_cos = []
    s = np.sin(np.pi * n * z / Lz)
    c = np.cos(np.pi * n * z / Lz)
    # Uniform grid weights are sufficient for phase; normalization cancels in ratios
    s_norm = float(np.sum(s**2)) + 1e-300
    c_norm = float(np.sum(c**2)) + 1e-300

    solver.stop_sim_time = tmax
    while solver.proceed:
        solver.step(tstep)
        if np.isnan(np.nanmax(np.abs(v['g'])) + np.nanmax(np.abs(b['g']))):
            raise RuntimeError("NaN encountered in fields")
        t = solver.sim_time
        if int(t / tstep) % 10 == 0:
            times.append(t)
            vg = np.real(v['g'])
            bg = np.real(b['g'])
            E = 0.5 * (np.mean(vg**2) + np.mean(bg**2))
            energies.append(E)
            proj_sin.append(float(np.sum(bg * s) / s_norm))
            proj_cos.append(float(np.sum(bg * c) / c_norm))
            # Instantaneous energetic estimator for physical damping
            mb2 = float(np.mean(bg*bg))
            kz = np.pi * n / Lz
            k2 = kx2 + kz**2
            denom = 2.0 * E + 1e-300
            gamma_inst = -eta * k2 * mb2 / denom
            gamma_inst_list.append(gamma_inst)

    times = np.array(times)
    energies = np.array(energies)
    proj_sin = np.array(proj_sin)
    proj_cos = np.array(proj_cos)

    # Choose the dominant spatial projection for frequency extraction (b has Neumann parity -> cos)
    pw_sin = float(np.mean(proj_sin * proj_sin)) if len(proj_sin) else 0.0
    pw_cos = float(np.mean(proj_cos * proj_cos)) if len(proj_cos) else 0.0
    series = proj_cos if pw_cos >= pw_sin else proj_sin
    omega_num = hilbert_phase_slope(times, series - np.mean(series))

    # Damping via log-slope of energy envelope (E ~ e^{2γ t})
    if len(times) > 10:
        # skip early transient ~1 period
        kz = np.pi * n / Lz
        T = 2*np.pi / (np.sqrt(tau) * B0 / np.sqrt(1.0 + rho_eff) * abs(kz) + 1e-16)
        mask = times > (1.0 * T)
        if np.count_nonzero(mask) < 5:
            mask = slice(None)
        slope, _ = np.polyfit(times[mask], np.log(np.clip(energies[mask], 1e-300, None)), 1)
        gamma_env = 0.5 * slope
        gamma_inst_avg = float(np.mean(np.asarray(gamma_inst_list)[mask]))
        gamma_num = float(gamma_env)
    else:
        gamma_num = np.nan

    # Theory
    kz = np.pi * n / Lz
    omega_th = np.sqrt(tau) * B0 * abs(kz) / np.sqrt(1.0 + rho_eff)
    gamma_th = -0.5 * eta * (kx2 + kz**2)

    return {
        'omega_num': float(omega_num) if np.isfinite(omega_num) else np.nan,
        'omega_th': float(omega_th),
        'gamma_num': float(gamma_num) if np.isfinite(gamma_num) else np.nan,
        'gamma_th': float(gamma_th),
        'times': times,
        'energies': energies,
        'proj_sin': proj_sin,
        'proj_cos': proj_cos,
    }


def main():
    p = argparse.ArgumentParser(description="Dedalus 2D slab Alfven IVP (param kx, Chebyshev z)")
    p.add_argument("--Lx", type=float, default=128.0)
    p.add_argument("--Lz", type=float, default=128.0)
    p.add_argument("--Nz", type=int, default=129)
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--tmax", type=float, default=512.0)
    p.add_argument("--dt", type=float, default=0.05)
    args = p.parse_args()

    res = run_ivp_2d(Lx=args.Lx, Lz=args.Lz, Nz=args.Nz, B0=args.B0, rho_eff=args.rho,
                     tau=args.tau, eta=args.eta, m=args.m, n=args.n, tmax=args.tmax, tstep=args.dt)

    werr = abs(res['omega_num'] - res['omega_th'])/res['omega_th'] if np.isfinite(res['omega_num']) and res['omega_th'] != 0 else np.nan
    gden = abs(res['gamma_th']) if res['gamma_th'] != 0 else 1.0
    gerr = abs(res['gamma_num'] - res['gamma_th'])/gden if np.isfinite(res['gamma_num']) else np.nan
    print(f"Numerical omega = {res['omega_num']:.6e}")
    print(f"Theory    omega = {res['omega_th']:.6e}")
    print(f"Numerical gamma = {res['gamma_num']:.6e}")
    print(f"Theory    gamma = {res['gamma_th']:.6e}")
    if np.isfinite(werr):
        print(f"omega % error = {100*werr:.3f}%")
    if np.isfinite(gerr):
        print(f"gamma % error = {100*gerr:.3f}%")

if __name__ == "__main__":
    main()
