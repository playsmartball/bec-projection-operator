#!/usr/bin/env python3
import argparse
import csv
import json
import math
import numpy as np


def is_finite(x):
    try:
        return math.isfinite(x)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description="Extract Phase-6 metrics (ΔE/E, ΔE/E_weak, max |Tau_*|, coupling vs boundary magnitude) from CSV. If E_weak column is absent, reconstruct weak-only energy by integrating energy-weighted couplings and dissipation.")
    p.add_argument("--csv", required=True)
    p.add_argument("--rho_perp", type=float, default=20.0)
    p.add_argument("--rho_par", type=float, default=5.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--omega_c", type=float, default=0.0)
    args = p.parse_args()

    E = []
    E_weak = []
    # For reconstruction
    t_series = []
    E_x_series = []
    E_y_series = []
    # Tau power/work and components
    P_tau_series = []
    W_tau_series = []
    Tau_vx_series = []
    Tau_vy_series = []
    Tau_bx_series = []
    Tau_by_series = []
    C_vx_bx_series = []
    C_vy_by_series = []
    C_bx_wx_series = []
    C_by_wy_series = []
    D_bx_series = []
    D_by_series = []
    P_eta_left_series = []
    P_eta_right_series = []
    P_eta_total_series = []
    max_abs_tau = 0.0
    tau_max_field = None
    max_abs_coupling = 0.0
    coup_max_field = None

    with open(args.csv, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row.get('t', 'nan'))
            except Exception:
                t = float('nan')
            if is_finite(t):
                t_series.append(t)
            try:
                Et = float(row.get('E_tot', 'nan'))
            except Exception:
                Et = float('nan')
            if is_finite(Et):
                E.append(Et)
            # Optional diagnostic weak-only energy
            try:
                Ew = float(row.get('E_weak', 'nan'))
            except Exception:
                Ew = float('nan')
            if is_finite(Ew):
                E_weak.append(Ew)
            # store components and powers for reconstruction
            try:
                Ex = float(row.get('E_x', 'nan'))
                Ey = float(row.get('E_y', 'nan'))
            except Exception:
                Ex, Ey = float('nan'), float('nan')
            E_x_series.append(Ex)
            E_y_series.append(Ey)
            # boundary power/work direct columns if present
            try:
                P_tau_val = float(row.get('P_tau', 'nan'))
            except Exception:
                P_tau_val = float('nan')
            P_tau_series.append(P_tau_val)
            try:
                W_tau_val = float(row.get('W_tau', 'nan'))
            except Exception:
                W_tau_val = float('nan')
            W_tau_series.append(W_tau_val)
            for key, target in (
                ('C_vx_bx', C_vx_bx_series),
                ('C_vy_by', C_vy_by_series),
                ('C_bx_wx', C_bx_wx_series),
                ('C_by_wy', C_by_wy_series),
                ('D_bx', D_bx_series),
                ('D_by', D_by_series),
            ):
                try:
                    val = float(row.get(key, 'nan'))
                except Exception:
                    val = float('nan')
                target.append(val)
            try:
                pL = float(row.get('P_eta_left', 'nan'))
            except Exception:
                pL = float('nan')
            P_eta_left_series.append(pL)
            try:
                pR = float(row.get('P_eta_right', 'nan'))
            except Exception:
                pR = float('nan')
            P_eta_right_series.append(pR)
            try:
                pT = float(row.get('P_eta_total', 'nan'))
            except Exception:
                pT = float('nan')
            P_eta_total_series.append(pT)
            # Tau terms
            for key in ('Tau_vx', 'Tau_vy', 'Tau_bx', 'Tau_by'):
                try:
                    val = abs(float(row.get(key, '0.0')))
                except Exception:
                    val = 0.0
                if is_finite(val) and val > max_abs_tau:
                    max_abs_tau = val
                    tau_max_field = key
            # Store signed Tau_* series for potential power reconstruction
            for key, target in (
                ('Tau_vx', Tau_vx_series),
                ('Tau_vy', Tau_vy_series),
                ('Tau_bx', Tau_bx_series),
                ('Tau_by', Tau_by_series),
            ):
                try:
                    sval = float(row.get(key, 'nan'))
                except Exception:
                    sval = float('nan')
                target.append(sval)
            # Coupling terms
            for key in ('C_vx_bx', 'C_bx_wx', 'C_vy_by', 'C_by_wy'):
                try:
                    val = abs(float(row.get(key, '0.0')))
                except Exception:
                    val = 0.0
                if is_finite(val) and val > max_abs_coupling:
                    max_abs_coupling = val
                    coup_max_field = key

    if len(E) == 0:
        deltaE_over_E = float('nan')
    else:
        E0 = E[0] if abs(E[0]) > 0 else 1.0
        deltaE_over_E = (max(E) - min(E)) / (abs(E0) + 1e-300)
    # Prefer reconstruction if possible (even if E_weak is present); otherwise fall back
    if len(t_series) >= 2 and all(len(s) == len(t_series) for s in [E_x_series, E_y_series, C_vx_bx_series, C_vy_by_series, C_bx_wx_series, C_by_wy_series, D_bx_series, D_by_series]):
        Ew_series = []
        Ew_prev = None
        for i in range(len(t_series)):
            if Ew_prev is None:
                # initialize to instantaneous total energy from components
                if is_finite(E_x_series[i]) and is_finite(E_y_series[i]):
                    Ew_prev = E_x_series[i] + E_y_series[i]
                else:
                    Ew_prev = float('nan')
            else:
                dt = t_series[i] - t_series[i-1]
                # weighted couplings and dissipation
                Csum = 0.0
                Dsum = 0.0
                if is_finite(C_vx_bx_series[i]):
                    Csum += (1.0 + args.rho_perp) * C_vx_bx_series[i]
                if is_finite(C_vy_by_series[i]):
                    Csum += (1.0 + args.rho_par) * C_vy_by_series[i]
                if is_finite(C_bx_wx_series[i]):
                    Csum += (1.0/args.tau) * C_bx_wx_series[i]
                if is_finite(C_by_wy_series[i]):
                    Csum += (1.0/args.tau) * C_by_wy_series[i]
                if is_finite(D_bx_series[i]):
                    Dsum += (1.0/args.tau) * D_bx_series[i]
                if is_finite(D_by_series[i]):
                    Dsum += (1.0/args.tau) * D_by_series[i]
                if is_finite(dt):
                    Ew_prev = Ew_prev + dt * (Csum + Dsum)
            Ew_series.append(Ew_prev)
        # compute weak-only drift
        if len(Ew_series) > 0 and is_finite(Ew_series[0]):
            Ew0 = Ew_series[0] if abs(Ew_series[0]) > 0 else 1.0
            deltaE_over_E_weak = (max([x for x in Ew_series if is_finite(x)]) - min([x for x in Ew_series if is_finite(x)])) / (abs(Ew0) + 1e-300)
        else:
            deltaE_over_E_weak = float('nan')
    else:
        if len(E_weak) == 0:
            deltaE_over_E_weak = float('nan')
        else:
            Ew0 = E_weak[0] if abs(E_weak[0]) > 0 else 1.0
            deltaE_over_E_weak = (max(E_weak) - min(E_weak)) / (abs(Ew0) + 1e-300)

    # Boundary cumulative work W_tau and normalized W_tau/E0
    W_tau_final = float('nan')
    W_tau_over_E0 = float('nan')
    # Prefer direct W_tau column if valid
    if len(W_tau_series) > 0 and any(is_finite(w) for w in W_tau_series):
        # take last finite value
        for w in reversed(W_tau_series):
            if is_finite(w):
                W_tau_final = w
                break
    else:
        # Reconstruct from P_tau or from Tau_* if needed
        have_ptau = (len(P_tau_series) == len(t_series)) and any(is_finite(p) for p in P_tau_series)
        if len(t_series) >= 2:
            acc = 0.0
            for i in range(1, len(t_series)):
                dt = t_series[i] - t_series[i-1]
                if have_ptau and is_finite(P_tau_series[i]):
                    p = P_tau_series[i]
                else:
                    # sum Tau_* if present
                    p = 0.0
                    for sarr in (Tau_vx_series, Tau_vy_series, Tau_bx_series, Tau_by_series):
                        try:
                            sval = sarr[i]
                        except Exception:
                            sval = float('nan')
                        if is_finite(sval):
                            p += sval
                if is_finite(dt) and is_finite(p):
                    acc += dt * p
            W_tau_final = acc
    # Normalize by initial total energy if available
    E0_for_norm = None
    if len(E) > 0 and is_finite(E[0]):
        E0_for_norm = E[0]
    elif len(E_x_series) > 0 and len(E_y_series) > 0 and is_finite(E_x_series[0]) and is_finite(E_y_series[0]):
        E0_for_norm = E_x_series[0] + E_y_series[0]
    if E0_for_norm is not None and abs(E0_for_norm) > 0 and is_finite(W_tau_final):
        W_tau_over_E0 = W_tau_final / (abs(E0_for_norm) + 1e-300)

    p_eta_left_final = float('nan')
    p_eta_right_final = float('nan')
    p_eta_total_final = float('nan')
    for arr, store in (
        (P_eta_left_series, 'L'),
        (P_eta_right_series, 'R'),
        (P_eta_total_series, 'T'),
    ):
        for v in reversed(arr):
            if is_finite(v):
                if store == 'L':
                    p_eta_left_final = v
                elif store == 'R':
                    p_eta_right_final = v
                else:
                    p_eta_total_final = v
                break

    p_eta_total_slope_lastk = float('nan')
    try:
        n = len(t_series)
        k = min(1000, n)
        if k >= 2:
            pts = []
            i = n - 1
            while i >= 0 and len(pts) < k:
                tval = t_series[i]
                pval = P_eta_total_series[i]
                if is_finite(tval) and is_finite(pval):
                    pts.append((tval, pval))
                i -= 1
            if len(pts) >= 2:
                xs = [x for (x, _) in pts]
                ys = [y for (_, y) in pts]
                xm = sum(xs) / len(xs)
                ym = sum(ys) / len(ys)
                num = sum((x - xm) * (y - ym) for x, y in pts)
                den = sum((x - xm) * (x - xm) for x in xs)
                if den != 0.0:
                    p_eta_total_slope_lastk = num / den
    except Exception:
        p_eta_total_slope_lastk = float('nan')

    peta_fft_power_total = float('nan')
    peta_fft_power_above = float('nan')
    peta_fft_fraction_above = float('nan')
    peta_fft_omega_peak = float('nan')
    try:
        if args.omega_c > 0.0 and len(t_series) >= 8 and len(P_eta_total_series) == len(t_series):
            ts = []
            ys = []
            for t, y in zip(t_series, P_eta_total_series):
                if is_finite(t) and is_finite(y):
                    ts.append(t)
                    ys.append(y)
            if len(ts) >= 8:
                dtc = []
                for i in range(1, len(ts)):
                    dti = ts[i] - ts[i-1]
                    if is_finite(dti) and dti > 0:
                        dtc.append(dti)
                if len(dtc) > 0:
                    dt = float(np.median(np.array(dtc)))
                    y = np.array(ys, dtype=float)
                    y = y - float(np.mean(y))
                    w = np.hanning(len(y))
                    yw = y * w
                    Y = np.fft.rfft(yw)
                    freqs = np.fft.rfftfreq(len(yw), d=dt)
                    omega = 2.0 * np.pi * freqs
                    P = (Y.conj() * Y).real
                    if P.size > 0:
                        peta_fft_power_total = float(np.sum(P))
                        mask = omega >= args.omega_c
                        if np.any(mask):
                            peta_fft_power_above = float(np.sum(P[mask]))
                            if peta_fft_power_total > 0:
                                peta_fft_fraction_above = peta_fft_power_above / peta_fft_power_total
                        kmax = int(np.argmax(P))
                        if 0 <= kmax < omega.size:
                            peta_fft_omega_peak = float(omega[kmax])
    except Exception:
        peta_fft_power_total = float('nan')
        peta_fft_power_above = float('nan')
        peta_fft_fraction_above = float('nan')
        peta_fft_omega_peak = float('nan')

    result = {
        "csv": args.csv,
        "deltaE_over_E": deltaE_over_E,
        "deltaE_over_E_weak": deltaE_over_E_weak,
        "W_tau_final": W_tau_final,
        "W_tau_over_E0": W_tau_over_E0,
        "P_eta_left_final": p_eta_left_final,
        "P_eta_right_final": p_eta_right_final,
        "P_eta_total_final": p_eta_total_final,
        "P_eta_total_slope_lastk": p_eta_total_slope_lastk,
        "max_abs_tau": max_abs_tau,
        "tau_max_field": tau_max_field,
        "max_abs_coupling": max_abs_coupling,
        "coupling_max_field": coup_max_field,
        "dominance": "boundary_tau" if (max_abs_tau >= max_abs_coupling) else "coupling",
        "peta_fft_power_total": peta_fft_power_total,
        "peta_fft_power_above_omega_c": peta_fft_power_above,
        "peta_fft_fraction_above_omega_c": peta_fft_fraction_above,
        "peta_fft_omega_peak": peta_fft_omega_peak
    }

    print("METRICS_JSON=" + json.dumps(result))


if __name__ == '__main__':
    main()
