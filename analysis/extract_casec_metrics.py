#!/usr/bin/env python3
import argparse
import os
import sys
import json
import numpy as np

# Try to import the local helper for Hilbert phase slope
try:
    from examples.utils.modal_projections import hilbert_phase_slope
except Exception:
    hilbert_phase_slope = None


def find_blowup_index(times, E):
    # Detect onset of sustained growth using log-energy slope
    # Smooth with a small window to avoid noise
    mask = np.isfinite(E) & (E > 0)
    if np.count_nonzero(mask) < 5:
        return None
    t = times[mask]
    lnE = np.log(E[mask])
    dlnE = np.gradient(lnE, t)
    # moving average over ~5 samples
    k = 5
    if len(dlnE) >= k:
        kernel = np.ones(k) / k
        dlnE_sm = np.convolve(dlnE, kernel, mode='same')
    else:
        dlnE_sm = dlnE
    # threshold for growth (per unit time). choose small positive value
    thr = 1e-3
    K = 10  # sustained for ~K samples (sampling every 10 steps)
    run = 0
    for i, val in enumerate(dlnE_sm):
        if np.isfinite(val) and val > thr:
            run += 1
            if run >= K:
                # map back to original times index
                idx_global = np.where(mask)[0][i - K + 1]
                return int(idx_global)
        else:
            run = 0
    return None


def compute_metrics(csv_path, t0_hint=10.0, margin=0.5, t1_max=None):
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    t = np.array(data['t'], dtype=float)
    E_tot = np.array(data['E_tot'], dtype=float)
    pbx = np.array(data['proj_bx'], dtype=float)

    # Finite masks
    mfin = np.isfinite(t) & np.isfinite(E_tot) & (t > 0)
    t = t[mfin]
    E_tot = E_tot[mfin]
    pbx = pbx[mfin]

    if t.size < 20:
        raise RuntimeError('Insufficient samples in CSV to compute metrics')

    # Determine stable window
    # If t1_max provided, use [t0_hint, t1_max]; else detect blow-up and stop before it
    if t1_max is not None:
        t_fail = float('nan')
        t0 = float(t0_hint)
        t1 = float(min(t[-1], t1_max))
    else:
        blow_idx = find_blowup_index(t, E_tot)
        if blow_idx is None:
            t_fail = t[-1]
        else:
            t_fail = float(t[blow_idx])
        # Choose stable window [t0, t1)
        t0 = float(t0_hint)
        t1 = float(min(t[-1], t_fail - margin))
        if not (t1 > t0 + 0.5):
            # fallback: use last 40% prior to failure
            t0 = float(t[ int(0.4 * len(t)) ])
            t1 = float(min(t[-1], t_fail - margin))
            if not (t1 > t0 + 0.25):
                # last resort: use first half
                t0 = float(t[int(0.2 * len(t))])
                t1 = float(t[int(0.6 * len(t))])

    w = (t >= t0) & (t <= t1)
    if np.count_nonzero(w) < 10:
        raise RuntimeError('Stable window too short for metrics')

    tw = t[w]
    Ew = E_tot[w]
    pbxw = pbx[w]

    # omega via Hilbert phase slope (if available)
    omega = float('nan')
    if hilbert_phase_slope is not None:
        series = pbxw - np.mean(pbxw)
        if np.any(np.isfinite(series)) and np.nanmax(np.abs(series)) > 0:
            omega = float(hilbert_phase_slope(tw, series))

    # gamma via energy decay fit: 0.5 * slope of ln(E)
    mask = np.isfinite(Ew) & (Ew > 0)
    gamma = float('nan')
    if np.count_nonzero(mask) > 10:
        coeffs = np.polyfit(tw[mask], np.log(Ew[mask] + 1e-300), 1)
        gamma = 0.5 * float(coeffs[0])

    # dE/dt max in window
    dE = np.gradient(Ew, tw)
    dE_dt_max = float(np.nanmax(np.abs(dE)))

    # energy drift (relative) in window
    E0 = float(Ew[0])
    drift = float((np.nanmax(Ew) - np.nanmin(Ew)) / (abs(E0) + 1e-300))

    result = {
        't0': float(t0),
        't1': float(t1),
        't_fail_est': float(t_fail),
        'omega': float(omega),
        'gamma': float(gamma),
        'dE_dt_max': float(dE_dt_max),
        'energy_drift_rel': float(drift),
        'samples': int(np.count_nonzero(w)),
        'window_length': float(t1 - t0),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--t0', type=float, default=10.0)
    ap.add_argument('--t1', type=float, default=None)
    args = ap.parse_args()

    res = compute_metrics(args.csv, t0_hint=args.t0, t1_max=args.t1)
    print('METRICS_JSON=' + json.dumps(res))


if __name__ == '__main__':
    main()
