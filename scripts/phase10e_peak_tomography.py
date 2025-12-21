#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _load_class_cl(file_path: Path):
    data = np.loadtxt(file_path)
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    return ell, tt, ee


def _fractional_residual(cl_bec, cl_lcdm):
    denom = np.where(cl_lcdm != 0.0, cl_lcdm, np.nan)
    r = (cl_bec - cl_lcdm) / denom
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _dlncl_dell(ell: np.ndarray, cl: np.ndarray):
    cl_safe = np.where(cl > 0.0, cl, np.nan)
    ln = np.log(cl_safe)
    d = np.gradient(ln, ell.astype(float))
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_local_delta_ell(residual: np.ndarray, dlncl: np.ndarray):
    """Fit residual ≈ -Δℓ * dlnCl/dℓ + b.

    Returns:
      delta_ell, b, sigma_delta_ell, r2

    Notes:
      - OLS with intercept.
      - sigma_delta_ell from (X^T X)^{-1} * s^2.
    """
    y = residual.astype(float)
    x = (-dlncl).astype(float)
    X = np.column_stack([x, np.ones_like(x)])

    # OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    delta_ell = float(beta[0])
    b = float(beta[1])

    yhat = X @ beta
    resid = y - yhat

    n = int(y.size)
    p = 2
    dof = max(n - p, 1)
    s2 = float(np.sum(resid**2) / dof)

    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = s2 * XtX_inv
    sigma_delta_ell = float(np.sqrt(max(var_beta[0, 0], 0.0)))

    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (sse / sst) if sst != 0.0 else 0.0

    return delta_ell, b, sigma_delta_ell, r2


def _sliding_window_fit(ell, r, dlncl, lmin, lmax, width, step):
    half = width // 2
    centers = np.arange(lmin + half, lmax - half + 1, step, dtype=int)

    deltas = np.zeros_like(centers, dtype=float)
    sigmas = np.zeros_like(centers, dtype=float)
    r2s = np.zeros_like(centers, dtype=float)
    bs = np.zeros_like(centers, dtype=float)

    for i, c in enumerate(centers):
        lo = c - half
        hi = c + half
        m = (ell >= lo) & (ell <= hi)

        delta, b, sigma, r2 = _fit_local_delta_ell(r[m], dlncl[m])
        deltas[i] = delta
        bs[i] = b
        sigmas[i] = sigma
        r2s[i] = r2

    return centers, deltas, sigmas, r2s, bs


def _plot_delta_ell(centers, d_tt, s_tt, d_ee, s_ee, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(centers, d_tt, color='C0', linewidth=2.0, label='TT')
    ax.fill_between(centers, d_tt - s_tt, d_tt + s_tt, color='C0', alpha=0.2)

    ax.plot(centers, d_ee, color='C1', linewidth=2.0, linestyle='--', label='EE')
    ax.fill_between(centers, d_ee - s_ee, d_ee + s_ee, color='C1', alpha=0.2)

    ax.axhline(0.0, color='k', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('Δℓ (local phase shift)')
    ax.set_title('Phase 10E: Scale-dependent phase-shift tomography Δℓ(ℓ)\n(unlensed, precise θₛ-matched BEC vs ΛCDM)')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_r2(centers, r2_tt, r2_ee, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(centers, r2_tt, color='C0', linewidth=2.0, label='TT')
    ax.plot(centers, r2_ee, color='C1', linewidth=2.0, linestyle='--', label='EE')

    ax.set_xlabel('ℓ')
    ax.set_ylabel('R² (local)')
    ax.set_title('Phase 10E: Local explanatory power of phase-shift model (R² vs ℓ)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_residual_collapse(ell, r, dlncl, center, width, delta, b, out_path: Path, spec_label: str):
    half = width // 2
    lo = center - half
    hi = center + half
    m = (ell >= lo) & (ell <= hi)

    r_pred = (-delta) * dlncl[m] + b
    r_corr = r[m] - r_pred

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ell[m], r[m], color='C0', linewidth=1.4, label='Original residual')
    ax.plot(ell[m], r_corr, color='C2', linewidth=1.4, label='Locally phase-corrected residual')
    ax.axhline(0.0, color='k', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('Fractional residual')
    ax.set_title(f'Phase 10E: Residual collapse test ({spec_label}) around ℓ≈{center}')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size or a.size < 2:
        return np.nan
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    # Use relative paths from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    base_dir = repo_root / 'output'
    data_dir = repo_root / 'data'
    lcdm_path = data_dir / 'lcdm_unlensed' / 'lcdm_zz_thetaS_reference_precise_00_cl.dat'
    bec_path = data_dir / 'bec_unlensed' / 'test_bec_zz_thetaS_matched_precise_00_cl.dat'

    ell_b, tt_b, ee_b = _load_class_cl(bec_path)
    ell_l, tt_l, ee_l = _load_class_cl(lcdm_path)

    # Interpolate BEC to LCDM grid
    tt_bi = np.interp(ell_l, ell_b, tt_b)
    ee_bi = np.interp(ell_l, ell_b, ee_b)

    # Analysis range
    lmin = 800
    lmax = 2500
    m = (ell_l >= lmin) & (ell_l <= lmax)

    ell = ell_l[m]
    r_tt = _fractional_residual(tt_bi[m], tt_l[m])
    r_ee = _fractional_residual(ee_bi[m], ee_l[m])

    dln_tt = _dlncl_dell(ell, tt_l[m])
    dln_ee = _dlncl_dell(ell, ee_l[m])

    # Sliding windows
    width = 120
    step = 20

    centers, d_tt, s_tt, r2_tt, b_tt = _sliding_window_fit(ell, r_tt, dln_tt, lmin, lmax, width, step)
    _, d_ee, s_ee, r2_ee, b_ee = _sliding_window_fit(ell, r_ee, dln_ee, lmin, lmax, width, step)

    # Plots
    out_delta = base_dir / 'phase10e_delta_ell_TT_EE.png'
    out_r2 = base_dir / 'phase10e_r2_vs_ell.png'
    _plot_delta_ell(centers, d_tt, s_tt, d_ee, s_ee, out_delta)
    _plot_r2(centers, r2_tt, r2_ee, out_r2)

    # Residual collapse test (TT, representative window around ell~1400)
    # Pick closest center
    target_center = 1400
    idx = int(np.argmin(np.abs(centers - target_center)))
    c = int(centers[idx])
    out_collapse = base_dir / 'phase10e_residual_collapse_TT_ell1400.png'
    _plot_residual_collapse(ell, r_tt, dln_tt, c, width, d_tt[idx], b_tt[idx], out_collapse, spec_label='TT')

    # Summary metrics
    mean_tt = float(np.mean(d_tt))
    mean_ee = float(np.mean(d_ee))
    drift_tt = float(np.max(d_tt) - np.min(d_tt))
    drift_ee = float(np.max(d_ee) - np.min(d_ee))
    drift = max(drift_tt, drift_ee)
    corr = _corr(d_tt, d_ee)

    # Compare to global 10D-style fit R^2 (same data, full range) for improvement metric
    # Here we interpret "improvement" as mean(local R^2) - global R^2.
    def global_r2(r, dln):
        y = r
        x = (-dln)
        X = np.column_stack([x, np.ones_like(x)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - (sse / sst) if sst != 0.0 else 0.0

    r2g_tt = global_r2(r_tt, dln_tt)
    r2g_ee = global_r2(r_ee, dln_ee)

    r2_imp_tt = float(np.mean(r2_tt) - r2g_tt)
    r2_imp_ee = float(np.mean(r2_ee) - r2g_ee)

    summary = []
    summary.append('PHASE 10E SUMMARY')
    summary.append('-----------------')
    summary.append(f'Mean Δℓ (TT):  {mean_tt:+.4f}')
    summary.append(f'Mean Δℓ (EE):  {mean_ee:+.4f}')
    summary.append(f'Δℓ drift amplitude (peak-to-peak):  {drift:.4f}')
    summary.append(f'Correlation(TT,EE):  {corr:.4f}')
    summary.append(f'R² improvement vs Phase 10D (TT):  {r2_imp_tt:+.4f}')
    summary.append(f'R² improvement vs Phase 10D (EE):  {r2_imp_ee:+.4f}')

    summary_txt = '\n'.join(summary) + '\n'

    print(summary_txt)

    out_txt = base_dir / 'phase10e_summary.txt'
    out_txt.write_text(summary_txt)

    out_npz = base_dir / 'phase10e_tomography.npz'
    np.savez(
        out_npz,
        centers=centers,
        delta_tt=d_tt,
        sigma_tt=s_tt,
        r2_tt=r2_tt,
        b_tt=b_tt,
        delta_ee=d_ee,
        sigma_ee=s_ee,
        r2_ee=r2_ee,
        b_ee=b_ee,
        lmin=lmin,
        lmax=lmax,
        width=width,
        step=step,
    )

    print(f'Saved: {out_delta}')
    print(f'Saved: {out_r2}')
    print(f'Saved: {out_collapse}')
    print(f'Saved: {out_txt}')
    print(f'Saved: {out_npz}')


if __name__ == '__main__':
    main()
