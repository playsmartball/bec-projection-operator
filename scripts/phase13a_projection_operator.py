#!/usr/bin/env python3
"""
Phase 13A: Projection-Space Convolution Kernel

Implements a fixed ℓ-ℓ′ convolution kernel applied AFTER k-integration:

    Cℓ_new = Σ_ℓ′ K(ℓ,ℓ′) Cℓ′

where K(ℓ,ℓ′) is a narrow kernel centered at ℓ′ ≈ ℓ with width σ = |ε|·ℓ.

Key properties:
- ε is LOCKED from Phase 10E tomography (no tuning)
- Kernel preserves total power (normalized)
- Acts at projection level, not Boltzmann level
- Should reproduce Phase 12B behavior if the effect is projection-space physics

This tests the hypothesis that the BEC residual is a coherent angular/projection
effect rather than a source-level modification.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import time


def _load_class_cl(file_path: Path):
    """Load CLASS Cℓ output file."""
    print(f"  Loading {file_path.name}...", end=" ", flush=True)
    t0 = time.time()
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    print(f"done ({time.time()-t0:.2f}s, n_ell={ell.size})")
    return ell, tt, ee


def _load_phase10e_epsilon():
    """Load locked ε from Phase 10E tomography."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    npz_path = repo_root / 'data' / 'phase10e_tomography.npz'
    data = np.load(npz_path)
    
    # Mean Δℓ from TT and EE
    delta_tt = data['delta_tt']
    delta_ee = data['delta_ee']
    centers = data['centers']
    
    # ε = mean(|Δℓ/ℓ|) over tomography range
    # Note: We use absolute value because BEC peaks are at lower ℓ (negative Δℓ),
    # but the operator ℓ → ℓ/(1+ε) with positive ε shifts ΛCDM toward BEC.
    epsilon_tt = np.abs(delta_tt) / centers
    epsilon_ee = np.abs(delta_ee) / centers
    epsilon = np.mean(np.concatenate([epsilon_tt, epsilon_ee]))
    
    print(f"  Phase 10E locked ε: {epsilon:.10e}")
    print(f"    (from mean(|Δℓ/ℓ|) over {len(centers)} windows)")
    
    # Also compute mean Δℓ for summary output
    mean_delta = (np.mean(np.abs(delta_tt)) + np.mean(np.abs(delta_ee))) / 2
    return epsilon, mean_delta


def _apply_projection_kernel_gaussian(ell, cl, epsilon, kernel_type='gaussian'):
    """
    Apply ℓ-ℓ′ convolution kernel to Cℓ spectrum.
    
    The kernel width at each ℓ is σ(ℓ) = |ε| · ℓ
    
    For a Gaussian kernel:
        K(ℓ,ℓ′) ∝ exp(-(ℓ-ℓ′)²/(2σ²))
    
    This is equivalent to a scale-dependent smoothing where the
    smoothing width grows linearly with ℓ.
    """
    cl_new = np.zeros_like(cl)
    abs_eps = abs(epsilon)
    
    for i, L in enumerate(ell):
        # Kernel width at this ℓ
        sigma = abs_eps * L
        
        if sigma < 0.5:
            # Width too small, no convolution needed
            cl_new[i] = cl[i]
            continue
        
        # Build kernel weights for all ℓ′
        # K(ℓ,ℓ′) = exp(-(ℓ-ℓ′)²/(2σ²))
        delta_ell = ell - L
        weights = np.exp(-0.5 * (delta_ell / sigma)**2)
        
        # Normalize to preserve power
        weights /= np.sum(weights)
        
        # Apply convolution
        cl_new[i] = np.sum(weights * cl)
    
    return cl_new


def _apply_projection_kernel_shift(ell, cl, epsilon):
    """
    Apply horizontal shift operator (Phase 12B style) at projection level.
    
    This is the ℓ → ℓ/(1+ε) remap applied to final Cℓ.
    Equivalent to interpolating Cℓ at shifted ℓ values.
    """
    ell_star = ell / (1 + epsilon)
    cl_new = np.interp(ell, ell_star, cl, left=cl[0], right=cl[-1])
    return cl_new


def _apply_projection_kernel_antisymmetric(ell, cl, epsilon):
    """
    Apply antisymmetric derivative-like kernel.
    
    This tests whether the effect is more like a phase gradient
    than a simple shift.
    
    K(ℓ,ℓ′) ∝ (ℓ′-ℓ) · exp(-(ℓ-ℓ′)²/(2σ²))
    
    This produces a derivative-weighted convolution.
    """
    cl_new = np.zeros_like(cl)
    abs_eps = abs(epsilon)
    sign_eps = np.sign(epsilon)
    
    for i, L in enumerate(ell):
        sigma = abs_eps * L
        
        if sigma < 0.5:
            cl_new[i] = cl[i]
            continue
        
        delta_ell = ell - L
        
        # Antisymmetric kernel: (ℓ′-ℓ) weighted Gaussian
        weights_asym = delta_ell * np.exp(-0.5 * (delta_ell / sigma)**2)
        
        # Normalize by RMS to get unit response
        norm = np.sqrt(np.sum(weights_asym**2))
        if norm > 0:
            weights_asym /= norm
        
        # The antisymmetric part adds a derivative-like correction
        # Scale by ε to get the right magnitude
        correction = sign_eps * abs_eps * np.sum(weights_asym * cl)
        
        cl_new[i] = cl[i] + correction
    
    return cl_new


def _fractional_residual(cl_a, cl_b):
    denom = np.where(np.abs(cl_b) > 0, cl_b, 1.0)
    return (cl_a - cl_b) / denom


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def _corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size or a.size < 2:
        return np.nan
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("=" * 70)
    print("PHASE 13A: PROJECTION-SPACE CONVOLUTION KERNEL")
    print("=" * 70)
    
    # Use relative paths from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    base_dir = repo_root / 'output'
    data_dir = repo_root / 'data'
    
    # Load spectra
    print("\n[1] Loading spectra...")
    lcdm_path = data_dir / 'lcdm_unlensed' / 'lcdm_zz_thetaS_reference_precise_00_cl.dat'
    bec_path = data_dir / 'bec_unlensed' / 'test_bec_zz_thetaS_matched_precise_00_cl.dat'
    
    ell_lcdm, tt_lcdm, ee_lcdm = _load_class_cl(lcdm_path)
    ell_bec, tt_bec, ee_bec = _load_class_cl(bec_path)
    
    # Load locked ε from Phase 10E
    print("\n[2] Loading locked ε from Phase 10E...")
    epsilon, mean_delta_ell = _load_phase10e_epsilon()
    
    # Align to common grid
    print("\n[3] Aligning to common ell grid...")
    common_ell = np.intersect1d(ell_lcdm, ell_bec)
    idx_lcdm = np.searchsorted(ell_lcdm, common_ell)
    idx_bec = np.searchsorted(ell_bec, common_ell)
    
    ell = common_ell
    tt_lcdm_c = tt_lcdm[idx_lcdm]
    ee_lcdm_c = ee_lcdm[idx_lcdm]
    tt_bec_c = tt_bec[idx_bec]
    ee_bec_c = ee_bec[idx_bec]
    
    # Analysis range
    lmin, lmax = 800, 2500
    m = (ell >= lmin) & (ell <= lmax)
    print(f"  Analysis range: [{lmin}, {lmax}], n={m.sum()}")
    
    # =========================================================================
    # Apply different kernel types
    # =========================================================================
    print("\n[4] Applying projection kernels to ΛCDM...")
    
    kernels = {}
    
    # Kernel 1: Horizontal shift (Phase 12B style)
    print("  [4a] Horizontal shift kernel (ℓ → ℓ/(1+ε))...")
    t0 = time.time()
    tt_shift = _apply_projection_kernel_shift(ell, tt_lcdm_c, epsilon)
    ee_shift = _apply_projection_kernel_shift(ell, ee_lcdm_c, epsilon)
    print(f"       done ({time.time()-t0:.2f}s)")
    kernels['shift'] = {'tt': tt_shift, 'ee': ee_shift, 'label': 'Horizontal shift (ε)'}
    
    # Kernel 1b: Horizontal shift with FLIPPED sign (sign convention test)
    print("  [4a'] Horizontal shift kernel with FLIPPED ε...")
    t0 = time.time()
    tt_shift_flip = _apply_projection_kernel_shift(ell, tt_lcdm_c, -epsilon)
    ee_shift_flip = _apply_projection_kernel_shift(ell, ee_lcdm_c, -epsilon)
    print(f"       done ({time.time()-t0:.2f}s)")
    kernels['shift_flip'] = {'tt': tt_shift_flip, 'ee': ee_shift_flip, 'label': 'Horizontal shift (-ε)'}
    
    # Kernel 2: Gaussian smoothing with ε·ℓ width
    print("  [4b] Gaussian convolution kernel (σ = |ε|·ℓ)...")
    t0 = time.time()
    tt_gauss = _apply_projection_kernel_gaussian(ell, tt_lcdm_c, epsilon)
    ee_gauss = _apply_projection_kernel_gaussian(ell, ee_lcdm_c, epsilon)
    print(f"       done ({time.time()-t0:.2f}s)")
    kernels['gaussian'] = {'tt': tt_gauss, 'ee': ee_gauss, 'label': 'Gaussian conv'}
    
    # Kernel 3: Antisymmetric (derivative-like)
    print("  [4c] Antisymmetric derivative kernel...")
    t0 = time.time()
    tt_asym = _apply_projection_kernel_antisymmetric(ell, tt_lcdm_c, epsilon)
    ee_asym = _apply_projection_kernel_antisymmetric(ell, ee_lcdm_c, epsilon)
    print(f"       done ({time.time()-t0:.2f}s)")
    kernels['antisym'] = {'tt': tt_asym, 'ee': ee_asym, 'label': 'Antisymmetric'}
    
    # =========================================================================
    # Compute residuals and metrics
    # =========================================================================
    print("\n[5] Computing residual statistics...")
    
    # Baseline: BEC - LCDM
    r_tt_baseline = _fractional_residual(tt_bec_c[m], tt_lcdm_c[m])
    r_ee_baseline = _fractional_residual(ee_bec_c[m], ee_lcdm_c[m])
    rms_tt_baseline = _rms(r_tt_baseline)
    rms_ee_baseline = _rms(r_ee_baseline)
    
    print(f"\n  Baseline (BEC - LCDM):")
    print(f"    RMS TT: {rms_tt_baseline:.6f}")
    print(f"    RMS EE: {rms_ee_baseline:.6f}")
    
    results = {}
    for name, kdata in kernels.items():
        r_tt = _fractional_residual(tt_bec_c[m], kdata['tt'][m])
        r_ee = _fractional_residual(ee_bec_c[m], kdata['ee'][m])
        
        rms_tt = _rms(r_tt)
        rms_ee = _rms(r_ee)
        
        red_tt = (rms_tt_baseline - rms_tt) / rms_tt_baseline * 100
        red_ee = (rms_ee_baseline - rms_ee) / rms_ee_baseline * 100
        
        # Correlation between kernel effect and BEC residual
        effect_tt = _fractional_residual(kdata['tt'][m], tt_lcdm_c[m])
        effect_ee = _fractional_residual(kdata['ee'][m], ee_lcdm_c[m])
        corr_tt = _corr(effect_tt, r_tt_baseline)
        corr_ee = _corr(effect_ee, r_ee_baseline)
        
        results[name] = {
            'rms_tt': rms_tt, 'rms_ee': rms_ee,
            'red_tt': red_tt, 'red_ee': red_ee,
            'corr_tt': corr_tt, 'corr_ee': corr_ee,
            'label': kdata['label']
        }
        
        print(f"\n  {kdata['label']} kernel (BEC - ΛCDM_kernel):")
        print(f"    RMS TT: {rms_tt:.6f}  (reduction: {red_tt:+.1f}%)")
        print(f"    RMS EE: {rms_ee:.6f}  (reduction: {red_ee:+.1f}%)")
        print(f"    Corr(effect, BEC-LCDM) TT: {corr_tt:+.4f}")
        print(f"    Corr(effect, BEC-LCDM) EE: {corr_ee:+.4f}")
    
    # =========================================================================
    # Plots
    # =========================================================================
    print("\n[6] Generating plots...")
    
    # Plot 1: Residual comparison for all kernels
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    colors = {'shift': 'C2', 'shift_flip': 'C5', 'gaussian': 'C3', 'antisym': 'C4'}
    
    ax = axes[0]
    ax.plot(ell[m], r_tt_baseline * 100, 'C0-', lw=1.5, alpha=0.7, label='BEC - LCDM (baseline)')
    for name, kdata in kernels.items():
        r_tt = _fractional_residual(tt_bec_c[m], kdata['tt'][m])
        ax.plot(ell[m], r_tt * 100, f'{colors[name]}--', lw=1.2, alpha=0.7, 
                label=f'BEC - LCDM_{kdata["label"]}')
    ax.axhline(0, color='k', lw=1, alpha=0.5)
    ax.set_ylabel('Fractional residual [%]')
    ax.set_title(f'Phase 13A: Projection Kernel Comparison (ε = {epsilon:.4e})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'TT', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    ax = axes[1]
    ax.plot(ell[m], r_ee_baseline * 100, 'C1-', lw=1.5, alpha=0.7, label='BEC - LCDM (baseline)')
    for name, kdata in kernels.items():
        r_ee = _fractional_residual(ee_bec_c[m], kdata['ee'][m])
        ax.plot(ell[m], r_ee * 100, f'{colors[name]}--', lw=1.2, alpha=0.7,
                label=f'BEC - LCDM_{kdata["label"]}')
    ax.axhline(0, color='k', lw=1, alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('Fractional residual [%]')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'EE', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    fig.tight_layout()
    out_residual = base_dir / 'phase13a_residual_comparison.png'
    fig.savefig(out_residual, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_residual}")
    
    # Plot 2: Kernel effect visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    ax = axes[0]
    for name, kdata in kernels.items():
        effect = _fractional_residual(kdata['tt'][m], tt_lcdm_c[m])
        ax.plot(ell[m], effect * 100, f'{colors[name]}-', lw=1.5, label=kdata['label'])
    ax.axhline(0, color='k', lw=1, alpha=0.5)
    ax.set_ylabel('(LCDM_kernel - LCDM) / LCDM [%]')
    ax.set_title('Phase 13A: Kernel Effect on ΛCDM Spectra')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'TT', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    ax = axes[1]
    for name, kdata in kernels.items():
        effect = _fractional_residual(kdata['ee'][m], ee_lcdm_c[m])
        ax.plot(ell[m], effect * 100, f'{colors[name]}-', lw=1.5, label=kdata['label'])
    ax.axhline(0, color='k', lw=1, alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('(LCDM_kernel - LCDM) / LCDM [%]')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, 'EE', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    fig.tight_layout()
    out_effect = base_dir / 'phase13a_kernel_effect.png'
    fig.savefig(out_effect, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_effect}")
    
    # Plot 3: RMS reduction bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    kernel_names = list(results.keys())
    x = np.arange(len(kernel_names))
    width = 0.35
    
    red_tt = [results[k]['red_tt'] for k in kernel_names]
    red_ee = [results[k]['red_ee'] for k in kernel_names]
    
    bars1 = ax.bar(x - width/2, red_tt, width, label='TT', color='C0')
    bars2 = ax.bar(x + width/2, red_ee, width, label='EE', color='C1')
    
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel('RMS Reduction [%]')
    ax.set_title('Phase 13A: RMS Reduction by Kernel Type')
    ax.set_xticks(x)
    ax.set_xticklabels([results[k]['label'] for k in kernel_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords='offset points',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords='offset points',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
    
    fig.tight_layout()
    out_bar = base_dir / 'phase13a_rms_reduction.png'
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_bar}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 13A SUMMARY")
    print("=" * 70)
    
    summary_lines = [
        "PHASE 13A: PROJECTION-SPACE CONVOLUTION KERNEL",
        "=" * 50,
        "",
        f"Locked ε from Phase 10E: {epsilon:.6e}",
        f"Mean Δℓ from Phase 10E: {mean_delta_ell:.4f}",
        "",
        "BASELINE (BEC - LCDM)",
        "-" * 30,
        f"  RMS TT: {rms_tt_baseline:.6f}",
        f"  RMS EE: {rms_ee_baseline:.6f}",
        "",
        "KERNEL RESULTS",
        "-" * 30,
    ]
    
    for name, r in results.items():
        summary_lines.extend([
            f"",
            f"  {r['label']}:",
            f"    RMS TT: {r['rms_tt']:.6f}  (reduction: {r['red_tt']:+.1f}%)",
            f"    RMS EE: {r['rms_ee']:.6f}  (reduction: {r['red_ee']:+.1f}%)",
            f"    Corr(effect, BEC-LCDM) TT: {r['corr_tt']:+.4f}",
            f"    Corr(effect, BEC-LCDM) EE: {r['corr_ee']:+.4f}",
        ])
    
    # Determine best kernel
    best_tt = max(results.keys(), key=lambda k: results[k]['red_tt'])
    best_ee = max(results.keys(), key=lambda k: results[k]['red_ee'])
    
    summary_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 30,
        f"Best TT reduction: {results[best_tt]['label']} ({results[best_tt]['red_tt']:+.1f}%)",
        f"Best EE reduction: {results[best_ee]['label']} ({results[best_ee]['red_ee']:+.1f}%)",
    ])
    
    # Physics interpretation
    if results['shift']['red_tt'] > 0 and results['shift']['red_ee'] > 0:
        summary_lines.append("")
        summary_lines.append("✓ Horizontal shift kernel REDUCES BEC residuals")
        summary_lines.append("  → Confirms Phase 12B: effect is projection-level, not Boltzmann-level")
    
    summary_txt = '\n'.join(summary_lines)
    print(summary_txt)
    
    out_summary = base_dir / 'phase13a_summary.txt'
    out_summary.write_text(summary_txt + '\n')
    print(f"\nSaved: {out_summary}")
    
    # Save NPZ
    out_npz = base_dir / 'phase13a_results.npz'
    np.savez(
        out_npz,
        ell=ell,
        epsilon=epsilon,
        mean_delta_ell=mean_delta_ell,
        tt_lcdm=tt_lcdm_c,
        ee_lcdm=ee_lcdm_c,
        tt_bec=tt_bec_c,
        ee_bec=ee_bec_c,
        tt_shift=kernels['shift']['tt'],
        ee_shift=kernels['shift']['ee'],
        tt_gauss=kernels['gaussian']['tt'],
        ee_gauss=kernels['gaussian']['ee'],
        tt_asym=kernels['antisym']['tt'],
        ee_asym=kernels['antisym']['ee'],
        rms_tt_baseline=rms_tt_baseline,
        rms_ee_baseline=rms_ee_baseline,
        **{f'{k}_rms_tt': v['rms_tt'] for k, v in results.items()},
        **{f'{k}_rms_ee': v['rms_ee'] for k, v in results.items()},
        **{f'{k}_red_tt': v['red_tt'] for k, v in results.items()},
        **{f'{k}_red_ee': v['red_ee'] for k, v in results.items()},
    )
    print(f"Saved: {out_npz}")


if __name__ == '__main__':
    main()
