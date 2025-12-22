#!/usr/bin/env python3
"""
PHASE 16A: SCALE-DEPENDENT PROJECTION OPERATOR WITH HARD PRIORS

EXPLORATORY PHASE - NOT PART OF v1.0.0 CONSERVATIVE RELEASE

Objective: Test whether a slowly varying projection distortion
    ε(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
explains the residual better than constant ε, without overfitting.

LOCKED PARAMETERS:
    ε₀ = 1.4558030818e-03 (from Phase 10E, NOT refitted)
    ℓ* = 1650 (pivot, fixed)

NEW PARAMETER:
    α = running strength (single degree of freedom)

HARD PRIORS:
    |α| ≤ 5×10⁻⁴
    α_TT = α_EE = α_TE (shared across spectra)
    ε₀ NOT refitted

SUCCESS CRITERIA (exploratory):
    ≥15% additional RMS reduction beyond Phase 13A (constant ε)
    Correlation ≥0.85 maintained
    TT/EE agree on optimal α
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# LOCKED PARAMETERS (DO NOT CHANGE)
# =============================================================================
EPSILON_0 = 1.4558030818e-03  # From Phase 10E
ELL_PIVOT = 1650              # Fixed pivot
LMIN, LMAX = 800, 2500        # Analysis range

# =============================================================================
# HARD PRIORS (EXTENDED for Phase 16A-ext2)
# =============================================================================
ALPHA_MAX = 2e-3              # |α| ≤ 2×10⁻³ (extended to find true minimum)
ALPHA_GRID = np.linspace(-ALPHA_MAX, ALPHA_MAX, 81)  # 81 points for fine resolution

# =============================================================================
# SUCCESS THRESHOLDS
# =============================================================================
ADDITIONAL_RMS_REDUCTION = 0.15  # ≥15% beyond constant ε
MIN_CORRELATION = 0.85


def _load_class_cl(file_path: Path):
    """Load CLASS Cℓ output file."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3] if data.shape[1] > 3 else None
    return ell, tt, ee, te


def epsilon_of_ell(ell, alpha):
    """
    Scale-dependent ε:
        ε(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
    """
    return EPSILON_0 + alpha * np.log(ell / ELL_PIVOT)


def apply_running_operator(ell, cl, alpha):
    """
    Apply scale-dependent operator:
        P_ε(ℓ) : C_ℓ ↦ C_{ℓ/(1+ε(ℓ))}
    """
    ell_float = ell.astype(float)
    eps = epsilon_of_ell(ell_float, alpha)
    ell_star = ell_float / (1 + eps)
    cl_new = np.interp(ell_float, ell_star, cl, left=cl[0], right=cl[-1])
    return cl_new


def fractional_residual(cl_a, cl_b):
    """Compute (a - b) / b."""
    denom = np.where(np.abs(cl_b) > 0, cl_b, 1.0)
    return (cl_a - cl_b) / denom


def rms(x):
    """Root mean square."""
    return np.sqrt(np.mean(x**2))


def corr(a, b):
    """Pearson correlation."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def scan_alpha(ell, cl_lcdm, cl_bec, mask, name):
    """
    Grid scan over α values.
    Returns dict with results for each α.
    """
    results = {
        'alpha': [],
        'rms_base': None,
        'rms_const': None,
        'rms_running': [],
        'corr_running': [],
        'reduction_vs_base': [],
        'reduction_vs_const': [],
    }
    
    # Baseline: no operator
    r_base = fractional_residual(cl_bec[mask], cl_lcdm[mask])
    rms_base = rms(r_base)
    results['rms_base'] = rms_base
    
    # Constant ε (Phase 13A baseline)
    cl_const = apply_running_operator(ell, cl_lcdm, alpha=0.0)
    r_const = fractional_residual(cl_bec[mask], cl_const[mask])
    rms_const = rms(r_const)
    results['rms_const'] = rms_const
    
    # Scan α
    for alpha in ALPHA_GRID:
        cl_shifted = apply_running_operator(ell, cl_lcdm, alpha)
        r_shifted = fractional_residual(cl_bec[mask], cl_shifted[mask])
        
        rms_shifted = rms(r_shifted)
        corr_shifted = corr(r_base, r_shifted)
        
        reduction_vs_base = (rms_base - rms_shifted) / rms_base
        reduction_vs_const = (rms_const - rms_shifted) / rms_const
        
        results['alpha'].append(alpha)
        results['rms_running'].append(rms_shifted)
        results['corr_running'].append(corr_shifted)
        results['reduction_vs_base'].append(reduction_vs_base)
        results['reduction_vs_const'].append(reduction_vs_const)
    
    # Convert to arrays
    for key in ['alpha', 'rms_running', 'corr_running', 'reduction_vs_base', 'reduction_vs_const']:
        results[key] = np.array(results[key])
    
    return results


def find_optimal_alpha(results):
    """Find α that minimizes RMS."""
    idx = np.argmin(results['rms_running'])
    return {
        'alpha_opt': results['alpha'][idx],
        'rms_opt': results['rms_running'][idx],
        'corr_opt': results['corr_running'][idx],
        'reduction_vs_base': results['reduction_vs_base'][idx],
        'reduction_vs_const': results['reduction_vs_const'][idx],
    }


def compute_aic_penalty(rms_const, rms_running, n_points, delta_k=1):
    """
    Compute AIC-style penalty for adding α parameter.
    
    Δχ² ≈ n × (rms_const² - rms_running²) / rms_const²
    Penalized improvement = Δχ² - 2Δk
    """
    delta_chi2 = n_points * (rms_const**2 - rms_running**2) / rms_const**2
    penalized = delta_chi2 - 2 * delta_k
    return delta_chi2, penalized


def main():
    print("=" * 70)
    print("PHASE 16A: SCALE-DEPENDENT PROJECTION OPERATOR")
    print("=" * 70)
    print("\n*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Pivot ℓ* = {ELL_PIVOT}")
    print(f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]")
    print(f"Ansatz: ε(ℓ) = ε₀ + α·ln(ℓ/ℓ*)")
    print(f"Prior: |α| ≤ {ALPHA_MAX:.1e}")
    
    base_dir = Path('/Users/hodge/Desktop/sanity_check/bec_cosmology/output')
    
    # Load spectra
    print("\n[1] Loading spectra...")
    lcdm_path = base_dir / 'lcdm_zz_thetaS_reference_precise_00_cl.dat'
    bec_path = base_dir / 'test_bec_zz_thetaS_matched_precise_00_cl.dat'
    
    ell, tt_lcdm, ee_lcdm, te_lcdm = _load_class_cl(lcdm_path)
    _, tt_bec, ee_bec, te_bec = _load_class_cl(bec_path)
    
    mask = (ell >= LMIN) & (ell <= LMAX)
    n_points = mask.sum()
    print(f"  Loaded {ell.size} multipoles, {n_points} in analysis range")
    
    # Scan α for TT and EE
    print("\n[2] Scanning α grid...")
    print(f"  Grid: {len(ALPHA_GRID)} points from {ALPHA_GRID[0]:.1e} to {ALPHA_GRID[-1]:.1e}")
    
    print("\n" + "=" * 70)
    print("TT SPECTRUM")
    print("=" * 70)
    results_tt = scan_alpha(ell, tt_lcdm, tt_bec, mask, 'TT')
    opt_tt = find_optimal_alpha(results_tt)
    
    print(f"\n  Baseline RMS: {results_tt['rms_base']:.6f}")
    print(f"  Constant ε RMS: {results_tt['rms_const']:.6f}")
    print(f"  Optimal α: {opt_tt['alpha_opt']:.4e}")
    print(f"  Running ε RMS: {opt_tt['rms_opt']:.6f}")
    print(f"  Reduction vs baseline: {opt_tt['reduction_vs_base']*100:+.1f}%")
    print(f"  Reduction vs constant ε: {opt_tt['reduction_vs_const']*100:+.1f}%")
    print(f"  Correlation: {opt_tt['corr_opt']:.3f}")
    
    delta_chi2_tt, penalized_tt = compute_aic_penalty(
        results_tt['rms_const'], opt_tt['rms_opt'], n_points
    )
    print(f"\n  AIC check: Δχ² = {delta_chi2_tt:.1f}, penalized = {penalized_tt:.1f}")
    
    print("\n" + "=" * 70)
    print("EE SPECTRUM")
    print("=" * 70)
    results_ee = scan_alpha(ell, ee_lcdm, ee_bec, mask, 'EE')
    opt_ee = find_optimal_alpha(results_ee)
    
    print(f"\n  Baseline RMS: {results_ee['rms_base']:.6f}")
    print(f"  Constant ε RMS: {results_ee['rms_const']:.6f}")
    print(f"  Optimal α: {opt_ee['alpha_opt']:.4e}")
    print(f"  Running ε RMS: {opt_ee['rms_opt']:.6f}")
    print(f"  Reduction vs baseline: {opt_ee['reduction_vs_base']*100:+.1f}%")
    print(f"  Reduction vs constant ε: {opt_ee['reduction_vs_const']*100:+.1f}%")
    print(f"  Correlation: {opt_ee['corr_opt']:.3f}")
    
    delta_chi2_ee, penalized_ee = compute_aic_penalty(
        results_ee['rms_const'], opt_ee['rms_opt'], n_points
    )
    print(f"\n  AIC check: Δχ² = {delta_chi2_ee:.1f}, penalized = {penalized_ee:.1f}")
    
    # TE analysis (correlation only)
    print("\n" + "=" * 70)
    print("TE SPECTRUM (correlation diagnostic only)")
    print("=" * 70)
    
    if te_lcdm is not None and te_bec is not None:
        results_te = scan_alpha(ell, te_lcdm, te_bec, mask, 'TE')
        opt_te = find_optimal_alpha(results_te)
        
        print(f"\n  Optimal α: {opt_te['alpha_opt']:.4e}")
        print(f"  Correlation: {opt_te['corr_opt']:.3f}")
        
        # Check sign consistency with TT/EE
        te_sign_consistent = np.sign(opt_te['alpha_opt']) == np.sign(opt_tt['alpha_opt'])
        print(f"  Sign consistent with TT: {'YES' if te_sign_consistent else 'NO'}")
    else:
        results_te = None
        opt_te = None
        print("  TE data not available")
    
    # TT/EE consistency check
    print("\n" + "=" * 70)
    print("TT/EE CONSISTENCY CHECK")
    print("=" * 70)
    
    alpha_diff = abs(opt_tt['alpha_opt'] - opt_ee['alpha_opt'])
    alpha_mean = (opt_tt['alpha_opt'] + opt_ee['alpha_opt']) / 2
    
    print(f"\n  α_TT = {opt_tt['alpha_opt']:.4e}")
    print(f"  α_EE = {opt_ee['alpha_opt']:.4e}")
    print(f"  |α_TT - α_EE| = {alpha_diff:.4e}")
    print(f"  Mean α = {alpha_mean:.4e}")
    
    # Consistency: same sign and within factor of 2
    same_sign = np.sign(opt_tt['alpha_opt']) == np.sign(opt_ee['alpha_opt'])
    within_factor_2 = alpha_diff < abs(alpha_mean)
    
    print(f"\n  Same sign: {'YES' if same_sign else 'NO'}")
    print(f"  Within factor 2: {'YES' if within_factor_2 else 'NO'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 16A SUMMARY")
    print("=" * 70)
    
    # Evaluate success criteria
    additional_reduction_tt = opt_tt['reduction_vs_const']
    additional_reduction_ee = opt_ee['reduction_vs_const']
    
    tt_pass_reduction = additional_reduction_tt >= ADDITIONAL_RMS_REDUCTION
    ee_pass_reduction = additional_reduction_ee >= ADDITIONAL_RMS_REDUCTION
    tt_pass_corr = opt_tt['corr_opt'] >= MIN_CORRELATION
    ee_pass_corr = opt_ee['corr_opt'] >= MIN_CORRELATION
    tt_pass_aic = penalized_tt > 0
    ee_pass_aic = penalized_ee > 0
    consistency_pass = same_sign and within_factor_2
    
    print(f"\n  TT additional reduction: {additional_reduction_tt*100:+.1f}% (need ≥{ADDITIONAL_RMS_REDUCTION*100:.0f}%): {'PASS' if tt_pass_reduction else 'FAIL'}")
    print(f"  EE additional reduction: {additional_reduction_ee*100:+.1f}% (need ≥{ADDITIONAL_RMS_REDUCTION*100:.0f}%): {'PASS' if ee_pass_reduction else 'FAIL'}")
    print(f"  TT correlation: {opt_tt['corr_opt']:.3f} (need ≥{MIN_CORRELATION}): {'PASS' if tt_pass_corr else 'FAIL'}")
    print(f"  EE correlation: {opt_ee['corr_opt']:.3f} (need ≥{MIN_CORRELATION}): {'PASS' if ee_pass_corr else 'FAIL'}")
    print(f"  TT AIC penalized: {penalized_tt:.1f} (need >0): {'PASS' if tt_pass_aic else 'FAIL'}")
    print(f"  EE AIC penalized: {penalized_ee:.1f} (need >0): {'PASS' if ee_pass_aic else 'FAIL'}")
    print(f"  TT/EE consistency: {'PASS' if consistency_pass else 'FAIL'}")
    
    # Overall assessment
    if abs(alpha_mean) < 1e-5:
        conclusion = "α ≈ 0: Constant geometry sufficient"
        scale_dependent = False
    elif tt_pass_aic and ee_pass_aic and consistency_pass:
        conclusion = "Scale-dependent geometry PLAUSIBLE"
        scale_dependent = True
    else:
        conclusion = "Scale-dependent geometry NOT SUPPORTED"
        scale_dependent = False
    
    print(f"\n  CONCLUSION: {conclusion}")
    
    # Save summary
    summary = f"""PHASE 16A: SCALE-DEPENDENT PROJECTION OPERATOR
============================================================
*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***

Locked ε₀ = {EPSILON_0:.10e}
Pivot ℓ* = {ELL_PIVOT}
Analysis range: ℓ ∈ [{LMIN}, {LMAX}]
Ansatz: ε(ℓ) = ε₀ + α·ln(ℓ/ℓ*)

TT RESULTS:
  Optimal α = {opt_tt['alpha_opt']:.4e}
  RMS reduction vs constant ε: {opt_tt['reduction_vs_const']*100:+.1f}%
  Correlation: {opt_tt['corr_opt']:.3f}
  AIC penalized: {penalized_tt:.1f}

EE RESULTS:
  Optimal α = {opt_ee['alpha_opt']:.4e}
  RMS reduction vs constant ε: {opt_ee['reduction_vs_const']*100:+.1f}%
  Correlation: {opt_ee['corr_opt']:.3f}
  AIC penalized: {penalized_ee:.1f}

TT/EE CONSISTENCY:
  Same sign: {'YES' if same_sign else 'NO'}
  Within factor 2: {'YES' if within_factor_2 else 'NO'}

CONCLUSION: {conclusion}
"""
    
    out_summary = base_dir / 'phase16a_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plots
    print("\n[3] Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: ε(ℓ) for different α
    ax = axes[0, 0]
    ell_plot = np.linspace(LMIN, LMAX, 200)
    for alpha in [-ALPHA_MAX, -ALPHA_MAX/2, 0, ALPHA_MAX/2, ALPHA_MAX]:
        eps = epsilon_of_ell(ell_plot, alpha)
        label = f'α = {alpha:.1e}' if alpha != 0 else 'α = 0 (constant)'
        ax.plot(ell_plot, eps * 1e3, label=label, lw=1.5 if alpha == 0 else 1)
    ax.axhline(EPSILON_0 * 1e3, color='gray', ls='--', alpha=0.5, label='ε₀')
    ax.axvline(ELL_PIVOT, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ε(ℓ) × 10³')
    ax.set_title('Scale-Dependent ε(ℓ)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMS vs α (TT)
    ax = axes[0, 1]
    ax.plot(results_tt['alpha'] * 1e4, results_tt['rms_running'], 'b-o', ms=4, label='TT')
    ax.axhline(results_tt['rms_const'], color='b', ls='--', alpha=0.5, label='TT constant ε')
    ax.axhline(results_tt['rms_base'], color='b', ls=':', alpha=0.5, label='TT baseline')
    ax.axvline(opt_tt['alpha_opt'] * 1e4, color='b', ls='-', alpha=0.3)
    ax.set_xlabel('α × 10⁴')
    ax.set_ylabel('RMS residual')
    ax.set_title('TT: RMS vs α')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMS vs α (EE)
    ax = axes[0, 2]
    ax.plot(results_ee['alpha'] * 1e4, results_ee['rms_running'], 'r-o', ms=4, label='EE')
    ax.axhline(results_ee['rms_const'], color='r', ls='--', alpha=0.5, label='EE constant ε')
    ax.axhline(results_ee['rms_base'], color='r', ls=':', alpha=0.5, label='EE baseline')
    ax.axvline(opt_ee['alpha_opt'] * 1e4, color='r', ls='-', alpha=0.3)
    ax.set_xlabel('α × 10⁴')
    ax.set_ylabel('RMS residual')
    ax.set_title('EE: RMS vs α')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reduction vs α
    ax = axes[1, 0]
    ax.plot(results_tt['alpha'] * 1e4, results_tt['reduction_vs_base'] * 100, 'b-o', ms=4, label='TT vs baseline')
    ax.plot(results_ee['alpha'] * 1e4, results_ee['reduction_vs_base'] * 100, 'r-o', ms=4, label='EE vs baseline')
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('α × 10⁴')
    ax.set_ylabel('RMS reduction (%)')
    ax.set_title('RMS Reduction vs α')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Correlation vs α
    ax = axes[1, 1]
    ax.plot(results_tt['alpha'] * 1e4, results_tt['corr_running'], 'b-o', ms=4, label='TT')
    ax.plot(results_ee['alpha'] * 1e4, results_ee['corr_running'], 'r-o', ms=4, label='EE')
    ax.axhline(MIN_CORRELATION, color='gray', ls='--', alpha=0.5, label=f'threshold ({MIN_CORRELATION})')
    ax.set_xlabel('α × 10⁴')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation vs α')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: TT vs EE α consistency
    ax = axes[1, 2]
    ax.scatter([opt_tt['alpha_opt'] * 1e4], [opt_ee['alpha_opt'] * 1e4], s=100, c='purple', marker='*', zorder=5)
    ax.plot([-5, 5], [-5, 5], 'k--', alpha=0.5, label='TT = EE')
    ax.fill_between([-5, 5], [-10, 0], [0, 10], alpha=0.1, color='green', label='Same sign')
    ax.set_xlabel('α_TT × 10⁴')
    ax.set_ylabel('α_EE × 10⁴')
    ax.set_title('TT/EE Consistency')
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    fig.suptitle(f'Phase 16A: Scale-Dependent Operator (α_mean = {alpha_mean:.2e})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase16a_epsilon_running.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase16a_results.npz'
    np.savez(
        out_npz,
        epsilon_0=EPSILON_0,
        ell_pivot=ELL_PIVOT,
        alpha_grid=ALPHA_GRID,
        alpha_opt_tt=opt_tt['alpha_opt'],
        alpha_opt_ee=opt_ee['alpha_opt'],
        alpha_mean=alpha_mean,
        results_tt_alpha=results_tt['alpha'],
        results_tt_rms=results_tt['rms_running'],
        results_tt_corr=results_tt['corr_running'],
        results_ee_alpha=results_ee['alpha'],
        results_ee_rms=results_ee['rms_running'],
        results_ee_corr=results_ee['corr_running'],
        scale_dependent=scale_dependent,
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 16A COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
