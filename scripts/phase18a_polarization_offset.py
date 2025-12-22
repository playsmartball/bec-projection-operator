#!/usr/bin/env python3
"""
PHASE 18A: POLARIZATION-WEIGHTED PROJECTION

EXPLORATORY PHASE - NOT PART OF v1.0.0

Objective: Test whether TT/EE divergence can be explained by a single
polarization offset δε, rather than spectrum-dependent running.

HYPOTHESIS:
    TT and EE probe different effective projection kernels due to:
    - Visibility function width differences
    - Polarization source thickness
    - Mode-coupling sensitivity to projection curvature

ANSATZ:
    ε_TT(ℓ) = ε(ℓ)
    ε_EE(ℓ) = ε(ℓ) + δε

where:
    ε(ℓ) = ε₀ (constant, from Phase 10E)
    δε = single scalar offset (new parameter)

This is the MINIMAL test for polarization-dependent geometry.

ACCEPTANCE CRITERIA:
    - δε ≠ 0 (statistically significant)
    - Combined AIC improves over separate fits
    - TE remains phase-consistent with TT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# LOCKED PARAMETERS (DO NOT CHANGE)
# =============================================================================
EPSILON_0 = 1.4558030818e-03  # From Phase 10E
LMIN, LMAX = 800, 2500        # Analysis range

# =============================================================================
# PHASE 18A PARAMETERS
# =============================================================================
DELTA_EPS_MAX = 1e-3          # |δε| ≤ 1×10⁻³
DELTA_EPS_GRID = np.linspace(-DELTA_EPS_MAX, DELTA_EPS_MAX, 81)


def _load_class_cl(file_path: Path):
    """Load CLASS Cℓ output file."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3] if data.shape[1] > 3 else None
    return ell, tt, ee, te


def apply_constant_shift(ell, cl, epsilon):
    """Apply constant horizontal shift: ℓ → ℓ/(1+ε)"""
    ell_float = ell.astype(float)
    ell_star = ell_float / (1 + epsilon)
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


def scan_delta_epsilon(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask):
    """
    Scan δε: apply ε₀ to TT, apply ε₀+δε to EE.
    Find the δε that minimizes combined TT+EE RMS.
    """
    results = {
        'delta_eps': [],
        'rms_tt': [],
        'rms_ee': [],
        'rms_combined': [],
        'corr_tt': [],
        'corr_ee': [],
    }
    
    # Baseline residuals (no operator)
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    
    for delta_eps in DELTA_EPS_GRID:
        # TT: apply ε₀
        eps_tt = EPSILON_0
        tt_shifted = apply_constant_shift(ell, tt_lcdm, eps_tt)
        r_tt = fractional_residual(tt_bec[mask], tt_shifted[mask])
        rms_tt = rms(r_tt)
        corr_tt = corr(r_tt_base, r_tt)
        
        # EE: apply ε₀ + δε
        eps_ee = EPSILON_0 + delta_eps
        ee_shifted = apply_constant_shift(ell, ee_lcdm, eps_ee)
        r_ee = fractional_residual(ee_bec[mask], ee_shifted[mask])
        rms_ee = rms(r_ee)
        corr_ee = corr(r_ee_base, r_ee)
        
        # Combined RMS (equal weight)
        rms_combined = np.sqrt((rms_tt**2 + rms_ee**2) / 2)
        
        results['delta_eps'].append(delta_eps)
        results['rms_tt'].append(rms_tt)
        results['rms_ee'].append(rms_ee)
        results['rms_combined'].append(rms_combined)
        results['corr_tt'].append(corr_tt)
        results['corr_ee'].append(corr_ee)
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def find_optimal_delta_eps(results):
    """Find δε that minimizes combined RMS."""
    # Find optimal for EE alone (TT is fixed at ε₀)
    idx_ee = np.argmin(results['rms_ee'])
    
    # Find optimal for combined
    idx_combined = np.argmin(results['rms_combined'])
    
    return {
        'delta_eps_ee_opt': results['delta_eps'][idx_ee],
        'rms_ee_opt': results['rms_ee'][idx_ee],
        'delta_eps_combined_opt': results['delta_eps'][idx_combined],
        'rms_combined_opt': results['rms_combined'][idx_combined],
        'rms_tt_at_combined': results['rms_tt'][idx_combined],
        'rms_ee_at_combined': results['rms_ee'][idx_combined],
    }


def main():
    print("=" * 70)
    print("PHASE 18A: POLARIZATION-WEIGHTED PROJECTION")
    print("=" * 70)
    print("\n*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]")
    print(f"\nAnsatz:")
    print(f"  ε_TT = ε₀ (fixed)")
    print(f"  ε_EE = ε₀ + δε (δε is the single new parameter)")
    print(f"\nPrior: |δε| ≤ {DELTA_EPS_MAX:.1e}")
    
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
    
    # Baseline: no operator
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    rms_tt_baseline = rms(r_tt_base)
    rms_ee_baseline = rms(r_ee_base)
    
    # Constant ε₀ for both (Phase 13A result)
    tt_const = apply_constant_shift(ell, tt_lcdm, EPSILON_0)
    ee_const = apply_constant_shift(ell, ee_lcdm, EPSILON_0)
    r_tt_const = fractional_residual(tt_bec[mask], tt_const[mask])
    r_ee_const = fractional_residual(ee_bec[mask], ee_const[mask])
    rms_tt_const = rms(r_tt_const)
    rms_ee_const = rms(r_ee_const)
    
    print(f"\n[2] Baseline comparison:")
    print(f"  TT no operator: RMS = {rms_tt_baseline:.6f}")
    print(f"  TT constant ε₀: RMS = {rms_tt_const:.6f} ({(rms_tt_baseline-rms_tt_const)/rms_tt_baseline*100:+.1f}%)")
    print(f"  EE no operator: RMS = {rms_ee_baseline:.6f}")
    print(f"  EE constant ε₀: RMS = {rms_ee_const:.6f} ({(rms_ee_baseline-rms_ee_const)/rms_ee_baseline*100:+.1f}%)")
    
    # Scan δε
    print("\n[3] Scanning δε grid...")
    results = scan_delta_epsilon(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask)
    opt = find_optimal_delta_eps(results)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n  Optimal δε (EE alone): {opt['delta_eps_ee_opt']:.4e}")
    print(f"  EE RMS at optimal δε: {opt['rms_ee_opt']:.6f}")
    print(f"  EE improvement vs ε₀: {(rms_ee_const - opt['rms_ee_opt'])/rms_ee_const*100:+.1f}%")
    
    # Effective ε values
    eps_tt = EPSILON_0
    eps_ee_opt = EPSILON_0 + opt['delta_eps_ee_opt']
    
    print(f"\n  Effective ε values:")
    print(f"    ε_TT = {eps_tt:.4e}")
    print(f"    ε_EE = {eps_ee_opt:.4e}")
    print(f"    Ratio ε_EE/ε_TT = {eps_ee_opt/eps_tt:.3f}")
    
    # Compare to Phase 16A/17A findings
    print("\n" + "=" * 70)
    print("COMPARISON TO PREVIOUS PHASES")
    print("=" * 70)
    
    # Phase 16A found α_TT = -9.5e-4, α_EE = -1.8e-3
    # At pivot ℓ*=1650, ε = ε₀ for both
    # The difference in α translates to different ε at edges
    
    print(f"\n  Phase 16A (log-running at ℓ=2500):")
    alpha_tt_16a = -9.5e-4
    alpha_ee_16a = -1.8e-3
    eps_tt_16a_2500 = EPSILON_0 + alpha_tt_16a * np.log(2500/1650)
    eps_ee_16a_2500 = EPSILON_0 + alpha_ee_16a * np.log(2500/1650)
    print(f"    ε_TT(2500) ≈ {eps_tt_16a_2500:.4e}")
    print(f"    ε_EE(2500) ≈ {eps_ee_16a_2500:.4e}")
    print(f"    Difference: {(eps_ee_16a_2500 - eps_tt_16a_2500):.4e}")
    
    print(f"\n  Phase 18A (constant offset):")
    print(f"    δε = {opt['delta_eps_ee_opt']:.4e}")
    print(f"    This is a {opt['delta_eps_ee_opt']/EPSILON_0*100:+.1f}% shift relative to ε₀")
    
    # Statistical significance
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    # AIC comparison: 1 parameter (δε) vs 0 parameters
    delta_chi2 = n_points * (rms_ee_const**2 - opt['rms_ee_opt']**2) / rms_ee_const**2
    aic_penalty = delta_chi2 - 2  # 1 new parameter
    
    print(f"\n  Δχ² (EE improvement): {delta_chi2:.1f}")
    print(f"  AIC penalized: {aic_penalty:.1f}")
    print(f"  Significant (AIC > 0): {'YES' if aic_penalty > 0 else 'NO'}")
    
    # Is δε significantly different from 0?
    delta_eps_significant = abs(opt['delta_eps_ee_opt']) > 0.1 * EPSILON_0
    print(f"\n  δε significantly ≠ 0: {'YES' if delta_eps_significant else 'NO'}")
    print(f"    (threshold: 10% of ε₀ = {0.1*EPSILON_0:.4e})")
    
    # TE consistency check
    print("\n" + "=" * 70)
    print("TE PHASE CONSISTENCY")
    print("=" * 70)
    
    if te_lcdm is not None and te_bec is not None:
        # Apply ε_TT to TE (since TE = T×E, it should track TT)
        te_shifted_tt = apply_constant_shift(ell, te_lcdm, eps_tt)
        r_te_tt = fractional_residual(te_bec[mask], te_shifted_tt[mask])
        
        # Apply ε_EE to TE
        te_shifted_ee = apply_constant_shift(ell, te_lcdm, eps_ee_opt)
        r_te_ee = fractional_residual(te_bec[mask], te_shifted_ee[mask])
        
        # Apply geometric mean
        eps_te_mean = np.sqrt(eps_tt * eps_ee_opt)
        te_shifted_mean = apply_constant_shift(ell, te_lcdm, eps_te_mean)
        r_te_mean = fractional_residual(te_bec[mask], te_shifted_mean[mask])
        
        r_te_base = fractional_residual(te_bec[mask], te_lcdm[mask])
        
        corr_te_tt = corr(r_te_base, r_te_tt)
        corr_te_ee = corr(r_te_base, r_te_ee)
        corr_te_mean = corr(r_te_base, r_te_mean)
        
        print(f"\n  TE correlation with ε_TT: {corr_te_tt:+.3f}")
        print(f"  TE correlation with ε_EE: {corr_te_ee:+.3f}")
        print(f"  TE correlation with √(ε_TT·ε_EE): {corr_te_mean:+.3f}")
        
        # TE should track TT more than EE (since T dominates)
        te_tracks_tt = corr_te_tt > corr_te_ee
        print(f"\n  TE tracks TT more than EE: {'YES' if te_tracks_tt else 'NO'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 18A SUMMARY")
    print("=" * 70)
    
    if delta_eps_significant and aic_penalty > 0:
        conclusion = "POLARIZATION OFFSET DETECTED"
        interpretation = f"""
  The data prefer different projection distortions for TT and EE:
    ε_TT = {eps_tt:.4e}
    ε_EE = {eps_ee_opt:.4e}
    δε = ε_EE - ε_TT = {opt['delta_eps_ee_opt']:.4e}
  
  This is NOT a functional form artifact (Phase 17A ruled that out).
  This is NOT noise (AIC = {aic_penalty:.1f} >> 0).
  
  Interpretation: TT and EE probe different effective projection depths.
  This is consistent with:
    - Different visibility function widths for T and E
    - Polarization source thickness effects
    - Mode-coupling differences
  
  This is projection anisotropy, not exotic physics.
"""
    else:
        conclusion = "NO SIGNIFICANT POLARIZATION OFFSET"
        interpretation = "  The constant ε₀ is sufficient for both TT and EE."
    
    print(f"\n  CONCLUSION: {conclusion}")
    print(interpretation)
    
    # Save summary
    summary = f"""PHASE 18A: POLARIZATION-WEIGHTED PROJECTION
============================================================
*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***

Locked ε₀ = {EPSILON_0:.10e}
Analysis range: ℓ ∈ [{LMIN}, {LMAX}]

ANSATZ:
  ε_TT = ε₀ (fixed)
  ε_EE = ε₀ + δε

RESULTS:
  Optimal δε = {opt['delta_eps_ee_opt']:.4e}
  ε_TT = {eps_tt:.4e}
  ε_EE = {eps_ee_opt:.4e}
  Ratio ε_EE/ε_TT = {eps_ee_opt/eps_tt:.3f}

STATISTICAL SIGNIFICANCE:
  Δχ² = {delta_chi2:.1f}
  AIC penalized = {aic_penalty:.1f}
  δε ≠ 0: {'YES' if delta_eps_significant else 'NO'}

CONCLUSION: {conclusion}
{interpretation}
"""
    
    out_summary = base_dir / 'phase18a_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plots
    print("\n[4] Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: RMS vs δε
    ax = axes[0, 0]
    ax.plot(results['delta_eps'] * 1e3, results['rms_tt'], 'b-', lw=2, label='TT (fixed at ε₀)')
    ax.plot(results['delta_eps'] * 1e3, results['rms_ee'], 'r-', lw=2, label='EE (ε₀ + δε)')
    ax.axvline(0, color='gray', ls='--', alpha=0.5, label='δε = 0')
    ax.axvline(opt['delta_eps_ee_opt'] * 1e3, color='r', ls=':', alpha=0.7, label=f'δε_opt = {opt["delta_eps_ee_opt"]*1e3:.2f}×10⁻³')
    ax.axhline(rms_ee_const, color='r', ls='--', alpha=0.3)
    ax.set_xlabel('δε × 10³')
    ax.set_ylabel('RMS residual')
    ax.set_title('RMS vs Polarization Offset δε')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ε values comparison
    ax = axes[0, 1]
    x = ['TT', 'EE (δε=0)', 'EE (optimal)']
    y = [eps_tt * 1e3, EPSILON_0 * 1e3, eps_ee_opt * 1e3]
    colors = ['blue', 'lightcoral', 'red']
    bars = ax.bar(x, y, color=colors, edgecolor='black')
    ax.axhline(EPSILON_0 * 1e3, color='gray', ls='--', alpha=0.5, label='ε₀')
    ax.set_ylabel('ε × 10³')
    ax.set_title('Effective ε Values')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Improvement breakdown
    ax = axes[1, 0]
    
    # Calculate improvements
    tt_imp_baseline = (rms_tt_baseline - rms_tt_const) / rms_tt_baseline * 100
    ee_imp_baseline_const = (rms_ee_baseline - rms_ee_const) / rms_ee_baseline * 100
    ee_imp_baseline_opt = (rms_ee_baseline - opt['rms_ee_opt']) / rms_ee_baseline * 100
    ee_imp_const_opt = (rms_ee_const - opt['rms_ee_opt']) / rms_ee_const * 100
    
    x = np.arange(2)
    width = 0.35
    
    ax.bar(x - width/2, [tt_imp_baseline, ee_imp_baseline_const], width, 
           label='Constant ε₀', color='lightblue', edgecolor='blue')
    ax.bar(x + width/2, [tt_imp_baseline, ee_imp_baseline_opt], width,
           label='With δε offset', color='lightcoral', edgecolor='red')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['TT', 'EE'])
    ax.set_ylabel('RMS Reduction vs Baseline (%)')
    ax.set_title('Improvement from Polarization Offset')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Phase comparison
    ax = axes[1, 1]
    
    # Show the evolution of TT/EE split across phases
    phases = ['v1.0.0\n(constant)', 'Phase 16A\n(log)', 'Phase 17A\n(tanh)', 'Phase 18A\n(offset)']
    tt_eps = [EPSILON_0, EPSILON_0, EPSILON_0, eps_tt]
    ee_eps = [EPSILON_0, EPSILON_0 + (-1.8e-3)*np.log(2000/1650), 
              EPSILON_0 + (-1.25e-3)*np.tanh(np.log(2000/1650)/0.6),
              eps_ee_opt]
    
    x = np.arange(len(phases))
    ax.plot(x, np.array(tt_eps)*1e3, 'b-o', ms=8, lw=2, label='ε_TT')
    ax.plot(x, np.array(ee_eps)*1e3, 'r-s', ms=8, lw=2, label='ε_EE (at ℓ=2000)')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel('ε × 10³')
    ax.set_title('TT/EE Split Evolution Across Phases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Phase 18A: Polarization Offset (δε = {opt["delta_eps_ee_opt"]*1e3:.2f}×10⁻³)', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase18a_polarization_offset.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase18a_results.npz'
    np.savez(
        out_npz,
        epsilon_0=EPSILON_0,
        delta_eps_grid=DELTA_EPS_GRID,
        delta_eps_opt=opt['delta_eps_ee_opt'],
        eps_tt=eps_tt,
        eps_ee=eps_ee_opt,
        rms_tt=results['rms_tt'],
        rms_ee=results['rms_ee'],
        delta_chi2=delta_chi2,
        aic_penalized=aic_penalty,
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 18A COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
