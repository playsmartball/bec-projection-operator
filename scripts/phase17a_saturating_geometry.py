#!/usr/bin/env python3
"""
PHASE 17A: SATURATING SIGMOID GEOMETRY

EXPLORATORY PHASE - NOT PART OF v1.0.0

Objective: Test whether TT/EE divergence from Phase 16A disappears when
using a more faithful functional form for ε(ℓ).

ANSATZ:
    ε(ℓ) = ε₀ + β · tanh(ln(ℓ/ℓ*) / Δ)

where:
    ε₀ = 1.4558030818e-03 (LOCKED from Phase 10E)
    ℓ* = 1650 (fixed pivot)
    Δ = 0.6 (fixed width parameter)
    β = free amplitude (single new parameter)

PROPERTIES:
    - Flat near pivot (ℓ ≈ ℓ*)
    - Strong curvature at edges
    - Natural saturation (prevents runaway)
    - tanh bounded in [-1, +1]

ACCEPTANCE CRITERIA:
    - TT AIC > +20
    - EE AIC > +20
    - TT/EE β consistency within factor 1.5
    - Correlation ≥ 0.6
    - No boundary hits
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
# PHASE 17A PARAMETERS
# =============================================================================
DELTA = 0.6                   # Fixed width parameter for tanh
BETA_MAX = 2e-3               # |β| ≤ 2×10⁻³
BETA_GRID = np.linspace(-BETA_MAX, BETA_MAX, 81)

# =============================================================================
# ACCEPTANCE THRESHOLDS
# =============================================================================
MIN_AIC = 20.0
MAX_BETA_RATIO = 1.5          # TT/EE β must be within factor 1.5
MIN_CORRELATION = 0.60


def _load_class_cl(file_path: Path):
    """Load CLASS Cℓ output file."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3] if data.shape[1] > 3 else None
    return ell, tt, ee, te


def f_tanh(ell, delta=DELTA):
    """
    Saturating sigmoid shape function:
        f(ℓ) = tanh(ln(ℓ/ℓ*) / Δ)
    
    Properties:
        f(ℓ*) = 0
        f → +1 as ℓ → ∞
        f → -1 as ℓ → 0
    """
    return np.tanh(np.log(ell / ELL_PIVOT) / delta)


def epsilon_of_ell(ell, beta):
    """
    Scale-dependent ε with saturating sigmoid:
        ε(ℓ) = ε₀ + β · tanh(ln(ℓ/ℓ*) / Δ)
    """
    return EPSILON_0 + beta * f_tanh(ell)


def apply_saturating_operator(ell, cl, beta):
    """
    Apply scale-dependent operator with saturating sigmoid:
        P_ε(ℓ) : C_ℓ ↦ C_{ℓ/(1+ε(ℓ))}
    """
    ell_float = ell.astype(float)
    eps = epsilon_of_ell(ell_float, beta)
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


def scan_beta(ell, cl_lcdm, cl_bec, mask, name):
    """
    Grid scan over β values.
    """
    results = {
        'beta': [],
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
    
    # Constant ε (β = 0)
    cl_const = apply_saturating_operator(ell, cl_lcdm, beta=0.0)
    r_const = fractional_residual(cl_bec[mask], cl_const[mask])
    rms_const = rms(r_const)
    results['rms_const'] = rms_const
    
    # Scan β
    for beta in BETA_GRID:
        cl_shifted = apply_saturating_operator(ell, cl_lcdm, beta)
        r_shifted = fractional_residual(cl_bec[mask], cl_shifted[mask])
        
        rms_shifted = rms(r_shifted)
        corr_shifted = corr(r_base, r_shifted)
        
        reduction_vs_base = (rms_base - rms_shifted) / rms_base
        reduction_vs_const = (rms_const - rms_shifted) / rms_const
        
        results['beta'].append(beta)
        results['rms_running'].append(rms_shifted)
        results['corr_running'].append(corr_shifted)
        results['reduction_vs_base'].append(reduction_vs_base)
        results['reduction_vs_const'].append(reduction_vs_const)
    
    for key in ['beta', 'rms_running', 'corr_running', 'reduction_vs_base', 'reduction_vs_const']:
        results[key] = np.array(results[key])
    
    return results


def find_optimal_beta(results):
    """Find β that minimizes RMS."""
    idx = np.argmin(results['rms_running'])
    return {
        'beta_opt': results['beta'][idx],
        'rms_opt': results['rms_running'][idx],
        'corr_opt': results['corr_running'][idx],
        'reduction_vs_base': results['reduction_vs_base'][idx],
        'reduction_vs_const': results['reduction_vs_const'][idx],
        'at_boundary': abs(results['beta'][idx]) >= 0.95 * BETA_MAX,
    }


def compute_aic_penalty(rms_const, rms_running, n_points, delta_k=1):
    """AIC-style penalty."""
    delta_chi2 = n_points * (rms_const**2 - rms_running**2) / rms_const**2
    penalized = delta_chi2 - 2 * delta_k
    return delta_chi2, penalized


def main():
    print("=" * 70)
    print("PHASE 17A: SATURATING SIGMOID GEOMETRY")
    print("=" * 70)
    print("\n*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Pivot ℓ* = {ELL_PIVOT}")
    print(f"Width Δ = {DELTA}")
    print(f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]")
    print(f"Ansatz: ε(ℓ) = ε₀ + β·tanh(ln(ℓ/ℓ*)/Δ)")
    print(f"Prior: |β| ≤ {BETA_MAX:.1e}")
    
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
    
    # Show shape function
    print("\n[2] Shape function f(ℓ) = tanh(ln(ℓ/ℓ*)/Δ):")
    for l_test in [800, 1000, 1200, 1650, 2000, 2500]:
        print(f"    f({l_test}) = {f_tanh(l_test):+.4f}")
    
    # Scan β for TT and EE
    print("\n[3] Scanning β grid...")
    print(f"  Grid: {len(BETA_GRID)} points from {BETA_GRID[0]:.1e} to {BETA_GRID[-1]:.1e}")
    
    print("\n" + "=" * 70)
    print("TT SPECTRUM")
    print("=" * 70)
    results_tt = scan_beta(ell, tt_lcdm, tt_bec, mask, 'TT')
    opt_tt = find_optimal_beta(results_tt)
    
    print(f"\n  Baseline RMS: {results_tt['rms_base']:.6f}")
    print(f"  Constant ε RMS: {results_tt['rms_const']:.6f}")
    print(f"  Optimal β: {opt_tt['beta_opt']:.4e}")
    print(f"  Running ε RMS: {opt_tt['rms_opt']:.6f}")
    print(f"  Reduction vs baseline: {opt_tt['reduction_vs_base']*100:+.1f}%")
    print(f"  Reduction vs constant ε: {opt_tt['reduction_vs_const']*100:+.1f}%")
    print(f"  Correlation: {opt_tt['corr_opt']:.3f}")
    print(f"  At boundary: {'YES' if opt_tt['at_boundary'] else 'NO'}")
    
    delta_chi2_tt, penalized_tt = compute_aic_penalty(
        results_tt['rms_const'], opt_tt['rms_opt'], n_points
    )
    print(f"\n  AIC check: Δχ² = {delta_chi2_tt:.1f}, penalized = {penalized_tt:.1f}")
    
    print("\n" + "=" * 70)
    print("EE SPECTRUM")
    print("=" * 70)
    results_ee = scan_beta(ell, ee_lcdm, ee_bec, mask, 'EE')
    opt_ee = find_optimal_beta(results_ee)
    
    print(f"\n  Baseline RMS: {results_ee['rms_base']:.6f}")
    print(f"  Constant ε RMS: {results_ee['rms_const']:.6f}")
    print(f"  Optimal β: {opt_ee['beta_opt']:.4e}")
    print(f"  Running ε RMS: {opt_ee['rms_opt']:.6f}")
    print(f"  Reduction vs baseline: {opt_ee['reduction_vs_base']*100:+.1f}%")
    print(f"  Reduction vs constant ε: {opt_ee['reduction_vs_const']*100:+.1f}%")
    print(f"  Correlation: {opt_ee['corr_opt']:.3f}")
    print(f"  At boundary: {'YES' if opt_ee['at_boundary'] else 'NO'}")
    
    delta_chi2_ee, penalized_ee = compute_aic_penalty(
        results_ee['rms_const'], opt_ee['rms_opt'], n_points
    )
    print(f"\n  AIC check: Δχ² = {delta_chi2_ee:.1f}, penalized = {penalized_ee:.1f}")
    
    # TE analysis
    print("\n" + "=" * 70)
    print("TE SPECTRUM (correlation diagnostic only)")
    print("=" * 70)
    
    if te_lcdm is not None and te_bec is not None:
        results_te = scan_beta(ell, te_lcdm, te_bec, mask, 'TE')
        opt_te = find_optimal_beta(results_te)
        print(f"\n  Optimal β: {opt_te['beta_opt']:.4e}")
        print(f"  Correlation: {opt_te['corr_opt']:.3f}")
    else:
        opt_te = None
        print("  TE data not available")
    
    # TT/EE consistency check
    print("\n" + "=" * 70)
    print("TT/EE CONSISTENCY CHECK (Critical for Phase 17A)")
    print("=" * 70)
    
    beta_tt = opt_tt['beta_opt']
    beta_ee = opt_ee['beta_opt']
    
    # Handle sign and ratio
    same_sign = np.sign(beta_tt) == np.sign(beta_ee)
    if same_sign and beta_tt != 0 and beta_ee != 0:
        beta_ratio = max(abs(beta_tt), abs(beta_ee)) / min(abs(beta_tt), abs(beta_ee))
    else:
        beta_ratio = np.inf
    
    beta_mean = (beta_tt + beta_ee) / 2
    
    print(f"\n  β_TT = {beta_tt:.4e}")
    print(f"  β_EE = {beta_ee:.4e}")
    print(f"  Same sign: {'YES' if same_sign else 'NO'}")
    print(f"  Ratio (max/min): {beta_ratio:.2f}")
    print(f"  Mean β = {beta_mean:.4e}")
    
    # Compare to Phase 16A
    print("\n  Comparison to Phase 16A (log-running):")
    print(f"    Phase 16A: α_TT/α_EE ratio ≈ 1.9 (factor ~2 divergence)")
    print(f"    Phase 17A: β_TT/β_EE ratio = {beta_ratio:.2f}")
    
    convergence_improved = beta_ratio < 1.9
    print(f"    Convergence improved: {'YES' if convergence_improved else 'NO'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 17A SUMMARY")
    print("=" * 70)
    
    # Evaluate acceptance criteria
    tt_pass_aic = penalized_tt > MIN_AIC
    ee_pass_aic = penalized_ee > MIN_AIC
    tt_pass_corr = opt_tt['corr_opt'] >= MIN_CORRELATION
    ee_pass_corr = opt_ee['corr_opt'] >= MIN_CORRELATION
    consistency_pass = same_sign and beta_ratio <= MAX_BETA_RATIO
    no_boundary_tt = not opt_tt['at_boundary']
    no_boundary_ee = not opt_ee['at_boundary']
    
    print(f"\n  TT AIC: {penalized_tt:.1f} (need >{MIN_AIC}): {'PASS' if tt_pass_aic else 'FAIL'}")
    print(f"  EE AIC: {penalized_ee:.1f} (need >{MIN_AIC}): {'PASS' if ee_pass_aic else 'FAIL'}")
    print(f"  TT correlation: {opt_tt['corr_opt']:.3f} (need ≥{MIN_CORRELATION}): {'PASS' if tt_pass_corr else 'FAIL'}")
    print(f"  EE correlation: {opt_ee['corr_opt']:.3f} (need ≥{MIN_CORRELATION}): {'PASS' if ee_pass_corr else 'FAIL'}")
    print(f"  TT/EE β ratio: {beta_ratio:.2f} (need ≤{MAX_BETA_RATIO}): {'PASS' if consistency_pass else 'FAIL'}")
    print(f"  TT no boundary: {'PASS' if no_boundary_tt else 'FAIL'}")
    print(f"  EE no boundary: {'PASS' if no_boundary_ee else 'FAIL'}")
    
    # Overall assessment
    all_pass = tt_pass_aic and ee_pass_aic and consistency_pass and no_boundary_tt and no_boundary_ee
    
    if all_pass:
        conclusion = "SATURATING GEOMETRY VALIDATED - TT/EE converged"
        proceed_to_18a = False
    elif consistency_pass:
        conclusion = "PARTIAL SUCCESS - TT/EE converged but other criteria failed"
        proceed_to_18a = False
    else:
        conclusion = "TT/EE STILL DIVERGENT - Proceed to Phase 18A"
        proceed_to_18a = True
    
    print(f"\n  CONCLUSION: {conclusion}")
    
    # ε(ℓ) profiles at optimal β
    print("\n" + "=" * 70)
    print("ε(ℓ) PROFILES AT OPTIMAL β")
    print("=" * 70)
    
    for name, beta in [('TT', beta_tt), ('EE', beta_ee)]:
        print(f"\n  {name} (β = {beta:.4e}):")
        for l_test in [800, 1200, 1650, 2000, 2500]:
            eps = epsilon_of_ell(l_test, beta)
            print(f"    ε({l_test}) = {eps:.4e}")
    
    # Save summary
    summary = f"""PHASE 17A: SATURATING SIGMOID GEOMETRY
============================================================
*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***

Locked ε₀ = {EPSILON_0:.10e}
Pivot ℓ* = {ELL_PIVOT}
Width Δ = {DELTA}
Ansatz: ε(ℓ) = ε₀ + β·tanh(ln(ℓ/ℓ*)/Δ)

TT RESULTS:
  Optimal β = {opt_tt['beta_opt']:.4e}
  RMS reduction vs constant ε: {opt_tt['reduction_vs_const']*100:+.1f}%
  Correlation: {opt_tt['corr_opt']:.3f}
  AIC penalized: {penalized_tt:.1f}
  At boundary: {'YES' if opt_tt['at_boundary'] else 'NO'}

EE RESULTS:
  Optimal β = {opt_ee['beta_opt']:.4e}
  RMS reduction vs constant ε: {opt_ee['reduction_vs_const']*100:+.1f}%
  Correlation: {opt_ee['corr_opt']:.3f}
  AIC penalized: {penalized_ee:.1f}
  At boundary: {'YES' if opt_ee['at_boundary'] else 'NO'}

TT/EE CONSISTENCY:
  Same sign: {'YES' if same_sign else 'NO'}
  Ratio: {beta_ratio:.2f} (threshold: {MAX_BETA_RATIO})
  Phase 16A ratio was ~1.9
  Convergence improved: {'YES' if convergence_improved else 'NO'}

CONCLUSION: {conclusion}
Proceed to Phase 18A: {'YES' if proceed_to_18a else 'NO'}
"""
    
    out_summary = base_dir / 'phase17a_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plots
    print("\n[4] Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Shape function and ε(ℓ) profiles
    ax = axes[0, 0]
    ell_plot = np.linspace(LMIN, LMAX, 200)
    
    # Show ε(ℓ) for different β
    for beta, ls, label in [
        (0, '--', 'β=0 (constant)'),
        (beta_tt, '-', f'β_TT={beta_tt:.2e}'),
        (beta_ee, '-.', f'β_EE={beta_ee:.2e}'),
    ]:
        eps = epsilon_of_ell(ell_plot, beta)
        ax.plot(ell_plot, eps * 1e3, ls=ls, lw=2, label=label)
    
    ax.axvline(ELL_PIVOT, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ε(ℓ) × 10³')
    ax.set_title('Saturating ε(ℓ) Profiles')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMS vs β (TT)
    ax = axes[0, 1]
    ax.plot(results_tt['beta'] * 1e3, results_tt['rms_running'], 'b-o', ms=3, label='TT')
    ax.axhline(results_tt['rms_const'], color='b', ls='--', alpha=0.5)
    ax.axvline(opt_tt['beta_opt'] * 1e3, color='b', ls='-', alpha=0.3)
    ax.set_xlabel('β × 10³')
    ax.set_ylabel('RMS residual')
    ax.set_title('TT: RMS vs β')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMS vs β (EE)
    ax = axes[0, 2]
    ax.plot(results_ee['beta'] * 1e3, results_ee['rms_running'], 'r-o', ms=3, label='EE')
    ax.axhline(results_ee['rms_const'], color='r', ls='--', alpha=0.5)
    ax.axvline(opt_ee['beta_opt'] * 1e3, color='r', ls='-', alpha=0.3)
    ax.set_xlabel('β × 10³')
    ax.set_ylabel('RMS residual')
    ax.set_title('EE: RMS vs β')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Shape function
    ax = axes[1, 0]
    ax.plot(ell_plot, f_tanh(ell_plot), 'k-', lw=2)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.axvline(ELL_PIVOT, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('f(ℓ) = tanh(ln(ℓ/ℓ*)/Δ)')
    ax.set_title(f'Shape Function (Δ={DELTA})')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: TT vs EE β comparison
    ax = axes[1, 1]
    ax.scatter([beta_tt * 1e3], [beta_ee * 1e3], s=150, c='purple', marker='*', zorder=5, label='Phase 17A')
    
    # Add Phase 16A point for comparison (approximate)
    alpha_tt_16a = -9.5e-4
    alpha_ee_16a = -1.8e-3
    ax.scatter([alpha_tt_16a * 1e3], [alpha_ee_16a * 1e3], s=100, c='orange', marker='o', zorder=4, label='Phase 16A (α)')
    
    ax.plot([-2, 2], [-2, 2], 'k--', alpha=0.5, label='TT = EE')
    ax.set_xlabel('β_TT × 10³ (or α_TT)')
    ax.set_ylabel('β_EE × 10³ (or α_EE)')
    ax.set_title('TT/EE Consistency Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 6: Reduction comparison
    ax = axes[1, 2]
    x = np.arange(2)
    width = 0.35
    
    const_reductions = [
        (results_tt['rms_base'] - results_tt['rms_const']) / results_tt['rms_base'] * 100,
        (results_ee['rms_base'] - results_ee['rms_const']) / results_ee['rms_base'] * 100,
    ]
    running_reductions = [
        opt_tt['reduction_vs_base'] * 100,
        opt_ee['reduction_vs_base'] * 100,
    ]
    
    ax.bar(x - width/2, const_reductions, width, label='Constant ε', color='lightblue', edgecolor='blue')
    ax.bar(x + width/2, running_reductions, width, label='Saturating ε(ℓ)', color='lightcoral', edgecolor='red')
    ax.set_xticks(x)
    ax.set_xticklabels(['TT', 'EE'])
    ax.set_ylabel('RMS Reduction vs Baseline (%)')
    ax.set_title('Constant vs Saturating Operator')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Phase 17A: Saturating Sigmoid Geometry (β_TT/β_EE = {beta_ratio:.2f})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase17a_saturating_geometry.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase17a_results.npz'
    np.savez(
        out_npz,
        epsilon_0=EPSILON_0,
        ell_pivot=ELL_PIVOT,
        delta=DELTA,
        beta_grid=BETA_GRID,
        beta_opt_tt=opt_tt['beta_opt'],
        beta_opt_ee=opt_ee['beta_opt'],
        beta_ratio=beta_ratio,
        results_tt_beta=results_tt['beta'],
        results_tt_rms=results_tt['rms_running'],
        results_ee_beta=results_ee['beta'],
        results_ee_rms=results_ee['rms_running'],
        convergence_improved=convergence_improved,
        proceed_to_18a=proceed_to_18a,
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 17A COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
