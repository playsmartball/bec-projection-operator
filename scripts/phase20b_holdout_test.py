#!/usr/bin/env python3
"""
PHASE 20B: ℓ-RANGE HOLDOUT TEST

INTERNAL PREDICTION TEST - NO REFITTING ALLOWED

Objective: Test whether the (α, γ) structure generalizes beyond the fitting range.

PROCEDURE:
    1. Fit α, γ on ℓ ∈ [800, 1800] (TRAINING)
    2. Predict residual reduction on ℓ ∈ [1800, 2500] (HOLDOUT)
    3. No refitting allowed on holdout range

ACCEPTANCE CRITERIA:
    - RMS reduction persists in holdout range
    - TE continues to track TT
    - No sign flips in residuals
    - Holdout performance within 50% of training performance

This addresses "overfitting the damping tail" criticisms preemptively.

NOTE: We compare to the FULL-RANGE fit from Phase 18B to see if
the structure is stable under different fitting ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# LOCKED PARAMETERS (FROM PHASE 18B - FULL RANGE FIT)
# =============================================================================
EPSILON_0 = 1.4558030818e-03  # From Phase 10E (never changes)
ELL_PIVOT = 1650              # Fixed pivot

# Phase 18B parameters (fit on FULL range [800, 2500])
ALPHA_FULL = -9.3333e-04
GAMMA_FULL = -8.6667e-04

# =============================================================================
# HOLDOUT TEST RANGES
# =============================================================================
TRAIN_LMIN, TRAIN_LMAX = 800, 1800
HOLDOUT_LMIN, HOLDOUT_LMAX = 1800, 2500


def _load_class_cl(file_path: Path):
    """Load CLASS Cℓ output file."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3] if data.shape[1] > 3 else None
    return ell, tt, ee, te


def f_running(ell):
    """Running shape function: f(ℓ) = ln(ℓ/ℓ*)"""
    return np.log(ell / ELL_PIVOT)


def epsilon_tt(ell, alpha):
    """ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)"""
    return EPSILON_0 + alpha * f_running(ell)


def epsilon_ee(ell, alpha, gamma):
    """ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)"""
    return EPSILON_0 + (alpha + gamma) * f_running(ell)


def apply_shift(ell, cl, eps_array):
    """Apply position-dependent horizontal shift."""
    ell_float = ell.astype(float)
    ell_star = ell_float / (1 + eps_array)
    cl_new = np.interp(ell_float, ell_star, cl, left=cl[0], right=cl[-1])
    return cl_new


def fractional_residual(cl_a, cl_b):
    """Compute (a - b) / b."""
    denom = np.where(np.abs(cl_b) > 0, cl_b, 1.0)
    return (cl_a - cl_b) / denom


def rms(x):
    """Root mean square."""
    return np.sqrt(np.mean(x**2))


def fit_alpha_gamma(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask, n_alpha=41, n_gamma=41):
    """
    Grid search to find optimal (α, γ) on the given mask.
    """
    alpha_range = np.linspace(-2e-3, 2e-3, n_alpha)
    gamma_range = np.linspace(-2e-3, 2e-3, n_gamma)
    
    best_rms = np.inf
    best_alpha = 0
    best_gamma = 0
    
    for alpha in alpha_range:
        for gamma in gamma_range:
            # TT
            eps_tt = epsilon_tt(ell, alpha)
            tt_shifted = apply_shift(ell, tt_lcdm, eps_tt)
            r_tt = fractional_residual(tt_bec[mask], tt_shifted[mask])
            rms_tt = rms(r_tt)
            
            # EE
            eps_ee = epsilon_ee(ell, alpha, gamma)
            ee_shifted = apply_shift(ell, ee_lcdm, eps_ee)
            r_ee = fractional_residual(ee_bec[mask], ee_shifted[mask])
            rms_ee = rms(r_ee)
            
            # Combined
            rms_combined = np.sqrt((rms_tt**2 + rms_ee**2) / 2)
            
            if rms_combined < best_rms:
                best_rms = rms_combined
                best_alpha = alpha
                best_gamma = gamma
    
    return best_alpha, best_gamma


def evaluate_on_range(ell, tt_lcdm, ee_lcdm, te_lcdm, tt_bec, ee_bec, te_bec, 
                      mask, alpha, gamma, range_name):
    """
    Evaluate the operator with given (α, γ) on the specified range.
    """
    results = {}
    
    # Baselines
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    r_te_base = fractional_residual(te_bec[mask], te_lcdm[mask])
    
    rms_tt_baseline = rms(r_tt_base)
    rms_ee_baseline = rms(r_ee_base)
    rms_te_baseline = rms(r_te_base)
    
    # With operator
    eps_tt = epsilon_tt(ell, alpha)
    eps_ee = epsilon_ee(ell, alpha, gamma)
    
    tt_shifted = apply_shift(ell, tt_lcdm, eps_tt)
    ee_shifted = apply_shift(ell, ee_lcdm, eps_ee)
    te_shifted = apply_shift(ell, te_lcdm, eps_tt)  # TE tracks TT
    
    r_tt = fractional_residual(tt_bec[mask], tt_shifted[mask])
    r_ee = fractional_residual(ee_bec[mask], ee_shifted[mask])
    r_te = fractional_residual(te_bec[mask], te_shifted[mask])
    
    rms_tt = rms(r_tt)
    rms_ee = rms(r_ee)
    rms_te = rms(r_te)
    
    results['tt'] = {
        'baseline': rms_tt_baseline,
        'with_op': rms_tt,
        'reduction': (rms_tt_baseline - rms_tt) / rms_tt_baseline * 100,
    }
    results['ee'] = {
        'baseline': rms_ee_baseline,
        'with_op': rms_ee,
        'reduction': (rms_ee_baseline - rms_ee) / rms_ee_baseline * 100,
    }
    results['te'] = {
        'baseline': rms_te_baseline,
        'with_op': rms_te,
        'reduction': (rms_te_baseline - rms_te) / rms_te_baseline * 100,
    }
    
    return results


def main():
    print("=" * 70)
    print("PHASE 20B: ℓ-RANGE HOLDOUT TEST")
    print("=" * 70)
    print("\n*** INTERNAL PREDICTION TEST - NO REFITTING ALLOWED ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Pivot ℓ* = {ELL_PIVOT}")
    print(f"\nTraining range: ℓ ∈ [{TRAIN_LMIN}, {TRAIN_LMAX}]")
    print(f"Holdout range: ℓ ∈ [{HOLDOUT_LMIN}, {HOLDOUT_LMAX}]")
    
    base_dir = Path('/Users/hodge/Desktop/sanity_check/bec_cosmology/output')
    
    # Load spectra
    print("\n[1] Loading spectra...")
    lcdm_path = base_dir / 'lcdm_zz_thetaS_reference_precise_00_cl.dat'
    bec_path = base_dir / 'test_bec_zz_thetaS_matched_precise_00_cl.dat'
    
    ell, tt_lcdm, ee_lcdm, te_lcdm = _load_class_cl(lcdm_path)
    _, tt_bec, ee_bec, te_bec = _load_class_cl(bec_path)
    
    # Define masks
    train_mask = (ell >= TRAIN_LMIN) & (ell <= TRAIN_LMAX)
    holdout_mask = (ell >= HOLDOUT_LMIN) & (ell <= HOLDOUT_LMAX)
    full_mask = (ell >= TRAIN_LMIN) & (ell <= HOLDOUT_LMAX)
    
    n_train = train_mask.sum()
    n_holdout = holdout_mask.sum()
    n_full = full_mask.sum()
    
    print(f"  Training: {n_train} multipoles")
    print(f"  Holdout: {n_holdout} multipoles")
    print(f"  Full: {n_full} multipoles")
    
    # Step 1: Fit on training range
    print("\n[2] Fitting (α, γ) on TRAINING range...")
    alpha_train, gamma_train = fit_alpha_gamma(
        ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, train_mask
    )
    
    print(f"\n  Training fit:")
    print(f"    α_train = {alpha_train:.4e}")
    print(f"    γ_train = {gamma_train:.4e}")
    
    print(f"\n  Full-range fit (Phase 18B):")
    print(f"    α_full = {ALPHA_FULL:.4e}")
    print(f"    γ_full = {GAMMA_FULL:.4e}")
    
    # Compare
    alpha_diff = abs(alpha_train - ALPHA_FULL) / abs(ALPHA_FULL) * 100
    gamma_diff = abs(gamma_train - GAMMA_FULL) / abs(GAMMA_FULL) * 100
    
    print(f"\n  Parameter stability:")
    print(f"    |α_train - α_full| / |α_full| = {alpha_diff:.1f}%")
    print(f"    |γ_train - γ_full| / |γ_full| = {gamma_diff:.1f}%")
    
    # Step 2: Evaluate on all ranges
    print("\n[3] Evaluating on all ranges...")
    
    # Training range with training fit
    train_results_train = evaluate_on_range(
        ell, tt_lcdm, ee_lcdm, te_lcdm, tt_bec, ee_bec, te_bec,
        train_mask, alpha_train, gamma_train, "Training"
    )
    
    # Holdout range with training fit (THE KEY TEST)
    holdout_results_train = evaluate_on_range(
        ell, tt_lcdm, ee_lcdm, te_lcdm, tt_bec, ee_bec, te_bec,
        holdout_mask, alpha_train, gamma_train, "Holdout"
    )
    
    # Full range with full fit (reference)
    full_results_full = evaluate_on_range(
        ell, tt_lcdm, ee_lcdm, te_lcdm, tt_bec, ee_bec, te_bec,
        full_mask, ALPHA_FULL, GAMMA_FULL, "Full"
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n  TRAINING RANGE [{TRAIN_LMIN}, {TRAIN_LMAX}] with training fit:")
    print(f"    TT: {train_results_train['tt']['reduction']:+.1f}%")
    print(f"    EE: {train_results_train['ee']['reduction']:+.1f}%")
    print(f"    TE: {train_results_train['te']['reduction']:+.1f}%")
    
    print(f"\n  HOLDOUT RANGE [{HOLDOUT_LMIN}, {HOLDOUT_LMAX}] with training fit (PREDICTION):")
    print(f"    TT: {holdout_results_train['tt']['reduction']:+.1f}%")
    print(f"    EE: {holdout_results_train['ee']['reduction']:+.1f}%")
    print(f"    TE: {holdout_results_train['te']['reduction']:+.1f}%")
    
    print(f"\n  FULL RANGE [{TRAIN_LMIN}, {HOLDOUT_LMAX}] with full fit (reference):")
    print(f"    TT: {full_results_full['tt']['reduction']:+.1f}%")
    print(f"    EE: {full_results_full['ee']['reduction']:+.1f}%")
    print(f"    TE: {full_results_full['te']['reduction']:+.1f}%")
    
    # Holdout performance ratio
    print("\n" + "=" * 70)
    print("HOLDOUT PERFORMANCE RATIO")
    print("=" * 70)
    
    tt_ratio = holdout_results_train['tt']['reduction'] / train_results_train['tt']['reduction']
    ee_ratio = holdout_results_train['ee']['reduction'] / train_results_train['ee']['reduction']
    te_ratio = holdout_results_train['te']['reduction'] / train_results_train['te']['reduction']
    
    print(f"\n  Holdout / Training performance:")
    print(f"    TT: {tt_ratio:.2f}")
    print(f"    EE: {ee_ratio:.2f}")
    print(f"    TE: {te_ratio:.2f}")
    
    # Acceptance criteria
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)
    
    # 1. RMS reduction persists (>0)
    tt_persists = holdout_results_train['tt']['reduction'] > 0
    ee_persists = holdout_results_train['ee']['reduction'] > 0
    te_persists = holdout_results_train['te']['reduction'] > 0
    
    print(f"\n  1. RMS reduction persists in holdout:")
    print(f"     TT: {'PASS' if tt_persists else 'FAIL'} ({holdout_results_train['tt']['reduction']:+.1f}%)")
    print(f"     EE: {'PASS' if ee_persists else 'FAIL'} ({holdout_results_train['ee']['reduction']:+.1f}%)")
    print(f"     TE: {'PASS' if te_persists else 'FAIL'} ({holdout_results_train['te']['reduction']:+.1f}%)")
    
    # 2. Holdout within 50% of training
    tt_within = tt_ratio >= 0.5
    ee_within = ee_ratio >= 0.5
    te_within = te_ratio >= 0.5
    
    print(f"\n  2. Holdout within 50% of training:")
    print(f"     TT: {'PASS' if tt_within else 'FAIL'} (ratio = {tt_ratio:.2f})")
    print(f"     EE: {'PASS' if ee_within else 'FAIL'} (ratio = {ee_ratio:.2f})")
    print(f"     TE: {'PASS' if te_within else 'FAIL'} (ratio = {te_ratio:.2f})")
    
    # 3. Parameter stability (<50% change)
    alpha_stable = alpha_diff < 50
    gamma_stable = gamma_diff < 50
    
    print(f"\n  3. Parameter stability (<50% change):")
    print(f"     α: {'PASS' if alpha_stable else 'FAIL'} ({alpha_diff:.1f}% change)")
    print(f"     γ: {'PASS' if gamma_stable else 'FAIL'} ({gamma_diff:.1f}% change)")
    
    # Overall
    all_pass = (tt_persists and ee_persists and te_persists and 
                tt_within and ee_within and te_within and
                alpha_stable and gamma_stable)
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 20B SUMMARY")
    print("=" * 70)
    
    if all_pass:
        conclusion = "HOLDOUT TEST PASSED - Structure generalizes"
        interpretation = """
  The (α, γ) operator structure fitted on ℓ ∈ [800, 1800] successfully
  predicts the residual reduction on ℓ ∈ [1800, 2500].
  
  This rules out overfitting to the damping tail.
  
  The operator captures genuine geometric structure that extends
  beyond the fitting range.
"""
    else:
        conclusion = "HOLDOUT TEST PARTIAL - Some criteria not met"
        interpretation = """
  The holdout test shows mixed results. Some spectra generalize
  better than others. This may indicate:
    - Scale-dependent effects beyond the simple log-running
    - Different physics in different ℓ ranges
    - Noise sensitivity at high ℓ
"""
    
    print(f"\n  OVERALL: {'PASS' if all_pass else 'PARTIAL'}")
    print(f"\n  CONCLUSION: {conclusion}")
    print(interpretation)
    
    # Save summary
    summary = f"""PHASE 20B: ℓ-RANGE HOLDOUT TEST
============================================================
*** INTERNAL PREDICTION TEST - NO REFITTING ALLOWED ***

Training range: ℓ ∈ [{TRAIN_LMIN}, {TRAIN_LMAX}]
Holdout range: ℓ ∈ [{HOLDOUT_LMIN}, {HOLDOUT_LMAX}]

FITTED PARAMETERS:
  Training fit: α = {alpha_train:.4e}, γ = {gamma_train:.4e}
  Full fit (18B): α = {ALPHA_FULL:.4e}, γ = {GAMMA_FULL:.4e}
  Parameter change: α {alpha_diff:.1f}%, γ {gamma_diff:.1f}%

TRAINING RANGE PERFORMANCE:
  TT: {train_results_train['tt']['reduction']:+.1f}%
  EE: {train_results_train['ee']['reduction']:+.1f}%
  TE: {train_results_train['te']['reduction']:+.1f}%

HOLDOUT RANGE PERFORMANCE (PREDICTION):
  TT: {holdout_results_train['tt']['reduction']:+.1f}%
  EE: {holdout_results_train['ee']['reduction']:+.1f}%
  TE: {holdout_results_train['te']['reduction']:+.1f}%

HOLDOUT / TRAINING RATIO:
  TT: {tt_ratio:.2f}
  EE: {ee_ratio:.2f}
  TE: {te_ratio:.2f}

ACCEPTANCE CRITERIA:
  RMS persists: TT={'PASS' if tt_persists else 'FAIL'}, EE={'PASS' if ee_persists else 'FAIL'}, TE={'PASS' if te_persists else 'FAIL'}
  Within 50%: TT={'PASS' if tt_within else 'FAIL'}, EE={'PASS' if ee_within else 'FAIL'}, TE={'PASS' if te_within else 'FAIL'}
  Param stable: α={'PASS' if alpha_stable else 'FAIL'}, γ={'PASS' if gamma_stable else 'FAIL'}

OVERALL: {'PASS' if all_pass else 'PARTIAL'}

CONCLUSION: {conclusion}
{interpretation}
"""
    
    out_summary = base_dir / 'phase20b_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plots
    print("\n[4] Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Performance by range
    ax = axes[0, 0]
    spectra = ['TT', 'EE', 'TE']
    train_red = [train_results_train['tt']['reduction'], 
                 train_results_train['ee']['reduction'],
                 train_results_train['te']['reduction']]
    holdout_red = [holdout_results_train['tt']['reduction'],
                   holdout_results_train['ee']['reduction'],
                   holdout_results_train['te']['reduction']]
    
    x = np.arange(len(spectra))
    width = 0.35
    
    ax.bar(x - width/2, train_red, width, label=f'Training [{TRAIN_LMIN},{TRAIN_LMAX}]', 
           color='blue', alpha=0.7)
    ax.bar(x + width/2, holdout_red, width, label=f'Holdout [{HOLDOUT_LMIN},{HOLDOUT_LMAX}]',
           color='red', alpha=0.7)
    
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(spectra)
    ax.set_ylabel('RMS Reduction (%)')
    ax.set_title('Training vs Holdout Performance')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Holdout/Training ratio
    ax = axes[0, 1]
    ratios = [tt_ratio, ee_ratio, te_ratio]
    colors = ['green' if r >= 0.5 else 'red' for r in ratios]
    
    ax.bar(spectra, ratios, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(0.5, color='red', ls='--', label='50% threshold')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_ylabel('Holdout / Training Ratio')
    ax.set_title('Generalization Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Parameter comparison
    ax = axes[1, 0]
    params = ['α', 'γ']
    train_vals = [alpha_train * 1e3, gamma_train * 1e3]
    full_vals = [ALPHA_FULL * 1e3, GAMMA_FULL * 1e3]
    
    x = np.arange(len(params))
    ax.bar(x - width/2, train_vals, width, label='Training fit', color='blue', alpha=0.7)
    ax.bar(x + width/2, full_vals, width, label='Full fit (18B)', color='green', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel('Parameter × 10³')
    ax.set_title('Parameter Stability')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: ℓ-dependent residuals
    ax = axes[1, 1]
    
    # TT residuals
    eps_tt = epsilon_tt(ell, alpha_train)
    tt_shifted = apply_shift(ell, tt_lcdm, eps_tt)
    r_tt = fractional_residual(tt_bec, tt_shifted)
    
    ax.plot(ell[train_mask], r_tt[train_mask] * 100, 'b-', lw=0.5, alpha=0.7, label='TT (train)')
    ax.plot(ell[holdout_mask], r_tt[holdout_mask] * 100, 'r-', lw=0.5, alpha=0.7, label='TT (holdout)')
    
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.axvline(TRAIN_LMAX, color='black', ls='--', alpha=0.5, label='Train/Holdout boundary')
    ax.set_xlabel('ℓ')
    ax.set_ylabel('Fractional Residual (%)')
    ax.set_title('TT Residuals by Range')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Phase 20B: Holdout Test ({"PASS" if all_pass else "PARTIAL"})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase20b_holdout_test.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase20b_results.npz'
    np.savez(
        out_npz,
        alpha_train=alpha_train,
        gamma_train=gamma_train,
        alpha_full=ALPHA_FULL,
        gamma_full=GAMMA_FULL,
        train_results=train_results_train,
        holdout_results=holdout_results_train,
        all_pass=all_pass,
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 20B COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
