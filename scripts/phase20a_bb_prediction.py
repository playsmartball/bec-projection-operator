#!/usr/bin/env python3
"""
PHASE 20A: BB NULL / PREDICTION TEST

PREDICTION TEST - NO NEW PARAMETERS

Objective: Test whether the (ε₀, α, γ) operator structure predicts BB behavior.

HYPOTHESIS:
    If γ reflects polarization projection sensitivity at recombination,
    BB should behave like EE (since BB is also a polarization auto-spectrum).
    
    BUT: BB at ℓ > 100 is dominated by lensing, not recombination.
    Lensing-induced BB should NOT show the same γ-type running.

TESTS:
    1. Apply ε_TT(ℓ) to BB
    2. Apply ε_EE(ℓ) to BB  
    3. Apply constant ε₀
    4. Apply no operator (baseline)

ACCEPTANCE LOGIC:
    - If BB prefers ε_TT → γ is recombination-specific (EE only)
    - If BB prefers ε_EE → γ is polarization-generic
    - If BB prefers constant ε₀ or baseline → lensing dominance confirmed

This is a HARD FALSIFICATION channel.

LOCKED PARAMETERS (from Phase 18B):
    ε₀ = 1.4558030818e-03
    α = -9.3333e-04
    γ = -8.6667e-04
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# LOCKED PARAMETERS (FROM PHASE 18B - NO CHANGES ALLOWED)
# =============================================================================
EPSILON_0 = 1.4558030818e-03  # From Phase 10E
ELL_PIVOT = 1650              # Fixed pivot
LMIN, LMAX = 800, 2500        # Analysis range (limited by lensed file)

# Phase 18B optimal parameters - LOCKED
ALPHA = -9.3333e-04           # Shared running
GAMMA = -8.6667e-04           # EE extra running


def _load_class_cl_lensed(file_path: Path):
    """Load CLASS lensed Cℓ output file (includes BB)."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3]
    bb = data[:, 4]
    return ell, tt, ee, te, bb


def f_running(ell):
    """Running shape function: f(ℓ) = ln(ℓ/ℓ*)"""
    return np.log(ell / ELL_PIVOT)


def epsilon_tt(ell):
    """ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)"""
    return EPSILON_0 + ALPHA * f_running(ell)


def epsilon_ee(ell):
    """ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)"""
    return EPSILON_0 + (ALPHA + GAMMA) * f_running(ell)


def epsilon_const(ell):
    """Constant ε₀"""
    return np.full_like(ell, EPSILON_0, dtype=float)


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


def corr(a, b):
    """Pearson correlation."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def main():
    print("=" * 70)
    print("PHASE 20A: BB NULL / PREDICTION TEST")
    print("=" * 70)
    print("\n*** PREDICTION TEST - NO NEW PARAMETERS ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Locked α = {ALPHA:.4e}")
    print(f"Locked γ = {GAMMA:.4e}")
    print(f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]")
    
    base_dir = Path('/Users/hodge/Desktop/sanity_check/bec_cosmology/output')
    
    # Load LENSED spectra (BB is lensing-induced)
    print("\n[1] Loading LENSED spectra...")
    lcdm_path = base_dir / 'lcdm_zz_thetaS_reference_precise_00_cl_lensed.dat'
    bec_path = base_dir / 'test_bec_zz_thetaS_matched_precise_00_cl_lensed.dat'
    
    ell_lcdm, tt_lcdm, ee_lcdm, te_lcdm, bb_lcdm = _load_class_cl_lensed(lcdm_path)
    ell_bec, tt_bec, ee_bec, te_bec, bb_bec = _load_class_cl_lensed(bec_path)
    
    # Lensed files go to ℓ=2500
    max_ell = min(ell_lcdm.max(), ell_bec.max(), LMAX)
    mask = (ell_lcdm >= LMIN) & (ell_lcdm <= max_ell)
    n_points = mask.sum()
    
    print(f"  LCDM: ℓ ∈ [{ell_lcdm.min()}, {ell_lcdm.max()}]")
    print(f"  BEC: ℓ ∈ [{ell_bec.min()}, {ell_bec.max()}]")
    print(f"  Analysis: {n_points} multipoles in [{LMIN}, {max_ell}]")
    
    # Check BB signal level
    bb_lcdm_mean = np.mean(bb_lcdm[mask])
    bb_bec_mean = np.mean(bb_bec[mask])
    print(f"\n  BB signal level (mean in analysis range):")
    print(f"    LCDM: {bb_lcdm_mean:.4e}")
    print(f"    BEC: {bb_bec_mean:.4e}")
    print(f"    Ratio BEC/LCDM: {bb_bec_mean/bb_lcdm_mean:.4f}")
    
    # Baseline BB residual
    r_bb_base = fractional_residual(bb_bec[mask], bb_lcdm[mask])
    rms_bb_baseline = rms(r_bb_base)
    print(f"\n  BB baseline RMS (no operator): {rms_bb_baseline:.6f}")
    
    # Test models
    print("\n[2] Testing BB under different ε models...")
    
    models = {
        'baseline': lambda ell: np.zeros_like(ell, dtype=float),  # No shift
        'const': epsilon_const,
        'tt': epsilon_tt,
        'ee': epsilon_ee,
    }
    
    model_names = {
        'baseline': 'No operator',
        'const': 'Constant ε₀',
        'tt': 'ε_TT(ℓ) = ε₀ + α·f(ℓ)',
        'ee': 'ε_EE(ℓ) = ε₀ + (α+γ)·f(ℓ)',
    }
    
    results = {}
    
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'RMS':>12} {'Reduction':>12} {'Corr':>10}")
    print("=" * 70)
    
    for model_key, eps_func in models.items():
        if model_key == 'baseline':
            # No shift applied
            bb_shifted = bb_lcdm.copy()
        else:
            eps = eps_func(ell_lcdm)
            bb_shifted = apply_shift(ell_lcdm, bb_lcdm, eps)
        
        r_bb = fractional_residual(bb_bec[mask], bb_shifted[mask])
        rms_bb = rms(r_bb)
        reduction = (rms_bb_baseline - rms_bb) / rms_bb_baseline * 100
        correlation = corr(r_bb_base, r_bb)
        
        results[model_key] = {
            'rms': rms_bb,
            'reduction': reduction,
            'corr': correlation,
        }
        
        print(f"{model_names[model_key]:<35} {rms_bb:>12.6f} {reduction:>+11.1f}% {correlation:>+10.3f}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda m: results[m]['rms'])
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL FOR BB: {model_names[best_model]}")
    print("=" * 70)
    
    # Compare to TT, EE, TE behavior
    print("\n[3] Cross-spectrum comparison...")
    
    # TT with ε_TT
    tt_shifted = apply_shift(ell_lcdm, tt_lcdm, epsilon_tt(ell_lcdm))
    r_tt = fractional_residual(tt_bec[mask], tt_shifted[mask])
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    rms_tt = rms(r_tt)
    rms_tt_baseline = rms(r_tt_base)
    
    # EE with ε_EE
    ee_shifted = apply_shift(ell_lcdm, ee_lcdm, epsilon_ee(ell_lcdm))
    r_ee = fractional_residual(ee_bec[mask], ee_shifted[mask])
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    rms_ee = rms(r_ee)
    rms_ee_baseline = rms(r_ee_base)
    
    # TE with ε_TT (from Phase 19)
    te_shifted = apply_shift(ell_lcdm, te_lcdm, epsilon_tt(ell_lcdm))
    r_te = fractional_residual(te_bec[mask], te_shifted[mask])
    r_te_base = fractional_residual(te_bec[mask], te_lcdm[mask])
    rms_te = rms(r_te)
    rms_te_baseline = rms(r_te_base)
    
    print(f"\n  {'Spectrum':<10} {'Baseline':>12} {'With operator':>14} {'Reduction':>12} {'Operator':>20}")
    print("  " + "-" * 70)
    print(f"  {'TT':<10} {rms_tt_baseline:>12.6f} {rms_tt:>14.6f} {(rms_tt_baseline-rms_tt)/rms_tt_baseline*100:>+11.1f}% {'ε_TT':>20}")
    print(f"  {'EE':<10} {rms_ee_baseline:>12.6f} {rms_ee:>14.6f} {(rms_ee_baseline-rms_ee)/rms_ee_baseline*100:>+11.1f}% {'ε_EE':>20}")
    print(f"  {'TE':<10} {rms_te_baseline:>12.6f} {rms_te:>14.6f} {(rms_te_baseline-rms_te)/rms_te_baseline*100:>+11.1f}% {'ε_TT':>20}")
    print(f"  {'BB':<10} {rms_bb_baseline:>12.6f} {results[best_model]['rms']:>14.6f} {results[best_model]['reduction']:>+11.1f}% {model_names[best_model]:>20}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("PHASE 20A INTERPRETATION")
    print("=" * 70)
    
    # Determine what BB behavior tells us
    if best_model == 'ee':
        conclusion = "BB FOLLOWS EE → γ is polarization-generic"
        interpretation = """
  BB prefers the same operator as EE: ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)
  
  This means:
    - The polarization offset γ applies to ALL polarization spectra
    - γ is NOT specific to recombination-sourced polarization
    - Even lensing-induced BB sees the same projection effect
    
  Physical implication:
    The projection distortion couples to polarization generically,
    not just to the primordial E-mode source.
"""
    elif best_model == 'tt':
        conclusion = "BB FOLLOWS TT → γ is recombination-specific"
        interpretation = """
  BB prefers the TT operator: ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
  
  This means:
    - The polarization offset γ is specific to recombination EE
    - Lensing-induced BB does NOT see the γ offset
    - γ reflects something about the E-mode source, not polarization generically
    
  Physical implication:
    The EE-specific offset γ is tied to the recombination surface,
    possibly visibility function thickness or source geometry.
"""
    elif best_model == 'const':
        conclusion = "BB PREFERS CONSTANT ε₀ → Scale-dependence is spectrum-specific"
        interpretation = """
  BB prefers constant ε₀, not scale-dependent running.
  
  This means:
    - The running (α, γ) is specific to TT/EE/TE
    - Lensing-induced BB does not share this structure
    - The running may be tied to recombination physics
    
  Physical implication:
    The scale-dependent projection is a recombination-era effect,
    not a generic geometric distortion.
"""
    else:  # baseline
        conclusion = "BB PREFERS NO OPERATOR → Lensing dominates"
        interpretation = """
  BB is best described with no projection operator.
  
  This means:
    - The ΛCDM-BEC difference in BB is not geometric
    - Lensing-induced BB is insensitive to the projection effect
    - The operator structure is specific to primary CMB
    
  Physical implication:
    The projection operator acts on the primary CMB only,
    not on the lensing-induced secondary signal.
"""
    
    print(f"\n  CONCLUSION: {conclusion}")
    print(interpretation)
    
    # Statistical significance
    print("\n[4] Statistical significance...")
    
    # AIC comparison vs baseline
    for model_key in ['const', 'tt', 'ee']:
        rms_model = results[model_key]['rms']
        delta_chi2 = n_points * (rms_bb_baseline**2 - rms_model**2) / rms_bb_baseline**2
        # No new parameters - using locked (α, γ)
        aic = delta_chi2 - 0
        results[model_key]['delta_chi2'] = delta_chi2
        results[model_key]['aic'] = aic
        print(f"  {model_names[model_key]}: Δχ² = {delta_chi2:+.1f}, AIC = {aic:+.1f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 20A SUMMARY")
    print("=" * 70)
    
    print(f"""
  OPERATOR STRUCTURE (locked from Phase 18B):
    ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
    ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)
    ε_TE(ℓ) = ε_TT(ℓ)  [from Phase 19]
    
  BB PREDICTION TEST:
    Best model: {model_names[best_model]}
    RMS reduction: {results[best_model]['reduction']:+.1f}%
    
  CONCLUSION: {conclusion}
""")
    
    # Save summary
    summary = f"""PHASE 20A: BB NULL / PREDICTION TEST
============================================================
*** PREDICTION TEST - NO NEW PARAMETERS ***

Locked parameters (from Phase 18B):
  ε₀ = {EPSILON_0:.10e}
  α = {ALPHA:.4e}
  γ = {GAMMA:.4e}

BB MODEL COMPARISON:
"""
    for model_key in models:
        r = results[model_key]
        summary += f"  {model_names[model_key]:<35}: RMS={r['rms']:.6f}, Reduction={r['reduction']:+.1f}%\n"
    
    summary += f"""
BEST MODEL FOR BB: {model_names[best_model]}

CROSS-SPECTRUM SUMMARY:
  TT: {(rms_tt_baseline-rms_tt)/rms_tt_baseline*100:+.1f}% with ε_TT
  EE: {(rms_ee_baseline-rms_ee)/rms_ee_baseline*100:+.1f}% with ε_EE
  TE: {(rms_te_baseline-rms_te)/rms_te_baseline*100:+.1f}% with ε_TT
  BB: {results[best_model]['reduction']:+.1f}% with {model_names[best_model]}

CONCLUSION: {conclusion}
{interpretation}
"""
    
    out_summary = base_dir / 'phase20a_summary.txt'
    out_summary.write_text(summary)
    print(f"Saved: {out_summary}")
    
    # Generate plots
    print("\n[5] Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: BB spectra
    ax = axes[0, 0]
    ell_masked = ell_lcdm[mask]
    
    ax.semilogy(ell_masked, bb_lcdm[mask], 'b-', lw=1, alpha=0.7, label='ΛCDM')
    ax.semilogy(ell_masked, bb_bec[mask], 'r-', lw=1, alpha=0.7, label='BEC')
    
    ax.set_xlabel('ℓ')
    ax.set_ylabel('BB')
    ax.set_title('BB Spectra (Lensed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: BB residuals by model
    ax = axes[0, 1]
    
    for model_key, color in [('baseline', 'gray'), ('const', 'blue'), ('tt', 'green'), ('ee', 'red')]:
        if model_key == 'baseline':
            bb_shifted = bb_lcdm.copy()
        else:
            eps = models[model_key](ell_lcdm)
            bb_shifted = apply_shift(ell_lcdm, bb_lcdm, eps)
        
        r_bb = fractional_residual(bb_bec[mask], bb_shifted[mask])
        ax.plot(ell_masked, r_bb * 100, color=color, lw=0.5, alpha=0.7, 
                label=f'{model_names[model_key]} (RMS={results[model_key]["rms"]:.4f})')
    
    ax.axhline(0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('Fractional Residual (%)')
    ax.set_title('BB Residuals by Model')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Model comparison bar chart
    ax = axes[1, 0]
    model_labels = [model_names[m] for m in models]
    rms_values = [results[m]['rms'] for m in models]
    colors = ['green' if m == best_model else 'lightblue' for m in models]
    
    bars = ax.barh(model_labels, rms_values, color=colors, edgecolor='black')
    ax.set_xlabel('RMS')
    ax.set_title('BB Model Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Cross-spectrum summary
    ax = axes[1, 1]
    spectra = ['TT', 'EE', 'TE', 'BB']
    reductions = [
        (rms_tt_baseline - rms_tt) / rms_tt_baseline * 100,
        (rms_ee_baseline - rms_ee) / rms_ee_baseline * 100,
        (rms_te_baseline - rms_te) / rms_te_baseline * 100,
        results[best_model]['reduction'],
    ]
    operators = ['ε_TT', 'ε_EE', 'ε_TT', model_names[best_model].split('=')[0].strip() if '=' in model_names[best_model] else model_names[best_model]]
    
    colors = ['blue', 'red', 'green', 'purple']
    bars = ax.bar(spectra, reductions, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.set_ylabel('RMS Reduction (%)')
    ax.set_title('Cross-Spectrum Improvement')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add operator labels
    for bar, op in zip(bars, operators):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, op, 
                ha='center', va='bottom', fontsize=9, rotation=0)
    
    fig.suptitle(f'Phase 20A: BB Prediction Test (Best: {model_names[best_model]})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase20a_bb_prediction.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase20a_results.npz'
    np.savez(
        out_npz,
        epsilon_0=EPSILON_0,
        alpha=ALPHA,
        gamma=GAMMA,
        best_model=best_model,
        results=results,
        conclusion=conclusion,
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 20A COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
