#!/usr/bin/env python3
"""
PHASE 19: TE CONSISTENCY UNDER THE (α, γ) MODEL

EXPLORATORY PHASE - NOT PART OF v1.0.0

Objective: Test how TE behaves under the polarization-dependent projection
operator discovered in Phase 18B.

BACKGROUND:
    Phase 18B established:
        ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
        ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)
    
    with α = -9.33×10⁻⁴, γ = -8.67×10⁻⁴

QUESTION:
    What ε should apply to TE?
    
    TE is a cross-correlation: TE = ⟨T × E⟩
    
    Possible models:
    1. ε_TE = ε_TT (TE tracks temperature)
    2. ε_TE = ε_EE (TE tracks polarization)
    3. ε_TE = (ε_TT + ε_EE)/2 (arithmetic mean)
    4. ε_TE = √(ε_TT × ε_EE) (geometric mean)
    5. ε_TE = ε₀ + (α + γ/2)·ln(ℓ/ℓ*) (half the offset)

PHYSICAL EXPECTATION:
    If the projection operator acts on the angular diameter distance,
    and TE correlates T and E at the same angular scale, then TE should
    see an effective ε between ε_TT and ε_EE.
    
    The geometric mean √(ε_TT × ε_EE) is natural if the effect is
    multiplicative in the transfer functions.

ACCEPTANCE CRITERIA:
    - One model clearly preferred by AIC
    - TE RMS improvement consistent with TT/EE
    - Phase alignment preserved (sign consistency)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# LOCKED PARAMETERS (FROM PHASE 18B)
# =============================================================================
EPSILON_0 = 1.4558030818e-03  # From Phase 10E
ELL_PIVOT = 1650              # Fixed pivot
LMIN, LMAX = 800, 2500        # Analysis range

# Phase 18B optimal parameters
ALPHA = -9.3333e-04           # Shared running
GAMMA = -8.6667e-04           # EE extra running


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


def epsilon_tt(ell):
    """ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)"""
    return EPSILON_0 + ALPHA * f_running(ell)


def epsilon_ee(ell):
    """ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)"""
    return EPSILON_0 + (ALPHA + GAMMA) * f_running(ell)


def epsilon_te_model(ell, model):
    """
    Compute ε_TE for different models.
    
    Models:
        'tt': ε_TE = ε_TT
        'ee': ε_TE = ε_EE
        'arith': ε_TE = (ε_TT + ε_EE) / 2
        'geom': ε_TE = √(ε_TT × ε_EE)
        'half': ε_TE = ε₀ + (α + γ/2)·ln(ℓ/ℓ*)
        'const': ε_TE = ε₀ (constant, v1.0.0)
    """
    eps_tt = epsilon_tt(ell)
    eps_ee = epsilon_ee(ell)
    
    if model == 'tt':
        return eps_tt
    elif model == 'ee':
        return eps_ee
    elif model == 'arith':
        return (eps_tt + eps_ee) / 2
    elif model == 'geom':
        return np.sqrt(eps_tt * eps_ee)
    elif model == 'half':
        return EPSILON_0 + (ALPHA + GAMMA/2) * f_running(ell)
    elif model == 'const':
        return np.full_like(ell, EPSILON_0, dtype=float)
    else:
        raise ValueError(f"Unknown model: {model}")


def apply_shift(ell, cl, eps_array):
    """Apply position-dependent horizontal shift."""
    ell_float = ell.astype(float)
    ell_star = ell_float / (1 + eps_array)
    cl_new = np.interp(ell_float, ell_star, cl, left=cl[0], right=cl[-1])
    return cl_new


def fractional_residual(cl_a, cl_b):
    """Compute (a - b) / b, handling sign changes."""
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


def sign_consistency(te_bec, te_shifted, mask):
    """
    Check sign consistency between BEC and shifted LCDM TE.
    TE has zero crossings, so we check if signs match.
    """
    signs_bec = np.sign(te_bec[mask])
    signs_shifted = np.sign(te_shifted[mask])
    
    # Fraction of points with matching signs
    match_frac = np.mean(signs_bec == signs_shifted)
    
    # Number of zero crossings
    crossings_bec = np.sum(np.diff(signs_bec) != 0)
    crossings_shifted = np.sum(np.diff(signs_shifted) != 0)
    
    return match_frac, crossings_bec, crossings_shifted


def main():
    print("=" * 70)
    print("PHASE 19: TE CONSISTENCY UNDER THE (α, γ) MODEL")
    print("=" * 70)
    print("\n*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Pivot ℓ* = {ELL_PIVOT}")
    print(f"Phase 18B parameters:")
    print(f"  α = {ALPHA:.4e} (shared running)")
    print(f"  γ = {GAMMA:.4e} (EE extra running)")
    
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
    
    if te_lcdm is None or te_bec is None:
        print("ERROR: TE data not available")
        return
    
    # Baseline TE
    r_te_base = fractional_residual(te_bec[mask], te_lcdm[mask])
    rms_te_baseline = rms(r_te_base)
    print(f"\n  TE baseline RMS: {rms_te_baseline:.6f}")
    
    # Test all models
    print("\n[2] Testing ε_TE models...")
    
    models = ['const', 'tt', 'ee', 'arith', 'geom', 'half']
    model_names = {
        'const': 'Constant ε₀ (v1.0.0)',
        'tt': 'ε_TE = ε_TT',
        'ee': 'ε_TE = ε_EE',
        'arith': 'ε_TE = (ε_TT + ε_EE)/2',
        'geom': 'ε_TE = √(ε_TT × ε_EE)',
        'half': 'ε_TE = ε₀ + (α + γ/2)·f(ℓ)',
    }
    
    results = {}
    
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'RMS':>10} {'Reduction':>12} {'Corr':>8} {'Sign Match':>12}")
    print("=" * 70)
    
    for model in models:
        eps_te = epsilon_te_model(ell, model)
        te_shifted = apply_shift(ell, te_lcdm, eps_te)
        
        r_te = fractional_residual(te_bec[mask], te_shifted[mask])
        rms_te = rms(r_te)
        reduction = (rms_te_baseline - rms_te) / rms_te_baseline * 100
        correlation = corr(r_te_base, r_te)
        sign_match, _, _ = sign_consistency(te_bec, te_shifted, mask)
        
        results[model] = {
            'rms': rms_te,
            'reduction': reduction,
            'corr': correlation,
            'sign_match': sign_match,
            'eps_te': eps_te,
        }
        
        print(f"{model_names[model]:<30} {rms_te:>10.6f} {reduction:>+11.1f}% {correlation:>+8.3f} {sign_match*100:>11.1f}%")
    
    # Find best model
    best_model = min(results.keys(), key=lambda m: results[m]['rms'])
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {model_names[best_model]}")
    print("=" * 70)
    
    # AIC comparison
    print("\n[3] AIC comparison...")
    
    # Reference: constant ε₀
    rms_ref = results['const']['rms']
    
    print(f"\n  Reference: {model_names['const']}")
    print(f"  {'Model':<30} {'Δχ²':>10} {'Δk':>6} {'AIC':>10}")
    print("  " + "-" * 60)
    
    for model in models:
        if model == 'const':
            continue
        
        rms_model = results[model]['rms']
        delta_chi2 = n_points * (rms_ref**2 - rms_model**2) / rms_ref**2
        
        # Parameter count relative to constant
        # const: 0 params (ε₀ is locked)
        # tt, ee: 0 extra (uses α, γ from TT/EE fit)
        # arith, geom, half: 0 extra (derived from α, γ)
        delta_k = 0  # All models use the same (α, γ) from Phase 18B
        
        aic = delta_chi2 - 2 * delta_k
        
        results[model]['delta_chi2'] = delta_chi2
        results[model]['aic'] = aic
        
        print(f"  {model_names[model]:<30} {delta_chi2:>+10.1f} {delta_k:>6} {aic:>+10.1f}")
    
    # ε profiles comparison
    print("\n[4] ε(ℓ) profiles at key multipoles...")
    
    ell_test = np.array([800, 1200, 1650, 2000, 2500])
    
    print(f"\n  {'ℓ':>6} {'ε_TT':>12} {'ε_EE':>12} {'ε_TE(geom)':>12} {'ε_TE(arith)':>12}")
    print("  " + "-" * 60)
    
    for l in ell_test:
        e_tt = epsilon_tt(l)
        e_ee = epsilon_ee(l)
        e_te_geom = epsilon_te_model(np.array([l]), 'geom')[0]
        e_te_arith = epsilon_te_model(np.array([l]), 'arith')[0]
        print(f"  {l:>6} {e_tt:>12.4e} {e_ee:>12.4e} {e_te_geom:>12.4e} {e_te_arith:>12.4e}")
    
    # Phase alignment analysis
    print("\n[5] Phase alignment analysis...")
    
    # Check zero crossings
    te_shifted_best = apply_shift(ell, te_lcdm, results[best_model]['eps_te'])
    sign_match, crossings_bec, crossings_shifted = sign_consistency(te_bec, te_shifted_best, mask)
    
    print(f"\n  Best model: {model_names[best_model]}")
    print(f"  Sign match: {sign_match*100:.1f}%")
    print(f"  Zero crossings (BEC): {crossings_bec}")
    print(f"  Zero crossings (shifted LCDM): {crossings_shifted}")
    print(f"  Crossing difference: {abs(crossings_bec - crossings_shifted)}")
    
    # Compare TT, EE, TE improvements
    print("\n[6] Cross-spectrum consistency...")
    
    # TT and EE with Phase 18B model
    tt_shifted = apply_shift(ell, tt_lcdm, epsilon_tt(ell))
    ee_shifted = apply_shift(ell, ee_lcdm, epsilon_ee(ell))
    
    r_tt = fractional_residual(tt_bec[mask], tt_shifted[mask])
    r_ee = fractional_residual(ee_bec[mask], ee_shifted[mask])
    
    rms_tt = rms(r_tt)
    rms_ee = rms(r_ee)
    rms_te_best = results[best_model]['rms']
    
    # Baselines
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    rms_tt_baseline = rms(r_tt_base)
    rms_ee_baseline = rms(r_ee_base)
    
    print(f"\n  {'Spectrum':<10} {'Baseline':>12} {'With (α,γ)':>12} {'Reduction':>12}")
    print("  " + "-" * 50)
    print(f"  {'TT':<10} {rms_tt_baseline:>12.6f} {rms_tt:>12.6f} {(rms_tt_baseline-rms_tt)/rms_tt_baseline*100:>+11.1f}%")
    print(f"  {'EE':<10} {rms_ee_baseline:>12.6f} {rms_ee:>12.6f} {(rms_ee_baseline-rms_ee)/rms_ee_baseline*100:>+11.1f}%")
    print(f"  {'TE':<10} {rms_te_baseline:>12.6f} {rms_te_best:>12.6f} {results[best_model]['reduction']:>+11.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 19 SUMMARY")
    print("=" * 70)
    
    # Determine conclusion
    if best_model in ['geom', 'arith', 'half']:
        conclusion = f"TE FOLLOWS GEOMETRIC/ARITHMETIC MEAN"
        interpretation = f"""
  The TE spectrum is best described by:
    ε_TE(ℓ) ≈ {model_names[best_model]}
  
  This is physically expected:
    - TE = ⟨T × E⟩ correlates temperature and polarization
    - The projection effect should be intermediate between TT and EE
    - The {'geometric' if best_model == 'geom' else 'arithmetic'} mean captures this naturally
  
  The (α, γ) model from Phase 18B successfully predicts TE behavior
  without any additional free parameters.
"""
    elif best_model == 'tt':
        conclusion = "TE TRACKS TEMPERATURE"
        interpretation = """
  The TE spectrum follows the TT projection:
    ε_TE(ℓ) = ε_TT(ℓ)
  
  This suggests temperature dominates the TE correlation,
  possibly due to the larger T signal amplitude.
"""
    elif best_model == 'ee':
        conclusion = "TE TRACKS POLARIZATION"
        interpretation = """
  The TE spectrum follows the EE projection:
    ε_TE(ℓ) = ε_EE(ℓ)
  
  This is unexpected and may indicate polarization-dominated
  projection effects in the cross-correlation.
"""
    else:
        conclusion = "TE PREFERS CONSTANT ε₀"
        interpretation = """
  The TE spectrum does not benefit from scale-dependent projection.
  The constant ε₀ from v1.0.0 remains optimal for TE.
"""
    
    print(f"\n  BEST MODEL: {model_names[best_model]}")
    print(f"  RMS reduction: {results[best_model]['reduction']:+.1f}%")
    print(f"  Sign consistency: {results[best_model]['sign_match']*100:.1f}%")
    print(f"\n  CONCLUSION: {conclusion}")
    print(interpretation)
    
    # Save summary
    summary = f"""PHASE 19: TE CONSISTENCY UNDER THE (α, γ) MODEL
============================================================
*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***

Phase 18B parameters:
  α = {ALPHA:.4e} (shared running)
  γ = {GAMMA:.4e} (EE extra running)

MODEL COMPARISON:
"""
    for model in models:
        r = results[model]
        summary += f"  {model_names[model]:<30}: RMS={r['rms']:.6f}, Reduction={r['reduction']:+.1f}%\n"
    
    summary += f"""
BEST MODEL: {model_names[best_model]}
  RMS: {results[best_model]['rms']:.6f}
  Reduction: {results[best_model]['reduction']:+.1f}%
  Sign consistency: {results[best_model]['sign_match']*100:.1f}%

CROSS-SPECTRUM CONSISTENCY:
  TT reduction: {(rms_tt_baseline-rms_tt)/rms_tt_baseline*100:+.1f}%
  EE reduction: {(rms_ee_baseline-rms_ee)/rms_ee_baseline*100:+.1f}%
  TE reduction: {results[best_model]['reduction']:+.1f}%

CONCLUSION: {conclusion}
{interpretation}
"""
    
    out_summary = base_dir / 'phase19_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plots
    print("\n[7] Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: ε profiles
    ax = axes[0, 0]
    ell_plot = np.linspace(LMIN, LMAX, 200)
    
    ax.plot(ell_plot, epsilon_tt(ell_plot) * 1e3, 'b-', lw=2, label='ε_TT')
    ax.plot(ell_plot, epsilon_ee(ell_plot) * 1e3, 'r-', lw=2, label='ε_EE')
    ax.plot(ell_plot, epsilon_te_model(ell_plot, 'geom') * 1e3, 'g--', lw=2, label='ε_TE (geom)')
    ax.plot(ell_plot, epsilon_te_model(ell_plot, 'arith') * 1e3, 'm:', lw=2, label='ε_TE (arith)')
    ax.axhline(EPSILON_0 * 1e3, color='gray', ls='--', alpha=0.5, label='ε₀')
    ax.axvline(ELL_PIVOT, color='gray', ls=':', alpha=0.5)
    
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ε(ℓ) × 10³')
    ax.set_title('ε(ℓ) Profiles: TT, EE, TE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Model comparison (RMS)
    ax = axes[0, 1]
    model_labels = [model_names[m].replace('ε_TE = ', '') for m in models]
    rms_values = [results[m]['rms'] for m in models]
    colors = ['gray' if m != best_model else 'green' for m in models]
    
    bars = ax.barh(model_labels, rms_values, color=colors, edgecolor='black')
    ax.axvline(rms_te_baseline, color='red', ls='--', label='Baseline')
    ax.set_xlabel('RMS')
    ax.set_title('TE Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Reduction comparison
    ax = axes[0, 2]
    reductions = [results[m]['reduction'] for m in models]
    ax.barh(model_labels, reductions, color=colors, edgecolor='black')
    ax.axvline(0, color='gray', ls='-', alpha=0.5)
    ax.set_xlabel('RMS Reduction (%)')
    ax.set_title('TE Improvement by Model')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: TE spectra comparison
    ax = axes[1, 0]
    ell_masked = ell[mask]
    
    ax.plot(ell_masked, te_bec[mask], 'k-', lw=0.5, alpha=0.7, label='BEC')
    ax.plot(ell_masked, te_lcdm[mask], 'b-', lw=0.5, alpha=0.7, label='ΛCDM')
    
    te_shifted_best = apply_shift(ell, te_lcdm, results[best_model]['eps_te'])
    ax.plot(ell_masked, te_shifted_best[mask], 'g-', lw=0.5, alpha=0.7, label=f'ΛCDM + {best_model}')
    
    ax.set_xlabel('ℓ')
    ax.set_ylabel('TE')
    ax.set_title('TE Spectra')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    ax = axes[1, 1]
    
    r_te_base = fractional_residual(te_bec[mask], te_lcdm[mask])
    r_te_best = fractional_residual(te_bec[mask], te_shifted_best[mask])
    
    ax.plot(ell_masked, r_te_base * 100, 'b-', lw=0.5, alpha=0.5, label='Baseline')
    ax.plot(ell_masked, r_te_best * 100, 'g-', lw=0.5, alpha=0.7, label=f'With {best_model}')
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    
    ax.set_xlabel('ℓ')
    ax.set_ylabel('Fractional Residual (%)')
    ax.set_title('TE Residuals')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Cross-spectrum summary
    ax = axes[1, 2]
    spectra = ['TT', 'EE', 'TE']
    baseline_rms = [rms_tt_baseline, rms_ee_baseline, rms_te_baseline]
    final_rms = [rms_tt, rms_ee, rms_te_best]
    
    x = np.arange(len(spectra))
    width = 0.35
    
    ax.bar(x - width/2, baseline_rms, width, label='Baseline', color='lightblue', edgecolor='blue')
    ax.bar(x + width/2, final_rms, width, label='With (α, γ)', color='lightgreen', edgecolor='green')
    
    ax.set_xticks(x)
    ax.set_xticklabels(spectra)
    ax.set_ylabel('RMS')
    ax.set_title('Cross-Spectrum Consistency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Phase 19: TE Consistency (Best: {model_names[best_model]})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase19_te_consistency.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase19_results.npz'
    np.savez(
        out_npz,
        epsilon_0=EPSILON_0,
        ell_pivot=ELL_PIVOT,
        alpha=ALPHA,
        gamma=GAMMA,
        models=models,
        best_model=best_model,
        rms_values={m: results[m]['rms'] for m in models},
        reductions={m: results[m]['reduction'] for m in models},
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 19 COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
