#!/usr/bin/env python3
"""
PHASE 18B: SCALE-DEPENDENT POLARIZATION OFFSET

EXPLORATORY PHASE - NOT PART OF v1.0.0

Objective: Test whether TT/EE divergence can be captured by a scale-dependent
polarization offset δε(ℓ), rather than separate running parameters.

BACKGROUND:
    Phase 18A showed constant δε ≈ 0 (not significant)
    But Phases 16A/17A showed TT/EE diverge in their RUNNING
    This suggests the offset itself is scale-dependent

ANSATZ:
    ε_TT(ℓ) = ε₀ + α·f(ℓ)           [shared running]
    ε_EE(ℓ) = ε₀ + α·f(ℓ) + γ·g(ℓ)  [shared running + polarization offset]

where:
    f(ℓ) = ln(ℓ/ℓ*)                 [running shape, same for both]
    g(ℓ) = ln(ℓ/ℓ*)                 [offset shape, same functional form]
    α = shared running amplitude
    γ = polarization offset amplitude (new parameter)

This tests: "Do TT and EE share a common running α, with EE having
an additional scale-dependent offset γ?"

ACCEPTANCE CRITERIA:
    - γ ≠ 0 (statistically significant)
    - Single α works for both TT and EE (within tolerance)
    - Combined model AIC > separate α_TT, α_EE fits
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize_scalar

# =============================================================================
# LOCKED PARAMETERS (DO NOT CHANGE)
# =============================================================================
EPSILON_0 = 1.4558030818e-03  # From Phase 10E
ELL_PIVOT = 1650              # Fixed pivot
LMIN, LMAX = 800, 2500        # Analysis range

# =============================================================================
# PHASE 18B PARAMETERS
# =============================================================================
ALPHA_RANGE = (-2e-3, 2e-3)   # Shared running amplitude
GAMMA_RANGE = (-1e-3, 1e-3)   # Polarization offset amplitude


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
    """ε_TT(ℓ) = ε₀ + α·f(ℓ)"""
    return EPSILON_0 + alpha * f_running(ell)


def epsilon_ee(ell, alpha, gamma):
    """ε_EE(ℓ) = ε₀ + α·f(ℓ) + γ·f(ℓ) = ε₀ + (α+γ)·f(ℓ)"""
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


def combined_rms(alpha, gamma, ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask):
    """Compute combined TT+EE RMS for given (α, γ)."""
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
    
    return np.sqrt((rms_tt**2 + rms_ee**2) / 2), rms_tt, rms_ee


def grid_search(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask, n_alpha=41, n_gamma=41):
    """Grid search over (α, γ) space."""
    alpha_grid = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], n_alpha)
    gamma_grid = np.linspace(GAMMA_RANGE[0], GAMMA_RANGE[1], n_gamma)
    
    results = np.zeros((n_alpha, n_gamma, 3))  # combined, tt, ee
    
    for i, alpha in enumerate(alpha_grid):
        for j, gamma in enumerate(gamma_grid):
            comb, tt, ee = combined_rms(alpha, gamma, ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask)
            results[i, j, 0] = comb
            results[i, j, 1] = tt
            results[i, j, 2] = ee
    
    return alpha_grid, gamma_grid, results


def main():
    print("=" * 70)
    print("PHASE 18B: SCALE-DEPENDENT POLARIZATION OFFSET")
    print("=" * 70)
    print("\n*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Pivot ℓ* = {ELL_PIVOT}")
    print(f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]")
    print(f"\nAnsatz:")
    print(f"  ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)")
    print(f"  ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)")
    print(f"\nParameters:")
    print(f"  α = shared running amplitude")
    print(f"  γ = polarization offset amplitude (EE extra running)")
    
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
    
    # Baseline comparisons
    print("\n[2] Computing baselines...")
    
    # No operator
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    rms_tt_baseline = rms(r_tt_base)
    rms_ee_baseline = rms(r_ee_base)
    
    # Constant ε₀ (v1.0.0)
    eps_const = np.full(ell.shape, EPSILON_0)
    tt_const = apply_shift(ell, tt_lcdm, eps_const)
    ee_const = apply_shift(ell, ee_lcdm, eps_const)
    rms_tt_const = rms(fractional_residual(tt_bec[mask], tt_const[mask]))
    rms_ee_const = rms(fractional_residual(ee_bec[mask], ee_const[mask]))
    
    print(f"  Baseline (no operator):")
    print(f"    TT RMS: {rms_tt_baseline:.6f}")
    print(f"    EE RMS: {rms_ee_baseline:.6f}")
    print(f"  Constant ε₀ (v1.0.0):")
    print(f"    TT RMS: {rms_tt_const:.6f} ({(rms_tt_baseline-rms_tt_const)/rms_tt_baseline*100:+.1f}%)")
    print(f"    EE RMS: {rms_ee_const:.6f} ({(rms_ee_baseline-rms_ee_const)/rms_ee_baseline*100:+.1f}%)")
    
    # Grid search
    print("\n[3] Grid search over (α, γ)...")
    alpha_grid, gamma_grid, results = grid_search(
        ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask, n_alpha=61, n_gamma=61
    )
    
    # Find optimal (α, γ)
    idx_flat = np.argmin(results[:, :, 0])
    idx_alpha, idx_gamma = np.unravel_index(idx_flat, results[:, :, 0].shape)
    
    alpha_opt = alpha_grid[idx_alpha]
    gamma_opt = gamma_grid[idx_gamma]
    rms_combined_opt = results[idx_alpha, idx_gamma, 0]
    rms_tt_opt = results[idx_alpha, idx_gamma, 1]
    rms_ee_opt = results[idx_alpha, idx_gamma, 2]
    
    print(f"\n  Optimal parameters:")
    print(f"    α = {alpha_opt:.4e} (shared running)")
    print(f"    γ = {gamma_opt:.4e} (EE extra running)")
    print(f"    α + γ = {alpha_opt + gamma_opt:.4e} (effective EE running)")
    
    # Compare to Phase 16A separate fits
    print("\n" + "=" * 70)
    print("COMPARISON TO PHASE 16A (SEPARATE FITS)")
    print("=" * 70)
    
    # Phase 16A found:
    alpha_tt_16a = -9.5e-4
    alpha_ee_16a = -1.8e-3
    
    print(f"\n  Phase 16A (separate α for each):")
    print(f"    α_TT = {alpha_tt_16a:.4e}")
    print(f"    α_EE = {alpha_ee_16a:.4e}")
    print(f"    Difference = {alpha_ee_16a - alpha_tt_16a:.4e}")
    
    print(f"\n  Phase 18B (shared α + offset γ):")
    print(f"    α = {alpha_opt:.4e}")
    print(f"    γ = {gamma_opt:.4e}")
    print(f"    Effective α_EE = α + γ = {alpha_opt + gamma_opt:.4e}")
    
    # Check if γ explains the difference
    gamma_expected = alpha_ee_16a - alpha_tt_16a
    gamma_match = abs(gamma_opt - gamma_expected) / abs(gamma_expected) < 0.3
    
    print(f"\n  Expected γ (from 16A difference): {gamma_expected:.4e}")
    print(f"  Measured γ: {gamma_opt:.4e}")
    print(f"  Match (within 30%): {'YES' if gamma_match else 'NO'}")
    
    # Results at optimal
    print("\n" + "=" * 70)
    print("RESULTS AT OPTIMAL (α, γ)")
    print("=" * 70)
    
    print(f"\n  TT RMS: {rms_tt_opt:.6f}")
    print(f"    vs baseline: {(rms_tt_baseline-rms_tt_opt)/rms_tt_baseline*100:+.1f}%")
    print(f"    vs constant ε₀: {(rms_tt_const-rms_tt_opt)/rms_tt_const*100:+.1f}%")
    
    print(f"\n  EE RMS: {rms_ee_opt:.6f}")
    print(f"    vs baseline: {(rms_ee_baseline-rms_ee_opt)/rms_ee_baseline*100:+.1f}%")
    print(f"    vs constant ε₀: {(rms_ee_const-rms_ee_opt)/rms_ee_const*100:+.1f}%")
    
    # ε profiles at optimal
    print("\n  ε(ℓ) profiles at optimal:")
    ell_test = np.array([800, 1200, 1650, 2000, 2500])
    print(f"    {'ℓ':>6} {'ε_TT':>12} {'ε_EE':>12} {'Δε':>12}")
    for l in ell_test:
        e_tt = epsilon_tt(l, alpha_opt)
        e_ee = epsilon_ee(l, alpha_opt, gamma_opt)
        print(f"    {l:>6} {e_tt:>12.4e} {e_ee:>12.4e} {e_ee-e_tt:>12.4e}")
    
    # Statistical significance
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    # Compare models:
    # Model 1: Constant ε₀ (0 parameters)
    # Model 2: Shared α only, γ=0 (1 parameter)
    # Model 3: Shared α + γ (2 parameters)
    # Model 4: Separate α_TT, α_EE (2 parameters) - Phase 16A
    
    # Model 2: γ = 0
    _, rms_tt_model2, rms_ee_model2 = combined_rms(
        alpha_opt, 0, ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask
    )
    rms_combined_model2 = np.sqrt((rms_tt_model2**2 + rms_ee_model2**2) / 2)
    
    # Model 4: Separate fits (use Phase 16A values)
    _, rms_tt_model4, _ = combined_rms(
        alpha_tt_16a, 0, ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask
    )
    # For EE in model 4, we need α_EE acting alone
    eps_ee_16a = EPSILON_0 + alpha_ee_16a * f_running(ell)
    ee_shifted_16a = apply_shift(ell, ee_lcdm, eps_ee_16a)
    rms_ee_model4 = rms(fractional_residual(ee_bec[mask], ee_shifted_16a[mask]))
    rms_combined_model4 = np.sqrt((rms_tt_model4**2 + rms_ee_model4**2) / 2)
    
    print(f"\n  Model comparison (combined RMS):")
    print(f"    Model 1 (constant ε₀): {np.sqrt((rms_tt_const**2 + rms_ee_const**2)/2):.6f} [0 params]")
    print(f"    Model 2 (shared α, γ=0): {rms_combined_model2:.6f} [1 param]")
    print(f"    Model 3 (shared α + γ): {rms_combined_opt:.6f} [2 params]")
    print(f"    Model 4 (separate α_TT, α_EE): {rms_combined_model4:.6f} [2 params]")
    
    # AIC comparison: Model 3 vs Model 2 (does γ help?)
    rms_model2_sq = rms_combined_model2**2
    rms_model3_sq = rms_combined_opt**2
    delta_chi2_gamma = 2 * n_points * (rms_model2_sq - rms_model3_sq) / rms_model2_sq
    aic_gamma = delta_chi2_gamma - 2  # 1 extra parameter
    
    print(f"\n  Does γ improve over shared α alone?")
    print(f"    Δχ² (Model 3 vs Model 2): {delta_chi2_gamma:.1f}")
    print(f"    AIC improvement: {aic_gamma:.1f}")
    print(f"    γ significant: {'YES' if aic_gamma > 0 else 'NO'}")
    
    # Is γ significantly different from 0?
    gamma_significant = abs(gamma_opt) > 0.1 * abs(alpha_opt) and aic_gamma > 0
    
    print(f"\n  γ/α ratio: {gamma_opt/alpha_opt:.2f}" if alpha_opt != 0 else "")
    print(f"  γ significantly ≠ 0: {'YES' if gamma_significant else 'NO'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 18B SUMMARY")
    print("=" * 70)
    
    if gamma_significant:
        conclusion = "SCALE-DEPENDENT POLARIZATION OFFSET DETECTED"
        interpretation = f"""
  The TT/EE divergence is captured by a scale-dependent offset:
    
    ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
    ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)
    
  with:
    α = {alpha_opt:.4e} (shared running)
    γ = {gamma_opt:.4e} (EE extra running)
    
  Physical interpretation:
    - TT and EE share a common geometric running
    - EE has additional sensitivity to projection curvature
    - This is consistent with polarization source thickness effects
    
  The offset γ is NOT constant (Phase 18A showed δε ≈ 0).
  The offset is SCALE-DEPENDENT: δε(ℓ) = γ·ln(ℓ/ℓ*)
"""
    else:
        conclusion = "NO SIGNIFICANT SCALE-DEPENDENT OFFSET"
        interpretation = "  The shared α model is sufficient; γ does not improve the fit."
    
    print(f"\n  CONCLUSION: {conclusion}")
    print(interpretation)
    
    # Save summary
    summary = f"""PHASE 18B: SCALE-DEPENDENT POLARIZATION OFFSET
============================================================
*** EXPLORATORY PHASE - NOT PART OF v1.0.0 ***

Locked ε₀ = {EPSILON_0:.10e}
Pivot ℓ* = {ELL_PIVOT}

ANSATZ:
  ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)
  ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)

OPTIMAL PARAMETERS:
  α = {alpha_opt:.4e} (shared running)
  γ = {gamma_opt:.4e} (EE extra running)

RESULTS:
  TT RMS: {rms_tt_opt:.6f} ({(rms_tt_baseline-rms_tt_opt)/rms_tt_baseline*100:+.1f}% vs baseline)
  EE RMS: {rms_ee_opt:.6f} ({(rms_ee_baseline-rms_ee_opt)/rms_ee_baseline*100:+.1f}% vs baseline)

STATISTICAL SIGNIFICANCE:
  Δχ² (γ contribution): {delta_chi2_gamma:.1f}
  AIC improvement: {aic_gamma:.1f}
  γ significant: {'YES' if gamma_significant else 'NO'}

COMPARISON TO PHASE 16A:
  Phase 16A: α_TT = {alpha_tt_16a:.4e}, α_EE = {alpha_ee_16a:.4e}
  Phase 18B: α = {alpha_opt:.4e}, γ = {gamma_opt:.4e}
  Expected γ = α_EE - α_TT = {gamma_expected:.4e}
  Match: {'YES' if gamma_match else 'NO'}

CONCLUSION: {conclusion}
{interpretation}
"""
    
    out_summary = base_dir / 'phase18b_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plots
    print("\n[4] Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: (α, γ) landscape - combined RMS
    ax = axes[0, 0]
    A, G = np.meshgrid(alpha_grid, gamma_grid, indexing='ij')
    c = ax.contourf(A * 1e3, G * 1e3, results[:, :, 0], levels=30, cmap='viridis')
    ax.plot(alpha_opt * 1e3, gamma_opt * 1e3, 'r*', ms=15, label='Optimal')
    ax.axhline(0, color='white', ls='--', alpha=0.5)
    ax.axvline(0, color='white', ls='--', alpha=0.5)
    ax.set_xlabel('α × 10³')
    ax.set_ylabel('γ × 10³')
    ax.set_title('Combined RMS Landscape')
    plt.colorbar(c, ax=ax, label='RMS')
    ax.legend()
    
    # Plot 2: ε profiles
    ax = axes[0, 1]
    ell_plot = np.linspace(LMIN, LMAX, 200)
    
    # Constant ε₀
    ax.axhline(EPSILON_0 * 1e3, color='gray', ls='--', label='ε₀ (v1.0.0)')
    
    # Phase 18B optimal
    eps_tt_plot = epsilon_tt(ell_plot, alpha_opt)
    eps_ee_plot = epsilon_ee(ell_plot, alpha_opt, gamma_opt)
    ax.plot(ell_plot, eps_tt_plot * 1e3, 'b-', lw=2, label=f'ε_TT (α={alpha_opt*1e3:.2f}×10⁻³)')
    ax.plot(ell_plot, eps_ee_plot * 1e3, 'r-', lw=2, label=f'ε_EE (α+γ={((alpha_opt+gamma_opt)*1e3):.2f}×10⁻³)')
    
    ax.axvline(ELL_PIVOT, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('ε(ℓ) × 10³')
    ax.set_title('ε(ℓ) Profiles at Optimal (α, γ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Offset δε(ℓ) = γ·ln(ℓ/ℓ*)
    ax = axes[0, 2]
    delta_eps_plot = gamma_opt * f_running(ell_plot)
    ax.plot(ell_plot, delta_eps_plot * 1e3, 'purple', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(ELL_PIVOT, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('ℓ')
    ax.set_ylabel('δε(ℓ) × 10³')
    ax.set_title(f'Scale-Dependent Offset δε(ℓ) = γ·ln(ℓ/ℓ*)\nγ = {gamma_opt*1e3:.2f}×10⁻³')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: RMS comparison
    ax = axes[1, 0]
    models = ['Baseline', 'ε₀\n(v1.0.0)', 'Shared α\n(γ=0)', 'α + γ\n(18B)', 'Separate\n(16A)']
    tt_rms = [rms_tt_baseline, rms_tt_const, rms_tt_model2, rms_tt_opt, rms_tt_model4]
    ee_rms = [rms_ee_baseline, rms_ee_const, rms_ee_model2, rms_ee_opt, rms_ee_model4]
    
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, tt_rms, width, label='TT', color='blue', alpha=0.7)
    ax.bar(x + width/2, ee_rms, width, label='EE', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('RMS')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Phase 16A vs 18B
    ax = axes[1, 1]
    ax.scatter([alpha_tt_16a * 1e3], [alpha_ee_16a * 1e3], s=150, c='orange', marker='o', 
               label='Phase 16A (separate)', zorder=5)
    ax.scatter([alpha_opt * 1e3], [(alpha_opt + gamma_opt) * 1e3], s=150, c='purple', marker='*',
               label='Phase 18B (α, α+γ)', zorder=5)
    ax.plot([-2, 2], [-2, 2], 'k--', alpha=0.5, label='TT = EE')
    ax.set_xlabel('Effective α_TT × 10³')
    ax.set_ylabel('Effective α_EE × 10³')
    ax.set_title('Phase 16A vs 18B Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 6: Improvement breakdown
    ax = axes[1, 2]
    improvements = [
        ('TT: ε₀→α', (rms_tt_const - rms_tt_opt) / rms_tt_const * 100),
        ('EE: ε₀→α', (rms_ee_const - rms_ee_model2) / rms_ee_const * 100),
        ('EE: α→α+γ', (rms_ee_model2 - rms_ee_opt) / rms_ee_model2 * 100),
    ]
    names = [x[0] for x in improvements]
    values = [x[1] for x in improvements]
    colors = ['blue', 'lightcoral', 'red']
    ax.barh(names, values, color=colors)
    ax.set_xlabel('RMS Improvement (%)')
    ax.set_title('Improvement Breakdown')
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(f'Phase 18B: Scale-Dependent Polarization Offset (γ = {gamma_opt*1e3:.2f}×10⁻³)', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase18b_scale_dependent_offset.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save results
    out_npz = base_dir / 'phase18b_results.npz'
    np.savez(
        out_npz,
        epsilon_0=EPSILON_0,
        ell_pivot=ELL_PIVOT,
        alpha_grid=alpha_grid,
        gamma_grid=gamma_grid,
        results=results,
        alpha_opt=alpha_opt,
        gamma_opt=gamma_opt,
        rms_tt_opt=rms_tt_opt,
        rms_ee_opt=rms_ee_opt,
        aic_gamma=aic_gamma,
        gamma_significant=gamma_significant,
    )
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 18B COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
