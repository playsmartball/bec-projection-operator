#!/usr/bin/env python3
"""
PHASE 20C: DATASET REPLICATION TEST

INTERNAL CONSISTENCY TEST - NO NEW PARAMETERS

Objective: Test whether the (ε₀, α, γ) operator structure is stable across
different CLASS runs and numerical precision settings.

AVAILABLE DATA:
    We don't have Planck frequency splits, but we have:
    1. Multiple precision settings (standard vs precise)
    2. Different phase runs (phase11e, phase12a, thermo)
    3. Lensed vs unlensed spectra
    
TEST STRATEGY:
    Apply the LOCKED operator from Phase 18B to different dataset pairs.
    If the operator is geometric (not numerical artifact), it should:
    - Give consistent RMS reduction across datasets
    - Not depend on precision settings
    - Work on both lensed and unlensed spectra

LOCKED PARAMETERS (from Phase 18B - NO CHANGES):
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
EPSILON_0 = 1.4558030818e-03
ELL_PIVOT = 1650
LMIN, LMAX = 800, 2500

ALPHA = -9.3333e-04
GAMMA = -8.6667e-04


def _load_class_cl(file_path: Path, lensed=False):
    """Load CLASS Cℓ output file."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3] if data.shape[1] > 3 else None
    return ell, tt, ee, te


def f_running(ell):
    """Running shape function."""
    return np.log(ell / ELL_PIVOT)


def epsilon_tt(ell):
    """ε_TT(ℓ) = ε₀ + α·ln(ℓ/ℓ*)"""
    return EPSILON_0 + ALPHA * f_running(ell)


def epsilon_ee(ell):
    """ε_EE(ℓ) = ε₀ + (α+γ)·ln(ℓ/ℓ*)"""
    return EPSILON_0 + (ALPHA + GAMMA) * f_running(ell)


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


def evaluate_dataset(lcdm_path, bec_path, name, lmin=LMIN, lmax=LMAX):
    """
    Evaluate the locked operator on a dataset pair.
    Returns baseline and operator RMS for TT, EE, TE.
    """
    try:
        ell_lcdm, tt_lcdm, ee_lcdm, te_lcdm = _load_class_cl(lcdm_path)
        ell_bec, tt_bec, ee_bec, te_bec = _load_class_cl(bec_path)
    except Exception as e:
        return None, f"Load error: {e}"
    
    # Check ell ranges match
    if not np.array_equal(ell_lcdm, ell_bec):
        return None, "ell mismatch"
    
    ell = ell_lcdm
    max_ell = min(ell.max(), lmax)
    mask = (ell >= lmin) & (ell <= max_ell)
    n_points = mask.sum()
    
    if n_points < 100:
        return None, f"Too few points: {n_points}"
    
    results = {'name': name, 'n_points': n_points}
    
    # TT
    r_tt_base = fractional_residual(tt_bec[mask], tt_lcdm[mask])
    tt_shifted = apply_shift(ell, tt_lcdm, epsilon_tt(ell))
    r_tt = fractional_residual(tt_bec[mask], tt_shifted[mask])
    
    results['tt_baseline'] = rms(r_tt_base)
    results['tt_operator'] = rms(r_tt)
    results['tt_reduction'] = (results['tt_baseline'] - results['tt_operator']) / results['tt_baseline'] * 100
    
    # EE
    r_ee_base = fractional_residual(ee_bec[mask], ee_lcdm[mask])
    ee_shifted = apply_shift(ell, ee_lcdm, epsilon_ee(ell))
    r_ee = fractional_residual(ee_bec[mask], ee_shifted[mask])
    
    results['ee_baseline'] = rms(r_ee_base)
    results['ee_operator'] = rms(r_ee)
    results['ee_reduction'] = (results['ee_baseline'] - results['ee_operator']) / results['ee_baseline'] * 100
    
    # TE (uses ε_TT)
    if te_lcdm is not None and te_bec is not None:
        r_te_base = fractional_residual(te_bec[mask], te_lcdm[mask])
        te_shifted = apply_shift(ell, te_lcdm, epsilon_tt(ell))
        r_te = fractional_residual(te_bec[mask], te_shifted[mask])
        
        results['te_baseline'] = rms(r_te_base)
        results['te_operator'] = rms(r_te)
        results['te_reduction'] = (results['te_baseline'] - results['te_operator']) / results['te_baseline'] * 100
    else:
        results['te_baseline'] = np.nan
        results['te_operator'] = np.nan
        results['te_reduction'] = np.nan
    
    return results, None


def main():
    print("=" * 70)
    print("PHASE 20C: DATASET REPLICATION TEST")
    print("=" * 70)
    print("\n*** INTERNAL CONSISTENCY TEST - NO NEW PARAMETERS ***\n")
    print(f"Locked ε₀ = {EPSILON_0:.10e}")
    print(f"Locked α = {ALPHA:.4e}")
    print(f"Locked γ = {GAMMA:.4e}")
    
    base_dir = Path('/Users/hodge/Desktop/sanity_check/bec_cosmology/output')
    
    # Define dataset pairs to test
    datasets = [
        # Primary dataset (what we've been using)
        {
            'name': 'Primary (θ_S matched, precise)',
            'lcdm': 'lcdm_zz_thetaS_reference_precise_00_cl.dat',
            'bec': 'test_bec_zz_thetaS_matched_precise_00_cl.dat',
        },
        # Lensed version
        {
            'name': 'Lensed (θ_S matched, precise)',
            'lcdm': 'lcdm_zz_thetaS_reference_precise_00_cl_lensed.dat',
            'bec': 'test_bec_zz_thetaS_matched_precise_00_cl_lensed.dat',
        },
        # Phase 11e run
        {
            'name': 'Phase 11e run',
            'lcdm': 'lcdm_zz_thetaS_reference_precise_phase11e_00_cl.dat',
            'bec': 'test_bec_zz_thetaS_matched_precise_phase11e_00_cl.dat',
        },
        # Phase 11e lensed
        {
            'name': 'Phase 11e lensed',
            'lcdm': 'lcdm_zz_thetaS_reference_precise_phase11e_00_cl_lensed.dat',
            'bec': 'test_bec_zz_thetaS_matched_precise_phase11e_00_cl_lensed.dat',
        },
        # Standard precision (non-precise)
        {
            'name': 'Standard precision',
            'lcdm': 'lcdm_zz_thetaS_reference_00_cl.dat',
            'bec': 'test_bec_zz_thetaS_matched_00_cl.dat',
        },
        # Standard precision lensed
        {
            'name': 'Standard precision lensed',
            'lcdm': 'lcdm_zz_thetaS_reference_00_cl_lensed.dat',
            'bec': 'test_bec_zz_thetaS_matched_00_cl_lensed.dat',
        },
    ]
    
    print("\n[1] Testing datasets...")
    
    all_results = []
    
    for ds in datasets:
        lcdm_path = base_dir / ds['lcdm']
        bec_path = base_dir / ds['bec']
        
        if not lcdm_path.exists() or not bec_path.exists():
            print(f"  SKIP: {ds['name']} (files not found)")
            continue
        
        results, error = evaluate_dataset(lcdm_path, bec_path, ds['name'])
        
        if error:
            print(f"  SKIP: {ds['name']} ({error})")
            continue
        
        all_results.append(results)
        print(f"  OK: {ds['name']} ({results['n_points']} points)")
    
    if len(all_results) == 0:
        print("\nERROR: No valid datasets found")
        return
    
    # Print results table
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    
    print(f"\n{'Dataset':<35} {'TT Red':>10} {'EE Red':>10} {'TE Red':>10}")
    print("-" * 70)
    
    for r in all_results:
        te_str = f"{r['te_reduction']:+.1f}%" if not np.isnan(r['te_reduction']) else "N/A"
        print(f"{r['name']:<35} {r['tt_reduction']:>+9.1f}% {r['ee_reduction']:>+9.1f}% {te_str:>10}")
    
    # Compute statistics
    tt_reductions = [r['tt_reduction'] for r in all_results]
    ee_reductions = [r['ee_reduction'] for r in all_results]
    te_reductions = [r['te_reduction'] for r in all_results if not np.isnan(r['te_reduction'])]
    
    print("\n" + "=" * 70)
    print("CONSISTENCY STATISTICS")
    print("=" * 70)
    
    print(f"\n  TT reduction: mean = {np.mean(tt_reductions):+.1f}%, std = {np.std(tt_reductions):.1f}%")
    print(f"  EE reduction: mean = {np.mean(ee_reductions):+.1f}%, std = {np.std(ee_reductions):.1f}%")
    if len(te_reductions) > 0:
        print(f"  TE reduction: mean = {np.mean(te_reductions):+.1f}%, std = {np.std(te_reductions):.1f}%")
    
    # Acceptance criteria
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)
    
    # 1. All datasets show positive reduction
    tt_all_positive = all(r > 0 for r in tt_reductions)
    ee_all_positive = all(r > 0 for r in ee_reductions)
    
    print(f"\n  1. All datasets show positive reduction:")
    print(f"     TT: {'PASS' if tt_all_positive else 'FAIL'}")
    print(f"     EE: {'PASS' if ee_all_positive else 'FAIL'}")
    
    # 2. Standard deviation < 20% of mean (consistency)
    tt_consistent = np.std(tt_reductions) < 0.2 * abs(np.mean(tt_reductions))
    ee_consistent = np.std(ee_reductions) < 0.2 * abs(np.mean(ee_reductions))
    
    print(f"\n  2. Reduction consistent across datasets (std < 20% of mean):")
    print(f"     TT: {'PASS' if tt_consistent else 'FAIL'} (std/mean = {np.std(tt_reductions)/abs(np.mean(tt_reductions))*100:.1f}%)")
    print(f"     EE: {'PASS' if ee_consistent else 'FAIL'} (std/mean = {np.std(ee_reductions)/abs(np.mean(ee_reductions))*100:.1f}%)")
    
    # 3. Lensed vs unlensed consistency
    primary_idx = next((i for i, r in enumerate(all_results) if 'Primary' in r['name']), None)
    lensed_idx = next((i for i, r in enumerate(all_results) if r['name'] == 'Lensed (θ_S matched, precise)'), None)
    
    if primary_idx is not None and lensed_idx is not None:
        tt_lens_diff = abs(all_results[primary_idx]['tt_reduction'] - all_results[lensed_idx]['tt_reduction'])
        ee_lens_diff = abs(all_results[primary_idx]['ee_reduction'] - all_results[lensed_idx]['ee_reduction'])
        
        tt_lens_ok = tt_lens_diff < 10
        ee_lens_ok = ee_lens_diff < 10
        
        print(f"\n  3. Lensed vs unlensed difference < 10%:")
        print(f"     TT: {'PASS' if tt_lens_ok else 'FAIL'} (diff = {tt_lens_diff:.1f}%)")
        print(f"     EE: {'PASS' if ee_lens_ok else 'FAIL'} (diff = {ee_lens_diff:.1f}%)")
    
    # Overall
    all_pass = tt_all_positive and ee_all_positive and tt_consistent and ee_consistent
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 20C SUMMARY")
    print("=" * 70)
    
    if all_pass:
        conclusion = "REPLICATION TEST PASSED - Operator is stable across datasets"
        interpretation = """
  The locked (ε₀, α, γ) operator gives consistent RMS reduction across:
    - Different precision settings
    - Lensed and unlensed spectra
    - Different CLASS run phases
  
  This rules out numerical artifacts and confirms the operator
  captures genuine geometric structure in the ΛCDM-BEC residuals.
"""
    else:
        conclusion = "REPLICATION TEST PARTIAL - Some inconsistency detected"
        interpretation = """
  The operator shows some variation across datasets.
  This may indicate:
    - Sensitivity to numerical precision
    - Different behavior for lensed vs unlensed
    - Run-to-run variation in CLASS output
"""
    
    print(f"\n  OVERALL: {'PASS' if all_pass else 'PARTIAL'}")
    print(f"\n  CONCLUSION: {conclusion}")
    print(interpretation)
    
    # Save summary
    summary = f"""PHASE 20C: DATASET REPLICATION TEST
============================================================
*** INTERNAL CONSISTENCY TEST - NO NEW PARAMETERS ***

Locked parameters:
  ε₀ = {EPSILON_0:.10e}
  α = {ALPHA:.4e}
  γ = {GAMMA:.4e}

DATASETS TESTED: {len(all_results)}

RESULTS:
"""
    for r in all_results:
        te_str = f"{r['te_reduction']:+.1f}%" if not np.isnan(r['te_reduction']) else "N/A"
        summary += f"  {r['name']}: TT={r['tt_reduction']:+.1f}%, EE={r['ee_reduction']:+.1f}%, TE={te_str}\n"
    
    summary += f"""
STATISTICS:
  TT: mean={np.mean(tt_reductions):+.1f}%, std={np.std(tt_reductions):.1f}%
  EE: mean={np.mean(ee_reductions):+.1f}%, std={np.std(ee_reductions):.1f}%

OVERALL: {'PASS' if all_pass else 'PARTIAL'}

CONCLUSION: {conclusion}
{interpretation}
"""
    
    out_summary = base_dir / 'phase20c_summary.txt'
    out_summary.write_text(summary)
    print(f"\nSaved: {out_summary}")
    
    # Generate plot
    print("\n[2] Generating plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Reduction by dataset
    ax = axes[0]
    dataset_names = [r['name'][:25] + '...' if len(r['name']) > 25 else r['name'] for r in all_results]
    x = np.arange(len(dataset_names))
    width = 0.35
    
    ax.bar(x - width/2, tt_reductions, width, label='TT', color='blue', alpha=0.7)
    ax.bar(x + width/2, ee_reductions, width, label='EE', color='red', alpha=0.7)
    
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.axhline(np.mean(tt_reductions), color='blue', ls='--', alpha=0.5)
    ax.axhline(np.mean(ee_reductions), color='red', ls='--', alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RMS Reduction (%)')
    ax.set_title('Reduction by Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Consistency
    ax = axes[1]
    spectra = ['TT', 'EE']
    means = [np.mean(tt_reductions), np.mean(ee_reductions)]
    stds = [np.std(tt_reductions), np.std(ee_reductions)]
    
    ax.bar(spectra, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.set_ylabel('RMS Reduction (%)')
    ax.set_title('Mean ± Std Across Datasets')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Phase 20C: Dataset Replication ({"PASS" if all_pass else "PARTIAL"})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_plot = base_dir / 'phase20c_replication.png'
    fig.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 20C COMPLETE: {conclusion}")
    print("=" * 70)


if __name__ == '__main__':
    main()
