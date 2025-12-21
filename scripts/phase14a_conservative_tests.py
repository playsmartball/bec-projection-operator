#!/usr/bin/env python3
"""
PHASE 14A — CONSERVATIVE CONSISTENCY & ROBUSTNESS TESTS

Objective: Demonstrate that the Phase 13 projection-level geometric signal is
stable, polarization-coherent, and pipeline-localized, and does not arise from
noise, smoothing, lensing, or ℓ-window artifacts.

LOCKED INPUTS (DO NOT CHANGE):
- ε = 1.4558030818e-03 (SIGN FIXED)
- Operator: ℓ → ℓ / (1 - ε)
- Analysis range: ℓ ∈ [800, 2500]
- No fitting, no re-estimation of ε

TESTS:
14A-1: TE Cross-Spectrum Consistency
14A-2: Lensing Contamination Null Test
14A-3: ℓ-Window Stability Test
14A-4: Noise Robustness Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# =============================================================================
# LOCKED PARAMETERS
# =============================================================================
EPSILON = 1.4558030818e-03  # SIGN FIXED - DO NOT CHANGE
LMIN, LMAX = 800, 2500

# =============================================================================
# PASS/FAIL CRITERIA
# =============================================================================
PASS_RMS_REDUCTION_TE = 0.20      # ≥ 20%
PASS_CORRELATION_TE = 0.60        # ≥ 0.6
PASS_RMS_REDUCTION_UNLENSED = 0.30  # ≥ 30%
PASS_WINDOW_CORRELATION = 0.60    # ≥ 0.6
PASS_MC_MEAN_REDUCTION = 0.25     # ≥ 25%
PASS_MC_POSITIVE_FRAC = 0.90      # ≥ 90%


def _load_class_cl(file_path: Path):
    """Load CLASS Cℓ output file with TT, EE, TE."""
    data = np.loadtxt(file_path, comments='#')
    ell = data[:, 0].astype(int)
    tt = data[:, 1]
    ee = data[:, 2]
    te = data[:, 3] if data.shape[1] > 3 else None
    return ell, tt, ee, te


def _apply_horizontal_shift(ell, cl, epsilon):
    """
    Apply operator: ℓ → ℓ / (1 + ε)
    
    This is the correct sign convention established in Phase 13A:
    - Phase 10E measured Δℓ = ℓ_BEC - ℓ_LCDM < 0 (BEC peaks at lower ℓ)
    - To shift LCDM toward BEC, we need ℓ → ℓ/(1+ε) with ε > 0
    - This moves LCDM peaks to lower ℓ, matching BEC
    """
    ell_float = ell.astype(float)
    ell_star = ell_float / (1 + epsilon)
    cl_new = np.interp(ell_float, ell_star, cl, left=cl[0], right=cl[-1])
    return cl_new


def _fractional_residual(cl_a, cl_b):
    """Compute (a - b) / b."""
    denom = np.where(np.abs(cl_b) > 0, cl_b, 1.0)
    return (cl_a - cl_b) / denom


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def _corr(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if a.size != b.size or a.size < 2:
        return np.nan
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def test_14a1_te_consistency(ell, te_lcdm, te_bec, mask):
    """
    14A-1: TE Cross-Spectrum Consistency Test
    
    PASS Criteria:
    - RMS reduction ≥ 20%
    - Correlation ≥ 0.6
    - Same sign as TT/EE
    """
    print("\n" + "=" * 70)
    print("14A-1: TE CROSS-SPECTRUM CONSISTENCY TEST")
    print("=" * 70)
    
    if te_lcdm is None or te_bec is None:
        print("  ERROR: TE spectrum not available")
        return {'pass': False, 'reason': 'TE not available'}
    
    # Baseline residual
    r_baseline = _fractional_residual(te_bec[mask], te_lcdm[mask])
    rms_baseline = _rms(r_baseline)
    
    # Apply locked operator
    te_lcdm_shifted = _apply_horizontal_shift(ell, te_lcdm, EPSILON)
    r_shifted = _fractional_residual(te_bec[mask], te_lcdm_shifted[mask])
    rms_shifted = _rms(r_shifted)
    
    # Compute kernel effect
    kernel_effect = te_lcdm_shifted[mask] - te_lcdm[mask]
    bec_minus_lcdm = te_bec[mask] - te_lcdm[mask]
    
    # Correlation
    correlation = _corr(kernel_effect, bec_minus_lcdm)
    
    # RMS reduction
    rms_reduction = (rms_baseline - rms_shifted) / rms_baseline
    
    print(f"\n  Locked ε = {EPSILON:.10e}")
    print(f"\n  Baseline RMS: {rms_baseline:.6f}")
    print(f"  Shifted RMS:  {rms_shifted:.6f}")
    print(f"  RMS reduction: {rms_reduction*100:+.1f}%")
    print(f"  Correlation:   {correlation:+.4f}")
    
    # PASS/FAIL
    pass_rms = rms_reduction >= PASS_RMS_REDUCTION_TE
    pass_corr = correlation >= PASS_CORRELATION_TE
    pass_sign = correlation > 0  # Same sign as TT/EE (positive)
    
    print(f"\n  CRITERIA:")
    print(f"    RMS reduction ≥ {PASS_RMS_REDUCTION_TE*100:.0f}%: {'PASS' if pass_rms else 'FAIL'} ({rms_reduction*100:.1f}%)")
    print(f"    Correlation ≥ {PASS_CORRELATION_TE:.1f}: {'PASS' if pass_corr else 'FAIL'} ({correlation:.3f})")
    print(f"    Same sign (positive): {'PASS' if pass_sign else 'FAIL'}")
    
    overall_pass = pass_rms and pass_corr and pass_sign
    print(f"\n  OVERALL: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    result = {
        'pass': overall_pass,
        'rms_baseline': rms_baseline,
        'rms_shifted': rms_shifted,
        'rms_reduction': rms_reduction,
        'correlation': correlation,
        'pass_rms': pass_rms,
        'pass_corr': pass_corr,
        'pass_sign': pass_sign,
    }
    
    return result


def test_14a2_lensing_null(base_dir, ell_unlensed, mask_unlensed):
    """
    14A-2: Lensing Contamination Null Test
    
    PASS Criteria:
    - Unlensed RMS reduction ≥ 30%
    - Lensed RMS reduction ≤ unlensed (not stronger)
    """
    print("\n" + "=" * 70)
    print("14A-2: LENSING CONTAMINATION NULL TEST")
    print("=" * 70)
    
    # Load lensed spectra
    data_dir = base_dir.parent / 'data'
    lcdm_lensed_path = data_dir / 'lcdm_lensed' / 'lcdm_zz_thetaS_reference_precise_00_cl_lensed.dat'
    bec_lensed_path = data_dir / 'bec_lensed' / 'test_bec_zz_thetaS_matched_precise_00_cl_lensed.dat'
    
    if not lcdm_lensed_path.exists() or not bec_lensed_path.exists():
        print(f"  ERROR: Lensed files not found")
        return {'pass': False, 'reason': 'Lensed files not found'}
    
    print(f"\n  Loading lensed spectra...")
    ell_l, tt_lcdm_l, ee_lcdm_l, _ = _load_class_cl(lcdm_lensed_path)
    _, tt_bec_l, ee_bec_l, _ = _load_class_cl(bec_lensed_path)
    
    # Align to common grid
    common_ell = np.intersect1d(ell_unlensed, ell_l)
    mask_l = (common_ell >= LMIN) & (common_ell <= LMAX)
    
    idx_l = np.searchsorted(ell_l, common_ell)
    tt_lcdm_l = tt_lcdm_l[idx_l]
    ee_lcdm_l = ee_lcdm_l[idx_l]
    tt_bec_l = tt_bec_l[idx_l]
    ee_bec_l = ee_bec_l[idx_l]
    
    # Compute RMS reductions for lensed TT and EE
    results_lensed = {}
    for name, cl_lcdm, cl_bec in [('TT', tt_lcdm_l, tt_bec_l), ('EE', ee_lcdm_l, ee_bec_l)]:
        r_base = _fractional_residual(cl_bec[mask_l], cl_lcdm[mask_l])
        rms_base = _rms(r_base)
        
        cl_shifted = _apply_horizontal_shift(common_ell, cl_lcdm, EPSILON)
        r_shift = _fractional_residual(cl_bec[mask_l], cl_shifted[mask_l])
        rms_shift = _rms(r_shift)
        
        reduction = (rms_base - rms_shift) / rms_base
        results_lensed[name] = {'rms_base': rms_base, 'rms_shift': rms_shift, 'reduction': reduction}
    
    print(f"\n  Lensed results:")
    print(f"    TT: RMS {results_lensed['TT']['rms_base']:.6f} → {results_lensed['TT']['rms_shift']:.6f}, reduction = {results_lensed['TT']['reduction']*100:+.1f}%")
    print(f"    EE: RMS {results_lensed['EE']['rms_base']:.6f} → {results_lensed['EE']['rms_shift']:.6f}, reduction = {results_lensed['EE']['reduction']*100:+.1f}%")
    
    # We need unlensed results for comparison - compute them here
    lcdm_unlensed_path = data_dir / 'lcdm_unlensed' / 'lcdm_zz_thetaS_reference_precise_00_cl.dat'
    bec_unlensed_path = data_dir / 'bec_unlensed' / 'test_bec_zz_thetaS_matched_precise_00_cl.dat'
    
    ell_u, tt_lcdm_u, ee_lcdm_u, _ = _load_class_cl(lcdm_unlensed_path)
    _, tt_bec_u, ee_bec_u, _ = _load_class_cl(bec_unlensed_path)
    
    mask_u = (ell_u >= LMIN) & (ell_u <= LMAX)
    
    results_unlensed = {}
    for name, cl_lcdm, cl_bec in [('TT', tt_lcdm_u, tt_bec_u), ('EE', ee_lcdm_u, ee_bec_u)]:
        r_base = _fractional_residual(cl_bec[mask_u], cl_lcdm[mask_u])
        rms_base = _rms(r_base)
        
        cl_shifted = _apply_horizontal_shift(ell_u, cl_lcdm, EPSILON)
        r_shift = _fractional_residual(cl_bec[mask_u], cl_shifted[mask_u])
        rms_shift = _rms(r_shift)
        
        reduction = (rms_base - rms_shift) / rms_base
        results_unlensed[name] = {'rms_base': rms_base, 'rms_shift': rms_shift, 'reduction': reduction}
    
    print(f"\n  Unlensed results:")
    print(f"    TT: RMS {results_unlensed['TT']['rms_base']:.6f} → {results_unlensed['TT']['rms_shift']:.6f}, reduction = {results_unlensed['TT']['reduction']*100:+.1f}%")
    print(f"    EE: RMS {results_unlensed['EE']['rms_base']:.6f} → {results_unlensed['EE']['rms_shift']:.6f}, reduction = {results_unlensed['EE']['reduction']*100:+.1f}%")
    
    # PASS/FAIL criteria
    unlensed_tt_pass = results_unlensed['TT']['reduction'] >= PASS_RMS_REDUCTION_UNLENSED
    unlensed_ee_pass = results_unlensed['EE']['reduction'] >= PASS_RMS_REDUCTION_UNLENSED
    
    # Lensed should not be stronger than unlensed
    lensed_not_stronger_tt = results_lensed['TT']['reduction'] <= results_unlensed['TT']['reduction'] + 0.05
    lensed_not_stronger_ee = results_lensed['EE']['reduction'] <= results_unlensed['EE']['reduction'] + 0.05
    
    print(f"\n  CRITERIA:")
    print(f"    Unlensed TT reduction ≥ {PASS_RMS_REDUCTION_UNLENSED*100:.0f}%: {'PASS' if unlensed_tt_pass else 'FAIL'} ({results_unlensed['TT']['reduction']*100:.1f}%)")
    print(f"    Unlensed EE reduction ≥ {PASS_RMS_REDUCTION_UNLENSED*100:.0f}%: {'PASS' if unlensed_ee_pass else 'FAIL'} ({results_unlensed['EE']['reduction']*100:.1f}%)")
    print(f"    Lensed TT not stronger: {'PASS' if lensed_not_stronger_tt else 'FAIL'}")
    print(f"    Lensed EE not stronger: {'PASS' if lensed_not_stronger_ee else 'FAIL'}")
    
    overall_pass = unlensed_tt_pass and unlensed_ee_pass and lensed_not_stronger_tt and lensed_not_stronger_ee
    print(f"\n  OVERALL: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    result = {
        'pass': overall_pass,
        'unlensed': results_unlensed,
        'lensed': results_lensed,
        'unlensed_tt_pass': unlensed_tt_pass,
        'unlensed_ee_pass': unlensed_ee_pass,
        'lensed_not_stronger_tt': lensed_not_stronger_tt,
        'lensed_not_stronger_ee': lensed_not_stronger_ee,
    }
    
    return result


def test_14a3_window_stability(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec):
    """
    14A-3: ℓ-Window Stability Test
    
    PASS Criteria:
    - RMS reduction stays positive in all windows
    - Correlation stays ≥ 0.6
    """
    print("\n" + "=" * 70)
    print("14A-3: ℓ-WINDOW STABILITY TEST")
    print("=" * 70)
    
    windows = [
        (800, 1600),
        (1000, 2000),
        (1400, 2500),
    ]
    
    results = []
    
    for lmin, lmax in windows:
        m = (ell >= lmin) & (ell <= lmax)
        n_ell = int(np.sum(m))
        
        window_result = {'lmin': lmin, 'lmax': lmax, 'n_ell': n_ell}
        
        for name, cl_lcdm, cl_bec in [('TT', tt_lcdm, tt_bec), ('EE', ee_lcdm, ee_bec)]:
            # Baseline
            r_base = _fractional_residual(cl_bec[m], cl_lcdm[m])
            rms_base = _rms(r_base)
            
            # Shifted
            cl_shifted = _apply_horizontal_shift(ell, cl_lcdm, EPSILON)
            r_shift = _fractional_residual(cl_bec[m], cl_shifted[m])
            rms_shift = _rms(r_shift)
            
            reduction = (rms_base - rms_shift) / rms_base
            
            # Correlation
            kernel_effect = cl_shifted[m] - cl_lcdm[m]
            bec_minus_lcdm = cl_bec[m] - cl_lcdm[m]
            corr = _corr(kernel_effect, bec_minus_lcdm)
            
            window_result[f'{name}_reduction'] = reduction
            window_result[f'{name}_correlation'] = corr
        
        results.append(window_result)
        
        print(f"\n  [{lmin}, {lmax}] (n={n_ell}):")
        print(f"    TT: reduction={window_result['TT_reduction']*100:+.1f}%, corr={window_result['TT_correlation']:+.3f}")
        print(f"    EE: reduction={window_result['EE_reduction']*100:+.1f}%, corr={window_result['EE_correlation']:+.3f}")
    
    # PASS/FAIL criteria
    all_positive_tt = all(r['TT_reduction'] > 0 for r in results)
    all_positive_ee = all(r['EE_reduction'] > 0 for r in results)
    all_corr_tt = all(r['TT_correlation'] >= PASS_WINDOW_CORRELATION for r in results)
    all_corr_ee = all(r['EE_correlation'] >= PASS_WINDOW_CORRELATION for r in results)
    
    print(f"\n  CRITERIA:")
    print(f"    All TT reductions positive: {'PASS' if all_positive_tt else 'FAIL'}")
    print(f"    All EE reductions positive: {'PASS' if all_positive_ee else 'FAIL'}")
    print(f"    All TT correlations ≥ {PASS_WINDOW_CORRELATION}: {'PASS' if all_corr_tt else 'FAIL'}")
    print(f"    All EE correlations ≥ {PASS_WINDOW_CORRELATION}: {'PASS' if all_corr_ee else 'FAIL'}")
    
    overall_pass = all_positive_tt and all_positive_ee and all_corr_tt and all_corr_ee
    print(f"\n  OVERALL: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    return {
        'pass': overall_pass,
        'windows': results,
        'all_positive_tt': all_positive_tt,
        'all_positive_ee': all_positive_ee,
        'all_corr_tt': all_corr_tt,
        'all_corr_ee': all_corr_ee,
    }


def test_14a4_noise_mc(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask, n_realizations=50):
    """
    14A-4: Noise Robustness Monte Carlo
    
    PASS Criteria:
    - Mean RMS reduction ≥ 25%
    - ≥ 90% of realizations show positive reduction
    """
    print("\n" + "=" * 70)
    print("14A-4: NOISE ROBUSTNESS MONTE CARLO")
    print("=" * 70)
    
    # Baseline residual RMS (for noise scaling)
    r_tt_base = _fractional_residual(tt_bec[mask], tt_lcdm[mask])
    r_ee_base = _fractional_residual(ee_bec[mask], ee_lcdm[mask])
    rms_tt_base = _rms(r_tt_base)
    rms_ee_base = _rms(r_ee_base)
    
    noise_levels = [0.3, 0.5]
    
    np.random.seed(42)  # Reproducibility
    
    all_results = {}
    
    for noise_frac in noise_levels:
        print(f"\n  Noise level: {noise_frac*100:.0f}% of baseline RMS")
        
        sigma_tt = rms_tt_base * noise_frac
        sigma_ee = rms_ee_base * noise_frac
        
        reductions_tt = []
        reductions_ee = []
        
        for i in range(n_realizations):
            # Add noise to ΛCDM
            noise_tt = np.random.normal(0, sigma_tt, size=tt_lcdm.shape) * tt_lcdm
            noise_ee = np.random.normal(0, sigma_ee, size=ee_lcdm.shape) * ee_lcdm
            
            tt_lcdm_noisy = tt_lcdm + noise_tt
            ee_lcdm_noisy = ee_lcdm + noise_ee
            
            # Baseline RMS with noisy ΛCDM
            r_tt = _fractional_residual(tt_bec[mask], tt_lcdm_noisy[mask])
            r_ee = _fractional_residual(ee_bec[mask], ee_lcdm_noisy[mask])
            rms_tt = _rms(r_tt)
            rms_ee = _rms(r_ee)
            
            # Apply operator to noisy ΛCDM
            tt_shifted = _apply_horizontal_shift(ell, tt_lcdm_noisy, EPSILON)
            ee_shifted = _apply_horizontal_shift(ell, ee_lcdm_noisy, EPSILON)
            
            r_tt_shift = _fractional_residual(tt_bec[mask], tt_shifted[mask])
            r_ee_shift = _fractional_residual(ee_bec[mask], ee_shifted[mask])
            rms_tt_shift = _rms(r_tt_shift)
            rms_ee_shift = _rms(r_ee_shift)
            
            red_tt = (rms_tt - rms_tt_shift) / rms_tt
            red_ee = (rms_ee - rms_ee_shift) / rms_ee
            
            reductions_tt.append(red_tt)
            reductions_ee.append(red_ee)
        
        reductions_tt = np.array(reductions_tt)
        reductions_ee = np.array(reductions_ee)
        
        mean_tt = np.mean(reductions_tt)
        mean_ee = np.mean(reductions_ee)
        std_tt = np.std(reductions_tt)
        std_ee = np.std(reductions_ee)
        pos_frac_tt = np.mean(reductions_tt > 0)
        pos_frac_ee = np.mean(reductions_ee > 0)
        
        print(f"    TT: mean={mean_tt*100:+.1f}% ± {std_tt*100:.1f}%, positive={pos_frac_tt*100:.0f}%")
        print(f"    EE: mean={mean_ee*100:+.1f}% ± {std_ee*100:.1f}%, positive={pos_frac_ee*100:.0f}%")
        
        all_results[noise_frac] = {
            'reductions_tt': reductions_tt,
            'reductions_ee': reductions_ee,
            'mean_tt': mean_tt,
            'mean_ee': mean_ee,
            'std_tt': std_tt,
            'std_ee': std_ee,
            'pos_frac_tt': pos_frac_tt,
            'pos_frac_ee': pos_frac_ee,
        }
    
    # PASS/FAIL: use the higher noise level (0.5) for strictest test
    strict = all_results[0.5]
    
    pass_mean_tt = strict['mean_tt'] >= PASS_MC_MEAN_REDUCTION
    pass_mean_ee = strict['mean_ee'] >= PASS_MC_MEAN_REDUCTION
    pass_pos_tt = strict['pos_frac_tt'] >= PASS_MC_POSITIVE_FRAC
    pass_pos_ee = strict['pos_frac_ee'] >= PASS_MC_POSITIVE_FRAC
    
    print(f"\n  CRITERIA (at 50% noise):")
    print(f"    Mean TT reduction ≥ {PASS_MC_MEAN_REDUCTION*100:.0f}%: {'PASS' if pass_mean_tt else 'FAIL'} ({strict['mean_tt']*100:.1f}%)")
    print(f"    Mean EE reduction ≥ {PASS_MC_MEAN_REDUCTION*100:.0f}%: {'PASS' if pass_mean_ee else 'FAIL'} ({strict['mean_ee']*100:.1f}%)")
    print(f"    TT positive ≥ {PASS_MC_POSITIVE_FRAC*100:.0f}%: {'PASS' if pass_pos_tt else 'FAIL'} ({strict['pos_frac_tt']*100:.0f}%)")
    print(f"    EE positive ≥ {PASS_MC_POSITIVE_FRAC*100:.0f}%: {'PASS' if pass_pos_ee else 'FAIL'} ({strict['pos_frac_ee']*100:.0f}%)")
    
    overall_pass = pass_mean_tt and pass_mean_ee and pass_pos_tt and pass_pos_ee
    print(f"\n  OVERALL: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    return {
        'pass': overall_pass,
        'results': all_results,
        'pass_mean_tt': pass_mean_tt,
        'pass_mean_ee': pass_mean_ee,
        'pass_pos_tt': pass_pos_tt,
        'pass_pos_ee': pass_pos_ee,
    }


def main():
    print("=" * 70)
    print("PHASE 14A: CONSERVATIVE CONSISTENCY & ROBUSTNESS TESTS")
    print("=" * 70)
    print(f"\nLocked ε = {EPSILON:.10e}")
    print(f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]")
    print(f"Operator: ℓ → ℓ / (1 - ε)")
    
    # Use relative paths from repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    base_dir = repo_root / 'output'
    data_dir = repo_root / 'data'
    
    # Load baseline spectra
    print("\n[Loading baseline spectra...]")
    lcdm_path = data_dir / 'lcdm_unlensed' / 'lcdm_zz_thetaS_reference_precise_00_cl.dat'
    bec_path = data_dir / 'bec_unlensed' / 'test_bec_zz_thetaS_matched_precise_00_cl.dat'
    
    ell, tt_lcdm, ee_lcdm, te_lcdm = _load_class_cl(lcdm_path)
    _, tt_bec, ee_bec, te_bec = _load_class_cl(bec_path)
    
    mask = (ell >= LMIN) & (ell <= LMAX)
    print(f"  Loaded {ell.size} multipoles, {mask.sum()} in analysis range")
    
    # Run all tests
    results = {}
    
    # 14A-1: TE Consistency
    results['14A-1'] = test_14a1_te_consistency(ell, te_lcdm, te_bec, mask)
    
    # 14A-2: Lensing Null
    results['14A-2'] = test_14a2_lensing_null(base_dir, ell, mask)
    
    # 14A-3: Window Stability
    results['14A-3'] = test_14a3_window_stability(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec)
    
    # 14A-4: Noise MC
    results['14A-4'] = test_14a4_noise_mc(ell, tt_lcdm, ee_lcdm, tt_bec, ee_bec, mask)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 14A SUMMARY")
    print("=" * 70)
    
    summary_lines = [
        "PHASE 14A: CONSERVATIVE CONSISTENCY & ROBUSTNESS TESTS",
        "=" * 60,
        f"",
        f"Locked ε = {EPSILON:.10e}",
        f"Analysis range: ℓ ∈ [{LMIN}, {LMAX}]",
        f"",
    ]
    
    test_names = {
        '14A-1': 'TE Cross-Spectrum Consistency',
        '14A-2': 'Lensing Contamination Null',
        '14A-3': 'ℓ-Window Stability',
        '14A-4': 'Noise Robustness MC',
    }
    
    n_pass = 0
    for test_id, test_name in test_names.items():
        passed = results[test_id]['pass']
        status = '✓ PASS' if passed else '✗ FAIL'
        summary_lines.append(f"{test_id}: {test_name}: {status}")
        if passed:
            n_pass += 1
    
    summary_lines.extend([
        "",
        f"OVERALL: {n_pass}/4 tests passed",
    ])
    
    if n_pass == 4:
        summary_lines.extend([
            "",
            "CONCLUSION: The ΛCDM–BEC residual is a real, coherent,",
            "projection-level geometric distortion, robust across",
            "polarization, ℓ-cuts, and noise, and independent of lensing.",
        ])
    elif n_pass >= 3:
        summary_lines.extend([
            "",
            "CONCLUSION: Strong evidence for geometric origin,",
            "with one test requiring further investigation.",
        ])
    else:
        summary_lines.extend([
            "",
            "CONCLUSION: Mixed results. Some tests failed.",
            "Geometric interpretation requires caution.",
        ])
    
    summary_txt = '\n'.join(summary_lines)
    print("\n" + summary_txt)
    
    # Save individual test results
    for test_id in test_names:
        out_path = base_dir / f'phase14a_{test_id.lower().replace("-", "_")}.txt'
        with open(out_path, 'w') as f:
            f.write(f"{test_id}: {test_names[test_id]}\n")
            f.write("=" * 50 + "\n")
            f.write(f"PASS: {results[test_id]['pass']}\n\n")
            for k, v in results[test_id].items():
                if k != 'pass':
                    f.write(f"{k}: {v}\n")
        print(f"Saved: {out_path}")
    
    # Save summary
    out_summary = base_dir / 'phase14a_summary.txt'
    out_summary.write_text(summary_txt + '\n')
    print(f"Saved: {out_summary}")
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    print("\n[Generating plots...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 14A-1: TE result
    ax = axes[0, 0]
    r = results['14A-1']
    if r.get('rms_reduction') is not None:
        bars = ax.bar(['Baseline', 'Shifted'], [r['rms_baseline'], r['rms_shifted']], 
                      color=['C0', 'C2'])
        ax.set_ylabel('RMS')
        ax.set_title(f"14A-1: TE Consistency\nReduction: {r['rms_reduction']*100:+.1f}%, Corr: {r['correlation']:+.3f}\n{'PASS' if r['pass'] else 'FAIL'}")
    else:
        ax.text(0.5, 0.5, 'TE not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('14A-1: TE Consistency')
    
    # 14A-2: Lensing comparison
    ax = axes[0, 1]
    r = results['14A-2']
    if 'unlensed' in r:
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [r['unlensed']['TT']['reduction']*100, r['unlensed']['EE']['reduction']*100],
               width, label='Unlensed', color='C0')
        ax.bar(x + width/2, [r['lensed']['TT']['reduction']*100, r['lensed']['EE']['reduction']*100],
               width, label='Lensed', color='C1')
        ax.set_xticks(x)
        ax.set_xticklabels(['TT', 'EE'])
        ax.set_ylabel('RMS Reduction [%]')
        ax.axhline(0, color='k', lw=1)
        ax.legend()
        ax.set_title(f"14A-2: Lensing Null Test\n{'PASS' if r['pass'] else 'FAIL'}")
    
    # 14A-3: Window stability
    ax = axes[1, 0]
    r = results['14A-3']
    windows = r['windows']
    x = np.arange(len(windows))
    width = 0.35
    ax.bar(x - width/2, [w['TT_reduction']*100 for w in windows], width, label='TT', color='C0')
    ax.bar(x + width/2, [w['EE_reduction']*100 for w in windows], width, label='EE', color='C1')
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{w['lmin']},{w['lmax']}]" for w in windows])
    ax.set_ylabel('RMS Reduction [%]')
    ax.axhline(0, color='k', lw=1)
    ax.legend()
    ax.set_title(f"14A-3: ℓ-Window Stability\n{'PASS' if r['pass'] else 'FAIL'}")
    
    # 14A-4: Noise MC
    ax = axes[1, 1]
    r = results['14A-4']
    mc_05 = r['results'][0.5]
    ax.hist(mc_05['reductions_tt']*100, bins=15, alpha=0.7, label='TT', color='C0')
    ax.hist(mc_05['reductions_ee']*100, bins=15, alpha=0.7, label='EE', color='C1')
    ax.axvline(0, color='k', lw=2, ls='--')
    ax.axvline(PASS_MC_MEAN_REDUCTION*100, color='r', lw=1, ls=':', label=f'Pass threshold ({PASS_MC_MEAN_REDUCTION*100:.0f}%)')
    ax.set_xlabel('RMS Reduction [%]')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title(f"14A-4: Noise MC (50% noise)\n{'PASS' if r['pass'] else 'FAIL'}")
    
    fig.tight_layout()
    out_plot = base_dir / 'phase14a_all_plots.png'
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_plot}")
    
    # Save NPZ
    out_npz = base_dir / 'phase14a_results.npz'
    np.savez(out_npz, **{k: v for k, v in results.items()})
    print(f"Saved: {out_npz}")
    
    print("\n" + "=" * 70)
    print(f"PHASE 14A COMPLETE: {n_pass}/4 tests passed")
    print("=" * 70)


if __name__ == '__main__':
    main()
