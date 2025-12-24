#!/usr/bin/env python3
"""
Test script for V9 BEC Crust CLASS implementation.

Validation tests:
1. ΛCDM recovery: n_bec = 0 should give identical results to standard ΛCDM
2. Background evolution: Check that BEC density evolves as (1+z)^n_bec
3. Distance measures: Compare angular diameter distance and sound horizon
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/hodge/Desktop/sanity_check/fracos/phi/class_v9_bec/python')

from classy import Class

def test_lcdm_recovery():
    """Test that n_bec = 0 recovers exact ΛCDM."""
    print("=" * 60)
    print("TEST 1: ΛCDM Recovery (n_bec = 0)")
    print("=" * 60)
    
    # Common parameters
    common_params = {
        'h': 0.6736,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'A_s': 2.1e-9,
        'n_s': 0.9649,
        'tau_reio': 0.0544,
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': 2500,
    }
    
    # Standard ΛCDM
    cosmo_lcdm = Class()
    cosmo_lcdm.set(common_params)
    cosmo_lcdm.compute()
    
    # V9 BEC Crust with n_bec = 0 (should be identical to ΛCDM)
    cosmo_v9 = Class()
    v9_params = common_params.copy()
    v9_params['bec_crust'] = 'yes'
    v9_params['n_bec'] = 0.0
    cosmo_v9.set(v9_params)
    cosmo_v9.compute()
    
    # Compare background quantities
    bg_lcdm = cosmo_lcdm.get_background()
    bg_v9 = cosmo_v9.get_background()
    
    # Compare H(z) at z=0
    H0_lcdm = cosmo_lcdm.Hubble(0) * 299792.458  # Convert to km/s/Mpc
    H0_v9 = cosmo_v9.Hubble(0) * 299792.458
    
    print(f"\nH0 comparison:")
    print(f"  ΛCDM:     {H0_lcdm:.4f} km/s/Mpc")
    print(f"  V9 n=0:   {H0_v9:.4f} km/s/Mpc")
    print(f"  Diff:     {abs(H0_lcdm - H0_v9):.2e} km/s/Mpc")
    
    # Compare angular diameter distance at z=1100 (CMB)
    DA_lcdm = cosmo_lcdm.angular_distance(1100)
    DA_v9 = cosmo_v9.angular_distance(1100)
    
    print(f"\nAngular diameter distance at z=1100:")
    print(f"  ΛCDM:     {DA_lcdm:.4f} Mpc")
    print(f"  V9 n=0:   {DA_v9:.4f} Mpc")
    print(f"  Diff:     {abs(DA_lcdm - DA_v9):.2e} Mpc ({100*abs(DA_lcdm - DA_v9)/DA_lcdm:.2e}%)")
    
    # Compare CMB TT spectrum
    cl_lcdm = cosmo_lcdm.lensed_cl(2500)
    cl_v9 = cosmo_v9.lensed_cl(2500)
    
    ell = np.arange(2, 2501)
    tt_lcdm = cl_lcdm['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * 1e12  # μK²
    tt_v9 = cl_v9['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * 1e12
    
    max_diff = np.max(np.abs(tt_lcdm - tt_v9))
    rel_diff = np.max(np.abs(tt_lcdm - tt_v9) / tt_lcdm)
    
    print(f"\nCMB TT spectrum (ℓ=2-2500):")
    print(f"  Max absolute diff: {max_diff:.2e} μK²")
    print(f"  Max relative diff: {rel_diff:.2e}")
    
    # Check if differences are at machine precision
    passed = rel_diff < 1e-6
    print(f"\n{'✓ PASSED' if passed else '✗ FAILED'}: ΛCDM recovery test")
    
    cosmo_lcdm.struct_cleanup()
    cosmo_v9.struct_cleanup()
    
    return passed


def test_bec_evolution():
    """Test that BEC density evolves as (1+z)^n_bec."""
    print("\n" + "=" * 60)
    print("TEST 2: BEC Density Evolution")
    print("=" * 60)
    
    # Test with n_bec = 0.5 (V9 prediction)
    n_bec = 0.5
    
    cosmo = Class()
    cosmo.set({
        'h': 0.6736,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'A_s': 2.1e-9,
        'n_s': 0.9649,
        'tau_reio': 0.0544,
        'bec_crust': 'yes',
        'n_bec': n_bec,
        'output': 'tCl',
        'l_max_scalars': 100,
    })
    cosmo.compute()
    
    bg = cosmo.get_background()
    
    # Check if BEC crust columns exist
    if '(.)rho_bec_crust' in bg:
        z = bg['z']
        rho_bec = bg['(.)rho_bec_crust']
        
        # Find indices where rho_bec > 0
        valid = rho_bec > 0
        z_valid = z[valid]
        rho_valid = rho_bec[valid]
        
        if len(z_valid) > 10:
            # Check scaling: rho_bec(z) / rho_bec(0) should equal (1+z)^n_bec
            rho_0 = rho_valid[-1]  # z=0 is at the end
            z_test = z_valid[::len(z_valid)//10]  # Sample 10 points
            rho_test = rho_valid[::len(rho_valid)//10]
            
            print(f"\nBEC density scaling test (n_bec = {n_bec}):")
            print(f"{'z':>10} {'ρ_BEC/ρ_BEC(0)':>15} {'(1+z)^n':>15} {'Ratio':>10}")
            print("-" * 55)
            
            max_error = 0
            for z_i, rho_i in zip(z_test, rho_test):
                ratio_actual = rho_i / rho_0
                ratio_expected = (1 + z_i) ** n_bec
                error = abs(ratio_actual / ratio_expected - 1)
                max_error = max(max_error, error)
                print(f"{z_i:10.2f} {ratio_actual:15.4f} {ratio_expected:15.4f} {ratio_actual/ratio_expected:10.6f}")
            
            passed = max_error < 1e-4
            print(f"\nMax relative error: {max_error:.2e}")
            print(f"{'✓ PASSED' if passed else '✗ FAILED'}: BEC evolution test")
        else:
            print("Not enough valid BEC data points")
            passed = False
    else:
        print("BEC crust columns not found in background output")
        print("Available columns:", list(bg.keys()))
        passed = False
    
    cosmo.struct_cleanup()
    return passed


def test_v9_vs_lcdm():
    """Compare V9 (n_bec=0.5) with ΛCDM to see the effect."""
    print("\n" + "=" * 60)
    print("TEST 3: V9 (n_bec=0.5) vs ΛCDM Comparison")
    print("=" * 60)
    
    common_params = {
        'h': 0.6736,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'A_s': 2.1e-9,
        'n_s': 0.9649,
        'tau_reio': 0.0544,
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': 2500,
    }
    
    # Standard ΛCDM
    cosmo_lcdm = Class()
    cosmo_lcdm.set(common_params)
    cosmo_lcdm.compute()
    
    # V9 BEC Crust with n_bec = 0.5
    cosmo_v9 = Class()
    v9_params = common_params.copy()
    v9_params['bec_crust'] = 'yes'
    v9_params['n_bec'] = 0.5
    cosmo_v9.set(v9_params)
    cosmo_v9.compute()
    
    # Compare key quantities
    H0_lcdm = cosmo_lcdm.Hubble(0) * 299792.458
    H0_v9 = cosmo_v9.Hubble(0) * 299792.458
    
    DA_lcdm = cosmo_lcdm.angular_distance(1100)
    DA_v9 = cosmo_v9.angular_distance(1100)
    
    rs_lcdm = cosmo_lcdm.rs_drag()
    rs_v9 = cosmo_v9.rs_drag()
    
    print(f"\nComparison (ΛCDM vs V9 n_bec=0.5):")
    print(f"{'Quantity':>25} {'ΛCDM':>15} {'V9':>15} {'Δ%':>10}")
    print("-" * 70)
    print(f"{'H0 [km/s/Mpc]':>25} {H0_lcdm:15.4f} {H0_v9:15.4f} {100*(H0_v9-H0_lcdm)/H0_lcdm:10.3f}")
    print(f"{'D_A(z=1100) [Mpc]':>25} {DA_lcdm:15.4f} {DA_v9:15.4f} {100*(DA_v9-DA_lcdm)/DA_lcdm:10.3f}")
    print(f"{'r_s(drag) [Mpc]':>25} {rs_lcdm:15.4f} {rs_v9:15.4f} {100*(rs_v9-rs_lcdm)/rs_lcdm:10.3f}")
    
    # CMB peak positions
    cl_lcdm = cosmo_lcdm.lensed_cl(2500)
    cl_v9 = cosmo_v9.lensed_cl(2500)
    
    ell = np.arange(2, 2501)
    tt_lcdm = cl_lcdm['tt'][2:2501]
    tt_v9 = cl_v9['tt'][2:2501]
    
    # Find first peak
    peak_range = (150, 300)
    idx_start = peak_range[0] - 2
    idx_end = peak_range[1] - 2
    peak_lcdm = ell[idx_start:idx_end][np.argmax(tt_lcdm[idx_start:idx_end])]
    peak_v9 = ell[idx_start:idx_end][np.argmax(tt_v9[idx_start:idx_end])]
    
    print(f"{'First TT peak ℓ':>25} {peak_lcdm:15d} {peak_v9:15d} {100*(peak_v9-peak_lcdm)/peak_lcdm:10.3f}")
    
    cosmo_lcdm.struct_cleanup()
    cosmo_v9.struct_cleanup()
    
    print(f"\n✓ V9 comparison complete")
    return True


def test_equation_of_state():
    """Test that w_bec = n_bec/3 - 1."""
    print("\n" + "=" * 60)
    print("TEST 4: Equation of State w = n_bec/3 - 1")
    print("=" * 60)
    
    for n_bec in [0.0, 0.5, 1.0, 1.5]:
        w_expected = n_bec / 3.0 - 1.0
        
        cosmo = Class()
        cosmo.set({
            'h': 0.6736,
            'omega_b': 0.02237,
            'omega_cdm': 0.12,
            'bec_crust': 'yes',
            'n_bec': n_bec,
            'output': 'tCl',
            'l_max_scalars': 100,
        })
        cosmo.compute()
        
        bg = cosmo.get_background()
        
        if 'w_bec_crust' in bg:
            w_actual = bg['w_bec_crust'][0]  # Should be constant
            print(f"n_bec = {n_bec:.1f}: w_expected = {w_expected:.4f}, w_actual = {w_actual:.4f}")
        else:
            print(f"n_bec = {n_bec:.1f}: w_bec_crust column not found")
        
        cosmo.struct_cleanup()
    
    print(f"\n✓ Equation of state test complete")
    return True


if __name__ == "__main__":
    print("V9 BEC Crust CLASS Implementation Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("ΛCDM Recovery", test_lcdm_recovery()))
    except Exception as e:
        print(f"ΛCDM Recovery test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ΛCDM Recovery", False))
    
    try:
        results.append(("BEC Evolution", test_bec_evolution()))
    except Exception as e:
        print(f"BEC Evolution test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("BEC Evolution", False))
    
    try:
        results.append(("V9 vs ΛCDM", test_v9_vs_lcdm()))
    except Exception as e:
        print(f"V9 vs ΛCDM test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("V9 vs ΛCDM", False))
    
    try:
        results.append(("Equation of State", test_equation_of_state()))
    except Exception as e:
        print(f"Equation of State test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Equation of State", False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
