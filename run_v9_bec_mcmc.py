#!/usr/bin/env python3
"""
V9 BEC Crust MCMC Analysis with Planck Likelihood

This script runs MCMC sampling to constrain n_bec using Planck 2018 data.

Key features:
- Uses modified CLASS with V9 BEC Crust implementation
- Single new parameter: n_bec (BEC dilution exponent)
- Ω_BEC_0 fixed by closure (not a free parameter)
- n_bec = 0 recovers exact ΛCDM

The question we're answering:
"Is n_bec ≈ 0.5 allowed by Planck, or is it ruled out?"
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/Users/hodge/Desktop/sanity_check/fracos/phi/class_v9_bec/python')

from cobaya.run import run
from cobaya.log import LoggedError

# Output directory
OUTPUT_DIR = '/Users/hodge/Desktop/sanity_check/fracos/phi/v9_bec_mcmc_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MCMC Configuration
info = {
    "likelihood": {
        "planck_2018_highl_plik.TTTEEE_lite": None,
        "planck_2018_lowl.TT": None,
    },
    
    "theory": {
        "classy": {
            "extra_args": {
                "bec_crust": "yes",
                "non_linear": "hmcode",
            },
        },
    },
    
    "params": {
        # Standard ΛCDM parameters
        "logA": {
            "prior": {"min": 2.5, "max": 3.5},
            "ref": {"dist": "norm", "loc": 3.044, "scale": 0.014},
            "proposal": 0.001,
            "latex": r"\log(10^{10} A_\mathrm{s})",
            "drop": True,
        },
        "A_s": {
            "value": "lambda logA: 1e-10*np.exp(logA)",
            "latex": r"A_\mathrm{s}",
        },
        "n_s": {
            "prior": {"min": 0.9, "max": 1.1},
            "ref": {"dist": "norm", "loc": 0.9649, "scale": 0.004},
            "proposal": 0.002,
            "latex": r"n_\mathrm{s}",
        },
        "H0": {
            "prior": {"min": 60, "max": 80},
            "ref": {"dist": "norm", "loc": 67.36, "scale": 0.5},
            "proposal": 0.2,
            "latex": r"H_0",
        },
        "omega_b": {
            "prior": {"min": 0.019, "max": 0.025},
            "ref": {"dist": "norm", "loc": 0.02237, "scale": 0.00015},
            "proposal": 0.0001,
            "latex": r"\Omega_\mathrm{b} h^2",
        },
        "omega_cdm": {
            "prior": {"min": 0.10, "max": 0.14},
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
            "proposal": 0.0005,
            "latex": r"\Omega_\mathrm{c} h^2",
        },
        "tau_reio": {
            "prior": {"min": 0.01, "max": 0.1},
            "ref": {"dist": "norm", "loc": 0.0544, "scale": 0.007},
            "proposal": 0.003,
            "latex": r"\tau_\mathrm{reio}",
        },
        
        # V9 BEC Crust parameter
        "n_bec": {
            "prior": {"min": -1.0, "max": 2.0},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 0.2},
            "proposal": 0.05,
            "latex": r"n_\mathrm{BEC}",
        },
        
        # Planck calibration parameter
        "A_planck": {
            "prior": {"dist": "norm", "loc": 1.0, "scale": 0.0025},
            "ref": {"dist": "norm", "loc": 1.0, "scale": 0.002},
            "proposal": 0.0005,
            "latex": r"y_\mathrm{cal}",
        },
    },
    
    "sampler": {
        "mcmc": {
            "max_samples": 50000,
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.1,
            "burn_in": 0,
            "learn_proposal": True,
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
            "covmat": "auto",
        },
    },
    
    "output": os.path.join(OUTPUT_DIR, "v9_bec_planck"),
    "force": True,
}


def run_quick_test():
    """Run a quick test to verify the setup works."""
    print("=" * 60)
    print("Quick Test: Verifying CLASS + Planck likelihood setup")
    print("=" * 60)
    
    from classy import Class
    
    # Test CLASS with V9 BEC Crust
    cosmo = Class()
    cosmo.set({
        'h': 0.6736,
        'omega_b': 0.02237,
        'omega_cdm': 0.12,
        'A_s': 2.1e-9,
        'n_s': 0.9649,
        'tau_reio': 0.0544,
        'bec_crust': 'yes',
        'n_bec': 0.0,
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': 2508,
    })
    cosmo.compute()
    
    print(f"✓ CLASS computed successfully")
    print(f"  H0 = {cosmo.Hubble(0) * 299792.458:.2f} km/s/Mpc")
    print(f"  D_A(z=1100) = {cosmo.angular_distance(1100):.2f} Mpc")
    
    cosmo.struct_cleanup()
    
    # Test Planck likelihood loading
    try:
        from cobaya.likelihood import Likelihood
        print(f"✓ Cobaya likelihood module available")
    except ImportError as e:
        print(f"✗ Cobaya import error: {e}")
        return False
    
    print(f"\n✓ Quick test passed - ready for MCMC")
    return True


def run_mcmc():
    """Run the full MCMC analysis."""
    print("=" * 60)
    print("V9 BEC Crust MCMC Analysis")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Parameter to constrain: n_bec")
    print(f"Prior: n_bec ∈ [-1, 2]")
    print(f"\nStarting MCMC sampling...")
    
    try:
        updated_info, sampler = run(info)
        
        print("\n" + "=" * 60)
        print("MCMC Complete!")
        print("=" * 60)
        
        # Print summary statistics
        if hasattr(sampler, 'products'):
            products = sampler.products()
            if 'sample' in products:
                samples = products['sample']
                
                # Get n_bec statistics
                n_bec_samples = samples['n_bec']
                print(f"\nn_bec constraints:")
                print(f"  Mean:   {np.mean(n_bec_samples):.4f}")
                print(f"  Std:    {np.std(n_bec_samples):.4f}")
                print(f"  Median: {np.median(n_bec_samples):.4f}")
                print(f"  16%:    {np.percentile(n_bec_samples, 16):.4f}")
                print(f"  84%:    {np.percentile(n_bec_samples, 84):.4f}")
                
                # Check if n_bec = 0.5 is within 2σ
                mean = np.mean(n_bec_samples)
                std = np.std(n_bec_samples)
                sigma_from_05 = abs(0.5 - mean) / std
                print(f"\n  n_bec = 0.5 is {sigma_from_05:.1f}σ from mean")
                
                if sigma_from_05 < 2:
                    print(f"  → V9 prediction (n_bec ≈ 0.5) is ALLOWED by Planck")
                else:
                    print(f"  → V9 prediction (n_bec ≈ 0.5) is DISFAVORED by Planck")
        
        return True
        
    except LoggedError as e:
        print(f"\nMCMC Error: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='V9 BEC Crust MCMC Analysis')
    parser.add_argument('--test', action='store_true', help='Run quick test only')
    parser.add_argument('--samples', type=int, default=50000, help='Max MCMC samples')
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    # Update max samples if specified
    info['sampler']['mcmc']['max_samples'] = args.samples
    
    # Run quick test first
    if not run_quick_test():
        print("\nQuick test failed. Please fix issues before running MCMC.")
        sys.exit(1)
    
    print("\n")
    
    # Run MCMC
    success = run_mcmc()
    sys.exit(0 if success else 1)
