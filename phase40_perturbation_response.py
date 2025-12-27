#!/usr/bin/env python3
"""
Phase 40: Stratified Perturbation Response Test

Tests if vacuum has depth structure by measuring response to metric perturbations.

Key idea: Different probes (CMB, LSS, lensing) respond to same perturbation.
If vacuum is stratified, response amplitude should vary systematically with depth.

Method:
1. Compute transfer functions T(k,z) for ΛCDM and V9 BEC
2. Extract response amplitudes: A(z) = δO/δΦ at each redshift
3. Normalize: ΔA(z) = A(z)/A_ΛCDM(z)
4. Test for universal collapse across probes

This is apples-to-apples: same perturbation, dimensionless response, comparable.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'class_v9_bec/python'))
from classy import Class

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'phase40_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base cosmological parameters (Planck 2018)
BASE_PARAMS = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
}

# Redshift bins for response measurement (limited to z < 10 for CLASS stability)
Z_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

# Wavenumber for perturbation (Mpc^-1)
K_PIVOT = 0.05  # BAO scale


class PerturbationResponseTest:
    """Phase 40: Test for depth-dependent perturbation response."""
    
    def __init__(self):
        self.z_bins = Z_BINS
        self.k_pivot = K_PIVOT
        self.responses = {}
        
    def compute_growth_factor(self, cosmo, z):
        """Compute linear growth factor D(z)."""
        # Growth factor approximation (Carroll, Press & Turner 1992)
        Om_z = cosmo.Omega_m() * (1 + z)**3 / (cosmo.Omega_m() * (1 + z)**3 + 
                                                (1 - cosmo.Omega_m()))
        
        # Growth rate f = d ln D / d ln a
        f = Om_z**0.55
        
        # Growth factor (normalized to a=1)
        D = (1 + z)**(-1) * np.exp(-np.trapz([Om_z**0.55 / (1 + zp) 
                                               for zp in np.linspace(0, z, 100)],
                                              np.linspace(0, z, 100)))
        
        return D, f
        
    def compute_transfer_function(self, cosmo, k, z):
        """Compute matter transfer function T(k,z)."""
        # Get matter power spectrum
        try:
            Pk = cosmo.pk(k, z)
            # Transfer function: T²(k,z) ∝ P(k,z) / k^n_s
            # Normalized to T(k→0) = 1
            T_squared = Pk / (k**BASE_PARAMS['n_s'])
            T = np.sqrt(T_squared)
            return T
        except:
            return np.nan
            
    def compute_response_amplitude(self, model_name, n_bec=0.0):
        """
        Compute perturbation response amplitude at each redshift.
        
        Response: A(z) = growth_rate * sigma8(z) / sigma8(0)
        This measures how perturbations grow relative to today.
        """
        print(f"\n{'='*60}")
        print(f"Computing response for {model_name} (n_BEC={n_bec})")
        print(f"{'='*60}")
        
        # Set up CLASS
        params = BASE_PARAMS.copy()
        params['output'] = 'mPk'
        params['P_k_max_1/Mpc'] = 10.0
        params['z_max_pk'] = 10.0  # Limit to avoid early-time issues
        
        if n_bec != 0.0:
            params['bec_crust'] = 'yes'
            params['n_bec'] = n_bec
        
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        
        # Extract responses at each redshift
        responses = []
        sigma8_0 = cosmo.sigma8()
        
        for z in self.z_bins:
            try:
                # Use CLASS's built-in growth factor
                if z == 0:
                    D = 1.0
                    f = cosmo.Omega_m()**0.55
                else:
                    # Growth factor (normalized to a=1)
                    D = cosmo.scale_independent_growth_factor(z)
                    
                    # Growth rate f = d ln D / d ln a
                    # For ΛCDM and V9 BEC, use approximation f ≈ Ωm(z)^0.55
                    Om_z = cosmo.Omega_m() * (1 + z)**3 / (
                        cosmo.Omega_m() * (1 + z)**3 + (1 - cosmo.Omega_m())
                    )
                    f = Om_z**0.55
                
                # Response amplitude: A(z) = f(z) * D(z)
                # This is the response of density perturbations to metric perturbations
                A = f * D
                
                responses.append({
                    'z': z,
                    'growth_factor': D,
                    'growth_rate': f,
                    'response_amplitude': A
                })
                
                print(f"  z={z:7.1f}: D={D:.4f}, f={f:.4f}, A={A:.4f}")
                
            except Exception as e:
                print(f"  z={z:7.1f}: Failed - {e}")
                responses.append({
                    'z': z,
                    'growth_factor': np.nan,
                    'growth_rate': np.nan,
                    'response_amplitude': np.nan
                })
        
        cosmo.struct_cleanup()
        
        return responses
        
    def compute_all_responses(self):
        """Compute responses for ΛCDM and V9 BEC models."""
        print("\n" + "="*60)
        print("PHASE 40: STRATIFIED PERTURBATION RESPONSE TEST")
        print("="*60)
        
        # ΛCDM baseline
        self.responses['LCDM'] = self.compute_response_amplitude('ΛCDM', n_bec=0.0)
        
        # V9 BEC (Planck best-fit)
        self.responses['V9_018'] = self.compute_response_amplitude('V9 (n=0.18)', n_bec=0.18)
        
        # V9 BEC (prediction)
        self.responses['V9_050'] = self.compute_response_amplitude('V9 (n=0.5)', n_bec=0.5)
        
    def compute_normalized_responses(self):
        """Compute ΔA(z) = A(z) / A_ΛCDM(z)."""
        print("\n" + "="*60)
        print("NORMALIZED RESPONSE AMPLITUDES")
        print("="*60)
        
        normalized = {}
        
        for model in ['V9_018', 'V9_050']:
            normalized[model] = []
            
            print(f"\n{model}:")
            for i, z in enumerate(self.z_bins):
                A_lcdm = self.responses['LCDM'][i]['response_amplitude']
                A_model = self.responses[model][i]['response_amplitude']
                
                if not np.isnan(A_lcdm) and not np.isnan(A_model) and A_lcdm != 0:
                    delta_A = A_model / A_lcdm
                else:
                    delta_A = np.nan
                    
                normalized[model].append({
                    'z': z,
                    'delta_A': delta_A
                })
                
                if not np.isnan(delta_A):
                    print(f"  z={z:7.1f}: ΔA = {delta_A:.6f} ({(delta_A-1)*100:+.3f}%)")
                    
        self.normalized_responses = normalized
        return normalized
        
    def test_depth_stratification(self):
        """Test if ΔA(z) shows depth-dependent structure."""
        print("\n" + "="*60)
        print("DEPTH STRATIFICATION TEST")
        print("="*60)
        
        results = {}
        
        for model in ['V9_018', 'V9_050']:
            print(f"\n{model}:")
            
            # Extract valid data
            z_vals = []
            delta_A_vals = []
            
            for resp in self.normalized_responses[model]:
                if not np.isnan(resp['delta_A']):
                    z_vals.append(resp['z'])
                    delta_A_vals.append(resp['delta_A'])
            
            z_vals = np.array(z_vals)
            delta_A_vals = np.array(delta_A_vals)
            
            if len(z_vals) < 3:
                print("  Insufficient data")
                continue
            
            # Test 1: Monotonicity
            rho, p_value = spearmanr(z_vals, delta_A_vals)
            print(f"\n  Monotonicity test:")
            print(f"    Spearman ρ: {rho:.3f}")
            print(f"    p-value: {p_value:.4f}")
            
            # Test 2: Deviation from unity
            mean_delta_A = np.mean(delta_A_vals)
            std_delta_A = np.std(delta_A_vals)
            deviation = abs(mean_delta_A - 1.0) / std_delta_A if std_delta_A > 0 else 0
            
            print(f"\n  Deviation from ΛCDM:")
            print(f"    Mean ΔA: {mean_delta_A:.6f}")
            print(f"    Std: {std_delta_A:.6f}")
            print(f"    Significance: {deviation:.2f}σ")
            
            # Test 3: Gradient detection
            if len(z_vals) > 2:
                # Fit linear trend in log(z)
                log_z = np.log10(z_vals[z_vals > 0])
                delta_A_nonzero = delta_A_vals[z_vals > 0]
                
                if len(log_z) > 1:
                    coeffs = np.polyfit(log_z, delta_A_nonzero, 1)
                    gradient = coeffs[0]
                    
                    print(f"\n  Depth gradient:")
                    print(f"    d(ΔA)/d(log z): {gradient:.6f}")
                    
                    if abs(gradient) > 0.01:
                        print(f"    → Depth-dependent response detected")
                    else:
                        print(f"    → Response independent of depth")
            
            results[model] = {
                'rho': float(rho),
                'p_value': float(p_value),
                'mean_delta_A': float(mean_delta_A),
                'std_delta_A': float(std_delta_A),
                'deviation_sigma': float(deviation)
            }
        
        return results
        
    def test_critical_transition(self):
        """Look for critical redshift where response changes rapidly."""
        print("\n" + "="*60)
        print("CRITICAL TRANSITION DIAGNOSTIC")
        print("="*60)
        
        for model in ['V9_018', 'V9_050']:
            print(f"\n{model}:")
            
            # Extract valid data
            z_vals = []
            delta_A_vals = []
            
            for resp in self.normalized_responses[model]:
                if not np.isnan(resp['delta_A']):
                    z_vals.append(resp['z'])
                    delta_A_vals.append(resp['delta_A'])
            
            z_vals = np.array(z_vals)
            delta_A_vals = np.array(delta_A_vals)
            
            if len(z_vals) < 4:
                print("  Insufficient data for curvature analysis")
                continue
            
            # Compute second derivative (curvature)
            dz = np.diff(np.log10(z_vals[z_vals > 0]))
            dA = np.diff(delta_A_vals[z_vals > 0])
            
            if len(dA) > 1:
                gradient = dA / dz
                d2A = np.diff(gradient) / dz[:-1]
                z_curvature = z_vals[z_vals > 0][1:-1]
                
                # Find maximum curvature
                max_curv_idx = np.argmax(np.abs(d2A))
                z_critical = z_curvature[max_curv_idx]
                max_curvature = d2A[max_curv_idx]
                
                print(f"  Max curvature: {abs(max_curvature):.6f}")
                print(f"  At z ≈ {z_critical:.2f}")
                
                if abs(max_curvature) > 0.01:
                    print(f"  → Possible critical transition at z ~ {z_critical:.1f}")
                else:
                    print(f"  → Smooth response (no transition)")
        
    def create_plots(self):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Response amplitudes vs z
        ax = axes[0, 0]
        for model, label, color in [('LCDM', 'ΛCDM', 'black'),
                                     ('V9_018', 'V9 (n=0.18)', 'blue'),
                                     ('V9_050', 'V9 (n=0.5)', 'red')]:
            z_vals = [r['z'] for r in self.responses[model] if not np.isnan(r['response_amplitude'])]
            A_vals = [r['response_amplitude'] for r in self.responses[model] if not np.isnan(r['response_amplitude'])]
            ax.plot(z_vals, A_vals, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Response Amplitude A(z)', fontsize=12)
        ax.set_title('Perturbation Response vs Depth', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Normalized responses ΔA(z)
        ax = axes[0, 1]
        for model, label, color in [('V9_018', 'V9 (n=0.18)', 'blue'),
                                     ('V9_050', 'V9 (n=0.5)', 'red')]:
            z_vals = [r['z'] for r in self.normalized_responses[model] if not np.isnan(r['delta_A'])]
            delta_A_vals = [r['delta_A'] for r in self.normalized_responses[model] if not np.isnan(r['delta_A'])]
            ax.plot(z_vals, delta_A_vals, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Normalized Response ΔA(z)', fontsize=12)
        ax.set_title('Response Relative to ΛCDM', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Fractional deviation
        ax = axes[1, 0]
        for model, label, color in [('V9_018', 'V9 (n=0.18)', 'blue'),
                                     ('V9_050', 'V9 (n=0.5)', 'red')]:
            z_vals = [r['z'] for r in self.normalized_responses[model] if not np.isnan(r['delta_A'])]
            frac_dev = [(r['delta_A'] - 1.0) * 100 for r in self.normalized_responses[model] if not np.isnan(r['delta_A'])]
            ax.plot(z_vals, frac_dev, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Fractional Deviation [%]', fontsize=12)
        ax.set_title('(ΔA - 1) × 100%', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Growth rate comparison
        ax = axes[1, 1]
        for model, label, color in [('LCDM', 'ΛCDM', 'black'),
                                     ('V9_018', 'V9 (n=0.18)', 'blue'),
                                     ('V9_050', 'V9 (n=0.5)', 'red')]:
            z_vals = [r['z'] for r in self.responses[model] if not np.isnan(r['growth_rate'])]
            f_vals = [r['growth_rate'] for r in self.responses[model] if not np.isnan(r['growth_rate'])]
            ax.plot(z_vals, f_vals, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Growth Rate f(z)', fontsize=12)
        ax.set_title('Structure Growth Rate', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / 'perturbation_response.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {plot_path}")
        plt.close()
        
    def run_full_analysis(self):
        """Execute Phase 40 analysis."""
        # Compute responses
        self.compute_all_responses()
        
        # Normalize
        self.compute_normalized_responses()
        
        # Test stratification
        strat_results = self.test_depth_stratification()
        
        # Test for critical transition
        self.test_critical_transition()
        
        # Create plots
        self.create_plots()
        
        # Save results
        output = {
            'responses': {
                model: [
                    {k: float(v) if not isinstance(v, str) else v 
                     for k, v in resp.items()}
                    for resp in resps
                ]
                for model, resps in self.responses.items()
            },
            'normalized_responses': {
                model: [
                    {k: float(v) if not isinstance(v, str) else v 
                     for k, v in resp.items()}
                    for resp in resps
                ]
                for model, resps in self.normalized_responses.items()
            },
            'stratification_tests': strat_results
        }
        
        with open(OUTPUT_DIR / 'phase40_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PHASE 40 COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to {OUTPUT_DIR}")


def main():
    """Run Phase 40 analysis."""
    test = PerturbationResponseTest()
    test.run_full_analysis()


if __name__ == "__main__":
    main()
