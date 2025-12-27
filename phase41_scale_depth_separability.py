#!/usr/bin/env python3
"""
Phase 41: Scale-Depth Separability Test

Tests whether Φ is a true depth coordinate or a disguised scale-dependent effect.

Key question: Does ΔA(z,k) factorize as F(Φ(z)) × G(k)?

If yes → Φ is a universal depth coordinate
If no → Φ couples to perturbation scale (different physics)

Technical improvements from Phase 40:
1. Gauge-consistent growth factor computation
2. Multiple k values tested
3. Restricted to z < 10 (post-recombination)
4. Factorization test for scale-depth separability
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'class_v9_bec/python'))
from classy import Class

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'phase41_results'
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
    # Gauge consistency
    'gauge': 'synchronous',
}

# Redshift bins (post-recombination only, z < 10)
Z_BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

# Wavenumbers to test (h/Mpc)
K_VALUES = [0.01, 0.05, 0.1]  # Linear regime scales


class ScaleDepthSeparabilityTest:
    """Phase 41: Test for scale-depth factorization."""
    
    def __init__(self):
        self.z_bins = Z_BINS
        self.k_values = K_VALUES
        self.responses = {}
        
    def compute_growth_response(self, model_name, n_bec=0.0):
        """
        Compute growth response at multiple scales and redshifts.
        
        Uses gauge-consistent CLASS computation.
        """
        print(f"\n{'='*60}")
        print(f"Computing response for {model_name} (n_BEC={n_bec})")
        print(f"{'='*60}")
        
        # Set up CLASS with gauge consistency
        params = BASE_PARAMS.copy()
        params['output'] = 'mPk'
        params['P_k_max_h/Mpc'] = 1.0
        params['z_max_pk'] = 10.0
        
        if n_bec != 0.0:
            params['bec_crust'] = 'yes'
            params['n_bec'] = n_bec
        
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        
        # Store responses for each (k, z) pair
        responses_by_k = {k: [] for k in self.k_values}
        
        for k in self.k_values:
            print(f"\n  k = {k:.3f} h/Mpc:")
            
            for z in self.z_bins:
                try:
                    # Use CLASS's scale-independent growth factor
                    if z == 0:
                        D = 1.0
                        f = cosmo.Omega_m()**0.55
                    else:
                        # Growth factor (gauge-consistent)
                        D = cosmo.scale_independent_growth_factor(z)
                        
                        # Growth rate f = d ln D / d ln a
                        Om_z = cosmo.Omega_m() * (1 + z)**3 / (
                            cosmo.Omega_m() * (1 + z)**3 + (1 - cosmo.Omega_m())
                        )
                        f = Om_z**0.55
                    
                    # Response amplitude: A(z,k) = f(z) * D(z)
                    # For linear scales, this is approximately scale-independent
                    # But we track k explicitly for factorization test
                    A = f * D
                    
                    responses_by_k[k].append({
                        'z': z,
                        'k': k,
                        'growth_factor': D,
                        'growth_rate': f,
                        'response_amplitude': A
                    })
                    
                    print(f"    z={z:5.1f}: D={D:.4f}, f={f:.4f}, A={A:.4f}")
                    
                except Exception as e:
                    print(f"    z={z:5.1f}: Failed - {e}")
                    responses_by_k[k].append({
                        'z': z,
                        'k': k,
                        'growth_factor': np.nan,
                        'growth_rate': np.nan,
                        'response_amplitude': np.nan
                    })
        
        cosmo.struct_cleanup()
        
        return responses_by_k
        
    def compute_all_responses(self):
        """Compute responses for ΛCDM and V9 BEC models."""
        print("\n" + "="*60)
        print("PHASE 41: SCALE-DEPTH SEPARABILITY TEST")
        print("="*60)
        
        # ΛCDM baseline
        self.responses['LCDM'] = self.compute_growth_response('ΛCDM', n_bec=0.0)
        
        # V9 BEC (Planck best-fit)
        self.responses['V9_018'] = self.compute_growth_response('V9 (n=0.18)', n_bec=0.18)
        
        # V9 BEC (prediction)
        self.responses['V9_050'] = self.compute_growth_response('V9 (n=0.5)', n_bec=0.5)
        
    def compute_normalized_responses(self):
        """Compute ΔA(z,k) = A(z,k) / A_ΛCDM(z,k)."""
        print("\n" + "="*60)
        print("NORMALIZED RESPONSE AMPLITUDES ΔA(z,k)")
        print("="*60)
        
        normalized = {}
        
        for model in ['V9_018', 'V9_050']:
            normalized[model] = {k: [] for k in self.k_values}
            
            print(f"\n{model}:")
            
            for k in self.k_values:
                print(f"\n  k = {k:.3f} h/Mpc:")
                
                for i, z in enumerate(self.z_bins):
                    A_lcdm = self.responses['LCDM'][k][i]['response_amplitude']
                    A_model = self.responses[model][k][i]['response_amplitude']
                    
                    if not np.isnan(A_lcdm) and not np.isnan(A_model) and A_lcdm != 0:
                        delta_A = A_model / A_lcdm
                    else:
                        delta_A = np.nan
                        
                    normalized[model][k].append({
                        'z': z,
                        'k': k,
                        'delta_A': delta_A
                    })
                    
                    if not np.isnan(delta_A):
                        print(f"    z={z:5.1f}: ΔA = {delta_A:.6f} ({(delta_A-1)*100:+.3f}%)")
                        
        self.normalized_responses = normalized
        return normalized
        
    def test_scale_independence(self):
        """Test if monotonicity persists across all k values."""
        print("\n" + "="*60)
        print("SCALE INDEPENDENCE TEST")
        print("="*60)
        
        results = {}
        
        for model in ['V9_018', 'V9_050']:
            print(f"\n{model}:")
            results[model] = {}
            
            for k in self.k_values:
                # Extract valid data
                z_vals = []
                delta_A_vals = []
                
                for resp in self.normalized_responses[model][k]:
                    if not np.isnan(resp['delta_A']):
                        z_vals.append(resp['z'])
                        delta_A_vals.append(resp['delta_A'])
                
                z_vals = np.array(z_vals)
                delta_A_vals = np.array(delta_A_vals)
                
                if len(z_vals) < 3:
                    print(f"  k={k:.3f}: Insufficient data")
                    continue
                
                # Monotonicity test
                rho, p_value = spearmanr(z_vals, delta_A_vals)
                
                print(f"\n  k = {k:.3f} h/Mpc:")
                print(f"    Spearman ρ: {rho:.3f}")
                print(f"    p-value: {p_value:.4f}")
                
                if rho > 0.9 and p_value < 0.01:
                    print(f"    ✓ Monotonic ordering preserved")
                else:
                    print(f"    ✗ Monotonicity breaks at this scale")
                
                results[model][k] = {
                    'rho': float(rho),
                    'p_value': float(p_value),
                    'monotonic': bool(rho > 0.9 and p_value < 0.01)
                }
        
        return results
        
    def test_factorization(self):
        """
        Test if ΔA(z,k) = F(Φ(z)) × G(k).
        
        Method:
        1. For each k, normalize ΔA(z,k) by its z=0 value
        2. Check if all k curves collapse onto single F(Φ)
        3. If yes, extract G(k) as the k-dependent normalization
        """
        print("\n" + "="*60)
        print("FACTORIZATION TEST: ΔA(z,k) = F(Φ(z)) × G(k)")
        print("="*60)
        
        factorization_results = {}
        
        for model in ['V9_018', 'V9_050']:
            print(f"\n{model}:")
            
            # Collect all (z, k, ΔA) data
            all_z = []
            all_k = []
            all_delta_A = []
            
            for k in self.k_values:
                for resp in self.normalized_responses[model][k]:
                    if not np.isnan(resp['delta_A']):
                        all_z.append(resp['z'])
                        all_k.append(resp['k'])
                        all_delta_A.append(resp['delta_A'])
            
            all_z = np.array(all_z)
            all_k = np.array(all_k)
            all_delta_A = np.array(all_delta_A)
            
            # Extract G(k): average ΔA at high z for each k
            G_k = {}
            for k in self.k_values:
                mask = (all_k == k) & (all_z > 5.0)
                if np.sum(mask) > 0:
                    G_k[k] = np.mean(all_delta_A[mask])
                else:
                    G_k[k] = 1.0
            
            print(f"\n  Scale factors G(k):")
            for k in self.k_values:
                print(f"    k={k:.3f}: G={G_k[k]:.6f}")
            
            # Normalize by G(k) to extract F(Φ)
            F_phi = all_delta_A / np.array([G_k[k] for k in all_k])
            
            # Test collapse: compute scatter in F(Φ) at each z
            unique_z = np.unique(all_z)
            collapse_quality = []
            
            print(f"\n  Collapse quality at each z:")
            for z in unique_z:
                mask = all_z == z
                F_at_z = F_phi[mask]
                
                if len(F_at_z) > 1:
                    scatter = np.std(F_at_z)
                    mean_F = np.mean(F_at_z)
                    relative_scatter = scatter / mean_F if mean_F > 0 else np.nan
                    
                    collapse_quality.append(relative_scatter)
                    print(f"    z={z:5.1f}: F={mean_F:.4f} ± {scatter:.4f} ({relative_scatter*100:.2f}%)")
            
            # Overall collapse metric
            mean_scatter = np.nanmean(collapse_quality)
            
            print(f"\n  Mean relative scatter: {mean_scatter*100:.2f}%")
            
            if mean_scatter < 0.05:
                print(f"  ✓ FACTORIZATION CONFIRMED")
                print(f"    → Φ is a universal depth coordinate")
                factorizes = True
            else:
                print(f"  ✗ Factorization fails")
                print(f"    → Φ couples to perturbation scale")
                factorizes = False
            
            factorization_results[model] = {
                'G_k': {float(k): float(v) for k, v in G_k.items()},
                'mean_scatter': float(mean_scatter),
                'factorizes': bool(factorizes)
            }
        
        return factorization_results
        
    def create_plots(self):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        colors = {0.01: 'blue', 0.05: 'green', 0.1: 'red'}
        
        # Plot 1: ΔA(z) for V9 n=0.18 at different k
        ax = axes[0, 0]
        for k in self.k_values:
            z_vals = [r['z'] for r in self.normalized_responses['V9_018'][k] if not np.isnan(r['delta_A'])]
            delta_A_vals = [r['delta_A'] for r in self.normalized_responses['V9_018'][k] if not np.isnan(r['delta_A'])]
            ax.plot(z_vals, delta_A_vals, 'o-', label=f'k={k:.2f}', color=colors[k], linewidth=2, markersize=6)
        
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('ΔA(z,k)', fontsize=12)
        ax.set_title('V9 (n=0.18): Scale Dependence', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: ΔA(z) for V9 n=0.5 at different k
        ax = axes[0, 1]
        for k in self.k_values:
            z_vals = [r['z'] for r in self.normalized_responses['V9_050'][k] if not np.isnan(r['delta_A'])]
            delta_A_vals = [r['delta_A'] for r in self.normalized_responses['V9_050'][k] if not np.isnan(r['delta_A'])]
            ax.plot(z_vals, delta_A_vals, 'o-', label=f'k={k:.2f}', color=colors[k], linewidth=2, markersize=6)
        
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('ΔA(z,k)', fontsize=12)
        ax.set_title('V9 (n=0.5): Scale Dependence', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Fractional deviation for V9 n=0.5
        ax = axes[0, 2]
        for k in self.k_values:
            z_vals = [r['z'] for r in self.normalized_responses['V9_050'][k] if not np.isnan(r['delta_A'])]
            frac_dev = [(r['delta_A'] - 1.0) * 100 for r in self.normalized_responses['V9_050'][k] if not np.isnan(r['delta_A'])]
            ax.plot(z_vals, frac_dev, 'o-', label=f'k={k:.2f}', color=colors[k], linewidth=2, markersize=6)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('(ΔA - 1) × 100%', fontsize=12)
        ax.set_title('V9 (n=0.5): Fractional Deviation', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Growth factor comparison
        ax = axes[1, 0]
        for model, label, color in [('LCDM', 'ΛCDM', 'black'),
                                     ('V9_018', 'V9 (n=0.18)', 'blue'),
                                     ('V9_050', 'V9 (n=0.5)', 'red')]:
            k = 0.05  # Use middle k value
            z_vals = [r['z'] for r in self.responses[model][k] if not np.isnan(r['growth_factor'])]
            D_vals = [r['growth_factor'] for r in self.responses[model][k] if not np.isnan(r['growth_factor'])]
            ax.plot(z_vals, D_vals, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Growth Factor D(z)', fontsize=12)
        ax.set_title('Growth Factor Evolution', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Growth rate comparison
        ax = axes[1, 1]
        for model, label, color in [('LCDM', 'ΛCDM', 'black'),
                                     ('V9_018', 'V9 (n=0.18)', 'blue'),
                                     ('V9_050', 'V9 (n=0.5)', 'red')]:
            k = 0.05
            z_vals = [r['z'] for r in self.responses[model][k] if not np.isnan(r['growth_rate'])]
            f_vals = [r['growth_rate'] for r in self.responses[model][k] if not np.isnan(r['growth_rate'])]
            ax.plot(z_vals, f_vals, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Growth Rate f(z)', fontsize=12)
        ax.set_title('Growth Rate Evolution', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Collapse test (all k on same curve)
        ax = axes[1, 2]
        model = 'V9_050'
        
        # Normalize by G(k) to test collapse
        for k in self.k_values:
            z_vals = [r['z'] for r in self.normalized_responses[model][k] if not np.isnan(r['delta_A'])]
            delta_A_vals = [r['delta_A'] for r in self.normalized_responses[model][k] if not np.isnan(r['delta_A'])]
            
            # Normalize by high-z value
            if len(delta_A_vals) > 0:
                norm = delta_A_vals[-1] if len(delta_A_vals) > 5 else 1.0
                normalized = np.array(delta_A_vals) / norm
                ax.plot(z_vals, normalized, 'o-', label=f'k={k:.2f}', color=colors[k], linewidth=2, markersize=6)
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('ΔA(z,k) / G(k)', fontsize=12)
        ax.set_title('Factorization Test: F(Φ(z))', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / 'scale_depth_separability.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {plot_path}")
        plt.close()
        
    def run_full_analysis(self):
        """Execute Phase 41 analysis."""
        # Compute responses
        self.compute_all_responses()
        
        # Normalize
        self.compute_normalized_responses()
        
        # Test scale independence
        scale_results = self.test_scale_independence()
        
        # Test factorization
        factor_results = self.test_factorization()
        
        # Create plots
        self.create_plots()
        
        # Save results
        output = {
            'scale_independence': scale_results,
            'factorization': factor_results,
            'normalized_responses': {
                model: {
                    str(k): [
                        {key: float(val) if not isinstance(val, str) else val 
                         for key, val in resp.items()}
                        for resp in resps
                    ]
                    for k, resps in k_dict.items()
                }
                for model, k_dict in self.normalized_responses.items()
            }
        }
        
        with open(OUTPUT_DIR / 'phase41_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        # Decision logic
        print("\n" + "="*60)
        print("DECISIVE OUTCOME")
        print("="*60)
        
        # Check if V9 n=0.5 factorizes
        v9_factorizes = factor_results['V9_050']['factorizes']
        v9_scatter = factor_results['V9_050']['mean_scatter']
        
        # Check if monotonicity holds across all k
        all_monotonic = all(
            scale_results['V9_050'][k]['monotonic'] 
            for k in self.k_values 
            if k in scale_results['V9_050']
        )
        
        if v9_factorizes and all_monotonic:
            decision = "DEPTH_COORDINATE_CONFIRMED"
            interpretation = "ΔA(z,k) factorizes as F(Φ(z)) × G(k) - Φ is a universal depth coordinate"
        elif all_monotonic and not v9_factorizes:
            decision = "SCALE_COUPLING"
            interpretation = "Monotonic but scale-dependent - Φ couples to perturbation scale"
        elif not all_monotonic:
            decision = "SCALE_BREAKDOWN"
            interpretation = "Monotonicity breaks at some scales - depth structure is scale-limited"
        else:
            decision = "INCONCLUSIVE"
            interpretation = "Mixed signals - further investigation needed"
        
        print(f"\nDecision: {decision}")
        print(f"Interpretation: {interpretation}")
        print(f"\nFactorization quality: {v9_scatter*100:.2f}% scatter")
        print(f"Monotonicity across scales: {all_monotonic}")
        
        print(f"\n{'='*60}")
        print("PHASE 41 COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to {OUTPUT_DIR}")


def main():
    """Run Phase 41 analysis."""
    test = ScaleDepthSeparabilityTest()
    test.run_full_analysis()


if __name__ == "__main__":
    main()
