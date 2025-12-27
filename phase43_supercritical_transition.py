#!/usr/bin/env python3
"""
Phase 43: Supercritical Transition & Vacuum Phase Structure Test

Tests whether Φ exhibits non-linear phase structure consistent with
supercritical transition (solid → liquid → supercritical vacuum).

Key tests:
1. Curvature detection: F''(Φ) ≠ 0
2. Inflection point: F''(Φ_c) = 0, F'''(Φ_c) ≠ 0
3. Effective potential reconstruction: V_eff(Φ) = -∫F(Φ)dΦ
4. Equation of state mapping: κ(Φ) = d ln F / d ln Φ

This is theory-internal - no new observational data required.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from pathlib import Path
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'class_v9_bec/python'))

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'phase43_results'
OUTPUT_DIR.mkdir(exist_ok=True)


class SupercriticalTransitionTest:
    """Phase 43: Test for vacuum phase structure in Φ(z)."""
    
    def __init__(self):
        # Load F(Φ) from Phase 41
        self.load_response_function()
        
        self.curvature_results = {}
        self.inflection_results = {}
        self.potential_results = {}
        
    def load_response_function(self):
        """Load F(Φ) = ΔA(z) from Phase 41."""
        print("\n" + "="*60)
        print("LOADING RESPONSE FUNCTION F(Φ) FROM PHASE 41")
        print("="*60)
        
        phase41_file = Path(__file__).parent / 'phase41_results' / 'phase41_results.json'
        
        if phase41_file.exists():
            with open(phase41_file, 'r') as f:
                phase41_data = json.load(f)
            
            # Extract V9 n=0.5 normalized responses
            v9_responses = phase41_data['normalized_responses']['V9_050']['0.05']
            
            z_vals = np.array([r['z'] for r in v9_responses if not np.isnan(r['delta_A'])])
            delta_A_vals = np.array([r['delta_A'] for r in v9_responses if not np.isnan(r['delta_A'])])
            
            # Φ(z) is defined by fractional deviation
            self.phi = delta_A_vals - 1.0  # Φ = ΔA - 1
            self.F_phi = delta_A_vals  # F(Φ) = ΔA
            self.z_vals = z_vals
            
            # Sort by Φ
            sort_idx = np.argsort(self.phi)
            self.phi = self.phi[sort_idx]
            self.F_phi = self.F_phi[sort_idx]
            self.z_vals = self.z_vals[sort_idx]
            
            print(f"\nLoaded F(Φ) from Phase 41:")
            print(f"  Φ range: [{self.phi[0]:.6f}, {self.phi[-1]:.6f}]")
            print(f"  F(Φ) range: [{self.F_phi[0]:.6f}, {self.F_phi[-1]:.6f}]")
            print(f"  Number of points: {len(self.phi)}")
            
            # Also load ΛCDM for control
            lcdm_responses = phase41_data['normalized_responses']['V9_018']['0.05']
            lcdm_delta_A = np.array([r['delta_A'] for r in lcdm_responses if not np.isnan(r['delta_A'])])
            self.F_phi_lcdm = lcdm_delta_A[sort_idx]
            
        else:
            raise FileNotFoundError("Phase 41 results not found - run Phase 41 first")
    
    def compute_derivatives_bootstrap(self, n_bootstrap=1000):
        """
        Compute F'(Φ) and F''(Φ) with bootstrap error estimates.
        
        Uses Savitzky-Golay filter for smoothing.
        """
        print("\n" + "="*60)
        print("COMPUTING DERIVATIVES WITH BOOTSTRAP")
        print("="*60)
        
        # Smooth F(Φ) using Savitzky-Golay filter
        # Window must be odd and >= polynomial order + 2
        window_length = min(7, len(self.phi) if len(self.phi) % 2 == 1 else len(self.phi) - 1)
        polyorder = 3
        
        F_smooth = savgol_filter(self.F_phi, window_length, polyorder)
        
        # First derivative
        F_prime = savgol_filter(self.F_phi, window_length, polyorder, deriv=1, 
                                delta=(self.phi[1] - self.phi[0]))
        
        # Second derivative
        F_double_prime = savgol_filter(self.F_phi, window_length, polyorder, deriv=2,
                                       delta=(self.phi[1] - self.phi[0]))
        
        # Bootstrap for error estimates
        F_prime_samples = []
        F_double_prime_samples = []
        
        print(f"\nRunning {n_bootstrap} bootstrap iterations...")
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(self.phi), size=len(self.phi), replace=True)
            phi_boot = self.phi[indices]
            F_boot = self.F_phi[indices]
            
            # Sort
            sort_idx = np.argsort(phi_boot)
            phi_boot = phi_boot[sort_idx]
            F_boot = F_boot[sort_idx]
            
            # Compute derivatives
            try:
                F_prime_boot = savgol_filter(F_boot, window_length, polyorder, deriv=1,
                                            delta=(phi_boot[1] - phi_boot[0]))
                F_double_prime_boot = savgol_filter(F_boot, window_length, polyorder, deriv=2,
                                                   delta=(phi_boot[1] - phi_boot[0]))
                
                F_prime_samples.append(F_prime_boot)
                F_double_prime_samples.append(F_double_prime_boot)
            except:
                continue
        
        # Compute confidence intervals
        F_prime_samples = np.array(F_prime_samples)
        F_double_prime_samples = np.array(F_double_prime_samples)
        
        F_prime_std = np.std(F_prime_samples, axis=0)
        F_double_prime_std = np.std(F_double_prime_samples, axis=0)
        
        print(f"\nDerivative statistics:")
        print(f"  F'(Φ) range: [{np.min(F_prime):.3f}, {np.max(F_prime):.3f}]")
        print(f"  F'(Φ) mean error: {np.mean(F_prime_std):.3f}")
        print(f"  F''(Φ) range: [{np.min(F_double_prime):.3f}, {np.max(F_double_prime):.3f}]")
        print(f"  F''(Φ) mean error: {np.mean(F_double_prime_std):.3f}")
        
        self.F_smooth = F_smooth
        self.F_prime = F_prime
        self.F_double_prime = F_double_prime
        self.F_prime_std = F_prime_std
        self.F_double_prime_std = F_double_prime_std
        
        return F_prime, F_double_prime, F_prime_std, F_double_prime_std
    
    def detect_curvature(self):
        """Test if F''(Φ) is significantly non-zero."""
        print("\n" + "="*60)
        print("CURVATURE DETECTION TEST")
        print("="*60)
        
        # Significance test: |F''| > 3σ
        significance = np.abs(self.F_double_prime) / self.F_double_prime_std
        
        # Find regions with significant curvature
        significant_mask = significance > 3.0
        
        if np.any(significant_mask):
            max_sig_idx = np.argmax(significance)
            max_curvature = self.F_double_prime[max_sig_idx]
            max_phi = self.phi[max_sig_idx]
            max_significance = significance[max_sig_idx]
            
            print(f"\n✓ Significant curvature detected:")
            print(f"  Maximum |F''(Φ)|: {abs(max_curvature):.6f}")
            print(f"  At Φ = {max_phi:.6f} (z ≈ {self.z_vals[max_sig_idx]:.1f})")
            print(f"  Significance: {max_significance:.1f}σ")
            
            curvature_detected = True
        else:
            print(f"\n✗ No significant curvature detected")
            print(f"  Maximum significance: {np.max(significance):.1f}σ")
            curvature_detected = False
        
        self.curvature_results = {
            'detected': bool(curvature_detected),
            'max_curvature': float(np.max(np.abs(self.F_double_prime))),
            'max_significance': float(np.max(significance)),
            'phi_at_max': float(self.phi[np.argmax(np.abs(self.F_double_prime))]),
            'z_at_max': float(self.z_vals[np.argmax(np.abs(self.F_double_prime))])
        }
        
        return curvature_detected
    
    def detect_inflection(self):
        """Find inflection point where F''(Φ_c) = 0 and F'''(Φ_c) ≠ 0."""
        print("\n" + "="*60)
        print("INFLECTION POINT DETECTION")
        print("="*60)
        
        # Find zero crossings of F''
        sign_changes = np.diff(np.sign(self.F_double_prime))
        zero_crossings = np.where(sign_changes != 0)[0]
        
        if len(zero_crossings) > 0:
            print(f"\nFound {len(zero_crossings)} potential inflection point(s):")
            
            inflections = []
            
            for idx in zero_crossings:
                phi_c = self.phi[idx]
                z_c = self.z_vals[idx]
                
                # Estimate F''' by finite difference
                if idx > 0 and idx < len(self.F_double_prime) - 1:
                    F_triple_prime = (self.F_double_prime[idx+1] - self.F_double_prime[idx-1]) / \
                                    (self.phi[idx+1] - self.phi[idx-1])
                else:
                    F_triple_prime = 0.0
                
                print(f"  Φ_c = {phi_c:.6f} (z ≈ {z_c:.1f})")
                print(f"    F''(Φ_c) ≈ {self.F_double_prime[idx]:.6f}")
                print(f"    F'''(Φ_c) ≈ {F_triple_prime:.3f}")
                
                inflections.append({
                    'phi_c': float(phi_c),
                    'z_c': float(z_c),
                    'F_triple_prime': float(F_triple_prime)
                })
            
            # Primary inflection (largest |F'''|)
            primary_idx = np.argmax([abs(inf['F_triple_prime']) for inf in inflections])
            primary = inflections[primary_idx]
            
            if abs(primary['F_triple_prime']) > 0.1:
                print(f"\n✓ Primary inflection at Φ_c = {primary['phi_c']:.6f}")
                print(f"  → Supercritical crossover signature")
                inflection_detected = True
            else:
                print(f"\n⚠ Weak inflection (F''' ≈ 0)")
                inflection_detected = False
            
            self.inflection_results = {
                'detected': bool(inflection_detected),
                'inflections': inflections,
                'primary': primary
            }
            
        else:
            print(f"\n✗ No inflection points detected")
            print(f"  F''(Φ) does not cross zero")
            inflection_detected = False
            
            self.inflection_results = {
                'detected': False,
                'inflections': []
            }
        
        return inflection_detected
    
    def reconstruct_effective_potential(self):
        """Reconstruct V_eff(Φ) = -∫F(Φ)dΦ."""
        print("\n" + "="*60)
        print("EFFECTIVE POTENTIAL RECONSTRUCTION")
        print("="*60)
        
        # Integrate -F(Φ)
        V_eff = -cumtrapz(self.F_smooth, self.phi, initial=0)
        
        # Normalize to V(Φ=0) = 0
        V_eff -= V_eff[0]
        
        # Analyze structure
        n_minima = len([i for i in range(1, len(V_eff)-1) 
                       if V_eff[i] < V_eff[i-1] and V_eff[i] < V_eff[i+1]])
        
        n_maxima = len([i for i in range(1, len(V_eff)-1)
                       if V_eff[i] > V_eff[i-1] and V_eff[i] > V_eff[i+1]])
        
        # Check for flat regions (supercritical)
        dV_dPhi = np.gradient(V_eff, self.phi)
        flat_mask = np.abs(dV_dPhi) < 0.01
        flat_fraction = np.sum(flat_mask) / len(flat_mask)
        
        print(f"\nPotential structure:")
        print(f"  V_eff range: [{np.min(V_eff):.6f}, {np.max(V_eff):.6f}]")
        print(f"  Number of minima: {n_minima}")
        print(f"  Number of maxima: {n_maxima}")
        print(f"  Flat region fraction: {flat_fraction*100:.1f}%")
        
        # Classify
        if n_minima == 0 and flat_fraction > 0.5:
            structure = "SUPERCRITICAL_FLAT"
            interpretation = "Broad flat potential - supercritical fluid"
        elif n_minima == 1 and n_maxima == 0:
            structure = "SINGLE_MINIMUM"
            interpretation = "Stable single-phase vacuum"
        elif n_minima == 2:
            structure = "DOUBLE_WELL"
            interpretation = "Mixed-phase structure"
        else:
            structure = "MONOTONIC"
            interpretation = "Simple stratification"
        
        print(f"\n  Structure: {structure}")
        print(f"  → {interpretation}")
        
        self.V_eff = V_eff
        self.potential_results = {
            'structure': structure,
            'interpretation': interpretation,
            'n_minima': int(n_minima),
            'n_maxima': int(n_maxima),
            'flat_fraction': float(flat_fraction)
        }
        
        return structure
    
    def compute_susceptibility_exponent(self):
        """Compute κ(Φ) = d ln F / d ln Φ."""
        print("\n" + "="*60)
        print("SUSCEPTIBILITY EXPONENT κ(Φ)")
        print("="*60)
        
        # Avoid log(0) by using Φ + offset
        phi_safe = self.phi - self.phi[0] + 1e-6
        
        # κ = d ln F / d ln Φ
        log_F = np.log(self.F_smooth)
        log_phi = np.log(phi_safe)
        
        kappa = np.gradient(log_F, log_phi)
        
        print(f"\nSusceptibility exponent:")
        print(f"  κ(Φ) range: [{np.min(kappa):.3f}, {np.max(kappa):.3f}]")
        print(f"  Mean κ: {np.mean(kappa):.3f}")
        
        # Check for plateau (incompressible → compressible)
        kappa_variation = np.std(kappa)
        
        if kappa_variation < 0.1:
            print(f"  → Constant susceptibility (single phase)")
        else:
            print(f"  → Varying susceptibility (phase transition)")
        
        self.kappa = kappa
        
        return kappa
    
    def run_lcdm_control(self):
        """Control test: ΛCDM should show flat F(Φ) = 1."""
        print("\n" + "="*60)
        print("ΛCDM CONTROL TEST")
        print("="*60)
        
        # For ΛCDM, F(Φ) should be constant ≈ 1
        F_lcdm_var = np.var(self.F_phi_lcdm)
        
        print(f"\nΛCDM F(Φ) statistics:")
        print(f"  Mean: {np.mean(self.F_phi_lcdm):.6f}")
        print(f"  Variance: {F_lcdm_var:.9f}")
        
        if F_lcdm_var < 1e-6:
            print(f"  ✓ ΛCDM shows flat response (control passed)")
            control_passed = True
        else:
            print(f"  ✗ ΛCDM shows variation (unexpected)")
            control_passed = False
        
        return control_passed
    
    def run_randomized_null(self):
        """Null test: randomize Φ assignments."""
        print("\n" + "="*60)
        print("RANDOMIZED Φ NULL TEST")
        print("="*60)
        
        # Shuffle Φ values
        phi_shuffled = np.random.permutation(self.phi)
        
        # Recompute derivatives
        sort_idx = np.argsort(phi_shuffled)
        phi_shuffled = phi_shuffled[sort_idx]
        F_shuffled = self.F_phi[sort_idx]
        
        window_length = min(7, len(phi_shuffled) if len(phi_shuffled) % 2 == 1 else len(phi_shuffled) - 1)
        polyorder = 3
        
        F_double_prime_null = savgol_filter(F_shuffled, window_length, polyorder, deriv=2,
                                           delta=(phi_shuffled[1] - phi_shuffled[0]))
        
        max_curvature_null = np.max(np.abs(F_double_prime_null))
        max_curvature_real = np.max(np.abs(self.F_double_prime))
        
        print(f"\nCurvature comparison:")
        print(f"  Real data: {max_curvature_real:.6f}")
        print(f"  Randomized: {max_curvature_null:.6f}")
        print(f"  Ratio: {max_curvature_real / max_curvature_null:.2f}x")
        
        if max_curvature_real > 2 * max_curvature_null:
            print(f"  ✓ Real signal exceeds random noise")
            null_passed = True
        else:
            print(f"  ✗ Signal consistent with noise")
            null_passed = False
        
        return null_passed
    
    def create_plots(self):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: F(Φ) response function
        ax = axes[0, 0]
        ax.plot(self.phi, self.F_phi, 'o', color='red', markersize=6, label='V9 BEC Data')
        ax.plot(self.phi, self.F_smooth, '-', color='red', linewidth=2, label='Smoothed')
        ax.axhline(1.0, color='blue', linestyle='--', alpha=0.5, label='ΛCDM')
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel('Response Function F(Φ)', fontsize=12)
        ax.set_title('Vacuum Response Function', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: First derivative F'(Φ)
        ax = axes[0, 1]
        ax.plot(self.phi, self.F_prime, '-', color='green', linewidth=2)
        ax.fill_between(self.phi, 
                        self.F_prime - 2*self.F_prime_std,
                        self.F_prime + 2*self.F_prime_std,
                        alpha=0.3, color='green', label='2σ confidence')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel("F'(Φ)", fontsize=12)
        ax.set_title('First Derivative (Compressibility)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Second derivative F''(Φ) - curvature
        ax = axes[0, 2]
        ax.plot(self.phi, self.F_double_prime, '-', color='purple', linewidth=2)
        ax.fill_between(self.phi,
                        self.F_double_prime - 2*self.F_double_prime_std,
                        self.F_double_prime + 2*self.F_double_prime_std,
                        alpha=0.3, color='purple', label='2σ confidence')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Mark inflection points
        if self.inflection_results.get('detected', False):
            for inf in self.inflection_results['inflections']:
                ax.axvline(inf['phi_c'], color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel("F''(Φ)", fontsize=12)
        ax.set_title('Curvature (Phase Transition Signature)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Effective potential V_eff(Φ)
        ax = axes[1, 0]
        ax.plot(self.phi, self.V_eff, '-', color='darkblue', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel('V_eff(Φ)', fontsize=12)
        ax.set_title(f'Effective Potential: {self.potential_results["structure"]}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Susceptibility exponent κ(Φ)
        ax = axes[1, 1]
        ax.plot(self.phi, self.kappa, '-', color='orange', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel('κ(Φ) = d ln F / d ln Φ', fontsize=12)
        ax.set_title('Susceptibility Exponent', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Phase diagram
        ax = axes[1, 2]
        
        # Color-code by curvature magnitude
        significance = np.abs(self.F_double_prime) / self.F_double_prime_std
        scatter = ax.scatter(self.phi, self.F_phi, c=significance, 
                           cmap='RdYlBu_r', s=100, edgecolors='black', linewidth=1)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Curvature Significance (σ)', fontsize=10)
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel('Response F(Φ)', fontsize=12)
        ax.set_title('Phase Structure Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / 'supercritical_transition.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {plot_path}")
        plt.close()
    
    def run_full_analysis(self):
        """Execute Phase 43 analysis."""
        print("\n" + "="*60)
        print("PHASE 43: SUPERCRITICAL TRANSITION TEST")
        print("="*60)
        
        # Compute derivatives
        self.compute_derivatives_bootstrap()
        
        # Test 1: Curvature
        curvature_detected = self.detect_curvature()
        
        # Test 2: Inflection
        inflection_detected = self.detect_inflection()
        
        # Test 3: Effective potential
        structure = self.reconstruct_effective_potential()
        
        # Test 4: Susceptibility
        self.compute_susceptibility_exponent()
        
        # Control tests
        lcdm_control = self.run_lcdm_control()
        null_test = self.run_randomized_null()
        
        # Create plots
        self.create_plots()
        
        # Decision logic
        print("\n" + "="*60)
        print("DECISIVE OUTCOME")
        print("="*60)
        
        if inflection_detected and structure in ["SUPERCRITICAL_FLAT", "DOUBLE_WELL"]:
            decision = "SUPERCRITICAL_TRANSITION"
            interpretation = "Φ exhibits phase transition structure - vacuum is stratified condensate"
        elif curvature_detected and structure == "SINGLE_MINIMUM":
            decision = "STRATIFIED_SINGLE_PHASE"
            interpretation = "Φ shows curvature but remains single-phase - smooth stratification"
        elif curvature_detected:
            decision = "CURVATURE_ONLY"
            interpretation = "Non-linear Φ-structure detected but phase unclear"
        else:
            decision = "LINEAR_STRATIFICATION"
            interpretation = "Φ is monotonic coordinate without phase structure"
        
        print(f"\nDecision: {decision}")
        print(f"Interpretation: {interpretation}")
        print(f"\nControl tests:")
        print(f"  ΛCDM control: {'PASS' if lcdm_control else 'FAIL'}")
        print(f"  Randomization null: {'PASS' if null_test else 'FAIL'}")
        
        # Save results
        output = {
            'decision': decision,
            'interpretation': interpretation,
            'curvature': self.curvature_results,
            'inflection': self.inflection_results,
            'potential': self.potential_results,
            'controls': {
                'lcdm_passed': bool(lcdm_control),
                'null_passed': bool(null_test)
            }
        }
        
        with open(OUTPUT_DIR / 'phase43_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        # Summary text file
        with open(OUTPUT_DIR / 'phase43_summary.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("PHASE 43: SUPERCRITICAL TRANSITION TEST - SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Decision: {decision}\n")
            f.write(f"Interpretation: {interpretation}\n\n")
            f.write(f"Curvature detected: {curvature_detected}\n")
            f.write(f"Inflection detected: {inflection_detected}\n")
            f.write(f"Potential structure: {structure}\n\n")
            f.write(f"ΛCDM control: {'PASS' if lcdm_control else 'FAIL'}\n")
            f.write(f"Randomization null: {'PASS' if null_test else 'FAIL'}\n")
        
        print(f"\n{'='*60}")
        print("PHASE 43 COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to {OUTPUT_DIR}")


def main():
    """Run Phase 43 analysis."""
    test = SupercriticalTransitionTest()
    test.run_full_analysis()


if __name__ == "__main__":
    main()
