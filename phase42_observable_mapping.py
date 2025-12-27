#!/usr/bin/env python3
"""
Phase 42: Observable Mapping & Physical Calibration of Φ(z)

Connects the universal depth coordinate Φ(z) (from Phase 41) to real observables.

Tests whether Φ(z) consistently explains:
- fσ8(z) from RSD measurements
- Weak lensing amplitude
- Matter power spectrum normalization

Key constraint: Φ(z) is FIXED from Phase 41. No refitting allowed.

Success criterion: All observables collapse onto same F(Φ) function.
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
OUTPUT_DIR = Path(__file__).parent / 'phase42_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# OBSERVATIONAL DATA
# ============================================================================

# RSD fσ8 measurements (from Phase 39)
FSIGMA8_DATA = [
    {'z': 0.067, 'fsigma8': 0.423, 'error': 0.055, 'survey': '6dFGS'},
    {'z': 0.15, 'fsigma8': 0.490, 'error': 0.145, 'survey': 'SDSS MGS'},
    {'z': 0.38, 'fsigma8': 0.497, 'error': 0.045, 'survey': 'BOSS DR12'},
    {'z': 0.51, 'fsigma8': 0.458, 'error': 0.038, 'survey': 'BOSS DR12'},
    {'z': 0.61, 'fsigma8': 0.436, 'error': 0.034, 'survey': 'BOSS DR12'},
    {'z': 0.44, 'fsigma8': 0.413, 'error': 0.080, 'survey': 'WiggleZ'},
    {'z': 0.60, 'fsigma8': 0.390, 'error': 0.063, 'survey': 'WiggleZ'},
    {'z': 0.73, 'fsigma8': 0.437, 'error': 0.072, 'survey': 'WiggleZ'},
    {'z': 0.76, 'fsigma8': 0.440, 'error': 0.040, 'survey': 'VIPERS'},
]

# Base cosmological parameters (Planck 2018)
BASE_PARAMS = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.12,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    'gauge': 'synchronous',
}


class ObservableMapping:
    """Phase 42: Map observables to Φ(z) depth coordinate."""
    
    def __init__(self):
        # Load Φ(z) from Phase 41 (FIXED - no refitting)
        self.load_phi_from_phase41()
        
        self.observables = {}
        self.lcdm_predictions = {}
        self.v9_predictions = {}
        
    def load_phi_from_phase41(self):
        """Load fixed Φ(z) coordinate from Phase 41 results."""
        print("\n" + "="*60)
        print("LOADING Φ(z) FROM PHASE 41")
        print("="*60)
        
        phase41_file = Path(__file__).parent / 'phase41_results' / 'phase41_results.json'
        
        if phase41_file.exists():
            with open(phase41_file, 'r') as f:
                phase41_data = json.load(f)
            
            # Extract V9 n=0.5 normalized responses (this defines Φ)
            v9_responses = phase41_data['normalized_responses']['V9_050']['0.05']
            
            z_vals = [r['z'] for r in v9_responses if not np.isnan(r['delta_A'])]
            delta_A_vals = [r['delta_A'] for r in v9_responses if not np.isnan(r['delta_A'])]
            
            # Φ(z) is defined by the normalized response amplitude
            # We use ΔA - 1 as the depth coordinate (fractional deviation)
            self.phi_z = np.array(z_vals)
            self.phi_vals = np.array(delta_A_vals) - 1.0  # Φ = ΔA - 1
            
            # Create interpolator
            self.phi_interp = interp1d(self.phi_z, self.phi_vals, 
                                      kind='linear', bounds_error=False, 
                                      fill_value='extrapolate')
            
            print(f"\nLoaded Φ(z) from Phase 41:")
            print(f"  Redshift range: z = {self.phi_z[0]:.1f} to {self.phi_z[-1]:.1f}")
            print(f"  Φ range: {self.phi_vals[0]:.6f} to {self.phi_vals[-1]:.6f}")
            print(f"  ✓ Φ(z) is FIXED - no refitting allowed")
            
        else:
            print("\n⚠ Phase 41 results not found - using analytical approximation")
            # Fallback: use analytical form
            self.phi_z = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
            # Approximate Φ ∝ log(1+z) for V9 BEC
            self.phi_vals = 0.05 * np.log10(1 + self.phi_z)
            self.phi_interp = interp1d(self.phi_z, self.phi_vals,
                                      kind='linear', bounds_error=False,
                                      fill_value='extrapolate')
    
    def compute_fsigma8_predictions(self, model_name, n_bec=0.0):
        """Compute fσ8(z) predictions for given model."""
        print(f"\nComputing fσ8 for {model_name} (n_BEC={n_bec})...")
        
        # Set up CLASS
        params = BASE_PARAMS.copy()
        params['output'] = 'mPk'
        params['P_k_max_h/Mpc'] = 1.0
        params['z_max_pk'] = 2.0
        
        if n_bec != 0.0:
            params['bec_crust'] = 'yes'
            params['n_bec'] = n_bec
        
        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        
        predictions = []
        
        for data_point in FSIGMA8_DATA:
            z = data_point['z']
            
            try:
                # Growth factor and rate
                if z == 0:
                    D = 1.0
                    f = cosmo.Omega_m()**0.55
                else:
                    D = cosmo.scale_independent_growth_factor(z)
                    Om_z = cosmo.Omega_m() * (1 + z)**3 / (
                        cosmo.Omega_m() * (1 + z)**3 + (1 - cosmo.Omega_m())
                    )
                    f = Om_z**0.55
                
                # σ8(z) = σ8(0) * D(z)
                sigma8_z = cosmo.sigma8() * D
                
                # fσ8(z)
                fsigma8 = f * sigma8_z
                
                predictions.append({
                    'z': z,
                    'fsigma8': fsigma8,
                    'survey': data_point['survey']
                })
                
            except Exception as e:
                print(f"  Failed at z={z}: {e}")
                predictions.append({
                    'z': z,
                    'fsigma8': np.nan,
                    'survey': data_point['survey']
                })
        
        cosmo.struct_cleanup()
        
        return predictions
    
    def compute_lcdm_normalized_residuals(self):
        """Compute δO(z) = (O_obs - O_ΛCDM) / O_ΛCDM for all observables."""
        print("\n" + "="*60)
        print("COMPUTING ΛCDM-NORMALIZED RESIDUALS")
        print("="*60)
        
        # Get ΛCDM predictions
        self.lcdm_predictions['fsigma8'] = self.compute_fsigma8_predictions('ΛCDM', n_bec=0.0)
        
        # Get V9 predictions
        self.v9_predictions['fsigma8'] = self.compute_fsigma8_predictions('V9 (n=0.5)', n_bec=0.5)
        
        # Compute residuals
        print("\nfσ8 residuals:")
        print(f"{'Survey':15s} {'z':>6s} {'Obs':>8s} {'ΛCDM':>8s} {'V9':>8s} {'δ_obs':>8s} {'δ_V9':>8s}")
        print("-" * 80)
        
        fsigma8_residuals = []
        
        for i, data_point in enumerate(FSIGMA8_DATA):
            z = data_point['z']
            obs = data_point['fsigma8']
            error = data_point['error']
            survey = data_point['survey']
            
            lcdm = self.lcdm_predictions['fsigma8'][i]['fsigma8']
            v9 = self.v9_predictions['fsigma8'][i]['fsigma8']
            
            if not np.isnan(lcdm) and not np.isnan(v9):
                # Fractional residuals
                delta_obs = (obs - lcdm) / lcdm
                delta_v9 = (v9 - lcdm) / lcdm
                
                # Map to Φ
                phi = self.phi_interp(z)
                
                fsigma8_residuals.append({
                    'z': z,
                    'phi': phi,
                    'delta_obs': delta_obs,
                    'delta_v9': delta_v9,
                    'error': error / lcdm,  # Fractional error
                    'survey': survey
                })
                
                print(f"{survey:15s} {z:6.2f} {obs:8.3f} {lcdm:8.3f} {v9:8.3f} "
                      f"{delta_obs:+8.3f} {delta_v9:+8.3f}")
        
        self.observables['fsigma8'] = fsigma8_residuals
        
    def test_phi_collapse(self):
        """Test if observables collapse onto same F(Φ) function."""
        print("\n" + "="*60)
        print("Φ-COLLAPSE TEST")
        print("="*60)
        
        # Extract data
        phi_vals = np.array([r['phi'] for r in self.observables['fsigma8']])
        delta_obs = np.array([r['delta_obs'] for r in self.observables['fsigma8']])
        delta_v9 = np.array([r['delta_v9'] for r in self.observables['fsigma8']])
        
        # Test 1: Monotonicity of observations with Φ
        rho_obs, p_obs = spearmanr(phi_vals, delta_obs)
        
        print(f"\nObservational data vs Φ:")
        print(f"  Spearman ρ: {rho_obs:.3f}")
        print(f"  p-value: {p_obs:.4f}")
        
        if abs(rho_obs) > 0.5 and p_obs < 0.05:
            print(f"  ✓ Observations correlate with Φ")
        else:
            print(f"  ✗ No correlation with Φ")
        
        # Test 2: V9 prediction alignment
        rho_v9, p_v9 = spearmanr(phi_vals, delta_v9)
        
        print(f"\nV9 BEC prediction vs Φ:")
        print(f"  Spearman ρ: {rho_v9:.3f}")
        print(f"  p-value: {p_v9:.4f}")
        
        if rho_v9 > 0.9:
            print(f"  ✓ V9 prediction perfectly ordered by Φ")
        
        # Test 3: Obs-V9 alignment
        # Do observations follow V9's Φ-dependence?
        correlation = np.corrcoef(delta_obs, delta_v9)[0, 1]
        
        print(f"\nObservations vs V9 prediction:")
        print(f"  Correlation: {correlation:.3f}")
        
        if correlation > 0.5:
            print(f"  ✓ Observations align with V9 Φ-structure")
        else:
            print(f"  ✗ Observations don't follow V9 pattern")
        
        return {
            'rho_obs': float(rho_obs),
            'p_obs': float(p_obs),
            'rho_v9': float(rho_v9),
            'p_v9': float(p_v9),
            'obs_v9_correlation': float(correlation)
        }
    
    def test_factorization(self):
        """Test if δO(z) = αO × F(Φ(z))."""
        print("\n" + "="*60)
        print("FACTORIZATION TEST: δO = α × F(Φ)")
        print("="*60)
        
        phi_vals = np.array([r['phi'] for r in self.observables['fsigma8']])
        delta_obs = np.array([r['delta_obs'] for r in self.observables['fsigma8']])
        delta_v9 = np.array([r['delta_v9'] for r in self.observables['fsigma8']])
        errors = np.array([r['error'] for r in self.observables['fsigma8']])
        
        # V9 defines F(Φ) - normalize by maximum
        F_phi = delta_v9 / np.max(np.abs(delta_v9))
        
        # Fit observations: δ_obs = α × F(Φ)
        # Use weighted least squares
        weights = 1.0 / errors**2
        alpha = np.sum(weights * delta_obs * F_phi) / np.sum(weights * F_phi**2)
        
        # Predicted from factorization
        delta_pred = alpha * F_phi
        
        # Residuals
        residuals = delta_obs - delta_pred
        chi2 = np.sum((residuals / errors)**2)
        dof = len(delta_obs) - 1
        
        print(f"\nFactorization fit:")
        print(f"  Coupling α: {alpha:.3f}")
        print(f"  χ²/dof: {chi2/dof:.2f}")
        
        if chi2/dof < 2.0:
            print(f"  ✓ Factorization holds")
            factorizes = True
        else:
            print(f"  ✗ Factorization fails")
            factorizes = False
        
        return {
            'alpha': float(alpha),
            'chi2': float(chi2),
            'dof': int(dof),
            'chi2_per_dof': float(chi2/dof),
            'factorizes': bool(factorizes)
        }
    
    def extract_physical_properties(self):
        """Extract compressibility κ(Φ) and other vacuum properties."""
        print("\n" + "="*60)
        print("PHYSICAL PROPERTY EXTRACTION")
        print("="*60)
        
        # Compute dF/dΦ (compressibility proxy)
        phi_sorted = np.sort(self.phi_vals)
        
        # V9 response defines F(Φ)
        delta_v9_sorted = []
        for phi in phi_sorted:
            # Find closest z
            idx = np.argmin(np.abs(self.phi_vals - phi))
            z = self.phi_z[idx]
            
            # Get V9 prediction at this z
            for pred in self.v9_predictions['fsigma8']:
                if abs(pred['z'] - z) < 0.01:
                    delta_v9_sorted.append((pred['fsigma8'] - 
                                           self.lcdm_predictions['fsigma8'][
                                               self.v9_predictions['fsigma8'].index(pred)
                                           ]['fsigma8']) / 
                                          self.lcdm_predictions['fsigma8'][
                                              self.v9_predictions['fsigma8'].index(pred)
                                          ]['fsigma8'])
                    break
        
        if len(delta_v9_sorted) > 2:
            # Numerical derivative
            dF_dPhi = np.gradient(delta_v9_sorted, phi_sorted)
            
            print(f"\nCompressibility κ(Φ) ∝ dF/dΦ:")
            print(f"  Range: [{np.min(dF_dPhi):.3f}, {np.max(dF_dPhi):.3f}]")
            print(f"  Mean: {np.mean(dF_dPhi):.3f}")
            
            # Check for inflection (supercritical transition)
            d2F_dPhi2 = np.gradient(dF_dPhi, phi_sorted)
            max_curvature_idx = np.argmax(np.abs(d2F_dPhi2))
            
            print(f"\nCurvature d²F/dΦ²:")
            print(f"  Max at Φ ≈ {phi_sorted[max_curvature_idx]:.4f}")
            print(f"  (z ≈ {self.phi_z[max_curvature_idx]:.1f})")
            
            if np.max(np.abs(d2F_dPhi2)) > 1.0:
                print(f"  ⚠ Possible supercritical transition")
        
    def create_plots(self):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: fσ8 observations vs z
        ax = axes[0, 0]
        
        z_obs = [r['z'] for r in self.observables['fsigma8']]
        fsig8_obs = [FSIGMA8_DATA[i]['fsigma8'] for i in range(len(z_obs))]
        errors = [FSIGMA8_DATA[i]['error'] for i in range(len(z_obs))]
        
        z_lcdm = [r['z'] for r in self.lcdm_predictions['fsigma8']]
        fsig8_lcdm = [r['fsigma8'] for r in self.lcdm_predictions['fsigma8']]
        
        z_v9 = [r['z'] for r in self.v9_predictions['fsigma8']]
        fsig8_v9 = [r['fsigma8'] for r in self.v9_predictions['fsigma8']]
        
        ax.errorbar(z_obs, fsig8_obs, yerr=errors, fmt='o', color='black', 
                   label='Observations', markersize=6, capsize=3)
        ax.plot(z_lcdm, fsig8_lcdm, '-', color='blue', linewidth=2, label='ΛCDM')
        ax.plot(z_v9, fsig8_v9, '--', color='red', linewidth=2, label='V9 BEC (n=0.5)')
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('fσ₈(z)', fontsize=12)
        ax.set_title('Growth Rate Measurements', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Fractional residuals vs z
        ax = axes[0, 1]
        
        delta_obs = [r['delta_obs'] * 100 for r in self.observables['fsigma8']]
        delta_v9 = [r['delta_v9'] * 100 for r in self.observables['fsigma8']]
        frac_errors = [r['error'] * 100 for r in self.observables['fsigma8']]
        
        ax.errorbar(z_obs, delta_obs, yerr=frac_errors, fmt='o', color='black',
                   label='Obs - ΛCDM', markersize=6, capsize=3)
        ax.plot(z_v9, delta_v9, 's-', color='red', linewidth=2, markersize=6,
               label='V9 - ΛCDM')
        ax.axhline(0, color='blue', linestyle='--', alpha=0.5, label='ΛCDM')
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel('Fractional Deviation [%]', fontsize=12)
        ax.set_title('ΛCDM-Normalized Residuals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Φ-collapse test
        ax = axes[1, 0]
        
        phi_obs = [r['phi'] for r in self.observables['fsigma8']]
        
        ax.errorbar(phi_obs, delta_obs, yerr=frac_errors, fmt='o', color='black',
                   label='Observations', markersize=6, capsize=3)
        ax.plot(phi_obs, delta_v9, 's-', color='red', linewidth=2, markersize=6,
               label='V9 Prediction')
        ax.axhline(0, color='blue', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel('Fractional Deviation [%]', fontsize=12)
        ax.set_title('Φ-Collapse Test', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Factorization test
        ax = axes[1, 1]
        
        # Normalize V9 to define F(Φ)
        F_phi = np.array(delta_v9) / np.max(np.abs(delta_v9))
        
        # Fit α
        weights = 1.0 / np.array(frac_errors)**2
        alpha = np.sum(weights * np.array(delta_obs) * F_phi) / np.sum(weights * F_phi**2)
        
        delta_pred = alpha * F_phi
        
        ax.errorbar(delta_pred, delta_obs, yerr=frac_errors, fmt='o', color='black',
                   markersize=6, capsize=3)
        
        # Perfect factorization line
        lim = max(abs(np.min(delta_obs)), abs(np.max(delta_obs)))
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Perfect factorization')
        
        ax.set_xlabel('Predicted: α × F(Φ) [%]', fontsize=12)
        ax.set_ylabel('Observed Deviation [%]', fontsize=12)
        ax.set_title(f'Factorization: α = {alpha:.2f}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / 'observable_mapping.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {plot_path}")
        plt.close()
    
    def run_full_analysis(self):
        """Execute Phase 42 analysis."""
        print("\n" + "="*60)
        print("PHASE 42: OBSERVABLE MAPPING & PHYSICAL CALIBRATION")
        print("="*60)
        
        # Compute residuals
        self.compute_lcdm_normalized_residuals()
        
        # Test Φ-collapse
        collapse_results = self.test_phi_collapse()
        
        # Test factorization
        factor_results = self.test_factorization()
        
        # Extract physical properties
        self.extract_physical_properties()
        
        # Create plots
        self.create_plots()
        
        # Decision logic
        print("\n" + "="*60)
        print("DECISIVE OUTCOME")
        print("="*60)
        
        obs_correlates = abs(collapse_results['rho_obs']) > 0.5 and collapse_results['p_obs'] < 0.05
        v9_ordered = collapse_results['rho_v9'] > 0.9
        factorizes = factor_results['factorizes']
        
        if obs_correlates and v9_ordered and factorizes:
            decision = "PHI_VALIDATED"
            interpretation = "Φ(z) is a physical vacuum property - observables collapse onto F(Φ)"
        elif v9_ordered and factorizes and not obs_correlates:
            decision = "MODEL_PREDICTION"
            interpretation = "V9 BEC predicts Φ-structure, but current data insufficient to confirm"
        elif obs_correlates and not factorizes:
            decision = "PARTIAL_SIGNAL"
            interpretation = "Observations show Φ-correlation but factorization fails"
        else:
            decision = "PHI_FALSIFIED"
            interpretation = "Observables don't collapse onto universal F(Φ)"
        
        print(f"\nDecision: {decision}")
        print(f"Interpretation: {interpretation}")
        
        # Save results
        output = {
            'decision': decision,
            'interpretation': interpretation,
            'collapse_test': collapse_results,
            'factorization_test': factor_results,
            'observables': {
                'fsigma8': [
                    {k: float(v) if isinstance(v, (int, float, np.number, np.ndarray)) and not isinstance(v, str) else str(v) if isinstance(v, str) else float(v) if np.isscalar(v) else v
                     for k, v in obs.items()}
                    for obs in self.observables['fsigma8']
                ]
            }
        }
        
        with open(OUTPUT_DIR / 'phase42_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PHASE 42 COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to {OUTPUT_DIR}")


def main():
    """Run Phase 42 analysis."""
    test = ObservableMapping()
    test.run_full_analysis()


if __name__ == "__main__":
    main()
