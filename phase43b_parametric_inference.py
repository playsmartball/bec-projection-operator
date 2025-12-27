#!/usr/bin/env python3
"""
Phase 43b: Parametric Supercriticality Test

Tests whether non-linear structure in F(Φ) is statistically required
using parametric model fitting instead of numerical derivatives.

Avoids ill-conditioned finite differences by fitting functional forms:
1. Linear: F = 1 + aΦ
2. Quadratic: F = 1 + aΦ + bΦ²
3. Log-enhanced: F = 1 + aΦ + bΦ log Φ
4. Crossover: F = 1 + a tanh(Φ/Φ_c)

Model selection via AIC/BIC/Δχ² determines if curvature is required.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'phase43b_results'
OUTPUT_DIR.mkdir(exist_ok=True)


class ParametricSupercriticalityTest:
    """Phase 43b: Parametric inference of vacuum phase structure."""
    
    def __init__(self):
        # Load F(Φ) from Phase 41
        self.load_response_function()
        
        self.models = {}
        self.best_model = None
        
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
            
            # Uniform errors (conservative estimate)
            self.F_errors = np.ones_like(self.F_phi) * 0.001  # 0.1% error
            
            print(f"\nLoaded F(Φ) from Phase 41:")
            print(f"  Φ range: [{self.phi[0]:.6f}, {self.phi[-1]:.6f}]")
            print(f"  F(Φ) range: [{self.F_phi[0]:.6f}, {self.F_phi[-1]:.6f}]")
            print(f"  Number of points: {len(self.phi)}")
            
            # Also load ΛCDM for control (n=0.18 is closer to ΛCDM)
            lcdm_responses = phase41_data['normalized_responses']['V9_018']['0.05']
            lcdm_delta_A = np.array([r['delta_A'] for r in lcdm_responses if not np.isnan(r['delta_A'])])
            self.F_phi_lcdm = lcdm_delta_A[sort_idx]
            
        else:
            raise FileNotFoundError("Phase 41 results not found - run Phase 41 first")
    
    # Model definitions
    def model_linear(self, phi, a):
        """Linear: F = 1 + aΦ"""
        return 1.0 + a * phi
    
    def model_quadratic(self, phi, a, b):
        """Quadratic: F = 1 + aΦ + bΦ²"""
        return 1.0 + a * phi + b * phi**2
    
    def model_log_enhanced(self, phi, a, b):
        """Log-enhanced: F = 1 + aΦ + bΦ log Φ"""
        # Avoid log(0) by using phi + small offset
        phi_safe = phi + 1e-6
        return 1.0 + a * phi + b * phi * np.log(phi_safe)
    
    def model_crossover(self, phi, a, phi_c):
        """Crossover: F = 1 + a tanh(Φ/Φ_c)"""
        return 1.0 + a * np.tanh(phi / phi_c)
    
    def fit_model(self, model_func, p0, bounds=None, name=""):
        """Fit a model to F(Φ) data."""
        try:
            if bounds is not None:
                popt, pcov = curve_fit(model_func, self.phi, self.F_phi, 
                                      p0=p0, sigma=self.F_errors, 
                                      absolute_sigma=True, bounds=bounds)
            else:
                popt, pcov = curve_fit(model_func, self.phi, self.F_phi, 
                                      p0=p0, sigma=self.F_errors, 
                                      absolute_sigma=True)
            
            # Compute residuals and chi-squared
            F_pred = model_func(self.phi, *popt)
            residuals = self.F_phi - F_pred
            chi2 = np.sum((residuals / self.F_errors)**2)
            
            # Degrees of freedom
            n_data = len(self.phi)
            n_params = len(popt)
            dof = n_data - n_params
            
            # AIC and BIC
            aic = chi2 + 2 * n_params
            bic = chi2 + n_params * np.log(n_data)
            
            # Parameter errors
            perr = np.sqrt(np.diag(pcov))
            
            return {
                'name': name,
                'params': popt,
                'errors': perr,
                'chi2': chi2,
                'dof': dof,
                'chi2_per_dof': chi2 / dof if dof > 0 else np.inf,
                'aic': aic,
                'bic': bic,
                'n_params': n_params,
                'success': True
            }
            
        except Exception as e:
            print(f"  ✗ {name} fit failed: {e}")
            return {
                'name': name,
                'success': False
            }
    
    def fit_all_models(self):
        """Fit all candidate models."""
        print("\n" + "="*60)
        print("FITTING PARAMETRIC MODELS")
        print("="*60)
        
        # Model 1: Linear
        print("\n1. Linear: F = 1 + aΦ")
        result = self.fit_model(self.model_linear, p0=[1.0], name="Linear")
        if result['success']:
            print(f"   a = {result['params'][0]:.3f} ± {result['errors'][0]:.3f}")
            print(f"   χ²/dof = {result['chi2_per_dof']:.3f}")
            print(f"   AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")
        self.models['linear'] = result
        
        # Model 2: Quadratic
        print("\n2. Quadratic: F = 1 + aΦ + bΦ²")
        result = self.fit_model(self.model_quadratic, p0=[1.0, 0.0], name="Quadratic")
        if result['success']:
            print(f"   a = {result['params'][0]:.3f} ± {result['errors'][0]:.3f}")
            print(f"   b = {result['params'][1]:.3f} ± {result['errors'][1]:.3f}")
            print(f"   χ²/dof = {result['chi2_per_dof']:.3f}")
            print(f"   AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")
            
            # Test significance of b
            b_significance = abs(result['params'][1]) / result['errors'][1]
            print(f"   b significance: {b_significance:.1f}σ")
        self.models['quadratic'] = result
        
        # Model 3: Log-enhanced
        print("\n3. Log-enhanced: F = 1 + aΦ + bΦ log Φ")
        result = self.fit_model(self.model_log_enhanced, p0=[1.0, 0.0], name="Log-enhanced")
        if result['success']:
            print(f"   a = {result['params'][0]:.3f} ± {result['errors'][0]:.3f}")
            print(f"   b = {result['params'][1]:.3f} ± {result['errors'][1]:.3f}")
            print(f"   χ²/dof = {result['chi2_per_dof']:.3f}")
            print(f"   AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")
        self.models['log_enhanced'] = result
        
        # Model 4: Crossover
        print("\n4. Crossover: F = 1 + a tanh(Φ/Φ_c)")
        # Initial guess: Φ_c at midpoint
        phi_c_guess = (self.phi[0] + self.phi[-1]) / 2
        result = self.fit_model(self.model_crossover, p0=[0.05, phi_c_guess], 
                               bounds=([0, self.phi[0]], [0.1, self.phi[-1]]),
                               name="Crossover")
        if result['success']:
            print(f"   a = {result['params'][0]:.3f} ± {result['errors'][0]:.3f}")
            print(f"   Φ_c = {result['params'][1]:.6f} ± {result['errors'][1]:.6f}")
            print(f"   χ²/dof = {result['chi2_per_dof']:.3f}")
            print(f"   AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")
        self.models['crossover'] = result
    
    def select_best_model(self):
        """Select best model using AIC/BIC."""
        print("\n" + "="*60)
        print("MODEL SELECTION")
        print("="*60)
        
        # Extract successful models
        successful = {k: v for k, v in self.models.items() if v['success']}
        
        if not successful:
            print("\n✗ No models fit successfully")
            return None
        
        # Compare by AIC (lower is better)
        aic_values = {k: v['aic'] for k, v in successful.items()}
        best_aic = min(aic_values, key=aic_values.get)
        
        # Compare by BIC (lower is better)
        bic_values = {k: v['bic'] for k, v in successful.items()}
        best_bic = min(bic_values, key=bic_values.get)
        
        print(f"\nModel comparison:")
        print(f"{'Model':<15} {'χ²/dof':<10} {'AIC':<10} {'BIC':<10} {'ΔAIC':<10} {'ΔBIC':<10}")
        print("-" * 70)
        
        for name, result in successful.items():
            delta_aic = result['aic'] - aic_values[best_aic]
            delta_bic = result['bic'] - bic_values[best_bic]
            
            print(f"{name:<15} {result['chi2_per_dof']:<10.3f} "
                  f"{result['aic']:<10.2f} {result['bic']:<10.2f} "
                  f"{delta_aic:<10.2f} {delta_bic:<10.2f}")
        
        # Decision rules
        print(f"\nBest by AIC: {best_aic}")
        print(f"Best by BIC: {best_bic}")
        
        # Use BIC (more conservative, penalizes complexity more)
        self.best_model = best_bic
        
        # Check if improvement is significant (ΔBIC > 2 is "positive", > 6 is "strong")
        if best_bic != 'linear':
            delta_bic = bic_values['linear'] - bic_values[best_bic]
            
            if delta_bic > 6:
                print(f"\n✓ {best_bic.upper()} strongly preferred (ΔBIC = {delta_bic:.1f})")
                preference = "STRONG"
            elif delta_bic > 2:
                print(f"\n✓ {best_bic.upper()} preferred (ΔBIC = {delta_bic:.1f})")
                preference = "MODERATE"
            else:
                print(f"\n⚠ {best_bic.upper()} only weakly preferred (ΔBIC = {delta_bic:.1f})")
                print(f"  → Linear model adequate")
                self.best_model = 'linear'
                preference = "WEAK"
        else:
            print(f"\n✓ LINEAR model preferred")
            preference = "LINEAR"
        
        return {
            'best_model': self.best_model,
            'preference': preference,
            'aic_values': {k: float(v) for k, v in aic_values.items()},
            'bic_values': {k: float(v) for k, v in bic_values.items()}
        }
    
    def compute_susceptibility_analytical(self):
        """Compute κ(Φ) = d ln F / d ln Φ analytically from best model."""
        print("\n" + "="*60)
        print("SUSCEPTIBILITY EXPONENT κ(Φ)")
        print("="*60)
        
        if self.best_model is None:
            print("\n✗ No best model selected")
            return None
        
        model = self.models[self.best_model]
        params = model['params']
        
        # Create dense Φ grid for plotting
        phi_dense = np.linspace(self.phi[0], self.phi[-1], 100)
        
        # Compute κ analytically for each model
        if self.best_model == 'linear':
            # F = 1 + aΦ
            # κ = d ln F / d ln Φ = (Φ/F) * dF/dΦ = (Φ/F) * a
            a = params[0]
            F_dense = self.model_linear(phi_dense, a)
            kappa_dense = (phi_dense / F_dense) * a
            
            print(f"\nLinear model: F = 1 + {a:.3f}Φ")
            print(f"  κ(Φ) = aΦ / (1 + aΦ)")
            
        elif self.best_model == 'quadratic':
            # F = 1 + aΦ + bΦ²
            # κ = (Φ/F) * (a + 2bΦ)
            a, b = params
            F_dense = self.model_quadratic(phi_dense, a, b)
            kappa_dense = (phi_dense / F_dense) * (a + 2*b*phi_dense)
            
            print(f"\nQuadratic model: F = 1 + {a:.3f}Φ + {b:.3f}Φ²")
            print(f"  κ(Φ) = Φ(a + 2bΦ) / (1 + aΦ + bΦ²)")
            
        elif self.best_model == 'log_enhanced':
            # F = 1 + aΦ + bΦ log Φ
            # κ = (Φ/F) * (a + b(1 + log Φ))
            a, b = params
            phi_safe = phi_dense + 1e-6
            F_dense = self.model_log_enhanced(phi_dense, a, b)
            kappa_dense = (phi_dense / F_dense) * (a + b * (1 + np.log(phi_safe)))
            
            print(f"\nLog-enhanced model: F = 1 + {a:.3f}Φ + {b:.3f}Φ log Φ")
            print(f"  κ(Φ) = Φ(a + b(1 + log Φ)) / F")
            
        elif self.best_model == 'crossover':
            # F = 1 + a tanh(Φ/Φ_c)
            # κ = (Φ/F) * (a/Φ_c) * sech²(Φ/Φ_c)
            a, phi_c = params
            F_dense = self.model_crossover(phi_dense, a, phi_c)
            sech2 = 1.0 / np.cosh(phi_dense / phi_c)**2
            kappa_dense = (phi_dense / F_dense) * (a / phi_c) * sech2
            
            print(f"\nCrossover model: F = 1 + {a:.3f} tanh(Φ/{phi_c:.6f})")
            print(f"  κ(Φ) = (Φ/F) * (a/Φ_c) * sech²(Φ/Φ_c)")
            print(f"  Critical depth: Φ_c = {phi_c:.6f}")
        
        print(f"\nκ(Φ) range: [{np.min(kappa_dense):.4f}, {np.max(kappa_dense):.4f}]")
        print(f"Mean κ: {np.mean(kappa_dense):.4f}")
        
        # Check for variation
        kappa_variation = np.std(kappa_dense) / np.mean(kappa_dense)
        
        if kappa_variation < 0.1:
            print(f"  → Nearly constant susceptibility (single phase)")
        else:
            print(f"  → Varying susceptibility ({kappa_variation*100:.1f}% variation)")
        
        self.phi_dense = phi_dense
        self.F_dense = F_dense
        self.kappa_dense = kappa_dense
        
        return kappa_dense
    
    def run_controls(self):
        """Run control tests on ΛCDM and randomized data."""
        print("\n" + "="*60)
        print("CONTROL TESTS")
        print("="*60)
        
        # Control 1: ΛCDM should prefer flat (constant)
        print("\n1. ΛCDM Control:")
        
        # Fit linear to ΛCDM data
        phi_lcdm = self.phi
        F_lcdm = self.F_phi_lcdm
        errors_lcdm = np.ones_like(F_lcdm) * 0.001
        
        # Fit constant model: F = c
        c_fit = np.mean(F_lcdm)
        residuals = F_lcdm - c_fit
        chi2_const = np.sum((residuals / errors_lcdm)**2)
        
        # Fit linear model
        popt_lin, _ = curve_fit(self.model_linear, phi_lcdm, F_lcdm, 
                               p0=[0.0], sigma=errors_lcdm)
        F_lin = self.model_linear(phi_lcdm, *popt_lin)
        chi2_lin = np.sum(((F_lcdm - F_lin) / errors_lcdm)**2)
        
        # Compare
        delta_chi2 = chi2_const - chi2_lin
        
        print(f"   Constant: χ² = {chi2_const:.2f}")
        print(f"   Linear: χ² = {chi2_lin:.2f}")
        print(f"   Δχ² = {delta_chi2:.2f}")
        
        if delta_chi2 < 4:  # Not significant
            print(f"   ✓ ΛCDM consistent with flat (control passed)")
            lcdm_control = True
        else:
            print(f"   ⚠ ΛCDM shows trend (unexpected)")
            lcdm_control = False
        
        # Control 2: Randomized Φ should not improve fit
        print("\n2. Randomized Φ Null Test:")
        
        # Shuffle Φ assignments
        phi_shuffled = np.random.permutation(self.phi)
        
        # Fit linear to shuffled data
        sort_idx = np.argsort(phi_shuffled)
        phi_shuffled = phi_shuffled[sort_idx]
        F_shuffled = self.F_phi[sort_idx]
        
        result_null = self.fit_model(self.model_linear, p0=[1.0], name="Null")
        
        if result_null['success']:
            chi2_null = result_null['chi2']
            chi2_real = self.models['linear']['chi2']
            
            print(f"   Real data: χ² = {chi2_real:.2f}")
            print(f"   Shuffled: χ² = {chi2_null:.2f}")
            
            if chi2_real < chi2_null:
                print(f"   ✓ Real data fits better (null passed)")
                null_passed = True
            else:
                print(f"   ⚠ Shuffled data fits as well (unexpected)")
                null_passed = False
        else:
            null_passed = False
        
        return {
            'lcdm_control': bool(lcdm_control),
            'null_passed': bool(null_passed)
        }
    
    def create_plots(self):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: All model fits
        ax = axes[0, 0]
        ax.errorbar(self.phi, self.F_phi, yerr=self.F_errors, fmt='o', 
                   color='black', markersize=8, capsize=3, label='Data', zorder=5)
        
        colors = {'linear': 'blue', 'quadratic': 'green', 
                 'log_enhanced': 'orange', 'crossover': 'red'}
        
        for name, model in self.models.items():
            if model['success']:
                if name == 'linear':
                    F_fit = self.model_linear(self.phi_dense, *model['params'])
                elif name == 'quadratic':
                    F_fit = self.model_quadratic(self.phi_dense, *model['params'])
                elif name == 'log_enhanced':
                    F_fit = self.model_log_enhanced(self.phi_dense, *model['params'])
                elif name == 'crossover':
                    F_fit = self.model_crossover(self.phi_dense, *model['params'])
                
                linestyle = '-' if name == self.best_model else '--'
                linewidth = 3 if name == self.best_model else 1.5
                
                ax.plot(self.phi_dense, F_fit, linestyle, color=colors[name],
                       linewidth=linewidth, label=name.replace('_', ' ').title())
        
        ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
        ax.set_ylabel('Response Function F(Φ)', fontsize=12)
        ax.set_title('Parametric Model Fits', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals for best model
        ax = axes[0, 1]
        
        if self.best_model and self.models[self.best_model]['success']:
            model = self.models[self.best_model]
            
            if self.best_model == 'linear':
                F_fit = self.model_linear(self.phi, *model['params'])
            elif self.best_model == 'quadratic':
                F_fit = self.model_quadratic(self.phi, *model['params'])
            elif self.best_model == 'log_enhanced':
                F_fit = self.model_log_enhanced(self.phi, *model['params'])
            elif self.best_model == 'crossover':
                F_fit = self.model_crossover(self.phi, *model['params'])
            
            residuals = (self.F_phi - F_fit) / self.F_errors
            
            ax.errorbar(self.phi, residuals, yerr=1.0, fmt='o', 
                       color='black', markersize=8, capsize=3)
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            ax.axhline(2, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(-2, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
            ax.set_ylabel('Normalized Residuals (σ)', fontsize=12)
            ax.set_title(f'Residuals: {self.best_model.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Susceptibility exponent κ(Φ)
        ax = axes[1, 0]
        
        if hasattr(self, 'kappa_dense'):
            ax.plot(self.phi_dense, self.kappa_dense, '-', color='purple', linewidth=2)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Depth Coordinate Φ', fontsize=12)
            ax.set_ylabel('κ(Φ) = d ln F / d ln Φ', fontsize=12)
            ax.set_title('Susceptibility Exponent (Analytical)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Model comparison (AIC/BIC)
        ax = axes[1, 1]
        
        successful = {k: v for k, v in self.models.items() if v['success']}
        model_names = list(successful.keys())
        aic_vals = [successful[k]['aic'] for k in model_names]
        bic_vals = [successful[k]['bic'] for k in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, aic_vals, width, label='AIC', color='steelblue')
        ax.bar(x + width/2, bic_vals, width, label='BIC', color='coral')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Information Criterion', fontsize=12)
        ax.set_title('Model Selection (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in model_names], fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / 'parametric_inference.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {plot_path}")
        plt.close()
    
    def run_full_analysis(self):
        """Execute Phase 43b analysis."""
        print("\n" + "="*60)
        print("PHASE 43b: PARAMETRIC SUPERCRITICALITY TEST")
        print("="*60)
        
        # Fit all models
        self.fit_all_models()
        
        # Select best model
        selection = self.select_best_model()
        
        # Compute susceptibility
        self.compute_susceptibility_analytical()
        
        # Run controls
        controls = self.run_controls()
        
        # Create plots
        self.create_plots()
        
        # Decision logic
        print("\n" + "="*60)
        print("DECISIVE OUTCOME")
        print("="*60)
        
        if selection is None:
            decision = "INCONCLUSIVE"
            interpretation = "Model fitting failed"
        elif self.best_model == 'linear':
            decision = "LINEAR_STRATIFICATION"
            interpretation = "Pure depth stratification - no curvature required"
        elif self.best_model == 'quadratic':
            decision = "WEAK_CURVATURE"
            interpretation = "Weak non-linearity detected - smooth stratification"
        elif self.best_model in ['log_enhanced', 'crossover']:
            decision = "SUPERCRITICAL_CROSSOVER"
            interpretation = "Non-linear phase structure - supercritical transition signature"
        else:
            decision = "UNKNOWN"
            interpretation = "Unexpected model selected"
        
        print(f"\nDecision: {decision}")
        print(f"Interpretation: {interpretation}")
        print(f"Best model: {self.best_model}")
        print(f"\nControl tests:")
        print(f"  ΛCDM control: {'PASS' if controls['lcdm_control'] else 'FAIL'}")
        print(f"  Randomization null: {'PASS' if controls['null_passed'] else 'FAIL'}")
        
        # Save results
        output = {
            'decision': decision,
            'interpretation': interpretation,
            'best_model': self.best_model,
            'selection': selection,
            'models': {
                k: {key: float(val) if isinstance(val, (int, float, np.number)) 
                    else val.tolist() if isinstance(val, np.ndarray) 
                    else val
                    for key, val in v.items() if key != 'name'}
                for k, v in self.models.items() if v['success']
            },
            'controls': controls
        }
        
        with open(OUTPUT_DIR / 'phase43b_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        # Summary text file
        with open(OUTPUT_DIR / 'phase43b_summary.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("PHASE 43b: PARAMETRIC SUPERCRITICALITY TEST - SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Decision: {decision}\n")
            f.write(f"Interpretation: {interpretation}\n\n")
            f.write(f"Best model: {self.best_model}\n")
            f.write(f"Preference: {selection['preference']}\n\n")
            
            if self.best_model and self.models[self.best_model]['success']:
                model = self.models[self.best_model]
                f.write(f"Model parameters:\n")
                for i, (param, error) in enumerate(zip(model['params'], model['errors'])):
                    f.write(f"  p{i} = {param:.6f} ± {error:.6f}\n")
                f.write(f"\nχ²/dof = {model['chi2_per_dof']:.3f}\n")
                f.write(f"AIC = {model['aic']:.2f}\n")
                f.write(f"BIC = {model['bic']:.2f}\n")
            
            f.write(f"\nΛCDM control: {'PASS' if controls['lcdm_control'] else 'FAIL'}\n")
            f.write(f"Randomization null: {'PASS' if controls['null_passed'] else 'FAIL'}\n")
        
        print(f"\n{'='*60}")
        print("PHASE 43b COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to {OUTPUT_DIR}")


def main():
    """Run Phase 43b analysis."""
    test = ParametricSupercriticalityTest()
    test.run_full_analysis()


if __name__ == "__main__":
    main()
