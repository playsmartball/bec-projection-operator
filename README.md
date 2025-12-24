# V9 BEC Crust CLASS Implementation

Rigorous implementation of the V9 BEC Crust model in CLASS for Planck likelihood analysis.

## Model Description

The V9 BEC Crust model replaces the cosmological constant (Λ) with an evolving vacuum component:

```
Ω_BEC(z) = Ω_BEC_0 × (1+z)^n_BEC
```

Where:
- `Ω_BEC_0` is fixed by closure: `1 - Ω_m - Ω_r`
- `n_BEC` is the single new parameter (dilution exponent)
- `n_BEC = 0` recovers exact ΛCDM
- Equation of state: `w = n_BEC/3 - 1`

## Key Results

### Planck-only (CMB TT+TE+EE)
```
n_BEC = 0.18 ± 0.40 (68% CL)
- ΛCDM (n_BEC=0): 0.4σ from mean ✓
- V9 prediction (n_BEC≈0.5): 0.8σ from mean ✓
```

### fσ₈ Growth Rate Check
```
| n_BEC | σ₈   | χ²/dof | Status     |
|-------|------|--------|------------|
| 0.00  | 0.823| 0.95   | ✓ Good fit |
| 0.18  | 0.815| 0.68   | ✓ Good fit |
| 0.50  | 0.800| 0.58   | ✓ Good fit |

Δχ² (V9 vs ΛCDM) = -3.72 → V9 slightly better fit
```

### Joint Planck + BAO + SNe (preliminary, ~1200 samples)
```
n_BEC = -0.015 ± 0.072 (68% CL)
- ΛCDM (n_BEC=0): 0.2σ from mean ✓
- V9 prediction (n_BEC≈0.5): 7.2σ from mean ✗
```

## Installation

### 1. Clone and build CLASS
```bash
cd class_v9_bec
make clean
make -j4
```

### 2. Create Python environment
```bash
python3 -m venv venv_new
source venv_new/bin/activate
pip install numpy scipy matplotlib cython
```

### 3. Install CLASS Python wrapper
```bash
cd class_v9_bec/python
python setup.py install
```

### 4. Install Cobaya and likelihoods
```bash
pip install cobaya
cobaya-install planck_2018_highl_plik.TTTEEE_lite planck_2018_lowl.TT
cobaya-install bao.sixdf_2011_bao bao.sdss_dr7_mgs bao.sdss_dr12_consensus_bao sn.pantheon
```

## Running the Analysis

### Quick validation test
```bash
python test_v9_bec_crust.py
```

### Planck-only MCMC
```bash
python run_v9_bec_mcmc.py --samples 10000
```

### Joint Planck + BAO + SNe MCMC
```bash
python run_v9_bec_joint_mcmc.py --samples 10000
```

### fσ₈ growth rate check
```bash
python run_v9_bec_fsigma8_check.py
```

## Files

- `class_v9_bec/` - Modified CLASS with V9 BEC Crust implementation
  - `include/background.h` - Added n_bec, has_bec_crust parameters
  - `source/background.c` - BEC density evolution implementation
  - `source/input.c` - Parameter parsing
- `test_v9_bec_crust.py` - Validation tests
- `run_v9_bec_mcmc.py` - Planck-only MCMC
- `run_v9_bec_joint_mcmc.py` - Joint Planck+BAO+SNe MCMC
- `run_v9_bec_fsigma8_check.py` - Growth rate consistency check

## Citation

This implementation is part of the BEC cosmology framework.
Repository: https://github.com/playsmartball/bec-projection-operator
