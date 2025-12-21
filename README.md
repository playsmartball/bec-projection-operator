# BEC Projection Operator Analysis

A conservative, reproducible analysis of a projection-level geometric operator acting on CMB angular power spectra.

## Summary

This repository contains the complete analysis pipeline for identifying and validating a projection-space operator that partially explains the residual between ΛCDM and BEC (Bose-Einstein Condensate dark energy) power spectra.

**Key Result:**

> A fixed, non-tunable, projection-level horizontal operator—parameterized by ε = 1.456 × 10⁻³ independently measured from peak displacements—removes approximately 40% of the ΛCDM–BEC residual in TT and EE power spectra.

## Operator Definition

```
P_ε : C_ℓ ↦ C_{ℓ/(1+ε)}
```

where:
- ε = 1.4558030818 × 10⁻³ (locked, not tuned)
- Equivalent to δD_A/D_A ≈ 0.15% angular diameter distance perturbation

## Validation Results

| Test | Status | Key Metric |
|------|--------|------------|
| Lensing Null (14A-2) | ✓ PASS | Effect not lensing-induced |
| Window Stability (14A-3) | ✓ PASS | Stable across all ℓ-cuts |
| Noise Robustness (14A-4) | ✓ PASS | 100% positive at 50% noise |
| TE Consistency (14A-1) | ✓ PASS | Correlation +0.91 |

## Repository Structure

```
bec-projection-operator/
├── README.md
├── LICENSE
├── CITATION.cff
├── data/
│   ├── lcdm_unlensed/      # ΛCDM reference spectra
│   ├── bec_unlensed/       # BEC target spectra
│   ├── lcdm_lensed/        # Lensed ΛCDM (for null test)
│   ├── bec_lensed/         # Lensed BEC (for null test)
│   └── phase10e_tomography.npz  # Peak displacement data
├── scripts/
│   ├── phase10e_peak_tomography.py
│   ├── phase12a_class_attempt.md
│   ├── phase13a_projection_operator.py
│   ├── phase14a_conservative_tests.py
│   ├── phase15a_formal_operator.py
│   └── phase15b_interpretation_boundary.py
└── output/
    ├── figures/
    ├── logs/
    └── summaries/
```

## Reproducibility

### Requirements

```
python >= 3.8
numpy
matplotlib
scipy
```

### Running the Analysis

```bash
# 1. Validate the projection operator
python scripts/phase13a_projection_operator.py

# 2. Run conservative robustness tests
python scripts/phase14a_conservative_tests.py

# 3. Generate formal documentation
python scripts/phase15a_formal_operator.py
python scripts/phase15b_interpretation_boundary.py
```

### Locked Parameters

**DO NOT MODIFY:**
- ε = 1.4558030818e-03
- Analysis range: ℓ ∈ [800, 2500]
- Operator: `ℓ → ℓ/(1+ε)`

## What Is Claimed

✓ Existence of a coherent, projection-level geometric pattern  
✓ Single-parameter characterization (ε ≈ 1.5 × 10⁻³)  
✓ Robustness across spectra, windows, and noise  
✓ Equivalence to ~0.15% D_A perturbation  

## What Is NOT Claimed

✗ Physical mechanism  
✗ Modified gravity  
✗ Dark energy microphysics  
✗ Inflationary modifications  
✗ Boltzmann equation changes  
✗ New fundamental physics  

See `output/summaries/phase15b_interpretation_boundary.txt` for full scope.

## Citation

If you use this analysis, please cite:

```bibtex
@software{bec_projection_operator,
  title = {BEC Projection Operator Analysis},
  year = {2024},
  url = {https://github.com/[username]/bec-projection-operator}
}
```

## License

MIT License. See LICENSE file.

## Contact

[Your contact information]
