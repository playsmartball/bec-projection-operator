# Phase 12A: CLASS Boltzmann-Level Attempt

## Summary

Phase 12A attempted to implement the horizontal phase operator at the Boltzmann
source level (inside CLASS `harmonic.c`). This approach **failed** to reproduce
the BEC residual pattern.

## What Was Tried

Modified CLASS to apply the transformation `ℓ → ℓ/(1+ε)` at the transfer
function level, before k-integration:

```c
// In harmonic.c, harmonic_compute_cl()
double l_star = phr->l[index_l] / (1.0 + phr->horizontal_phase_epsilon);
// Binary search for interpolation indices
// Apply interpolated transfer functions
```

## Result

- **RMS increased** (made residuals worse, not better)
- **Correlation was negative** (opposite direction)
- The operator at this level probes **wrong physics**

## Interpretation

The Boltzmann-level modification acts on the source terms before projection.
The BEC effect is a **projection-level** phenomenon that occurs after
k-integration, in the geometric mapping from 3D perturbations to 2D angular
correlations.

This failure is **informative**: it localizes the effect to the projection
layer and rules out early-universe microphysics explanations.

## Files Modified (for reference only)

- `fracos/phi/class_public/source/harmonic.c` (lines 920-967)
- `fracos/phi/class_public/include/harmonic.h` (added `horizontal_phase_epsilon`)

## Conclusion

Phase 12A demonstrates that the BEC residual pattern **cannot** be explained
by Boltzmann-level modifications. The effect must live in the projection
layer (Phase 13A), not the source layer.

This is a **negative result** that strengthens the projection-level interpretation.
