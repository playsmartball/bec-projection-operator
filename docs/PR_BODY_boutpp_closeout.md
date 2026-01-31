# Close out BOUT++ inertia validation; add minimal and Dedalus 1D Alfvén benchmarks

## Summary
This PR closes the BOUT++ inertia-scaling validation branch and introduces two independent, reviewer-safe benchmarks (a minimal 1D solver and a Dedalus replica) that validate Alfvén dispersion with effective inertia. It resolves the ambiguity encountered in the BOUT++ Orszag–Tang example and establishes a robust control framework for future extensions (including NIMROD).

## Key Findings
- Orszag–Tang correctly parses/applies `rho_eff` in the momentum equation, but normalization/linearization fixes the Alfvén eigenfrequency independently of inertia, preventing unambiguous dispersion validation.
- This is an architectural limitation of the example, not a user or implementation error.
- Minimal 1D solver reproduces theory `omega = k * B0 / sqrt(1 + rho_eff)` to <0.3%.
- Dedalus replica reproduces both theory and the minimal solver to <0.5%.

## What’s Included
- `examples/alfven_1d.py` — Minimal explicit RK solver (specification benchmark)
- `examples/dedalus_alfven_1d.py` — Dedalus v3 spectral replica with identical equations/diagnostics
- `docs/BOUTPP_CLOSEOUT.md` — Appended “Dedalus Control-Parity Validation” section and results table
- `docs/SBIR_Appendix_Inertia_Validation.md` — SBIR-ready appendix (drop-in)

## Representative Results (N=128, L=128, k=1, B0=1)
- Dedalus ρ=0: numerical 4.914755e-02 vs theory 4.908739e-02 (0.12%)
- Dedalus ρ=20: numerical 1.068703e-02 vs theory 1.071175e-02 (0.23%)
- Ratio ω(0)/ω(20): 4.599 (theory √21 = 4.5826)
- Minimal solver matches theory within <0.3% and agrees with Dedalus.

## Impact
- Definitive, reviewer-safe close-out of the BOUT++ branch
- Establishes a validated control benchmark for SBIR documentation
- Provides a clean bridge to 2D Dedalus studies and to future NIMROD validation

## Next (not part of this PR)
- 2D slab Dedalus extension
- Eigenvalue formulation (no time-stepping)
- Mapping to NIMROD once access is granted
