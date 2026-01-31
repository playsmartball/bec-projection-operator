# Phase-4 Implementation Checklist — Nonlinear Mode Coupling and Saturation (2D Slab Alfvén)

Status: DRAFT (Not Implemented)
Scope: Code-level plan aligned with the Phase-4 Design Specification
Prereq: Phase-1–3 FROZEN (no edits to frozen files)

## Guardrails
- Do not modify frozen controls (Phase-1–3 scripts and docs).
- Implement Phase-4 in new example files and new doc sections only.
- Keep linear recovery tests in place (ε → 0) for every nonlinear run.

## Files to Add (proposed)
- examples/dedalus_alfven_2d_nl_ivp.py
- examples/utils/modal_projections.py (shared diagnostics)
- analysis/phase4_runs/ (CSV logs and plots)

## Model Formulation
- Equations (η ≥ 0):
  - dt(v) + (v·∇)v = (B0·∇)b + ε C[v,b]
  - dt(b) = (B0·∇)v + ∇×(v×b) + η ∇² b
- Geometry: 2D slab (x periodic, z Chebyshev line-tied for v), B0 = ẑ B0, τ = 1.
- Inertia: (ρ⊥, ρ∥) = (20, 5).
- Coupling: begin with ε from Phase-3B; nonlinear terms are quadratic and energy-conserving.

Phase-4 nonlinear runs employ a reduced 2D closure with vz ≡ 0; this preserves transverse Alfvén dynamics and energy symmetry while enabling controlled quadratic nonlinearity.

## Discretization
- Dedalus IVP, RK443 timestepper.
- Basis: Fourier in x, Chebyshev in z with tau for line-tied v.
- Dealiasing: 3/2-rule in both x and z (configure dealias in Dedalus builders, or pad/unpad operators).
- Time step: CFL with safety factor; start from linear stable dt and reduce if needed.

## Diagnostics
- Energy (η=0):
  - E = 1/2 ∫ [(1+ρ⊥)|vx|² + (1+ρ∥)|vy|² + |b|²/τ] dz (Dedalus Integrate; Chebyshev quadrature).
  - Acceptance: relative drift < 1e-8 over ≥ 1000 Alfvén times.
- Modal energies E_mn(t):
  - Compute Fourier coefficients in x and projections in z onto sin/cos(nπz/Lz).
  - Track primary (1,1) and secondary (2,1), (1,2), (2,2) modes.
- Frequency shift:
  - Hilbert-phase slope on dominant modal time series.
  - Fit ω(ε) = ω0 + α ε²; report α with resolution convergence.
- Exchange symmetry:
  - Compare envelopes of Ex and Ey; report symmetry metric as in Phase-3B.

## Test Matrix
- ε ∈ {1e-3, 3e-3, 1e-2}
- η ∈ {0, 1e-3}
- Resolutions: 128² and 256² (increase Nz to ensure BC accuracy under dealiasing).
- Duration: ≥ 1000 Alfvén times (based on ω0 for n=1).

## Acceptance Criteria
- Linear recovery: as ε → 0, frequencies and energies match Phase-3B.
- Energy conservation: η=0, max(E)−min(E) < 1e-8 E0.
- Nonlinear frequency shift: α finite and stable under refinement.
- Mode energy transfer: bounded oscillatory exchange or saturation; no secular growth.
- Resolution robustness: results reproducible at 256² within tolerance.

## Runbook (Templates)
- Baseline (η=0, ε=1e-3, 128²):
  - python examples/dedalus_alfven_2d_nl_ivp.py --Lx 128 --Lz 128 --Nx 128 --Nz 129 --B0 1 --tau 1 \
    --rho_perp 20 --rho_par 5 --eps 1e-3 --m 1 --n 1 --tmax <T> --dt <dt> --dealias 1.5 \
    --log analysis/phase4_runs/nl_eta0_eps1e-3_128.csv
- Refinement (256²) and η=1e-3 variants: replicate with adjusted Nx/Nz/tmax.

## Task Breakdown
1) Scaffolding
- Create new NL IVP script with argument parser and dealiased builders.
- Implement energy integrals and CSV logging hooks.

2) Operators
- Implement (v·∇)v and ∇×(v×b) with dealiased spectral products.
- Validate that η=0 linear limit reproduces Phase-3B.

3) Diagnostics
- Modal projection utilities; Hilbert-phase slope; envelope extraction.
- Unit tests for projections on manufactured data.

4) Runs & Plots
- Execute test matrix; store CSV; generate plots for E(t), E_x/E_y, modal energies, ω(ε) fit.

5) Acceptance & Freeze
- Check criteria; document results in docs/BOUTPP_CLOSEOUT.md Phase-4 section.
- Tag release v0.2-nonlinear-init upon PASS; freeze Phase-4.

## Risks & Mitigations
- Aliasing: strict 3/2 dealiasing; confirm with energy test.
- Spurious damping: cross-check with η=0 controls; adjust dt.
- False chaos: resolution sweep; modal spectra check for exponential tails.

## Linear Recovery Gate (Must Pass Early)
- With ε → 0, disable nonlinear terms and compare directly to Phase-3B EVP/IVP.
- Block progression until recovery is exact within tolerance.
