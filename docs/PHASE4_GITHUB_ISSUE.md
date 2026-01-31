# GitHub Issue Template — Phase-4 Nonlinear Mode Coupling and Saturation (2D Slab Alfvén)

Title: Phase-4: Nonlinear Mode Coupling and Saturation in 2D Slab Alfvén System
Labels: enhancement, validation, phase-4, nonlinear
Milestone: Phase-4 Nonlinear Validation (see milestone draft below)
Assignees: (assign)

## Summary
Extend the validated linear/weakly-coupled slab Alfvén model into the nonlinear regime to measure ε² frequency corrections, characterize energy exchange between polarizations, and assess saturation pathways under strict conservation diagnostics. Phase-1–3 are FROZEN; this work must not modify any frozen artifacts.

Refs:
- docs/PHASE4_IMPLEMENTATION_CHECKLIST.md
- docs/BOUTPP_CLOSEOUT.md (Phases 1–3)

## Scope (Hard Constraints)
- Quadratic nonlinearities only ((v·∇)v and ∇×(v×b)); energy-conserving forms
- 2D slab (x periodic Fourier, z Chebyshev line-tied for v)
- Parameters fixed to Phase-3B values unless specified: Lx=Lz=128, B0=1, τ=1, (ρ⊥,ρ∥)=(20,5)
- ε ∈ {1e-3, 3e-3, 1e-2}; η ∈ {0, 1e-3}
- No changes to frozen files from Phases 1–3

Out of Scope: 3D geometry, turbulence/forcing, dissipative nonlinear closures, background gradients/mean-field evolution.

## Acceptance Criteria
- Linear Recovery: As ε→0, frequencies and energies match Phase-3B within tolerance
- Energy Conservation: η=0, relative drift max(E)−min(E) < 1e-8 E0 over ≥1000 Alfvén times
- Nonlinear Frequency Shift: Fit ω(ε)=ω0+αε²; α finite and stable under refinement
- Mode Energy Transfer: Bounded exchange or saturation; no secular growth
- Resolution Robustness: 128² vs 256² parity within tolerance

## Deliverables
- New NL IVP example(s)
- Shared modal diagnostics utility
- CSV logs and minimal plots under analysis/phase4_runs/
- Documentation: Phase-4 section in docs/BOUTPP_CLOSEOUT.md with concise tables and explicit PASS/FAIL

## Tasks
- [ ] Scaffolding: new NL IVP script (examples/dedalus_alfven_2d_nl_ivp.py) with parser, dealiased ops, energy integrals, CSV logging
- [ ] Operators: implement dealiased (v·∇)v and ∇×(v×b) terms; verify η=0 linear limit recovers Phase-3B
- [ ] Diagnostics: modal projections, Hilbert-phase slope, envelopes; unit tests on manufactured data
- [ ] Runs: execute test matrix (ε, η, resolution), store CSVs; generate minimal plots
- [ ] Acceptance: check criteria; append Phase-4 results to docs; prepare PASS/FAIL statement
- [ ] Release: tag v0.2-nonlinear-init upon PASS; freeze Phase-4

## Risks & Mitigations
- Aliasing instability → strict 3/2 dealiasing; confirm with η=0 energy test
- Spurious damping → cross-check against η=0 controls; adjust dt
- False chaos → resolution sweep; verify exponential modal spectra tails

---

# Milestone Draft — Phase-4 Nonlinear Validation

Milestone: Phase-4 Nonlinear Validation
Due date: (set)
Labels: phase-4, milestone

### Milestone Description
This milestone extends the slab Alfvén control benchmark into the nonlinear regime with energy-conserving quadratic terms. Success criteria are conservation at η=0, ε² scaling of frequency detuning, bounded/symmetric modal energy exchange, and robust recovery of Phase-3B in the ε→0 limit.

### Entry Conditions
- Phases 1–3 are CLOSED and FROZEN (tag v0.1-control-parity)
- Checklist available: docs/PHASE4_IMPLEMENTATION_CHECKLIST.md

### Exit Conditions (Freeze Criteria)
- All Acceptance Criteria in the Issue are met
- No regressions vs Phase-3 results
- Results reproducible at 256²
- Documentation updated and tag v0.2-nonlinear-init pushed

### Linked Artifacts
- Issue: Phase-4 Nonlinear Mode Coupling and Saturation
- Checklist: docs/PHASE4_IMPLEMENTATION_CHECKLIST.md
- Reference: docs/BOUTPP_CLOSEOUT.md (Phases 1–3)
