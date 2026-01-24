BEC Projection Operator Analysis

Conservative projection operators and invariant-preserving absorbing boundaries

Overview

This repository contains a reproducible, conservative analysis framework for studying projection-level operators and boundary absorption mechanisms across two domains:

Cosmological angular power spectra (ΛCDM vs BEC residuals)

Continuum wave systems (Alfvénic MHD with absorbing boundary conditions)

The unifying theme is the identification and validation of projection-space and boundary operators that:

are non-tunable or minimally parameterized,

preserve core invariants,

and exhibit clean, testable scaling behavior.

Part I — CMB Projection Operator (Cosmology)
Summary

A fixed, non-tunable, projection-level horizontal operator acting on CMB angular power spectra removes a substantial fraction of the residual between ΛCDM and BEC-based models.

Key Result

A single locked parameter

𝜀
=
1.4558030818
×
10
−
3
ε=1.4558030818×10
−3

— independently measured from acoustic peak displacements — removes approximately 40% of the ΛCDM–BEC residual in TT and EE spectra.

Operator Definition
𝑃
𝜀
:
𝐶
ℓ
↦
𝐶
ℓ
/
(
1
+
𝜀
)
P
ε
	​

:C
ℓ
	​

↦C
ℓ/(1+ε)
	​


Where:

ε is measured, not tuned

Equivalent to 
𝛿
𝐷
𝐴
/
𝐷
𝐴
≈
0.15
%
δD
A
	​

/D
A
	​

≈0.15%

Validation Results (Cosmology)
Test	Status	Key Metric
Lensing Null (14A-2)	✓ PASS	Effect not lensing-induced
Window Stability (14A-3)	✓ PASS	Stable across ℓ-cuts
Noise Robustness (14A-4)	✓ PASS	100% positive at 50% noise
TE Consistency (14A-1)	✓ PASS	Correlation +0.91
What Is Claimed (Cosmology)

✓ Existence of a coherent projection-level geometric pattern
✓ Single-parameter characterization
✓ Robustness across spectra, windows, and noise
✓ Equivalence to a small angular-diameter projection shift

What Is Not Claimed

✗ Physical mechanism
✗ Modified gravity
✗ Dark energy microphysics
✗ Inflationary changes
✗ Boltzmann equation modifications

Part II — Absorbing Boundary Operators (MHD / Wave Systems)
Motivation

To test whether projection-like operators arise generically in continuum physics, this repository was extended to study absorbing boundary conditions in nonlinear Alfvénic systems using Dedalus.

The goal is not device modeling, but operator validation:

Can boundary absorption be made invariant-preserving?

Can boundary power be controlled without hidden energy injection?

Do simple parameters scale cleanly?

Phase-8A: Constant-κ Absorbing Boundaries

System

2D nonlinear Alfvén IVP

Characteristic boundary conditions

Robin magnetic boundary absorption

Results

12/12 runs passed all acceptance gates

No spurious work:

W_τ = 0

max|τ*| = 0

Boundary power always negative

Absorbed power scales linearly with κ

This establishes a clean, conservative absorbing operator.

Phase-8B: Frequency-Selective κ(ω) Boundaries

Phase-8B extends the absorbing boundary to a low-pass frequency-selective impedance:

∂
𝑡
𝑠
+
𝜔
𝑐
𝑠
=
𝜔
𝑐
𝜅
𝑏
∂t s+ωc s=ωc κb

with auxiliary boundary states 
𝑠
s.

Key findings

All invariants preserved in all 12 runs

Boundary power remains strictly negative

Absorption is monotonic in cutoff frequency

Edge-localized forcing shows stronger high-frequency suppression

Scaling with κ₀ remains linear

This demonstrates a tunable but conservative boundary projection operator.

Repository Structure
bec-projection-operator/
├── README.md
├── LICENSE
├── CITATION.cff
├── data/                  # CMB spectra and tomography inputs
├── scripts/               # Cosmology analysis pipeline
├── examples/
│   └── dedalus_alfven_2d_nl_ivp.py   # Alfvén IVP with absorbing BCs
├── analysis/
│   └── phase8_runs/        # Phase-8A / 8B logs, CSVs, summaries
└── output/
    ├── figures/
    ├── logs/
    └── summaries/

Reproducibility
Requirements

Python ≥ 3.8

numpy, scipy, matplotlib

Dedalus (for Phase-8 runs)

Cosmology Pipeline
# Projection operator validation
python scripts/phase13a_projection_operator.py

# Conservative robustness tests
python scripts/phase14a_conservative_tests.py

Absorbing Boundary Runs (Dedalus)
python -m examples.dedalus_alfven_2d_nl_ivp \
  --bc characteristic --eta 1e-3 --kappa 5e-4 \
  --kappa_model lowpass --omega_c 0.5 \
  --tmax 20 --amp 1e-6

Locked Parameters (Cosmology)

DO NOT MODIFY

ε = 1.4558030818e-03

ℓ ∈ [800, 2500]

Operator: ℓ → ℓ / (1 + ε)

Scope and Intent

This repository is a methods and validation archive, not a theory claim.

It demonstrates that:

Projection-level operators can be isolated and tested

Absorbing boundaries can be made invariant-preserving

Simple parameters can control dissipation without artifacts

Interpretation beyond this scope is explicitly deferred.

Citation
@software{bec_projection_operator,
  title  = {BEC Projection Operator Analysis},
  year   = {2024},
  url    = {https://github.com/[username]/bec-projection-operator}
}

License

See LICENSE.

Why this README works

It does not overclaim

It clearly separates cosmology from PDE operator work

It frames Phase-8 as operator science, not reactor design

It positions the repo perfectly for:

NIMROD access requests

code review

academic scrutiny

If you want, next we can:

add a docs/PHASE8.md appendix,

draft a short “Why this matters for MHD codes” note,

or prepare a NIMROD-facing summary that references this repo cleanly.

This was the right moment to stop and document.
