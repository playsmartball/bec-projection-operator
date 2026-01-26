# Phase-21 Reframing and Rung-4 Bounding (Governance-Only)

Status: DESIGN-ONLY — NO EXECUTION — NO NUMERICS — NO FITS — NO KERNEL EDITS — NO COMPENSATIONS

This document performs a structural reframing of Phase-21 conclusions under a “Rung‑3 sandwich” and defines, formally, how Rung‑4 acts as a bounding (non‑activated) layer. It introduces no code, parameters, or execution logic.

---

## 1) Phase‑21 Re‑Framed Under a Rung‑3 Sandwich

### Original Phase‑21 (Single‑Sided Framing)

Phase‑21 evaluated Rung‑3 primarily against Rung‑2 (from below):

- **Rung‑2 locks**
  - σ8(z=0) containment
  - Low‑k power normalization
- **Global invariants**
  - BAO phase integrity
  - Ladder separability
  - No compensation via EARLY, PROJ, or PERT

Within that framing, Phase‑21 classified Rung‑3 mechanisms as:

- **Admitted**: GROWTH(z) — homogeneous, k‑independent timing (g(0)=1; σ8‑preserving)
- **Excluded/Blocked**: GEO — background geometry modifications

This was logically correct but incomplete, because Rung‑3 was treated as an open layer bounded only from below ("do not touch Rung‑4" implicitly), rather than as a contained interface.

### Sandwich Framing: Rung‑3 Between Rung‑2 and Rung‑4

Under a sandwich methodology (mirroring earlier σ8 treatment), Rung‑3 is bounded on both sides:

- **Below (Rung‑2)**: Fixes normalization/amplitude/low‑k; forbids growth rescaling at z=0.
- **Above (Rung‑4)**: Fixes how any Rung‑3 modification manifests observationally; forbids reshaping, reweighting, or reinterpretation of the response.

Therefore, Rung‑3 must satisfy both:

- **Internal consistency** with Rung‑2 (σ8 containment, low‑k invariants)
- **External consistency** through an immutable Rung‑4 response operator

This tightens Phase‑21 conclusions by requiring transmission through a fixed projection layer, not merely avoidance of violations at the source.

### Re‑Classification Under the Sandwich

- **A. GROWTH(z) — Homogeneous Timing**
  - Under sandwich constraints, timing shifts:
    - Change the epoch distribution of structure formation
    - Do not change final amplitude at z=0 (σ8 preserved by g(0)=1)
  - With fixed Rung‑4 kernels, the lensing amplitude responds to the line‑of‑sight integral of growth history; no kernel deformation is required to "see" the effect.
  - **Result**: GROWTH(z) remains ADMITTED, now with stronger justification: it is admissible because it survives projection through an immutable response operator, not only because it avoids σ8/low‑k violations.

- **B. GEO — Background/Geometry**
  - Geometry changes alter distance–redshift mappings and implicitly modify the lensing efficiency kernel.
  - With Rung‑4 frozen, such implicit kernel shifts cannot be absorbed, corrected, or reinterpreted.
  - **Result**: GEO is not merely "blocked under current invariants"; it is structurally incompatible with ladder separability when Rung‑4 is treated as a fixed operator. Hence, GEO fails the sandwich test.

### Refined Phase‑21 Conclusion (Sandwich Form)

When Rung‑3 is bounded below by σ8/low‑k containment (Rung‑2) and above by immutable projection response (Rung‑4), the only coherent Rung‑3 mechanism is homogeneous growth timing. Geometric background modifications fail for structural reasons under the sandwich constraints.

---

## 2) Formal Definition: Rung‑4 as a Bounding (Non‑Activated) Layer

### Definition (Bounding, Not Activated)

Rung‑4 acts as a fixed linear response operator that:

- **Accepts** Rung‑3 outputs (e.g., growth history) and **produces** observables
- **Does not** change shape, weighting, or interpretation
- **Is not** parameterized, tuned, rescaled, decomposed, or otherwise adjusted

Symbolically (conceptual only): Rung‑3 provides a spacetime field evolution; Rung‑4 applies a fixed functional mapping to observables. No degrees of freedom exist at Rung‑4 in governance mode.

This mirrors the earlier σ8 discipline: σ8 was not adjusted; it was checked.

### Allowed vs Forbidden Uses of Rung‑4

- **Allowed (Bounding/Checks)**
  - Determine whether a Rung‑3 change is observable at all
  - Check coherence/monotonicity/sign of response under fixed projection
  - Verify no hidden degeneracy is being exploited

- **Forbidden (Activation/Assistance)**
  - Kernel reshaping or redshift reweighting
  - Lensing‑efficiency tuning or projection‑space compensations
  - "Explaining away" effects via response modification

Governance principle: Rung‑4 may veto; it may not assist.

### Why This Matters for Execution Discipline

Fixing Rung‑4 eliminates the risk that a Rung‑3 effect "works" only because projection freedom was implicitly assumed. Any observed sensitivity (direction‑only if later authorized) must be due to Rung‑3 physics, not interpretive latitude at higher rungs.

---

## Enablement for WindSurf Execution (Statement of Scope Only)

- The sandwich reframing isolates **GROWTH(z)** as the sole coherent Rung‑3 transmission channel under current invariants and a fixed Rung‑4 operator.
- This enables, if and only if explicitly authorized, the previously specified option:
  - "Authorize minimal Phase‑21C execution under protocol (direction‑only A_L probe; no fits)."
- No parameters, numerics, or forward optimization are specified here. This is solely a structural readiness statement under governance locks.

---

## Stop Statement

“Reframing complete. Rung‑3 sandwich and Rung‑4 bounding defined. No execution performed. Governance preserved. Awaiting explicit authorization.”
