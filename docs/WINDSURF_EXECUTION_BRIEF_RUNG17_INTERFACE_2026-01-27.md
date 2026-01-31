# Windsurf Execution Brief — Interface-Level Tests (Rung-17 Mode)

Project: Density-Stratified Continuity Ladder  
Status: Ladder Closed (Rung-1 → Rung-16)  
Mode: Interface-Level Execution Only  
Governance: Strict (No fitting / No tuning / No ontology)

---

## 1. Current State (Authoritative)

The following are already complete and committed:

- ✔ Ladder Closure
  - Rungs 1–16 passed under governance.
  - Classification result: A single, continuous, density-stratified late-time growth invariant.

- ✔ Authorized Execution Already Performed
  - Phase-21C direction-only test
  - Frozen kernel: `data/kernels/WL_PLANCK2018_fiducial_z.txt`
  - Output artifact: `output/summaries/phase21c_direction_only_execution.txt`
  - Result: Positive ΔA_L, all invariants preserved.

- ✔ Governance Status
  - No fitting
  - No tuning
  - No kernel modification
  - No likelihoods
  - No ontology committed

This state is locked.

---

## 2. Objective of the Next Execution Phase

We are not “doing more cosmology.” We are doing interface verification.

Goal: Demonstrate that the ladder closure is mathematically actionable by executing interface-level tests that:
- operate only on frozen inputs,
- return sign / ordering / boolean outputs,
- and can be falsified without model commitment.

This is the Rosetta-stone layer between GR observables and optional future microphysics.

---

## 3. What Is Being Executed (and What Is Not)

- ✅ Allowed
  - Linear operators
  - Ratios
  - Finite differences
  - Convolutions with frozen kernels
  - Rank ordering
  - Sign checks
  - GPU acceleration for throughput only

- ❌ Forbidden
  - Likelihoods
  - Optimizers
  - Parameter scans
  - Priors
  - Model fitting
  - New kernels
  - Any ontology declaration

If a test requires guessing → do not run it.

---

## 4. Execution Environment

- Hardware: DLW + eGPU
  - GPU used only for batching / convolution / slicing
  - No ML, no training, no inference

- Execution Host: Windsurf (authoritative executor)
  - GitHub = system of record

---

## 5. Directory Structure (to create if missing)

```
windsurf/
├── pipelines/
│   └── rung17_interface/
│       ├── sign_test.py
│       ├── kernel_support_test.py
│       ├── orthogonality_test.py
│       ├── ordering_test.py
│       └── __init__.py
├── io/
│   ├── load_frozen_kernel.py
│   ├── load_growth_data.py
│   └── hashes.py
├── governance/
│   └── guard.py
├── data/
│   └── frozen/
└── output/
    └── interface_tests/
```

---

## 6. Mandatory Governance Guard (Must Be Imported)

File: `governance/guard.py`

```python
FORBIDDEN = [
    "fit", "optimize", "likelihood",
    "prior", "scan", "mcmc", "train"
]

def assert_governance(kwargs=None):
    if kwargs is None:
        return
    for k in kwargs:
        for word in FORBIDDEN:
            if word in k.lower():
                raise RuntimeError("Governance violation detected")
```

Every pipeline must import and call this.

---

## 7. Executable Pipelines (Exact)

- Pipeline 1 — Sign Consistency Test
  - File: `sign_test.py`
  - Purpose: Confirm that the observed deviation has the same sign across probes and redshift bins.
  - Operation: Compute ΔO(z_high) − ΔO(z_low); report sign only.
  - Output (JSON):

```json
{
  "test": "sign_consistency",
  "expected": "positive",
  "observed": "positive",
  "result": "PASS"
}
```

- Pipeline 2 — Kernel Support Localization
  - File: `kernel_support_test.py`
  - Purpose: Confirm the signal peaks where the frozen lensing kernel has support.
  - Operation: Convolve residuals with `WL_PLANCK2018_fiducial_z.txt`; compare peak redshift locations.
  - Output (JSON):

```json
{
  "test": "kernel_support",
  "kernel_peak_z": 0.6,
  "signal_peak_z": 0.6,
  "aligned": true,
  "result": "PASS"
}
```

- Pipeline 3 — Scale Orthogonality Test
  - File: `orthogonality_test.py`
  - Purpose: Verify k-independence (density-stratified, not scale-driven).
  - Operation: ∂/∂k (ΔP/P) ≈ 0 via finite differences; threshold defined a priori.
  - Output (JSON):

```json
{
  "test": "k_orthogonality",
  "max_slope": 1.1e-4,
  "threshold": 1e-3,
  "result": "PASS"
}
```

- Pipeline 4 — Tomographic Ordering Test
  - File: `ordering_test.py`
  - Purpose: Confirm monotonic redshift stratification.
  - Operation: Rank ΔO(z) across bins; check ordering consistency.
  - Output (JSON):

```json
{
  "test": "redshift_ordering",
  "ordering": ["z0.2", "z0.5", "z0.8"],
  "monotonic": true,
  "result": "PASS"
}
```

---

## 8. Outputs to Commit

Each run produces:

```
output/interface_tests/
├── sign_consistency.json
├── kernel_support.json
├── k_orthogonality.json
├── redshift_ordering.json
└── interface_summary.md
```

Required Commit Message:

```
Interface-level execution: density-stratified invariant verification. No fitting. No tuning. Governance preserved.
```

---

## 9. Why This Is the Correct Next Step

- You are not guessing
- You are not unifying prematurely
- You are making the ladder executable
- You are shrinking theory space, not expanding it

After this:
- Any GR–QFT bridge must reproduce these interface facts
- Any failure reopens a specific rung
- Ontology remains optional, not assumed

---

## 10. Stop Statement

“Execute only the four interface pipelines above. Produce only the listed artifacts. Do not introduce models, fits, or new kernels. Governance preserved.”
