# Formal Mathematical Interface (FMI) v0.1 — Phi Kernel

Deterministic math kernel implementing Path C: a formal, testable interface that maps a single global depth coordinate Φ to a state vector S(Φ) and a stability functional L(Φ), with no free parameters.

- Units: Planck-normalized (c = ħ = G = k_B = 1)
- Inputs: Φ ∈ ℝ⁺ only
- State: S(Φ) = [ρ(Φ), P(Φ), c_s(Φ), λ_c(Φ), ε(Φ)]
- Operators: D_Φ (finite differences), L(Φ) = ρ(Φ) − D_Φ²P(Φ)
- Phase transition: L(Φ) < 0 ⇒ instability/collapse
- Determinism: CPU == GPU within tolerance; no fitting/tuning; no hidden degrees of freedom

## Usage

WSL (CuPy GPU preferred, CPU fallback is deterministic):

```
# CPU
~/.micromamba/envs/dedalus/bin/python -m bec_phi_kernel.cli --phi 0.0 10.0 --resolution 10000 --backend cpu

# GPU (CuPy)
~/.micromamba/envs/dedalus/bin/python -m bec_phi_kernel.cli --phi 0.0 10.0 --resolution 10000 --backend gpu
```

Outputs are written to `bec_phi_kernel/out/`:

- `phi_trace.csv` — numeric trace
- `phi_trace.json` — metadata and L<0 intervals
- `hash.txt` — SHA256 of the CSV (determinism check)

## Tests

```
~/.micromamba/envs/dedalus/bin/python -m pip install -r bec_phi_kernel/requirements.txt
~/.micromamba/envs/dedalus/bin/python -m pytest -q bec_phi_kernel/tests
```

- `test_determinism.py` — repeatability and CPU/GPU parity (if GPU available)
- `test_no_free_params.py` — ensures no tunable parameters exist
- `test_phase_boundaries.py` — verifies an L<0 interval exists in [0, 10]
