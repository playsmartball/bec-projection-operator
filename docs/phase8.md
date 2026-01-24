# Phase-8 Summary (Absorbing Robin BCs + Frequency-Selective Walls)

Status: Phase-8A and Phase-8B validated and closed.

- Phase-8A: Constant-κ absorbing Robin BCs (characteristic hyperbolic BCs + Robin diffusion)
- Phase-8B: Frequency-selective low-pass absorber κ(ω) via auxiliary boundary states

## Invariants (held in all runs)
- W_tau = 0
- max|Tau_*| = 0
- ΔE/E ≈ ΔE/E_weak (within ~1e-6–1e-5 across sweeps; ~1e-10 in smoke)
- P_eta,total < 0 (no energy injection)
- dominance = coupling

## Implementation
- File: examples/dedalus_alfven_2d_nl_ivp.py
- New CLI flags:
  - --kappa_model {constant, lowpass}
  - --omega_c <float>
  - --forcing {bulk, edge}
- Low-pass boundary model (boundary ODEs):
  - dt(s) - omega_c*kappa*b + omega_c*s = 0 at z=left/right
  - -∂z b(z=left) + s_left = 0;  ∂z b(z=right) + s_right = 0
- Edge forcing envelope (option): env(z) = cos(π z/Lz)^2 applied to initial pattern

## Smoke tests (tmax=2)
- constant κ: ΔE/E ≈ 5.50e-10; P_eta,total ≈ -1.53e-19
- low-pass κ(ω=0.5): ΔE/E ≈ 4.81e-10; P_eta,total ≈ -5.60e-20

## Phase-8B Sweep (12 runs)
- Fixed: bc=characteristic, nl=1, eta=1e-3, dt=1e-3, tmax=20, Nx=128, Nz=129, m=1, n=1, eps=0, amp=1e-6
- Sweep:
  - κ0 ∈ {5e-4, 2e-3}
  - ωc ∈ {0.25, 0.5, 1.0} × ω0 (ω0 measured via Hilbert)
  - forcing ∈ {bulk, edge}
- Results summary: analysis/phase8_runs/p8b_summary.csv
- Key findings:
  - For fixed κ0, decreasing ωc monotonically reduces |P_eta,total|
  - |P_eta,total| scales ~linearly with κ0
  - Edge forcing shows slightly stronger ω-dependence than bulk once quantified by spectral fraction above ωc

## Reproduction

Environment (WSL recommended; OMP pinned):
```
export OMP_NUM_THREADS=1
export PYTHONPATH=/mnt/e/CascadeProjects/bec-projection-operator-git
PY="$HOME/.micromamba/envs/dedalus/bin/python"; [ -x "$PY" ] || PY=python3
```

Smoke (constant):
```
$PY -m examples.dedalus_alfven_2d_nl_ivp --bc characteristic --nl 1 --eta 1e-3 \
  --kappa 5e-4 --kappa_model constant --omega_c 0 \
  --dt 1e-3 --tmax 2 --Lx 128 --Nx 128 --Lz 128 --Nz 129 --m 1 --n 1 --amp 1e-6 --eps 0 \
  --csv analysis/phase8_runs/p8b_const_smoke.csv
```

Smoke (low-pass):
```
$PY -m examples.dedalus_alfven_2d_nl_ivp --bc characteristic --nl 1 --eta 1e-3 \
  --kappa 5e-4 --kappa_model lowpass --omega_c 0.5 \
  --dt 1e-3 --tmax 2 --Lx 128 --Nx 128 --Lz 128 --Nz 129 --m 1 --n 1 --amp 1e-6 --eps 0 \
  --csv analysis/phase8_runs/p8b_lowpass_smoke.csv
```

Sweep (12 runs):
```
bash analysis/phase8b_sweep.sh | tee analysis/phase8_runs/phase8b_sweep.log
```
Live watcher (optional):
```
$PY analysis/phase8b_watch_log.py | tee analysis/phase8_runs/p8b_watch.log
```

Artifacts:
- analysis/phase8_runs/phase8b_sweep.log
- analysis/phase8_runs/p8b_summary.csv
- analysis/phase8_runs/p8b_*.csv (per-case outputs)

## Interpretation checklist
- Invariants satisfied in all cases
- |P_eta,total| decreases with ωc for fixed κ0
- Edge forcing shows stronger ω-selectivity than bulk
- Spectral fraction above ωc ~O(5e-2) at 0.25·ω0 → ~O(1e-5) at 1.0·ω0

## Provenance
- Tag this commit as: phase-8b-validated
- Cite this tag for external requests (e.g., NIMROD access/validation)
