# FMI Depth Interface

**A deterministic, parameter-free mathematical interface for depth-stratified physical systems, with CPU/GPU parity and frozen reference outputs.**

---

## What this is

- **Deterministic mathematical interface**
- **No tunable parameters**
- **Depth coordinate Φ**
- **Ladder closure**

## What this is not

- Not a field theory
- Not a numerical fit
- Not ML
- Not speculative cosmology (yet)

## Executable proof

```bash
python execution/run_cpu.py
python execution/run_gpu.py
python execution/validate_parity.py
```

## Frozen reference

- SHA256 hashes
- FMI v0.1 declaration
- Immutable reference outputs

## Why it matters

- CPU/GPU parity
- Parameter-free
- Cross-domain applicability

---

## Quick Start

1. Clone and install dependencies
2. Run the executable proof scripts above
3. Validate parity with hash verification

## Structure

```
fmi-depth-interface/
├── interface/     # Core mathematical interface
├── execution/     # CPU/GPU execution scripts
├── data/         # Reference outputs and hashes
├── tests/        # Determinism and closure tests
└── docs/         # Interface mathematics and contracts
```

## License

MIT License - see LICENSE file

## Citation

See CITATION.cff for proper citation format.
