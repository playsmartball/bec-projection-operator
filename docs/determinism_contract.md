# Determinism Contract

**FMI v0.1 - Immutable Reference Standard**

---

## Core Principles

1. **Identical inputs → identical outputs**
2. **CPU/GPU allowed different floating paths only if hash-stable**
3. **Reference outputs are immutable**
4. **Any deviation increments FMI version**

---

## Contract Terms

### Input Determinism
- All random seeds are fixed
- Initial conditions are exactly specified
- No external dependencies on system state

### Output Determinism
- SHA256 hashes of all outputs are computed
- CPU and GPU outputs must match within numerical tolerance
- Reference traces are stored in `data/reference/`

### Version Control
- FMI version increments if any reference output changes
- All changes must be traceable to specific mathematical modifications
- No silent modifications allowed

---

## Validation Protocol

```python
python execution/validate_parity.py
```

This script:
- Computes SHA256 hashes of all outputs
- Compares against reference hashes
- Reports any deviations
- Fails fast on non-deterministic behavior

---

## Reference Data

- `data/reference/phi_trace.json` - Canonical Φ response trace
- `data/reference/phi_trace.sha256` - Immutable hash reference
- `data/outputs/` - Runtime outputs (not tracked in version control)

---

## Compliance Requirements

Any modification to the interface must:
1. Preserve deterministic behavior
2. Update reference hashes if needed
3. Increment FMI version
4. Pass all validation tests

---

**This contract is enforced by the test suite and CI pipeline.**
