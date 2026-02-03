# Φ-Integrity: Projection / Embedding Integrity Layer

**A deterministic, model-agnostic integrity wrapper that guarantees invariant-safe outputs or a mathematically justified refusal.**

---

## 🎯 Product Definition (Locked)

**A projection-integrity and interface-constraint layer that sits around existing LLMs and guarantees invariant-safe outputs or a mathematically justified refusal.**

---

## 🚫 Out of Scope (Explicit)

❌ Training new models  
❌ AGI claims  
❌ Physics claims  
❌ Compression benchmarks as a product  

---

## 🏗️ Architecture Overview

```
Prompt P + Domain D + Reference R
   ↓
Raw byte ingestion (no preprocessing magic)
   ↓
Φ projection (fixed dimensional, locked)
   ↓
Constraint evaluation
   ↓
Interface check
   ↓
Decision: ALLOW | REFUSE (with trace)
```

---

## 📋 Φ-Integrity Contract (Locked)

### Inputs
- `prompt: str` - User input
- `domain: str` - Explicit domain (required)
- `reference_data: optional` - Domain-specific reference
- `run_id: deterministic hash` - Reproducibility identifier

### Outputs (Exhaustive)
```json
{
  "status": "ALLOW | REFUSE",
  "output": "... | null",
  "reason": "invariant_violation | instability | ambiguity",
  "trace": {
    "phi_hash": "...",
    "metrics": {...},
    "constraints_checked": [...]
  }
}
```

**There is no silent failure mode.**

---

## 🎯 Primary Domain (Locked)

**Accounting / Numeric Reasoning**

### Core Invariants (Non-negotiable)
- Arithmetic closure
- Conservation of totals  
- Ledger balance
- Unit consistency
- Deterministic replay

### Design Principle
If the model is unsure, it must refuse — not approximate.

---

## 🤖 Model Selection (Locked)

**Phi-2 / Phi-3 class (≤7B parameters)**

### Rationale
- Small, fast, local
- Deterministic enough for auditing
- Symbolically capable
- No fine-tuning required
- Ironically aligned with Φ framing

**Important Clarification:** The model is replaceable. The Φ-Integrity layer is not.

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/your-username/phi-integrity-fork-a.git
cd phi-integrity-fork-a

# Run accounting demo
python examples/accounting_demo.py

# See refusal-first reliability in action
```

---

## 📦 Components

### Core Modules
- `src/projection.py` - Fixed-dimensional Φ projection
- `src/constraints.py` - Domain-specific constraint engine  
- `src/interfaces.py` - Interface compliance checker
- `src/wrapper.py` - Model-agnostic wrapper

### Examples
- `examples/accounting_demo.py` - Demonstrates refusal-first reliability

---

## 🎯 Demonstration Results

### What to Expect
✅ **Correct arithmetic → allowed**  
✅ **Incorrect arithmetic → refused**  
✅ **Ambiguous prompt → refused**  
✅ **Out-of-domain request → refused**

### Required Evidence
- ✅ Logs with full traceability
- ✅ Hashes for reproducibility
- ✅ Replayable runs
- ✅ No parameter tuning between runs

**If it refuses everything, that's fine initially — it proves honesty.**

---

## 📊 Evaluation Metrics (Locked)

We are **NOT** optimizing for:
- Accuracy
- Fluency  
- Creativity

We are optimizing for:
- ✅ **Integrity**
- ✅ **Determinism**
- ✅ **Reproducibility**
- ✅ **Correct refusal**

---

## 🛡️ Why This Works

**Low cost:** Runs on consumer hardware  
**Code-based:** Fast iteration  
**Immediately useful:** Accounting, compliance, audits  
**Extensible:** Physics, fusion, simulation later  
**Credible:** Refusal beats wrong answers

---

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
- [THEORY.md](THEORY.md) - Fork B summary (frozen)
- [examples/](examples/) - Working demonstrations

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Key Rule:** Nothing hidden. Nothing tuned. Nothing magical.

---

*This is how you go from "interesting theory" to "this changes how systems are built."*
