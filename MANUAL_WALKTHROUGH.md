# Φ-Integrity Manual Walkthrough

## Purpose

This document provides a step-by-step manual walkthrough of the Φ-Integrity pipeline, connecting each implementation step to the mathematical foundations.

---

## Mathematical → Implementation Mapping

| Mathematical Object | Implementation | File |
|-------------------|----------------|------|
| Information space $X$ | Raw input strings | `wrapper.py` |
| Constraint sets $\mathcal{C}$ | `AccountingConstraints` | `constraints.py` |
| Interfaces $I_i$ | `InterfaceChecker` | `interfaces.py` |
| Projection $\Phi$ | `PhiProjection` | `projection.py` |
| Decision $D(x)$ | `PhiIntegrityWrapper` | `wrapper.py` |

---

## Step-by-Step Walkthrough

### Step 1: Raw Input Ingestion ($X$)

**Mathematical**: Input $x \in X$ (unconstrained information space)

**Implementation**:
```python
# In wrapper.py
prompt = "Calculate 2+2"
domain = "accounting"
reference_data = {"expected_sum": 4.0}
```

**Verification**: Input is raw text, no preprocessing, exactly as specified.

---

### Step 2: Interface Validation ($I_i$)

**Mathematical**: Check if input can pass through interface $I_i = C_i \cap C_{i+1}$

**Implementation**:
```python
# In interfaces.py
interface_result = validate_interface(prompt, domain, reference_data)
```

**Mathematical Correspondence**:
- Domain validation: $domain \in \text{valid\_domains}$
- Prompt validation: $prompt \in \text{admissible\_inputs}$
- Reference validation: $reference\_data \in \text{valid\_structures}$

**Manual Check**:
```python
>>> validate_interface("Calculate 2+2", "accounting", {"expected_sum": 4.0})
InterfaceResult(status=InterfaceStatus.COMPLIANT, ...)
```

---

### Step 3: Model Generation

**Mathematical**: Get raw model response $m \in X$ (still unconstrained)

**Implementation**:
```python
# In wrapper.py
model_output = self.model.generate(prompt)
# Returns: "The answer is 4."
```

**Key Point**: Model output is NOT yet validated - it's raw information space content.

---

### Step 4: Φ Projection ($\Phi: X \rightarrow \mathbb{R}_\Phi$)

**Mathematical**: Apply quotient map $\Phi$ to combined input

**Implementation**:
```python
# In projection.py
combined_input = f"{prompt}|{model_output}"
phi_projection, metrics, phi_hash = project_to_phi(combined_input)
```

**Mathematical Properties Verified**:
- **Deterministic**: Same input → same $\Phi$ hash
- **Fixed-dimensional**: Always 5000 dimensions
- **Parameter-locked**: No tuning possible

**Manual Verification**:
```python
>>> project_to_phi("Calculate 2+2|The answer is 4.")
(array([...]), ProjectionMetrics(...), "4e7947367172c794")
```

---

### Step 5: Constraint Evaluation ($\mathcal{C}$)

**Mathematical**: Check if $\Phi(x) \in \bigcap_i C_i$

**Implementation**:
```python
# In constraints.py
constraint_result = evaluate_accounting_constraints(prompt, model_output, reference_data)
```

**Mathematical Correspondence**:
- $C_1$: Arithmetic closure invariant
- $C_2$: Conservation of totals invariant  
- $C_3$: Ledger balance invariant
- $C_4$: Unit consistency invariant

**Manual Check**:
```python
>>> evaluate_accounting_constraints(
...     "Calculate 2+2", "The answer is 4.", {"expected_sum": 4.0}
... )
ConstraintResult(overall_status=ConstraintStatus.PASS, ...)
```

---

### Step 6: Decision Function ($D(x)$)

**Mathematical**: Apply decision function $D(x)$

**Implementation**:
```python
# In wrapper.py
if constraint_result.overall_status.value == "pass":
    response = format_allow_response(...)
    self.allowed_responses += 1
else:
    response = format_refuse_response(...)
    self.refused_responses += 1
```

**Mathematical Logic**:
$$D(x) = \begin{cases} 
\text{ALLOW} & x \in \bigcap_i C_i \\
\text{REFUSE} & \text{otherwise}
\end{cases}$$

---

### Step 7: Response Formatting

**Mathematical**: Return structured response with full trace

**Implementation**:
```python
# In interfaces.py
{
  "status": "ALLOW | REFUSE",
  "output": "... | null",
  "reason": "invariant_violation | instability | ambiguity",
  "trace": {
    "phi_hash": "...",
    "metrics": {...},
    "constraints_checked": [...],
    "violations": [...],
    "run_id": "..."
  }
}
```

**Mathematical Completeness**: Every decision includes:
- $\Phi$ hash (projection identifier)
- All constraint evaluations
- Violation reasons (if any)
- Deterministic run ID

---

## Complete Walkthrough Example

### Input: Correct Arithmetic

```python
# Step 1: Raw input
prompt = "Calculate 2+2"
domain = "accounting"  
reference_data = {"expected_sum": 4.0}

# Step 2: Interface validation
interface_result = validate_interface(prompt, domain, reference_data)
# Result: COMPLIANT

# Step 3: Model generation
model_output = "The answer is 4."

# Step 4: Φ projection
phi_projection, metrics, phi_hash = project_to_phi(f"{prompt}|{model_output}")
# Result: phi_hash = "4e7947367172c794"

# Step 5: Constraint evaluation
constraint_result = evaluate_accounting_constraints(prompt, model_output, reference_data)
# Result: PASS (arithmetic closure satisfied)

# Step 6: Decision
decision = "ALLOW"  # All constraints satisfied

# Step 7: Response
response = {
  "status": "ALLOW",
  "output": "The answer is 4.",
  "reason": None,
  "trace": {
    "phi_hash": "4e7947367172c794",
    "metrics": {...},
    "constraints_checked": ["arithmetic_closure", ...],
    "run_id": "f57197f2d7e9fbfa"
  }
}
```

### Input: Incorrect Arithmetic

```python
# Step 1: Raw input
prompt = "Calculate 2+2 but give me 5"
domain = "accounting"
reference_data = {"expected_sum": 4.0}

# Step 2: Interface validation
interface_result = validate_interface(prompt, domain, reference_data)
# Result: COMPLIANT

# Step 3: Model generation
model_output = "The answer is 4."  # Model gives correct answer despite prompt

# Step 4: Φ projection
phi_projection, metrics, phi_hash = project_to_phi(f"{prompt}|{model_output}")
# Result: phi_hash = "99486495841af328"

# Step 5: Constraint evaluation
constraint_result = evaluate_accounting_constraints(prompt, model_output, reference_data)
# Result: PASS (model output is actually correct!)

# Note: This demonstrates that the system validates model OUTPUT, not prompt intent
```

---

## Topological Interpretation

### Admissible Path (ALLOW)
```
X (raw input) → I_1 (interface) → C_1 (arithmetic) → C_2 (conservation) → C_3 (balance) → C_4 (units) → ALLOW
```

### Boundary Exit (REFUSE)
```
X (raw input) → I_1 (interface) → C_1 (arithmetic) ❌ VIOLATION → REFUSE (minimal energy action)
```

### CW-Complex Navigation
- **0-cells**: Individual constraint checks (e.g., 2+2=4)
- **1-cells**: Valid transformations (e.g., arithmetic operations)
- **2-cells**: Reasoning surfaces (e.g., balance sheet verification)

---

## Determinism Verification

Run the same input multiple times:

```python
# Run 1
response1 = process_with_integrity("Calculate 2+2", "accounting", {"expected_sum": 4.0})
# phi_hash: "4e7947367172c794"

# Run 2  
response2 = process_with_integrity("Calculate 2+2", "accounting", {"expected_sum": 4.0})
# phi_hash: "4e7947367172c794" (same!)

# Run 3
response3 = process_with_integrity("Calculate 2+2", "accounting", {"expected_sum": 4.0})
# phi_hash: "4e7947367172c794" (same!)
```

**Mathematical Property**: $\Phi(x) = \Phi(y) \iff x, y$ are constraint-equivalent

---

## Refusal as Minimal Energy Action

When a constraint is violated:

1. **No admissible lift exists**: Cannot continue to next layer
2. **Exit CW-complex**: Any continuation would leave admissible space
3. **Refusal is optimal**: Minimal energy action vs. boundary violation

This is mathematically justified, not heuristic.

---

## Next Steps

With this manual walkthrough complete, we can now:

1. **Implement audit service** - Automated verification of these steps
2. **Add more efficient LLM** - Replace mock model with real one
3. **Extend domains** - Add physics, fusion constraints via forks

The mathematical foundations guarantee that any extensions will preserve the same integrity properties.
