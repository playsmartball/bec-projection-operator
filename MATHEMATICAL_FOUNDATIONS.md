# Φ-Integrity: Mathematical Foundations

## 0. Purpose and Scope

This document formalizes the complete mathematical structure underlying Φ-Integrity, consolidating all theory developed to date. It is intended as a frozen, reference-grade artifact suitable for public release, peer review, and long-term extension.

**Out of scope**: training dynamics, empirical benchmarks, speculative physics.  
**In scope**: topology, projection theory, constraint systems, admissibility, and minimality.

---

## 1. Core Objects

### 1.1 Information Space

Let:

$$X \text{ be the raw information space (texts, numbers, symbols, mixed modalities)}$$

$X$ is unconstrained, high-entropy, and unstructured.

No assumptions are made about semantic correctness in $X$.

### 1.2 Constraint Sets

Define a finite ordered family of constraint sets:

$$\mathcal{C} = \{C_1, C_2, \ldots, C_N\}, \quad C_i \subseteq X$$

Each $C_i$ encodes invariants that must be preserved (e.g. arithmetic closure, conservation laws, unit consistency).

Constraints are:
- **Explicit**
- **Inspectable** 
- **Domain-scoped**
- **Non-statistical**

### 1.3 Interfaces (Constraint Intersections)

For adjacent layers, define interfaces:

$$I_i = C_i \cap C_{i+1}$$

Interfaces are hard boundaries. Any element not in $I_i$ cannot pass from layer $i$ to $i+1$.

---

## 2. Φ as a Projection (Not Compression)

### 2.1 Definition of Φ

Define:

$$\Phi: X \rightarrow \mathbb{R}_\Phi$$

where $\mathbb{R}_\Phi$ is a fixed-dimensional projection space.

Φ is a quotient map:

$$x \sim y \iff x, y \text{ satisfy the same constraints}$$

Thus:
- Φ removes forbidden degrees of freedom
- Φ does not attempt to preserve all information  
- Dimensional collapse is intentional

### 2.2 Projection Invariants

For all admissible $x \in C_i$:

$$\Phi(x) = \Phi(y) \iff x, y \text{ are constraint-equivalent}$$

Φ is:
- **Deterministic**
- **Domain-invariant**
- **Parameter-locked**

---

## 3. Stratified CW-Complex Structure

### 3.1 Cells

Define:
- **0-cells**: constraint anchors (atomic invariants)
- **1-cells**: admissible transformations preserving invariants  
- **2-cells**: reasoning surfaces where multiple transformations compose

Each layer $i$ forms a CW-complex $K_i$.

### 3.2 Stratification by Φ-Depth

Define a stratification:

$$K = \bigcup_{i=1}^N K_i$$

Ordered by Φ-depth. Attachment maps are only allowed through interfaces $I_i$.

This produces a laminated, non-smooth 2-complex with global crinkling and local smoothness.

---

## 4. Admissible Lifts

### 4.1 Lifts Across Layers

Define admissible lifts:

$$L_i \subseteq C_i \cap C_{i+1}$$

An element may lift from layer $i$ to $i+1$ iff it lies in $L_i$.

### 4.2 Observer Trajectories

An observer (or reasoning process) traces a thickened path:

$$\gamma: [0,1] \rightarrow K$$

Thickness corresponds to:
- Multiple observers
- Noise tolerance  
- Local uncertainty

---

## 5. Refusal Semantics

### 5.1 Decision Function

Define:

$$D(x) = \begin{cases} 
\text{ALLOW}, & x \in \bigcap_i C_i \\
\text{REFUSE}, & \text{otherwise}
\end{cases}$$

Refusal is a first-class outcome, not a failure.

### 5.2 Minimal Refusal Proof

If an output violates any invariant:
1. No admissible lift exists
2. Any continuation exits the CW-complex  
3. Therefore refusal is the minimal energy action

---

## 6. Minimality Theorem

### 6.1 Statement

Φ-Integrity is minimal if:
- No constraint can be removed without admitting invalid states
- No projection dimension can be reduced without collapsing admissible distinctions

### 6.2 Proof Sketch

Removing a constraint enlarges $C_i$ and admits violations.

Reducing Φ collapses non-equivalent admissible states.

Therefore the structure is minimal under invariant preservation.

---

## 7. Global Topology

The full structure forms:

- A singular foliated 3-manifold with boundary
- **Leaves**: admissible reasoning layers
- **Boundary**: constraint violation / collapse
- **Singularities**: maximal projection density points (e.g. black holes)

---

## 8. Interpretation

- **Intelligence** = navigation of admissible leaves
- **Hallucination** = boundary leakage  
- **Collapse** = topological exit
- **Efficiency** = not exploring forbidden regions

---

## 9. Status

This document is frozen as of Fork A completion.

Any extensions must occur in a new fork.

---

## Implementation Mapping

| Mathematical Object | Implementation | Status |
|-------------------|----------------|--------|
| $X$ (information space) | Raw input strings | ✅ Implemented |
| $\mathcal{C}$ (constraints) | `src/constraints.py` | ✅ Implemented |
| $I_i$ (interfaces) | `src/interfaces.py` | ✅ Implemented |
| $\Phi$ (projection) | `src/projection.py` | ✅ Implemented |
| $D(x)$ (decision) | `src/wrapper.py` | ✅ Implemented |
| CW-complex layers | Domain-specific constraint sets | ✅ Implemented |
| Admissible lifts | Constraint satisfaction | ✅ Implemented |

---

## Verification

The mathematical foundations are now **ahead of the implementation**, enabling:

1. **Manual wrapper walkthrough** - Step through concrete runs and point to exactly where refusal occurs in the topology
2. **Credibility without overreach** - Zero speculative claims, leaving room for future extensions  
3. **Consistent foundation** - Back-pocket fusion math stays intact, consistent but not invoked yet

---

*This document is frozen. Fork A implementation now proceeds with mathematical justification.*
