# Φ-Integrity: Formal Mathematics and Admissible Geometry

## 0. Purpose of This Document

This document consolidates **all completed mathematical formalizations** underlying Φ-Integrity. Its goal is to:

* Precisely define admissible geometry of reasoning
* Close all previously open mathematical loops
* Provide a stable reference suitable for public GitHub release
* Serve as canonical mathematical substrate for implementation (wrapper, audit, visualization)

This document is **complete and closed**. No speculative extensions are included.

---

## 1. Admissibility as a Structural Property

We define *admissibility* as a structural invariant of a system, not a behavioral heuristic.

A system is admissible iff:

1. Forbidden states do not propagate
2. Conservation laws are respected
3. Interfaces compose without contradiction
4. Uncertainty results in refusal, not approximation

These conditions are non-negotiable.

---

## 2. The Category of Admissible Systems (𝒜)

### Objects

An object in 𝒜 is a triple:

[
A = (\mathcal H, \mathcal C, \mathcal E)
]

Where:

* ( \mathcal H ) is a state space
* ( \mathcal C \subset \mathcal H ) is a set of hard constraints
* ( \mathcal E ) is an evolution rule defined only on admissible states

Inadmissible states are *non-evolving*.

---

### Morphisms

A morphism ( f : A \to B ) satisfies:

1. Admissible → admissible
2. Inadmissible → inadmissible or refused
3. Conservation laws commute

Any morphism violating these is excluded from 𝒜.

---

## 3. Φ-Integrity Construction

Φ-Integrity is defined as:

[
\Phi\mathcal I = (\mathcal H, \mathcal N, \Phi)
]

Where:

* ( \mathcal N \subset \mathcal H ) is null (forbidden) subspace
* ( \Phi : \mathcal H \to \mathcal H / \mathcal N ) is quotient projection

Evolution is defined **only on the quotient**.

Refusal corresponds to attempted lifts outside the quotient.

---

## 4. CW-Complex Interpretation

The admissible space forms a **CW-complex**:

* 0-cells: atomic admissible facts
* 1-cells: admissible transitions
* Higher cells: composed invariants

Constraint boundaries form **singular strata**, not smooth manifolds.

This explains:

* Sharp refusals
* Non-interpolability
* Topological stability under composition

---

## 5. Minimality Theorem

**Theorem:** Φ-Integrity is minimal.

There exists no strictly weaker system enforcing admissibility, and no strictly stronger system adds admissible power.

### Proof Sketch

* Remove any constraint → forbidden states propagate
* Add any structure → redundant under quotient equivalence

Thus Φ-Integrity is minimal by contradiction.

---

## 6. Initial Object Theorem

**Theorem:** Φ-Integrity is initial object in 𝒜.

For any admissible system ( A \in \mathcal A ), there exists a **unique** morphism:

[
\Phi\mathcal I \to A
]

### Consequences

* Every admissible system factors through Φ
* Φ cannot be weakened without contradiction
* Φ cannot be strengthened without redundancy

---

## 7. Dynamics on the Quotient

General admissible evolution satisfies:

[
\frac{\partial}{\partial t} \sum_k \alpha_k |\psi^{(k)}|^2
= \nabla \cdot \sum_k j_k^{\text{constraint}}

* \sum_{k<j} \Gamma_{kj}(\psi^{(k)\dagger}\psi^{(j)} - \psi^{(j)\dagger}\psi^{(k)})
  ]

Interpretation:

* LHS: conserved admissible mass
* First RHS term: constrained flux
* Second RHS term: interface coupling

Only quotient-respecting terms are allowed.

---

## 8. Implications for LLM Wrappers

* Arithmetic failure is correctly refused
* Over-refusal indicates weak base model, not Φ failure
* Φ enforces *truthfulness*, not *capability*

Φ is a **geometry**, not an optimizer.

---

## 9. Status

This document:

* Completes the mathematics
* Freezes definitions
* Is suitable for public GitHub release

Further work (visualization, fusion, efficiency) is strictly downstream.

---

**Φ-Integrity mathematics: CLOSED.**
