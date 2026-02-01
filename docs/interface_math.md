# Interface Mathematics

**FMI v0.1 - Mathematical Foundation**

---

## Core Components

### Φ Coordinate System

The depth coordinate Φ provides a deterministic stratification parameter:

\[
\Phi \in [0, \Phi_{\text{max}}]
\]

Discretization:
\[
\Phi_i = i \cdot \Delta\Phi, \quad \Delta\Phi = \frac{\Phi_{\text{max}}}{N-1}
\]

### Density Stratification

Density distribution over Φ:
\[
\rho(\Phi) : \Phi \mapsto \mathbb{R}^+
\]

Stratification operation:
\[
\tilde{\rho}(\Phi) = \rho(\Phi) \cdot \Phi
\]

### Cumulative Integral

\[
C(\Phi) = \int_0^\Phi \rho(\Phi') \, d\Phi'
\]

Discrete implementation:
\[
C_i = \sum_{j=0}^i \rho_j \cdot \Delta\Phi
\]

---

## Ladder Constraints

### σ₈ Containment

Variance normalization:
\[
\text{Var}[\rho] \leq 1.0
\]

Implementation:
\[
\rho \leftarrow \frac{\rho}{\sqrt{\text{Var}[\rho] + \epsilon}}
\]

### Kernel Support

Window function for compact support:
\[
W(\Phi) = \exp\left(-\frac{(\Phi - \Phi_c)^2}{\sigma^2}\right)
\]

Supported density:
\[
\rho_{\text{supported}}(\Phi) = \rho(\Phi) \cdot W(\Phi)
\]

### k-Orthogonality

Gram-Schmidt orthogonalization for modes \(M_k\):
\[
M_k^{\perp} = M_k - \sum_{j<k} \frac{\langle M_k, M_j \rangle}{\langle M_j, M_j \rangle} M_j
\]

---

## Response Functionals

### Kernel Response

\[
R(\Phi) = \int_0^{\Phi_{\text{max}}} K(\Phi, \Phi') \rho(\Phi') \, d\Phi'
\]

Kernel definition:
\[
K(\Phi, \Phi') = \exp\left(-\frac{|\Phi - \Phi'|}{\lambda}\right)
\]

### Spectral Projection

Fourier decomposition:
\[
\hat{\rho}(k) = \int_0^{\Phi_{\text{max}}} \rho(\Phi) e^{-2\pi i k \Phi} \, d\Phi
\]

### Energy Redistribution

Conservative redistribution:
\[
E(\Phi) = \frac{|R(\Phi)|^2}{\int_0^{\Phi_{\text{max}}} |R(\Phi')|^2 \, d\Phi'}
\]

---

## Determinism Guarantees

### Hash Stability

SHA256 hash of output arrays:
\[
H = \text{SHA256}(\text{bytes}(\text{output}))
\]

### CPU/GPU Parity

Numerical tolerance:
\[
\|R_{\text{CPU}} - R_{\text{GPU}}\|_\infty \leq \tau
\]

Typical tolerance: \(\tau = 10^{-12}\)

---

## Interface Contract

All operations satisfy:
1. **Deterministic**: Same inputs → same outputs
2. **Parameter-free**: No tunable parameters
3. **Conservative**: Energy/mass conservation
4. **Causal**: Φ ordering preserved
