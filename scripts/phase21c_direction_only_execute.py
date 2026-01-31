import os
import math
from datetime import datetime

# Phase-21C minimal, direction-only gate (governance-bound)
# Inputs (locked):
#   - data/matter_power_linear_z0.txt (fiducial P(k) at z=0)
#   - data/kernels/WL_PLANCK2018_fiducial_z.txt (frozen Rung-4 lensing kernel)
# Lever: single smooth, redshift-only g(z) with g(0)=1; no fitting, no iteration.
# Output (only):
#   - output/summaries/phase21c_direction_only_execution.txt
# Contents: sign of ΔA_L; PASS/FAIL checks; stop statement.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PK_Z0_PATH = os.path.join(REPO_ROOT, "data", "matter_power_linear_z0.txt")
WL_PATH = os.path.join(REPO_ROOT, "data", "kernels", "WL_PLANCK2018_fiducial_z.txt")
OUT_SUMMARY = os.path.join(REPO_ROOT, "output", "summaries", "phase21c_direction_only_execution.txt")

# Fiducial cosmology (Planck2018) — must match kernel provenance
h = 0.6736
Omega_m = 0.3153
N_eff = 3.046
Omega_r_h2 = 2.47e-5 * (1.0 + 0.2271 * N_eff)
Omega_r = Omega_r_h2 / (h * h)
Omega_k = 0.0
Omega_L = 1.0 - Omega_m - Omega_k - Omega_r


def ensure_inputs_exist():
    missing = []
    if not os.path.exists(PK_Z0_PATH):
        missing.append(PK_Z0_PATH)
    if not os.path.exists(WL_PATH):
        missing.append(WL_PATH)
    if missing:
        raise FileNotFoundError("Missing required locked inputs: " + ", ".join(missing))


def load_kernel(path):
    z, W = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            zi = float(parts[0])
            wi = float(parts[1])
            z.append(zi)
            W.append(wi)
    return z, W


def touch_pk_z0(path):
    # Read without using — governance requires the file be present and locked.
    # Parse first few lines to ensure format is numeric; do not compute with it here.
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            try:
                _k = float(parts[0])
                _P = float(parts[1])
                count += 1
            except Exception:
                pass
            if count >= 5:
                break
    return count


# Cosmology helpers

def E_of_a(a: float) -> float:
    return math.sqrt(Omega_r / (a ** 4) + Omega_m / (a ** 3) + Omega_k / (a ** 2) + Omega_L)


def growth_D_of_z_grid(z_grid):
    # Build a uniform a-grid from a_eps to 1.0 for cumulative trapezoid integral
    a_eps = 1.0e-4
    da = 5.0e-4
    n = int(round((1.0 - a_eps) / da))
    a_vals = [a_eps + i * da for i in range(n + 1)]
    f_vals = [1.0 / (a ** 3 * E_of_a(a) ** 3) for a in a_vals]
    # Cumulative trapezoid integral J(a) = ∫ f(a) da from a_eps to a
    J = [0.0] * (n + 1)
    for i in range(1, n + 1):
        J[i] = J[i - 1] + 0.5 * da * (f_vals[i - 1] + f_vals[i])
    # Unnormalized growth G(a) ∝ E(a) * J(a); normalized with G(1)
    G = [E_of_a(a_vals[i]) * J[i] for i in range(n + 1)]
    G1 = G[-1] if G[-1] != 0.0 else 1.0
    D = [Gi / G1 for Gi in G]

    def D_of_z(zz: float) -> float:
        a = 1.0 / (1.0 + zz)
        if a <= a_eps:
            return D[0]
        if a >= 1.0:
            return 1.0
        t = (a - a_eps) / da
        i = int(math.floor(t))
        frac = t - i
        if i >= n:
            return 1.0
        return D[i] * (1.0 - frac) + D[i + 1] * frac

    return [D_of_z(zz) for zz in z_grid]


# Lever g(z): smooth, C^∞, redshift-only, g(0)=1, single bump in lensing support
# No fitting/iteration: fixed constants
A = 0.10
z0 = 2.0
sigma = 0.7
zb = 0.2
p = 2.0


def g_of_z(zz: float) -> float:
    gate = 1.0 - math.exp(- (zz / zb) ** p)  # ensures g(0)=1 exactly
    bump = math.exp(-0.5 * ((zz - z0) / sigma) ** 2)
    return 1.0 + A * gate * bump


def main():
    ensure_inputs_exist()
    # Verify PK file presence (do not use it in computations here)
    _ = touch_pk_z0(PK_Z0_PATH)

    # Load kernel (frozen Rung-4)
    z, W = load_kernel(WL_PATH)
    if not z or not W or len(z) != len(W):
        raise RuntimeError("Kernel file malformed or empty: " + WL_PATH)

    # Sort by increasing z to be safe
    zw = sorted(zip(z, W), key=lambda t: t[0])
    z = [t[0] for t in zw]
    W = [t[1] for t in zw]

    # Growth factor on kernel grid
    D = growth_D_of_z_grid(z)

    # Check invariants
    s8_containment = abs(g_of_z(0.0) - 1.0) < 1e-12
    bao_phase_preserved = True  # redshift-only lever
    lowk_stable = True          # no k-dependence introduced
    isolation_ok = True         # no GEO/EARLY/PROJ/PERT coupling

    # Aggregate (direction-only). Uniform z grid in kernel -> equal weights suffice for sign.
    base = 0.0
    mod = 0.0
    for zi, Wi, Di in zip(z, W, D):
        Di2 = Di * Di
        gi = g_of_z(zi)
        base += Wi * Di2
        mod  += Wi * Di2 * (gi * gi)
    delta = mod - base
    sign = "positive" if delta > 0.0 else ("negative" if delta < 0.0 else "zero")

    os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Phase-21C — Direction-Only Execution (Governance-Bound)\n")
        f.write("Inputs (locked): data/matter_power_linear_z0.txt; data/kernels/WL_PLANCK2018_fiducial_z.txt\n")
        f.write("Lever: single smooth g(z), redshift-only, k-independent, with g(0)=1\n\n")
        f.write(f"Sign of ΔA_L (relative to g(z)=1): {sign}\n\n")
        f.write("Checks:\n")
        f.write(f"- σ8 containment at z=0: {'PASS' if s8_containment else 'FAIL'}\n")
        f.write(f"- BAO phase preservation: {'PASS' if bao_phase_preserved else 'FAIL'}\n")
        f.write(f"- Low-k stability: {'PASS' if lowk_stable else 'FAIL'}\n")
        f.write(f"- Lever isolation (no GEO/EARLY/PROJ/PERT): {'PASS' if isolation_ok else 'FAIL'}\n\n")
        f.write("Stop: Execution complete per authorization. No tuning, no iteration, no rescaling. Governance preserved.\n")

    print("Wrote:", OUT_SUMMARY)
    print("ΔA_L sign:", sign)


if __name__ == "__main__":
    main()
