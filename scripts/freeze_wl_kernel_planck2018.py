import os
import math
from datetime import datetime

# Governance: one-time generation of CMB lensing kernel W_L(z) for Planck 2018 fiducial cosmology.
# Output path (relative to repo root): data/kernels/WL_PLANCK2018_fiducial_z.txt
# Format: two-column ASCII, headerless: z  W_L(z)
# Grid: z in [0, 10], monotonic, dense enough to cover lensing support (CMB lensing peaks at z~2)
# Normalization: absolute per standard expression, no rescaling.
# Expression (flat LCDM): W(z) = (3/2) * Ω_m * (H0/c)^2 * (1+z) * χ(z) * (χ_* - χ(z)) / χ_*
# where χ(z) = (c/H0) * ∫_0^z dz'/E(z'), E(z) = sqrt(Ω_r(1+z)^4 + Ω_m(1+z)^3 + Ω_Λ)

# Planck 2018 fiducial parameters (approximate best-fit)
h = 0.6736
Omega_m = 0.3153
N_eff = 3.046
T_cmb_K = 2.7255  # K (not used explicitly; use standard Ω_r approximation)
# Radiation density today: Ω_r h^2 ≈ Ω_γ h^2 (1 + 0.2271 N_eff), Ω_γ h^2 ≈ 2.47e-5
Omega_r_h2 = 2.47e-5 * (1.0 + 0.2271 * N_eff)
Omega_r = Omega_r_h2 / (h * h)
Omega_k = 0.0
Omega_L = 1.0 - Omega_m - Omega_k - Omega_r

c_km_s = 299792.458
H0_km_s_Mpc = 100.0 * h
c_over_H0_Mpc = c_km_s / H0_km_s_Mpc
H0_over_c_Mpc_inv = H0_km_s_Mpc / c_km_s

# E(z)
def Ez(z: float) -> float:
    return math.sqrt(
        Omega_r * (1.0 + z) ** 4 + Omega_m * (1.0 + z) ** 3 + Omega_L
    )

# Simpson integration on uniform grid

def simpson_integral_uniform(fvals, dz):
    n = len(fvals)
    if n < 2:
        return 0.0
    if (n - 1) % 2 == 1:
        # Ensure even number of intervals by dropping the last point for Simpson, then trapezoid for last
        main_n = n - 2
        s = fvals[0] + fvals[main_n]
        s += 4.0 * sum(fvals[1:main_n:2])
        s += 2.0 * sum(fvals[2:main_n-1:2]) if main_n > 2 else 0.0
        I = (dz / 3.0) * s
        # trapezoid for the last interval
        I += 0.5 * dz * (fvals[main_n] + fvals[main_n + 1])
        return I
    else:
        s = fvals[0] + fvals[-1]
        s += 4.0 * sum(fvals[1:-1:2])
        s += 2.0 * sum(fvals[2:-2:2]) if n > 3 else 0.0
        return (dz / 3.0) * s

# Build chi(z) grid from 0..zmax with uniform dz (efficient cumulative Simpson)

def chi_grid(zmax: float, dz: float):
    n = int(round(zmax / dz))
    z = [i * dz for i in range(n + 1)]
    invE = [1.0 / Ez(zi) for zi in z]
    # cumulative Simpson: integrate from 0 to each z_i by Simpson on the prefix
    chi = []
    for i in range(n + 1):
        if i == 0:
            chi.append(0.0)
        else:
            # Simpson on prefix [0..i]
            I = simpson_integral_uniform(invE[: i + 1], dz)
            chi.append(c_over_H0_Mpc * I)
    return z, chi

# Compute chi*(z_*) with coarse but sufficient step (radiation included)

def chi_star(z_star: float = 1090.0, dz: float = 0.5) -> float:
    n = int(round(z_star / dz))
    z = [i * dz for i in range(n + 1)]
    invE = [1.0 / Ez(zi) for zi in z]
    I = simpson_integral_uniform(invE, dz)
    return c_over_H0_Mpc * I


def main():
    # Output path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(repo_root, "data", "kernels", "WL_PLANCK2018_fiducial_z.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Grids
    zmax = 10.0
    dz = 0.005  # 2001 points
    z_arr, chi_arr = chi_grid(zmax, dz)
    chi_star_val = chi_star(1090.0, 0.5)

    # Prefactor
    pref = 1.5 * Omega_m * (H0_over_c_Mpc_inv ** 2)

    # Write file: two-column ASCII, headerless: z  W_L(z)
    with open(out_path, "w", encoding="utf-8") as f:
        for z, chi in zip(z_arr, chi_arr):
            a = 1.0 / (1.0 + z)
            W = pref * (1.0 / a) * chi * max(chi_star_val - chi, 0.0) / chi_star_val
            f.write(f"{z:.6f} {W:.12e}\n")

    # Provenance note to stdout
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print("Wrote:", out_path)
    print("Provenance: CMB lensing kernel; Planck2018 fiducial ΛCDM; h=0.6736, Ωm=0.3153, ΩΛ=", round(Omega_L, 6), 
          ", Ωr=", round(Omega_r, 8), "; z-grid [0, 10], dz=", dz, "; χ* via z*=1090, d z*=0.5; UTC:", now)


if __name__ == "__main__":
    main()
