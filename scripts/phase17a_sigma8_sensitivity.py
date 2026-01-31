import os
import numpy as np
import matplotlib.pyplot as plt

# Phase-17A: Classification and bounding only.
# No models, no fits, no new physics, no tuning to match an observed value.
# Reads locked fiducial P(k) from data/matter_power_linear_z0.txt and applies
# controlled, symbolic envelope perturbations to assess sigma8 sensitivity.


def load_pk_txt(path_txt: str):
    if not os.path.exists(path_txt):
        raise FileNotFoundError(
            "Missing fiducial matter power spectrum: data/matter_power_linear_z0.txt"
        )
    d = np.loadtxt(path_txt)
    k = np.asarray(d[:, 0], dtype=float)
    P = np.asarray(d[:, 1], dtype=float)
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(P))):
        raise ValueError("Non-finite k or P found.")
    if np.any(k <= 0):
        raise ValueError("Require k > 0 grid.")
    order = np.argsort(k)
    return k[order], P[order]


def window_tophat(k, R=8.0):
    x = k * R
    w = np.ones_like(x)
    small = np.isclose(x, 0.0)
    xs = x[~small]
    w[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / (xs ** 3)
    w[small] = 1.0
    return w


def sigma8_from_pk(k, P):
    W8 = window_tophat(k, R=8.0)
    delta2 = (k ** 3) * P / (2.0 * np.pi ** 2)
    lnk = np.log(k)
    integrand = delta2 * (np.abs(W8) ** 2)
    s2 = np.trapz(integrand, lnk)
    return float(np.sqrt(max(s2, 0.0)))


# Envelopes (symbolic, non-physical). Each returns a multiplicative factor F(k) in [0,1].

def env_scale_local(k, alpha: float, k1=0.10, k2=0.30):
    """Scale-local suppression: P' = (1 - alpha) P for k in [k1, k2], unchanged otherwise."""
    F = np.ones_like(k)
    band = (k >= k1) & (k <= k2)
    F[band] = 1.0 - alpha
    return F


def env_highk_rolloff(k, alpha: float, k0=0.20, kmin_noeffect=0.05, width=0.05):
    """High-k attenuation: smooth, monotone rolloff above k0; no effect below ~0.05 h/Mpc."""
    x = (k - k0) / max(width, 1e-6)
    s = 1.0 / (1.0 + np.exp(-x))  # logistic: ~0 below k0, ~1 above k0
    s[k < kmin_noeffect] = 0.0
    F = 1.0 - alpha * s
    return F


def env_broadband_tilt(k, alpha: float, kt=0.20):
    """Broadband amplitude tilt: weak k-dependent tilt, preserves large-scale normalization."""
    s = (k ** 2) / (k ** 2 + kt ** 2)  # 0 at k->0, ->1 at high k
    F = 1.0 - alpha * s
    return F


def env_late_growth_uniform(k, alpha: float):
    """Late-time only growth scaling (z<1): approximate as a uniform fractional suppression on P.
    This is a descriptive envelope on P(k) at z=0 (no growth evolution performed here)."""
    return np.full_like(k, 1.0 - alpha)


def classify_flag(delta_frac: float):
    """Descriptive bins (not fitted):
    INSUFFICIENT: |Δσ8| < 2%
    POTENTIALLY RELEVANT: 2%–8%
    EXCESSIVE: > 8%
    INCOMPATIBLE: reserved for structural violations (none used here)."""
    a = abs(delta_frac)
    if a < 0.02:
        return "INSUFFICIENT"
    if a <= 0.08:
        return "POTENTIALLY RELEVANT"
    return "EXCESSIVE"


def main():
    proj_root = "."
    data_txt = os.path.join(proj_root, "data", "matter_power_linear_z0.txt")
    out_fig = os.path.join(proj_root, "output", "figures", "phase17a_sigma8_response.png")
    out_sum = os.path.join(proj_root, "output", "summaries", "phase17a_sigma8_sensitivity.txt")
    out_cls = os.path.join(proj_root, "output", "summaries", "phase17a_density_response_classes.txt")

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_sum), exist_ok=True)

    k, P = load_pk_txt(data_txt)
    s8_base = sigma8_from_pk(k, P)

    # Envelopes and metadata
    envelopes = [
        ("scale-local", env_scale_local, {"k1": 0.10, "k2": 0.30}, "none", "[0.10, 0.30]"),
        ("high-k attenuation", env_highk_rolloff, {"k0": 0.20, "kmin_noeffect": 0.05, "width": 0.05}, "none", "k \u2265 0.20 (none below 0.05)"),
        ("broadband tilt", env_broadband_tilt, {"kt": 0.20}, "none", "broad (normalized at low k)"),
        ("late-time growth (z<1)", env_late_growth_uniform, {}, "late-time only", "all k (uniform)")
    ]

    magnitudes = [0.02, 0.05, 0.08]  # fixed, not tuned

    rows = []  # (class, k_range, z_dep, alpha, s8, delta, side_effects, flag)

    for name, func, params, z_dep, k_desc in envelopes:
        for alpha in magnitudes:
            F = func(k, alpha=alpha, **params)
            Pp = F * P
            s8 = sigma8_from_pk(k, Pp)
            delta = (s8 / s8_base) - 1.0
            # Side effects (descriptive)
            if name == "late-time growth (z<1)":
                large_scale_affected = True
            elif name == "high-k attenuation":
                large_scale_affected = False  # explicitly none below ~0.05
            elif name == "broadband tilt":
                large_scale_affected = False  # constructed to preserve low-k normalization
            else:  # scale-local
                large_scale_affected = False
            bao_shift = False  # multiplicative, smooth/constant factors do not shift BAO peak positions
            side = f"BAO shift: {'YES' if bao_shift else 'NO'}; large-scale affected: {'YES' if large_scale_affected else 'NO'}"
            flag = classify_flag(delta)
            rows.append((name, k_desc, z_dep, alpha, s8, delta, side, flag))

    # Write sensitivity summary
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write("Phase-17A — σ8 Sensitivity Scan (Classification)\n")
        f.write("Inputs: data/matter_power_linear_z0.txt (locked), linear theory; no growth evolution.\n")
        f.write(f"Baseline sigma8 = {s8_base:.6f}\n\n")
        f.write("class\talpha\tDelta_sigma8[%]\tsigma8\tk-range\tz-dependence\tside-effects\tflag\n")
        for name, k_desc, z_dep, alpha, s8, delta, side, flag in rows:
            f.write(f"{name}\t{alpha:.0%}\t{delta*100:+.2f}\t{s8:.6f}\t{k_desc}\t{z_dep}\t{side}\t{flag}\n")
        f.write("\nPhase-17A completed: σ₈ sensitivity classified without model assumptions or fitting.\n")

    # Plot: Δσ8 vs envelope class for each alpha
    classes = [e[0] for e in envelopes]
    x = np.arange(len(classes))
    width = 0.22

    # Aggregate by class
    deltas_by_class = {name: [] for name in classes}
    for name, _k, _z, alpha, _s8, delta, _side, _flag in rows:
        deltas_by_class[name].append(delta * 100.0)

    plt.figure(figsize=(10, 5))
    for i, alpha in enumerate(magnitudes):
        vals = [deltas_by_class[name][i] for name in classes]
        plt.bar(x + (i - 1) * width, vals, width=width, label=f"{int(alpha*100)}%")
    plt.axhline(0.0, color='k', lw=1)
    plt.xticks(x, classes, rotation=15, ha='right')
    plt.ylabel(r"Δ$\sigma_8$ [%]")
    plt.title("Phase-17A: σ8 response to density envelopes (no fits)")
    plt.legend(title="Envelope magnitude")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()

    # Density-response typing and rung eligibility
    with open(out_cls, "w", encoding="utf-8") as f:
        f.write("Phase-17A — Density-Response Typing\n\n")
        f.write("- scale-local: Observer-space redistribution\n")
        f.write("- high-k attenuation: Boundary-like leakage (spectral, monotone high-k sink)\n")
        f.write("- broadband tilt: Observer-space redistribution (preserves low-k normalization)\n")
        f.write("- late-time growth (z<1): Growth-rate modification (bulk)\n\n")
        f.write("Rung-2 eligibility (boundary/coupling) by structural criteria:\n")
        f.write("- Eligible: high-k attenuation (locality at boundary scales; monotone; conservative form)\n")
        f.write("- Possibly eligible (context-dependent): scale-local (if band aligns with boundary-controlled scales)\n")
        f.write("- Not eligible: broadband tilt (global reweighting without boundary locality), late-time growth (bulk evolution)\n")

    print(f"Wrote {out_sum}\nWrote {out_cls}\nSaved {out_fig}")


if __name__ == "__main__":
    main()
