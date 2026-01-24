import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from phase16a_operator_class import KOperator
except Exception as e:
    raise

# LOCKED operator choice (prior to evaluation)
LOCKED_FORM = "exponential"  # {'soft_cutoff','exponential'}
LOCKED_N = 2                  # {1,2}
LOCKED_KC = 0.20              # h/Mpc; chosen from observational window scales


def load_pk():
    npz_path = os.path.join("data", "matter_power_linear_z0.npz")
    txt_path = os.path.join("data", "matter_power_linear_z0.txt")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        k = np.asarray(d["k"], dtype=float)
        P = np.asarray(d["P"], dtype=float)
    elif os.path.exists(txt_path):
        d = np.loadtxt(txt_path)
        k = np.asarray(d[:, 0], dtype=float)
        P = np.asarray(d[:, 1], dtype=float)
    else:
        raise FileNotFoundError(
            "Missing fiducial matter power spectrum. Place data/matter_power_linear_z0.(npz|txt)."
        )
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


def sigma8(k, P):
    W8 = window_tophat(k, R=8.0)
    delta2 = (k ** 3) * P / (2.0 * np.pi ** 2)
    lnk = np.log(k)
    s2 = np.trapz(delta2 * (np.abs(W8) ** 2), lnk)
    return float(np.sqrt(max(s2, 0.0)))


def bao_peak_position(k, P, kmin=0.05, kmax=0.30):
    mask = (k >= kmin) & (k <= kmax)
    if mask.sum() < 5:
        return np.nan
    ks = k[mask]
    Ps = P[mask]
    # Find local maxima via sign change of derivative
    dP = np.gradient(Ps, ks)
    peaks = []
    for i in range(1, len(ks) - 1):
        if dP[i - 1] > 0 and dP[i] <= 0:
            peaks.append((Ps[i], ks[i]))
    if not peaks:
        return np.nan
    # Largest peak in the window
    peaks.sort(reverse=True)
    return peaks[0][1]


def main():
    os.makedirs(os.path.join("output", "figures"), exist_ok=True)
    os.makedirs(os.path.join("output", "summaries"), exist_ok=True)

    k, P = load_pk()

    base_s8 = sigma8(k, P)

    kop = KOperator(form=LOCKED_FORM, kc=LOCKED_KC, n=LOCKED_N)
    Wk = kop.W(k)
    Pobs = Wk * P
    obs_s8 = sigma8(k, Pobs)
    delta = (obs_s8 / base_s8) - 1.0

    # Abort conditions
    # 1) BAO scale moves
    k_bao_0 = bao_peak_position(k, P)
    k_bao_1 = bao_peak_position(k, Pobs)
    bao_shift_ok = True
    bao_shift_rel = np.nan
    if np.isfinite(k_bao_0) and np.isfinite(k_bao_1) and k_bao_0 > 0:
        bao_shift_rel = abs(k_bao_1 - k_bao_0) / k_bao_0
        bao_shift_ok = bao_shift_rel <= 0.01  # 1% tolerance

    # 2) Large-scale power suppressed (k < 0.02 h/Mpc)
    mask_ls = k < 0.02
    ls_ok = True
    if np.any(mask_ls):
        w_ls = np.mean(Wk[mask_ls])
        ls_ok = (w_ls >= 0.98)

    aborted = not (bao_shift_ok and ls_ok)

    # Save summary
    txt = os.path.join("output", "summaries", "phase16c_sigma8_operator.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Phase 16C — Minimal σ8 Operator (LOCKED)\n")
        f.write(f"operator: {LOCKED_FORM}, n={LOCKED_N}, kc={LOCKED_KC:.3f} h/Mpc\n")
        f.write(f"sigma8_base = {base_s8:.6f}\n")
        f.write(f"sigma8_obs  = {obs_s8:.6f}\n")
        f.write(f"delta = {delta:+.4%}\n")
        f.write(f"bao_k_base = {k_bao_0:.6f}\n")
        f.write(f"bao_k_obs  = {k_bao_1:.6f}\n")
        f.write(f"bao_shift_rel = {bao_shift_rel if np.isfinite(bao_shift_rel) else np.nan}\n")
        f.write(f"large_scale_W_mean(k<0.02) >= 0.98 ? {ls_ok}\n")
        if aborted:
            f.write("ABORT: Inconsistency detected (BAO shift or large-scale suppression).\n")
        else:
            f.write("OK: Operator passes minimal locked checks.\n")

    # Plot if not aborted
    fig = os.path.join("output", "figures", "phase16c_sigma8_operator.png")
    try:
        import matplotlib as mpl
        mpl.rcParams.update({"figure.dpi": 150})
        plt.figure(figsize=(6, 4))
        plt.loglog(k, P, label="P(k) baseline")
        plt.loglog(k, Pobs, label="P_obs(k)")
        plt.xlabel("k [h/Mpc]")
        plt.ylabel("P [Mpc^3/h^3]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig)
        plt.close()

        plt.figure(figsize=(6, 3))
        plt.semilogx(k, Wk)
        plt.xlabel("k [h/Mpc]")
        plt.ylabel("W(k)")
        plt.tight_layout()
        plt.savefig(fig.replace(".png", "_W.png"))
        plt.close()
    except Exception as e:
        with open(txt, "a", encoding="utf-8") as f:
            f.write(f"[warning] plotting failed: {e}\n")

    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
