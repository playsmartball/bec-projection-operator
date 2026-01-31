import os
import numpy as np
import matplotlib.pyplot as plt

# Phase-17B Execution (under governance):
# Apply boundary-like, k-local, monotone envelopes (B1–B3) to locked fiducial P(k),
# compute sigma8 once per envelope, check constraint flags, classify, and write outputs.
# No fitting, no tuning, no growth/equation changes, no commits.


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


# Envelope definitions (k-local, monotone)

def env_B1(k, k_b: float, A: float):
    """Hard high-k sink: W=1 for k<=k_b; W=1-A for k>k_b."""
    F = np.ones_like(k)
    F[k > k_b] = 1.0 - A
    return F


def env_B2(k, k_b: float, n: int):
    """Smooth boundary roll-off: W = 1 / (1 + (k/k_b)^n)."""
    return 1.0 / (1.0 + (k / k_b) ** n)


def env_B3(k, k_list, A_i: float):
    """Multi-shell integration: W = Π_i [1 - A_i Θ(k - k_i)]."""
    F = np.ones_like(k)
    for k_i in k_list:
        F = F * (1.0 - A_i * (k >= k_i).astype(float))
    return F


def monotone_nonincreasing(F):
    d = np.diff(F)
    return bool(np.all(d <= 1e-8))


def moving_average(y, win: int):
    if win < 3:
        return y.copy()
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    # pad with edge values
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def find_first_bao_peak_k(k, P, kmin=0.05, kmax=0.30):
    mask = (k >= kmin) & (k <= kmax)
    if not np.any(mask):
        return None
    idx = np.where(mask)[0]
    kk = k[idx]
    yy = P[idx]
    # mild smoothing: window ~ min(51, ~N/8), odd
    N = len(kk)
    win = max(5, min(51, (N // 8) * 2 + 1))
    ys = moving_average(yy, win)
    # peak at global maximum in the window (BAO-1 region)
    imax = int(np.argmax(ys))
    return float(kk[imax])


def bao_shift_flag(k, P_base, P_obs, thr_frac=0.01):
    kb = find_first_bao_peak_k(k, P_base)
    ko = find_first_bao_peak_k(k, P_obs)
    if kb is None or ko is None or kb <= 0:
        return False  # cannot assess; treat as no shift detected
    frac = abs(ko - kb) / kb
    return bool(frac > thr_frac)


def lowk_change_flag(k, F, kthr=0.05, thr_frac=0.01):
    m = k < kthr
    if not np.any(m):
        return False
    mean_abs = float(np.mean(np.abs(F[m] - 1.0)))
    return bool(mean_abs > thr_frac)


def classify(delta_frac: float, bao_violation: bool, lowk_violation: bool, mono_ok: bool):
    if (not mono_ok) or bao_violation or lowk_violation:
        return "DISQUALIFIED"
    a = abs(delta_frac)
    if a < 0.02:
        return "INSUFFICIENT"
    if a < 0.05:
        return "MARGINAL"
    return "STRUCTURALLY POTENT"


def main():
    proj_root = "."
    data_txt = os.path.join(proj_root, "data", "matter_power_linear_z0.txt")
    out_fig = os.path.join(proj_root, "output", "figures", "phase17b_sigma8_vs_kb.png")
    out_sum = os.path.join(proj_root, "output", "summaries", "phase17b_boundary_sigma8.txt")
    out_cls = os.path.join(proj_root, "output", "summaries", "phase17b_boundary_classification.txt")

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_sum), exist_ok=True)

    k, P = load_pk_txt(data_txt)
    s8_base = sigma8_from_pk(k, P)

    # Envelope grids (locked)
    B1_grid = [(k_b, A) for k_b in [0.3, 0.5, 1.0] for A in [0.05, 0.10]]
    B2_grid = [(k_b, n) for k_b in [0.3, 0.5] for n in [2, 4]]
    B3_def = {"k_list": [0.2, 0.4, 0.6, 0.8], "A_i": 0.02}

    rows = []  # (family, params_str, s8, delta, bao, lowk, mono, classification)

    # B1 cases
    for k_b, A in B1_grid:
        F = env_B1(k, k_b=k_b, A=A)
        Pp = F * P
        s8 = sigma8_from_pk(k, Pp)
        delta = (s8 / s8_base) - 1.0
        bao = bao_shift_flag(k, P, Pp)
        lowk = lowk_change_flag(k, F)
        mono = monotone_nonincreasing(F)
        cls = classify(delta, bao, lowk, mono)
        rows.append(("B1", f"k_b={k_b:.2f}, A={A:.2f}", s8, delta, bao, lowk, mono, cls))

    # B2 cases
    for k_b, n in B2_grid:
        F = env_B2(k, k_b=k_b, n=n)
        Pp = F * P
        s8 = sigma8_from_pk(k, Pp)
        delta = (s8 / s8_base) - 1.0
        bao = bao_shift_flag(k, P, Pp)
        lowk = lowk_change_flag(k, F)
        mono = monotone_nonincreasing(F)
        cls = classify(delta, bao, lowk, mono)
        rows.append(("B2", f"k_b={k_b:.2f}, n={int(n)}", s8, delta, bao, lowk, mono, cls))

    # B3 case
    F = env_B3(k, **B3_def)
    Pp = F * P
    s8 = sigma8_from_pk(k, Pp)
    delta = (s8 / s8_base) - 1.0
    bao = bao_shift_flag(k, P, Pp)
    lowk = lowk_change_flag(k, F)
    mono = monotone_nonincreasing(F)
    cls = classify(delta, bao, lowk, mono)
    rows.append(("B3", f"k_i={B3_def['k_list']}, A_i={B3_def['A_i']:.2f}", s8, delta, bao, lowk, mono, cls))

    # Write detailed summary
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write("Phase-17B — Boundary-Integrated Density Response Test for σ8\n")
        f.write("Status: EXECUTED UNDER GOVERNANCE\n")
        f.write("Scope: Classification and bounding only (no models, no fits, no tuning)\n\n")
        f.write(f"Baseline sigma8 = {s8_base:.6f}\n\n")
        f.write("family\tparams\tDelta_sigma8[%]\tsigma8\tBAO_shift>1%\tlow-k_changed>1%\tmonotone\tclassification\n")
        for fam, params, s8v, dlt, bao, lowk, mono, cls in rows:
            f.write(
                f"{fam}\t{params}\t{dlt*100:+.2f}\t{s8v:.6f}\t{'YES' if bao else 'NO'}\t{'YES' if lowk else 'NO'}\t{'YES' if mono else 'NO'}\t{cls}\n"
            )
        f.write("\nPhase-17B completed: boundary-integrated density response bounded without models or tuning.\n")

    # Classification table (machine-friendly)
    with open(out_cls, "w", encoding="utf-8") as f:
        f.write("class\tparams\tsigma8\tDelta_sigma8[%]\tBAO_shift\tlow-k_changed\tmonotone\tclassification\n")
        for fam, params, s8v, dlt, bao, lowk, mono, cls in rows:
            f.write(
                f"{fam}\t{params}\t{s8v:.6f}\t{dlt*100:.2f}\t{('YES' if bao else 'NO')}\t{('YES' if lowk else 'NO')}\t{('YES' if mono else 'NO')}\t{cls}\n"
            )

    # Plot: Δσ8 vs k_b (B1 and B2); B3 as a single marker at effective k
    plt.figure(figsize=(10, 5))
    # Aggregate
    B1_points = {}
    B2_points = {}
    B3_point = None
    for fam, params, _s8v, dlt, _bao, _lowk, _mono, _cls in rows:
        if fam == "B1":
            # parse k_b and A
            parts = dict([p.split("=") for p in params.replace(" ", "").split(",")])
            k_b = float(parts["k_b"])
            A = float(parts["A"])
            B1_points.setdefault(A, []).append((k_b, dlt * 100.0))
        elif fam == "B2":
            parts = dict([p.split("=") for p in params.replace(" ", "").split(",")])
            k_b = float(parts["k_b"])
            n = int(parts["n"])
            B2_points.setdefault(n, []).append((k_b, dlt * 100.0))
        else:
            # B3 effective k: geometric mean of k_i
            k_list = [0.2, 0.4, 0.6, 0.8]
            gm = float(np.exp(np.mean(np.log(k_list))))
            B3_point = (gm, dlt * 100.0)

    # plot B1 series by A
    markers = {0.05: "o", 0.10: "s"}
    colors = {0.05: "tab:blue", 0.10: "tab:cyan"}
    for A, pts in sorted(B1_points.items()):
        pts = sorted(pts)
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker=markers.get(A, "o"), color=colors.get(A, "tab:blue"), label=f"B1 A={A:.2f}")

    # plot B2 series by n
    markers2 = {2: "^", 4: "v"}
    colors2 = {2: "tab:orange", 4: "tab:red"}
    for n, pts in sorted(B2_points.items()):
        pts = sorted(pts)
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker=markers2.get(n, "^"), color=colors2.get(n, "tab:orange"), label=f"B2 n={n}")

    # plot B3 single point
    if B3_point is not None:
        plt.scatter([B3_point[0]], [B3_point[1]], marker="*", s=160, color="tab:green", label="B3 (eff k)")
        plt.annotate("B3",
                     xy=(B3_point[0], B3_point[1]),
                     xytext=(5, -10), textcoords="offset points",
                     fontsize=10, color="tab:green")

    plt.axhline(0.0, color='k', lw=1)
    plt.xticks([0.3, 0.4, 0.5, 1.0], ["0.3", "0.4", "0.5", "1.0"])  # include B3 eff k ~0.44 near 0.4
    plt.xlabel(r"k_b [h/Mpc]")
    plt.ylabel(r"Δ$\sigma_8$ [%]")
    plt.title("Phase-17B: σ8 suppression vs boundary scale (no fits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()

    print(f"Wrote {out_sum}\nWrote {out_cls}\nSaved {out_fig}")


if __name__ == "__main__":
    main()
