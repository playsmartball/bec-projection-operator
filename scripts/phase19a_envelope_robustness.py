import os
import numpy as np
import matplotlib.pyplot as plt

# Phase-19A: Envelope Robustness Execution (σ8 only)
# Executes ONLY the approved smooth, k-local, monotone B2 envelope on the locked P(k).
# No new envelopes, no tuning, no optimization. Outputs stability/constraint checks only.


def load_pk_txt(path_txt: str):
    if not os.path.exists(path_txt):
        raise FileNotFoundError("Missing fiducial matter power spectrum: data/matter_power_linear_z0.txt")
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


# Envelope: B2 smooth boundary roll-off

def env_B2(k, k_b: float, n: int):
    return 1.0 / (1.0 + (k / k_b) ** n)


def monotone_nonincreasing(F, tol=1e-8):
    d = np.diff(F)
    return bool(np.all(d <= tol))


def moving_average(y, win: int):
    if win < 3:
        return y.copy()
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
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
    N = len(kk)
    win = max(5, min(51, (N // 8) * 2 + 1))
    ys = moving_average(yy, win)
    imax = int(np.argmax(ys))
    return float(kk[imax])


def bao_shift_flag(k, P_base, P_obs, thr_frac=0.01):
    kb = find_first_bao_peak_k(k, P_base)
    ko = find_first_bao_peak_k(k, P_obs)
    if kb is None or ko is None or kb <= 0:
        return False
    frac = abs(ko - kb) / kb
    return bool(frac > thr_frac)


def lowk_change_flag(k, F, kthr=0.05, thr_frac=0.01):
    m = k < kthr
    if not np.any(m):
        return False
    mean_abs = float(np.mean(np.abs(F[m] - 1.0)))
    return bool(mean_abs > thr_frac)


def main():
    proj_root = "."
    data_txt = os.path.join(proj_root, "data", "matter_power_linear_z0.txt")
    out_sum = os.path.join(proj_root, "output", "summaries", "phase19a_sigma8_envelope_robustness.txt")
    out_tab = os.path.join(proj_root, "output", "summaries", "phase19a_sigma8_envelope_robustness_table.txt")
    out_fig = os.path.join(proj_root, "output", "figures", "phase19a_robustness_b2.png")

    os.makedirs(os.path.dirname(out_sum), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    # Locked grid
    kb_values = [0.30, 0.50]
    n_values = [2, 4]

    k, P = load_pk_txt(data_txt)
    s8_base = sigma8_from_pk(k, P)

    # Compute rows
    rows = []  # (k_b, n, s8, delta, bao, lowk, mono, in_range, w0_ok, repeat_ok)
    for k_b in kb_values:
        for n in n_values:
            F = env_B2(k, k_b=k_b, n=n)
            # Flags
            mono = monotone_nonincreasing(F)
            in_range = (float(np.min(F)) >= -1e-8) and (float(np.max(F)) <= 1.0 + 1e-8)
            w0_ok = bool(abs(F[0] - 1.0) <= 1e-6)

            Pp = F * P
            s8_1 = sigma8_from_pk(k, Pp)
            s8_2 = sigma8_from_pk(k, Pp)
            repeat_ok = bool(abs(s8_2 - s8_1) <= max(1e-12, 1e-12 * s8_1))
            delta = (s8_1 / s8_base) - 1.0

            bao = bao_shift_flag(k, P, Pp)
            lowk = lowk_change_flag(k, F)

            rows.append((k_b, int(n), float(s8_1), float(delta), bao, lowk, mono, in_range, w0_ok, repeat_ok))

    # Ordering checks (no fits)
    def find_row(kb, nn):
        for r in rows:
            if abs(r[0] - kb) < 1e-12 and r[1] == nn:
                return r
        return None

    ord_checks = []
    # For fixed n: |Δσ8(0.30)| >= |Δσ8(0.50)|
    for nn in n_values:
        d030 = abs(find_row(0.30, nn)[3])
        d050 = abs(find_row(0.50, nn)[3])
        ord_checks.append(d030 >= d050 - 1e-12)
    # For fixed k_b: |Δσ8(n=2)| >= |Δσ8(n=4)|
    for kb in kb_values:
        d_n2 = abs(find_row(kb, 2)[3])
        d_n4 = abs(find_row(kb, 4)[3])
        ord_checks.append(d_n2 >= d_n4 - 1e-12)
    ordering_ok = bool(np.all(ord_checks))

    # Overall pass per row and global
    def row_pass(r):
        _kb, _n, _s8, _d, bao, lowk, mono, in_range, w0_ok, repeat_ok = r
        return (not bao) and (not lowk) and mono and in_range and w0_ok and repeat_ok and np.isfinite(_s8)

    rows_pass = [row_pass(r) for r in rows]
    all_rows_pass = bool(np.all(rows_pass))

    # Write summary
    with open(out_sum, "w", encoding="utf-8") as f:
        f.write("Phase-19A — Envelope Robustness Execution (σ8 only)\n")
        f.write("Status: EXECUTED UNDER GOVERNANCE\n")
        f.write("Scope: B2 smooth roll-off only; no tuning; σ8 diagnostic and constraint checks.\n\n")
        f.write(f"Baseline sigma8 = {s8_base:.6f}\n\n")
        f.write("k_b\tn\tDelta_sigma8[%]\tsigma8\tBAO_shift>1%\tlow-k>1%\tmonotone\tin_range\tW0_ok\trepeat_ok\trow_status\n")
        for (k_b, n, s8, dlt, bao, lowk, mono, in_range, w0_ok, repeat_ok), rp in zip(rows, rows_pass):
            f.write(
                f"{k_b:.2f}\t{n}\t{dlt*100:+.2f}\t{s8:.6f}\t{'YES' if bao else 'NO'}\t{'YES' if lowk else 'NO'}\t{'YES' if mono else 'NO'}\t{'YES' if in_range else 'NO'}\t{'YES' if w0_ok else 'NO'}\t{'YES' if repeat_ok else 'NO'}\t{'PASS' if rp else 'FAIL'}\n"
            )
        f.write("\nOrdering checks (no fits):\n")
        f.write("- For fixed n: |Δσ8(0.30)| ≥ |Δσ8(0.50)|\n")
        f.write("- For fixed k_b: |Δσ8(n=2)| ≥ |Δσ8(n=4)|\n")
        f.write(f"Ordering verdict: {'PASS' if ordering_ok else 'FAIL'}\n")
        f.write(f"All rows constraints: {'PASS' if all_rows_pass else 'FAIL'}\n")
        f.write("\nPhase-19A completed: robustness checked; no models, no tuning, no broader interpretation.\n")

    # Machine table
    with open(out_tab, "w", encoding="utf-8") as f:
        f.write("params\tsigma8\tDelta_sigma8[%]\tBAO_shift\tlow-k_changed\tmonotone\tin_range\tW0_ok\trepeat_ok\trow_status\n")
        for (k_b, n, s8, dlt, bao, lowk, mono, in_range, w0_ok, repeat_ok), rp in zip(rows, rows_pass):
            f.write(
                f"k_b={k_b:.2f}, n={n}\t{s8:.6f}\t{dlt*100:.2f}\t{('YES' if bao else 'NO')}\t{('YES' if lowk else 'NO')}\t{('YES' if mono else 'NO')}\t{('YES' if in_range else 'NO')}\t{('YES' if w0_ok else 'NO')}\t{('YES' if repeat_ok else 'NO')}\t{('PASS' if rp else 'FAIL')}\n"
            )

    # Plot
    plt.figure(figsize=(9, 5))
    series = {2: [], 4: []}
    for (k_b, n, _s8, dlt, *_rest) in rows:
        series[n].append((k_b, dlt * 100.0))
    markers = {2: "o", 4: "s"}
    colors = {2: "tab:blue", 4: "tab:orange"}
    for n, pts in series.items():
        pts = sorted(pts)
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker=markers[n], color=colors[n], label=f"B2 n={n}")
    plt.axhline(0.0, color='k', lw=1)
    plt.xlabel("k_b [h/Mpc]")
    plt.ylabel("Δσ8 [%]")
    plt.title("Phase-19A: σ8 robustness under B2 envelope (no tuning)")
    verdict = "PASS" if (ordering_ok and all_rows_pass) else "CHECK"
    plt.annotate(f"ordering: {'PASS' if ordering_ok else 'FAIL'}\nconstraints: {'PASS' if all_rows_pass else 'FAIL'}",
                 xy=(0.32, 0.05), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()

    print(f"Wrote {out_sum}\nWrote {out_tab}\nSaved {out_fig}")


if __name__ == "__main__":
    main()
