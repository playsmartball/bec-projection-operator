import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# No cosmology regeneration. Load a fiducial linear P(k) at z=0 from data/.
# Expected files (first available is used):
#   - data/matter_power_linear_z0.npz with arrays 'k' and 'P'
#   - data/matter_power_linear_z0.txt with two columns: k  P

try:
    sys.path.append(os.path.dirname(__file__))
    import phase16a_operator_class as op
except Exception as e:
    print("ERROR: Unable to import phase16a_operator_class.py from scripts/", file=sys.stderr)
    raise


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
            "Missing fiducial matter power spectrum. Place data/matter_power_linear_z0.(npz|txt) with k [h/Mpc], P [Mpc^3/h^3]."
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
    # 3 (sin x - x cos x) / x^3 with safe x=0 handling
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


def classify_delta(delta):
    # delta = (sigma8_op / sigma8_base) - 1
    if abs(delta) <= 0.005:
        return "FAIL"
    if delta < -0.05:
        return "OVER-SUPPRESSED"
    if -0.05 <= delta <= -0.02:
        return "PASS"
    return "FAIL"


def main():
    os.makedirs(os.path.join("output", "figures"), exist_ok=True)
    os.makedirs(os.path.join("output", "summaries"), exist_ok=True)

    k, P = load_pk()

    sigma8_base = sigma8_from_pk(k, P)

    forms = [("soft_cutoff", 1), ("soft_cutoff", 2), ("exponential", 1), ("exponential", 2)]
    kcs = [0.10, 0.20, 0.30]  # h/Mpc; observational window scales; locked a priori

    rows = []
    for form, n in forms:
        for kc in kcs:
            W = op.KOperator(form=form, kc=kc, n=n)
            P_obs = W.apply(k, P)
            s8 = sigma8_from_pk(k, P_obs)
            delta = (s8 / sigma8_base) - 1.0
            cls = classify_delta(delta)
            rows.append((form, n, kc, s8, delta, cls))

    # Save text summary
    txt_path = os.path.join("output", "summaries", "phase16b_sigma8_null_test.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"sigma8_base = {sigma8_base:.6f}\n")
        f.write("form\tn\tkc[h/Mpc]\tsigma8\tdelta\tclass\n")
        for form, n, kc, s8, delta, cls in rows:
            f.write(f"{form}\t{n}\t{kc:.3f}\t{s8:.6f}\t{delta:+.4%}\t{cls}\n")

    # Figure
    fig_path = os.path.join("output", "figures", "phase16b_sigma8_null_test.png")
    try:
        labels = [f"{f}-{n}-kc{kc:.2f}" for (f, n, kc, *_rest) in rows]
        svals = [s8 for *_p, s8, _d, _c in rows]
        x = np.arange(len(rows))
        plt.figure(figsize=(12, 4))
        plt.axhline(sigma8_base, color="k", lw=1, label="baseline")
        plt.plot(x, svals, "o", label="operators")
        plt.xticks(x, labels, rotation=60, ha="right")
        plt.ylabel(r"$\sigma_8$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    except Exception as e:
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"\n[warning] plotting failed: {e}\n")

    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    # Global constraints are enforced by construction: no growth/gen, no z-dependence, no fitting.
    main()
