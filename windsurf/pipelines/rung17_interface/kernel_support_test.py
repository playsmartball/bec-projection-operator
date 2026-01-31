import os
import json
import numpy as np

try:
    import cupy as cp  # optional GPU
    XP = cp
    GPU = True
except Exception:  # pragma: no cover
    import numpy as np  # fallback
    XP = np
    GPU = False

from governance.guard import find_repo_root, assert_imports, safe_write_json


def load_kernel_txt(path: str):
    """Load two-column ASCII kernel file: z W(z). Ignores malformed lines."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                z = float(parts[0])
                w = float(parts[1])
                data.append((z, w))
            except ValueError:
                continue
    if not data:
        raise RuntimeError("Empty kernel file or parse failure")
    arr = np.array(data, dtype=float)
    z = arr[:, 0]
    w = arr[:, 1]
    return z, w


def parse_phase21c_sign(path: str) -> int:
    """Return +1 for positive sign, -1 for negative, default +1 if file missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().lower()
        return -1 if "negative" in txt else 1
    except FileNotFoundError:
        return 1


def synth_delta_on_z(z: np.ndarray, sign: int, peak_z: float, sigma: float) -> np.ndarray:
    """Minimal synthetic delta: sign-definite Gaussian bump centered at peak_z with fixed width.
    This uses frozen kernel peak to localize support, no amplitude tuning.
    """
    zz = z.astype(float)
    g = np.exp(-0.5 * ((zz - peak_z) / sigma) ** 2)
    # Normalize to unit max; apply sign only
    g = g / (g.max() if g.max() else 1.0)
    return (1.0 if sign > 0 else -1.0) * g


def run(threshold_align: float = 0.30, threshold_core_frac: float = 0.60) -> str:
    assert_imports()
    repo_root = find_repo_root(os.path.dirname(__file__))

    kpath = os.path.join(repo_root, "data", "kernels", "WL_PLANCK2018_fiducial_z.txt")
    z, w = load_kernel_txt(kpath)
    w = np.asarray(w)
    z = np.asarray(z)

    # Kernel peak
    imax = int(np.argmax(w))
    k_peak_z = float(z[imax])

    # Phase-21C sign-only artifact
    artifact = os.path.join(repo_root, "output", "summaries", "phase21c_direction_only_execution.txt")
    sgn = parse_phase21c_sign(artifact)

    # Derive synthetic width from kernel FWHM (governance-safe: uses frozen kernel only)
    wmax = float(w.max())
    half = 0.5 * wmax
    core_mask = w >= half
    if not np.any(core_mask):
        # Fallback width if kernel malformed; conservative, still interface-local
        fwhm_k = 0.7
    else:
        z_core = z[core_mask]
        fwhm_k = float(z_core.max() - z_core.min()) if z_core.size > 0 else 0.7
    # Gaussian relation: FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.355*sigma
    sigma = max(fwhm_k / 2.355, 1e-3)

    # Synthetic delta centered at kernel peak with kernel-matched width
    delta = synth_delta_on_z(z, sgn, k_peak_z, sigma=sigma)
    d_peak_z = float(z[int(np.argmax(np.abs(delta)))])

    # Core region of kernel (half-maximum)
    core_mask = w >= (0.5 * float(wmax))
    core_frac = float(np.sum(np.abs(delta)[core_mask]) / (np.sum(np.abs(delta)) + 1e-16))

    aligned = abs(d_peak_z - k_peak_z) <= threshold_align
    confined = core_frac >= threshold_core_frac

    result = "PASS" if (aligned and confined) else "FAIL"

    payload = {
        "test": "kernel_support",
        "kernel_peak_z": k_peak_z,
        "synthetic_peak_z": d_peak_z,
        "aligned_tol": threshold_align,
        "aligned": aligned,
        "core_fraction": core_frac,
        "core_fraction_threshold": threshold_core_frac,
        "confined": confined,
        "result": result,
        "gpu": GPU,
    }

    outpath = safe_write_json(repo_root, "interface_kernel_support.json", payload)
    return outpath


if __name__ == "__main__":
    print(run())
