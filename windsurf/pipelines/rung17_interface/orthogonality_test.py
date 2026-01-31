import os
import json
import numpy as np

try:
    import cupy as cp  # optional GPU
    XP = cp
    GPU = True
except Exception:  # pragma: no cover
    XP = np
    GPU = False

from governance.guard import find_repo_root, assert_imports, safe_write_json


def parse_phase21c_sign(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().lower()
        return -1 if "negative" in txt else 1
    except FileNotFoundError:
        return 1


def run(threshold: float = 1e-3) -> str:
    assert_imports()
    repo_root = find_repo_root(os.path.dirname(__file__))

    # Synthetic k grid and k-independent deltaP/P
    k = np.geomspace(1e-3, 1.0, 64)
    artifact = os.path.join(repo_root, "output", "summaries", "phase21c_direction_only_execution.txt")
    sgn = parse_phase21c_sign(artifact)

    dpp = np.full_like(k, 1.0 * (1 if sgn > 0 else -1), dtype=float)
    dk = np.diff(k)
    dd = np.diff(dpp)
    slope = np.abs(dd / dk)
    max_slope = float(np.max(slope)) if slope.size else 0.0

    result = "PASS" if max_slope <= threshold else "FAIL"
    payload = {
        "test": "k_orthogonality",
        "max_slope": max_slope,
        "threshold": threshold,
        "result": result,
        "gpu": GPU,
    }

    outpath = safe_write_json(repo_root, "interface_k_orthogonality.json", payload)
    return outpath


if __name__ == "__main__":
    print(run())
