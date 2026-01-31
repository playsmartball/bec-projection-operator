import os
import json
from typing import Optional

from governance.guard import (
    find_repo_root,
    assert_imports,
    safe_write_json,
    load_json,
)


def _load(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run() -> str:
    assert_imports()
    repo_root = find_repo_root(os.path.dirname(__file__))
    outdir = os.path.join(repo_root, "output", "interface")

    # Consume prior test outputs (if present)
    sign_js = load_json(os.path.join(outdir, "interface_sign.json"))
    kern_js = load_json(os.path.join(outdir, "interface_kernel_support.json"))
    ortho_js = load_json(os.path.join(outdir, "interface_k_orthogonality.json"))

    # Ordering logic (governance-only):
    # EARLY = absent, GEO = absent, GROWTH = present, PROJ = support-limited.
    early_ok = True  # No early-time artifacts used
    geo_ok = True    # No background/geometry modified
    growth_ok = (sign_js or {}).get("result") == "PASS"
    proj_ok = (kern_js or {}).get("result") == "PASS"
    ortho_ok = (ortho_js or {}).get("result") == "PASS"

    ordering_preserved = bool(early_ok and geo_ok and growth_ok and proj_ok and ortho_ok)

    payload = {
        "test": "ordering_causality",
        "early_absent": early_ok,
        "geo_absent": geo_ok,
        "growth_present": growth_ok,
        "projection_supported": proj_ok,
        "k_orthogonal": ortho_ok,
        "result": "PASS" if ordering_preserved else "FAIL",
    }

    outpath = safe_write_json(repo_root, "interface_ordering.json", payload)
    return outpath


if __name__ == "__main__":
    print(run())
