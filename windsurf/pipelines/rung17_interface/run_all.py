import os
from governance.guard import (
    find_repo_root,
    assert_imports,
    safe_write_text,
    sha256_file,
)

from .sign_test import run as run_sign
from .kernel_support_test import run as run_kernel
from .orthogonality_test import run as run_ortho
from .ordering_test import run as run_order


def main() -> str:
    assert_imports()
    repo_root = find_repo_root(os.path.dirname(__file__))

    p1 = run_sign()
    p2 = run_kernel()
    p3 = run_ortho()
    p4 = run_order()

    # Hashes
    h1 = sha256_file(p1)
    h2 = sha256_file(p2)
    h3 = sha256_file(p3)
    h4 = sha256_file(p4)

    # Minimal roll-up with PASS/FAIL table and stop statement
    md = []
    md.append("# Rung-17 Interface Verification Roll-up")
    md.append("")
    md.append("Governance: preserved. No fitting, no tuning, no model execution.")
    md.append("")
    md.append("## Results (PASS/FAIL)")
    md.append("")
    def _load_json(pp: str):
        import json
        with open(pp, "r", encoding="utf-8") as f:
            return json.load(f)
    j1 = _load_json(p1)
    j2 = _load_json(p2)
    j3 = _load_json(p3)
    j4 = _load_json(p4)

    rows = [
        ("Sign Consistency", j1.get("result"), os.path.basename(p1), h1),
        ("Kernel Support", j2.get("result"), os.path.basename(p2), h2),
        ("k-Orthogonality", j3.get("result"), os.path.basename(p3), h3),
        ("Ordering/Causality", j4.get("result"), os.path.basename(p4), h4),
    ]

    md.append("| Test | Result | File | SHA256 |")
    md.append("|------|--------|------|--------|")
    for name, res, fn, hh in rows:
        md.append(f"| {name} | {res} | {fn} | `{hh}` |")

    md.append("")
    md.append("## Stop Statement")
    md.append("")
    md.append("Interface tests complete. No fitting, no tuning, no model execution. Governance preserved.")
    md.append("")

    text = "\n".join(md)
    outpath = safe_write_text(repo_root, "INTERFACE_ROLLUP_RUNG17.md", text)
    return outpath


if __name__ == "__main__":
    print(main())
