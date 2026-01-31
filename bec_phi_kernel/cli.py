from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import List, Tuple

from bec_phi_kernel.sim import solver_cpu as cpu
from bec_phi_kernel.sim import solver_gpu as gpu


def _ensure_outdir(base: str) -> str:
    # Write directly into the provided base directory
    os.makedirs(base, exist_ok=True)
    return base


def _write_csv(path: str, rows: List[Tuple[float, ...]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phi", "rho", "P", "c_s", "lambda_c", "epsilon", "L"])
        for r in rows:
            w.writerow([f"{x:.17g}" for x in r])


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="FMI v0.1 CLI (Phi kernel)")
    p.add_argument("--phi", nargs=2, type=float, required=True, metavar=("PHI0", "PHI1"))
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--outbase", default=os.path.dirname(__file__))
    args = p.parse_args()

    phi0, phi1 = float(args.phi[0]), float(args.phi[1])
    n = int(args.resolution)

    if args.backend == "gpu":
        st, intervals = gpu.solve(phi0, phi1, n)
    else:
        st, intervals = cpu.solve(phi0, phi1, n)

    outdir = _ensure_outdir(args.outbase)

    # rows for CSV
    rows = list(
        zip(
            st["phi"].tolist(),
            st["rho"].tolist(),
            st["P"].tolist(),
            st["c_s"].tolist(),
            st["lambda_c"].tolist(),
            st["epsilon"].tolist(),
            st["L"].tolist(),
        )
    )

    csv_path = os.path.join(outdir, "phi_trace.csv")
    _write_csv(csv_path, rows)

    j = {
        "phi_start": phi0,
        "phi_end": phi1,
        "resolution": n,
        "backend": args.backend,
        "intervals_L_negative": intervals,
    }
    json_path = os.path.join(outdir, "phi_trace.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2, sort_keys=True)

    sha = _sha256_file(csv_path)
    with open(os.path.join(outdir, "hash.txt"), "w", encoding="utf-8") as f:
        f.write(sha + "\n")

    summary = {
        "csv": os.path.abspath(csv_path),
        "json": os.path.abspath(json_path),
        "sha256": sha,
        "interval_count": len(intervals),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
