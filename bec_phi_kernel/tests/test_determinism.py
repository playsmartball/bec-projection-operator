import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

from bec_phi_kernel import cli as kernel_cli
import csv


def _run_cli(tmpdir, backend="cpu"):
    outbase = os.fspath(tmpdir)
    argv = [
        "cli.py",
        "--phi",
        "0.0",
        "10.0",
        "--resolution",
        "5000",
        "--backend",
        backend,
        "--outbase",
        outbase,
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            kernel_cli.main()
        s = f.getvalue()
        j = json.loads(s)
        return j
    finally:
        sys.argv = old_argv


def test_cpu_determinism(tmp_path):
    j1 = _run_cli(tmp_path / "run1", backend="cpu")
    j2 = _run_cli(tmp_path / "run2", backend="cpu")
    # Hashes must be identical
    assert j1["sha256"] == j2["sha256"]


def test_cpu_gpu_parity_if_available(tmp_path):
    try:
        import cupy as cp  # noqa: F401
        try:
            from cupy.cuda import runtime as _runtime  # type: ignore
            from cupy_backends.cuda.libs import nvrtc as _nvrtc  # type: ignore
            _ = _nvrtc.getVersion()
            if _runtime.getDeviceCount() == 0:
                return
        except Exception:
            # CuPy present but runtime/NVRTC not operational; skip parity
            return
    except Exception:
        # GPU not available; parity not applicable but determinism already tested above
        return
    j_cpu = _run_cli(tmp_path / "cpu", backend="cpu")
    j_gpu = _run_cli(tmp_path / "gpu", backend="gpu")
    # Numerical parity within tolerance (CPU==GPU to 1e-12 absolute on all columns)
    cpu_csv = os.path.join(tmp_path, "cpu", "phi_trace.csv")
    gpu_csv = os.path.join(tmp_path, "gpu", "phi_trace.csv")
    with open(cpu_csv, "r", encoding="utf-8") as f1, open(gpu_csv, "r", encoding="utf-8") as f2:
        r1 = list(csv.reader(f1))
        r2 = list(csv.reader(f2))
    assert r1[0] == r2[0]  # header
    for a, b in zip(r1[1:], r2[1:]):
        # compare each numeric column
        for x, y in zip(a, b):
            assert abs(float(x) - float(y)) <= 1e-12
