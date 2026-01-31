import io
import json
import os
import sys
from contextlib import redirect_stdout

import bec_phi_kernel.cli as kernel_cli


def _run_cli(tmpdir, backend="cpu"):
    outbase = os.fspath(tmpdir)
    argv = [
        "cli.py",
        "--phi",
        "0.0",
        "10.0",
        "--resolution",
        "2000",
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
        return json.loads(s)
    finally:
        sys.argv = old_argv


def test_phase_transition_exists(tmp_path):
    j = _run_cli(tmp_path, backend="cpu")
    intervals = j.get("interval_count", 0)
    assert intervals >= 1, "Expected at least one L<0 interval in [0,10]"
