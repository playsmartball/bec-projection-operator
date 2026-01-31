import io
import os
import re
import sys
import inspect
from contextlib import redirect_stdout

import numpy as np

from bec_phi_kernel.math import phi_state as ps
from bec_phi_kernel.math import operators as ops
from bec_phi_kernel.math import normalization as norm
import bec_phi_kernel.cli as kernel_cli


def test_no_alpha_beta_parameters():
    # No tunable alpha/beta in operators
    assert not hasattr(ops, "ALPHA")
    assert not hasattr(ops, "BETA")


def test_operator_signature_minimal():
    sig = inspect.signature(ops.L_function)
    params = list(sig.parameters.keys())
    assert params == ["phi", "rho", "P", "use_gpu"], params
    # default
    assert sig.parameters["use_gpu"].default is False


def test_phi_state_signature_minimal():
    sig = inspect.signature(ps.phi_state)
    params = list(sig.parameters.keys())
    assert params == ["phi", "xp"], params
    assert sig.parameters["xp"].default is None


def test_normalization_only_units_uppercase():
    upper = [n for n in dir(norm) if n.isupper()]
    # Only UNITS defined as uppercase public constant
    assert upper == ["UNITS"], upper


def test_cli_options_are_fixed():
    old_argv = sys.argv
    sys.argv = [
        "cli.py",
        "--help",
    ]
    # Capture help output; argparse will sys.exit
    try:
        f = io.StringIO()
        try:
            with redirect_stdout(f):
                kernel_cli.main()
        except SystemExit:
            pass
        help_text = f.getvalue()
    finally:
        sys.argv = old_argv
    # Only these options must appear
    for opt in ("--phi", "--resolution", "--backend", "--outbase"):
        assert opt in help_text
    # No common tuning words
    bad_words = ["alpha", "beta", "fit", "tune", "scan", "prior"]
    tlow = help_text.lower()
    for w in bad_words:
        assert w not in tlow
