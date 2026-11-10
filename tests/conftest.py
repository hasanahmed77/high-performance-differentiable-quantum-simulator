"""Test-suite configuration."""

import importlib.util

# PennyLane is an optional extra (ketjet[pennylane]) and part of the test group. Tests that need it
# use pytest.importorskip, so the suite still runs without it; the shared helper modules must not
# even be collected (pytest imports every module for --doctest-modules) when it's missing.
collect_ignore = (
    []
    if importlib.util.find_spec("pennylane")
    else ["pennylane_support.py", "regression/generate_golden.py"]
)
