"""Test-suite configuration."""

import importlib.util

# PennyLane is a test-only dependency that isn't available on every Python version (e.g. 3.15).
# Tests that need it use pytest.importorskip; the shared helper module must not even be collected
# (pytest imports every module for --doctest-modules) when PennyLane is missing.
collect_ignore = (
    []
    if importlib.util.find_spec("pennylane")
    else ["pennylane_support.py", "regression/generate_golden.py"]
)
