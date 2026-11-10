"""Root test configuration: applies to the doctests collected from ``src/``."""

import importlib.util

# ketjet.pennylane imports PennyLane, an optional extra; skip its doctests when it's missing.
collect_ignore = [] if importlib.util.find_spec("pennylane") else ["src/ketjet/pennylane.py"]
