"""Root test configuration: applies to the tests and to the doctests collected from ``src/``."""

import importlib.util

# ketjet.pennylane and ketjet.backends.jax_backend import optional extras; skip their doctests
# when the extra is missing.
collect_ignore = [] if importlib.util.find_spec("pennylane") else ["src/ketjet/pennylane.py"]

if importlib.util.find_spec("jax"):
    import jax

    # complex128 on the JAX backend needs JAX's 64-bit mode, and ketjet never switches it on for
    # the user (ADR-0006 D5). The suite does, once, here.
    jax.config.update("jax_enable_x64", True)
else:
    collect_ignore.append("src/ketjet/backends/jax_backend.py")
