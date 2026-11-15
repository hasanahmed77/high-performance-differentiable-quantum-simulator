"""Simulator backends behind one typed protocol (ADR-0001 D4)."""

from typing import Any

from ketjet.backends.base import Backend, StateT
from ketjet.backends.numpy_backend import NumPyBackend

__all__ = ["Backend", "NumPyBackend", "StateT", "get_backend"]


def get_backend(name: str) -> Backend[Any]:
    """Return a new backend by name: ``"numpy"`` or ``"jax"``.

    The JAX backend is imported only when asked for, so ``import ketjet`` never imports JAX
    (ADR-0006 D1).

    Raises
    ------
    ValueError
        If ``name`` isn't a known backend.
    ImportError
        If the JAX backend is requested and JAX isn't installed.

    Examples
    --------
    >>> get_backend("numpy").name
    'numpy'
    """
    if name == "numpy":
        return NumPyBackend()
    if name == "jax":
        try:
            from ketjet.backends.jax_backend import JaxBackend  # noqa: PLC0415 - optional extra
        except ImportError as exc:
            raise ImportError(
                'The JAX backend needs JAX. Install it with: pip install "ketjet[jax]"'
            ) from exc
        return JaxBackend()
    raise ValueError(f"Unknown backend {name!r}. Available: jax, numpy.")
