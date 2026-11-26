"""Benchmark cases (QS-1102): a benchmark, a runner, a size, a device and a precision.

:func:`prepare` turns a case into the call to time and a check of its result against the NumPy
oracle in complex128 (ADR-0008 D1). Runners:

``ketjet-numpy``
    The NumPy backend: ``Circuit.state()``, or ``gradients.grad(..., method="adjoint")``.
``ketjet-jax``
    The JAX backend, compiled: ``compile_circuit(circuit)(x)``, or ``jax.grad`` of ``expval_fn``
    with backprop. Synchronized with ``block_until_ready``.
``ketjet-jax-adjoint``
    As ``ketjet-jax``, with ``diff_method="adjoint"``; gradient benchmarks only.

The other simulators' runners (QS-1103) are in :mod:`benchmarks.libraries`. They run on the CPU,
in the precisions each library has.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from tests.tolerances import Tolerance, tolerance_for  # one source of tolerances (§6.3)

from benchmarks.circuits import BENCHMARKS, GRADIENT_BENCHMARKS, QML_BATCH, Spec
from benchmarks.libraries import LIBRARY_RUNNERS, library_options, prepare_library, supports
from ketjet import gradients

KETJET_RUNNERS: Final = ("ketjet-numpy", "ketjet-jax", "ketjet-jax-adjoint")
RUNNERS: Final = (*KETJET_RUNNERS, *LIBRARY_RUNNERS)
JAX_RUNNERS: Final = frozenset({"ketjet-jax", "ketjet-jax-adjoint"})
DEVICES: Final = ("cpu", "gpu")
PRECISIONS: Final = ("complex128", "complex64")

SIZES: Final[dict[str, tuple[int, ...]]] = {
    "gate-h": (4, 8, 12, 16, 20),
    "gate-rz": (4, 8, 12, 16, 20),
    "gate-cnot": (4, 8, 12, 16, 20),
    "qft": (4, 8, 12, 16, 20),
    "random": (4, 8, 12, 16, 20),
    "vqe-grad": (4, 8, 12, 16, 20),
    "qml-batch": (4, 8, 12, 16),
}
"""Qubit counts of the published run, per benchmark."""

SMOKE_SIZE: Final = 3
"""The tiny size CI runs every benchmark at."""

GRADIENT_TOLERANCES: Final = {
    "complex128": Tolerance(atol=1e-9, rtol=1e-8),
    "complex64": Tolerance(atol=1e-3, rtol=1e-3),
}
"""Gradients against the NumPy adjoint in complex128; complex64 as for the GPU suite."""


class WrongResultError(AssertionError):
    """A case's result disagreed with the oracle, so it wasn't timed (ADR-0008 D1)."""


@dataclass(frozen=True, order=True)
class Case:
    """One thing to time."""

    benchmark: str
    runner: str
    n_qubits: int
    device: str = "cpu"
    precision: str = "complex128"

    @property
    def id(self) -> str:
        """A unique, readable name, e.g. ``qft/ketjet-jax/cpu/complex128/n12``."""
        return f"{self.benchmark}/{self.runner}/{self.device}/{self.precision}/n{self.n_qubits}"


def cases(
    benchmarks: Iterable[str] = BENCHMARKS,
    runners: Iterable[str] = RUNNERS,
    sizes: Sequence[int] | None = None,
    device: str = "cpu",
    precision: str = "complex128",
) -> list[Case]:
    """Every case of a run: ``sizes`` overrides :data:`SIZES` for every benchmark.

    The adjoint runner is used only for gradient benchmarks, and the GPU only by JAX runners.
    Other libraries run where :func:`benchmarks.libraries.supports` says they can, on the CPU.
    """
    chosen_runners = tuple(runners)
    for name in (*chosen_runners, device, precision):
        if name not in (*RUNNERS, *DEVICES, *PRECISIONS):
            raise ValueError(f"Unknown runner, device or precision {name!r}.")
    result: list[Case] = []
    for benchmark in benchmarks:
        if benchmark not in BENCHMARKS:
            known = ", ".join(BENCHMARKS)
            raise ValueError(f"Unknown benchmark {benchmark!r}; choose from {known}.")
        for runner in chosen_runners:
            if runner == "ketjet-jax-adjoint" and benchmark not in GRADIENT_BENCHMARKS:
                continue
            if device == "gpu" and runner not in JAX_RUNNERS:
                continue
            gradient = benchmark in GRADIENT_BENCHMARKS
            if runner in LIBRARY_RUNNERS and not supports(runner, benchmark, precision, gradient):
                continue
            result.extend(
                Case(benchmark, runner, n, device, precision)
                for n in (sizes if sizes is not None else SIZES[benchmark])
            )
    return result


@dataclass
class Prepared:
    """A case ready to time: the call, the check of its result, and facts for the record."""

    call: Callable[[], Any]
    check: Callable[[Any], None]
    info: dict[str, Any] = field(default_factory=dict)


def _checker(
    reference: npt.NDArray[Any], tolerance: Tolerance, case: Case
) -> Callable[[Any], None]:
    def check(result: Any) -> None:
        try:
            np.testing.assert_allclose(
                np.asarray(result), reference, atol=tolerance.atol, rtol=tolerance.rtol
            )
        except AssertionError as error:
            message = f"{case.id} disagrees with the oracle; not timed.\n{error}"
            raise WrongResultError(message) from error

    return check


def _values(spec: Spec, values: npt.NDArray[np.float64]) -> dict[str, float]:
    return {p.name: float(v) for p, v in zip(spec.circuit.parameters, values, strict=True)}


def _qml_values(spec: Spec, inputs: npt.NDArray[np.float64]) -> dict[str, float]:
    return _values(spec, np.concatenate([inputs, spec.values]))


def _reference(benchmark: str, spec: Spec) -> npt.NDArray[Any]:
    """Return the NumPy oracle's answer, in complex128."""
    if benchmark == "vqe-grad":
        assert spec.observable is not None
        return gradients.grad(spec.circuit, spec.observable, _values(spec, spec.values))
    if benchmark == "qml-batch":
        return _numpy_qml(spec, "complex128")()
    return np.asarray(spec.bound().state())


def _numpy_qml(spec: Spec, precision: str) -> Callable[[], npt.NDArray[np.float64]]:
    assert spec.observable is not None
    assert spec.inputs is not None
    observable, inputs, n = spec.observable, spec.inputs, spec.circuit.n_wires

    def call() -> npt.NDArray[np.float64]:
        total = np.zeros(len(spec.values))
        for x in inputs:
            total += gradients.grad(
                spec.circuit, observable, _qml_values(spec, x), dtype=precision
            )[n:]
        return total / len(inputs)

    return call


def _numpy(case: Case, spec: Spec) -> Callable[[], Any]:
    if case.benchmark == "vqe-grad":
        assert spec.observable is not None
        observable, values = spec.observable, _values(spec, spec.values)
        return lambda: gradients.grad(spec.circuit, observable, values, dtype=case.precision)
    if case.benchmark == "qml-batch":
        return _numpy_qml(spec, case.precision)
    bound = spec.bound()
    return lambda: bound.state(dtype=case.precision)


def _jax(case: Case, spec: Spec, info: dict[str, Any]) -> Callable[[], Any]:
    import jax  # noqa: PLC0415 - only JAX runners need JAX
    import jax.numpy as jnp  # noqa: PLC0415

    from ketjet.backends import jax_backend  # noqa: PLC0415

    # The backend never switches 64-bit mode on itself (ADR-0006 D5): a user asking for
    # complex128 does, and one asking for complex64 needn't.
    x64 = case.precision == "complex128"
    jax.config.update("jax_enable_x64", x64)  # type: ignore[no-untyped-call]
    backend = jax.default_backend()
    if (backend == "cpu") != (case.device == "cpu"):
        raise RuntimeError(f"{case.id} needs JAX on the {case.device}, but JAX runs on {backend}.")
    info.update(jax_backend=backend, jax_device=str(jax.devices()[0]), jax_x64=x64)
    real = np.float64 if case.precision == "complex128" else np.float32
    if case.benchmark not in GRADIENT_BENCHMARKS:
        bound = spec.bound()
        run = jax_backend.compile_circuit(bound, dtype=case.precision)
        x = jnp.asarray(jax_backend.parameter_values(bound), dtype=real)
        return lambda: run(x).block_until_ready()
    assert spec.observable is not None
    adjoint = case.runner == "ketjet-jax-adjoint"
    method: jax_backend.DiffMethod = "adjoint" if adjoint else "backprop"
    f = jax_backend.expval_fn(
        spec.circuit, [spec.observable], dtype=case.precision, diff_method=method
    )
    weights = jnp.asarray(spec.values, dtype=real)
    if case.benchmark == "vqe-grad":
        grad = jax.jit(jax.grad(lambda v: f(v)[0]))
        return lambda: grad(weights).block_until_ready()
    assert spec.inputs is not None
    inputs = jnp.asarray(spec.inputs, dtype=real)

    def loss(w: jax.Array, xs: jax.Array) -> jax.Array:
        return jax.vmap(lambda x: f(jnp.concatenate([x, w]))[0])(xs).mean()

    batch_grad = jax.jit(jax.grad(loss))
    return lambda: batch_grad(weights, inputs).block_until_ready()


def prepare(case: Case) -> Prepared:
    """Build ``case``'s call and its check against the oracle (ADR-0008 D1)."""
    spec = BENCHMARKS[case.benchmark](case.n_qubits)
    reference = _reference(case.benchmark, spec)
    gradient = case.benchmark in GRADIENT_BENCHMARKS
    tolerance = GRADIENT_TOLERANCES[case.precision] if gradient else tolerance_for(case.precision)
    info: dict[str, Any] = {
        "gates": spec.gates,
        "parameters": len(spec.circuit.parameters),
        "batch": QML_BATCH if case.benchmark == "qml-batch" else None,
    }
    if case.runner in LIBRARY_RUNNERS:
        if case.device != "cpu":
            raise RuntimeError(f"{case.id}: the other libraries run on the CPU only (QS-1103).")
        call = prepare_library(case.runner, case.benchmark, spec, case.precision)
        info["options"] = library_options(case.runner, case.precision)
    elif case.runner in JAX_RUNNERS:
        call = _jax(case, spec, info)
    else:
        call = _numpy(case, spec)
    return Prepared(call, _checker(reference, tolerance, case), info)
