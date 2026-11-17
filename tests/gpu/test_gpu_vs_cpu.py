"""The JAX backend on a GPU gives the CPU's results (QS-607, ADR-0007 D4).

Every test runs the same computation twice, once on the GPU and once on the CPU, and compares
them at plan.md §6.3's tolerances. Gradients use the bounds the CPU suite uses against the NumPy
adjoint (``test_backprop.py``): complex64 undoes every gate on the way back, ~1e-6 per gate.

These tests need an NVIDIA GPU, so the default run deselects them (``-m "not gpu"``). Run them in
the GPU image::

    docker run --rm --gpus all ketjet-gpu pytest -p no:cacheprovider -q -m gpu tests/gpu
"""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ketjet.backends import get_backend  # noqa: E402
from ketjet.backends.jax_backend import (  # noqa: E402
    DiffMethod,
    clear_compile_cache,
    compile_circuit,
    expval_fn,
    parameter_map,
    probabilities,
)
from ketjet.circuit import Circuit  # noqa: E402
from ketjet.observables import Observable  # noqa: E402
from tests.random_circuits import random_circuit, random_observable, symbolize  # noqa: E402
from tests.tolerances import assert_close  # noqa: E402
from tests.unit.test_jax_adjoint import _temp_states  # noqa: E402

pytestmark = pytest.mark.gpu

DTYPES = [np.complex128, np.complex64]
GRAD_TOL = {np.complex128: 1e-10, np.complex64: 2e-4}
SEEDS = range(6)
BATCH = 8


def _gpu() -> Any:
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return None


GPU = _gpu()
CPU = jax.devices("cpu")[0]
if GPU is None:
    pytest.skip("JAX sees no GPU", allow_module_level=True)


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_compile_cache()
    yield
    clear_compile_cache()


def _real(dtype: Any) -> Any:
    return np.float64 if dtype == np.complex128 else np.float32


def _on(device: Any, f: Callable[..., Any], *args: Any) -> Any:
    """Run ``f`` with its inputs and new arrays on ``device``; return NumPy results."""
    with jax.default_device(device):
        placed = [jax.device_put(a, device) for a in args]
        out = f(*placed)
    leaves = jax.tree.leaves(out)
    assert all(leaf.devices() == {device} for leaf in leaves), f"not on {device}"
    return jax.tree.map(np.asarray, out)


def _problem(seed: int, dtype: Any) -> tuple[Circuit, list[Observable], np.ndarray]:
    """A random circuit of every gate, controls and shared symbols, on 2 to 12 qubits."""
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 13))
    symbolic, values, _ = symbolize(random_circuit(n_wires, 60, rng), rng)
    circuit = Circuit(n_wires, symbolic)
    observables = [random_observable(n_wires, rng) for _ in range(2)]
    x = np.asarray([values[p.name] for p in circuit.parameters], dtype=_real(dtype))
    return circuit, observables, x


def test_the_gpu_is_jaxs_default_device() -> None:
    """In the GPU image JAX places everything on the GPU, with no ketjet setting (D4)."""
    assert jax.default_backend() == "gpu"
    circuit, _, x = _problem(0, np.complex128)
    run, to_slots = compile_circuit(circuit), parameter_map(circuit)
    assert run(to_slots(x)).devices() == {GPU}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_states_and_probabilities_equal_the_cpu_and_the_numpy_oracle(seed: int, dtype: Any) -> None:
    circuit, _, x = _problem(seed, dtype)
    run, to_slots = compile_circuit(circuit, dtype), parameter_map(circuit)

    def state_and_probs(v: Any) -> tuple[Any, Any]:
        state = run(to_slots(v))
        return state, probabilities(state)

    gpu, cpu = _on(GPU, state_and_probs, x), _on(CPU, state_and_probs, x)
    oracle = circuit.bind(dict(zip([p.name for p in circuit.parameters], x.tolist(), strict=True)))
    assert_close(gpu[0], cpu[0], dtype)
    assert_close(gpu[1], cpu[1], dtype)
    assert_close(gpu[0], oracle.state(backend=get_backend("numpy"), dtype=dtype), dtype)


@pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_expectations_and_jacobians_equal_the_cpu(
    seed: int, dtype: Any, diff_method: DiffMethod
) -> None:
    circuit, observables, x = _problem(seed, dtype)
    f = expval_fn(circuit, observables, dtype=dtype, diff_method=diff_method)

    def values_and_jacobian(v: Any) -> tuple[Any, Any]:
        return f(v), jax.jacrev(f)(v)

    gpu = _on(GPU, jax.jit(values_and_jacobian), x)
    cpu = _on(CPU, jax.jit(values_and_jacobian), x)
    assert_close(gpu[0], cpu[0], dtype)
    np.testing.assert_allclose(gpu[1], cpu[1], atol=GRAD_TOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vmap_batches_equal_the_cpu(dtype: Any) -> None:
    circuit, observables, x = _problem(3, dtype)
    f = expval_fn(circuit, observables, dtype=dtype, diff_method="adjoint")

    def energy(v: Any) -> Any:
        return jnp.sum(f(v))

    rng = np.random.default_rng(1)
    batch = (x + rng.normal(0, 0.3, (BATCH, len(x)))).astype(_real(dtype))
    step = jax.jit(jax.vmap(jax.value_and_grad(energy)))
    gpu, cpu = _on(GPU, step, batch), _on(CPU, step, batch)
    assert_close(gpu[0], cpu[0], dtype)
    np.testing.assert_allclose(gpu[1], cpu[1], atol=GRAD_TOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_20_qubit_circuit_equals_the_cpu(dtype: Any) -> None:
    """Large enough that the GPU splits the work differently from the CPU."""
    rng = np.random.default_rng(20)
    symbolic, values, _ = symbolize(random_circuit(20, 120, rng), rng)
    circuit = Circuit(20, symbolic)
    x = np.asarray([values[p.name] for p in circuit.parameters], dtype=_real(dtype))
    run, to_slots = compile_circuit(circuit, dtype), parameter_map(circuit)

    def state(v: Any) -> Any:
        return run(to_slots(v))

    assert_close(_on(GPU, state, x), _on(CPU, state, x), dtype)


def test_adjoint_memory_on_the_gpu_is_independent_of_depth_and_backprops_is_not() -> None:
    """At 18 qubits the adjoint gradient needs ~12.5 states at any depth; backprop ~16 per layer.

    The GPU keeps a fixed scratch area of ~4 MB, which hides a 12-qubit state, so this measures at
    18 qubits, where a state is 4 MB itself (ADR-0007 D2).
    """
    shallow, deep = 4, 12
    adjoint = [_temp_states(18, depth, "adjoint") for depth in (shallow, deep)]
    backprop = [_temp_states(18, depth, "backprop") for depth in (shallow, deep)]

    assert adjoint[1] < 16
    assert adjoint[1] - adjoint[0] < 1
    assert backprop[1] - backprop[0] > 6 * (deep - shallow)  # 18 RY gates per layer


# --------------------------------------------------------------------------- PennyLane


@pytest.mark.pennylane
@pytest.mark.parametrize("dtype", DTYPES)
def test_ketjet_qubit_on_the_gpu_equals_the_cpu_and_samples_the_same_bits(dtype: Any) -> None:
    """Sampling stays on the host (D4), so a seed gives the same bits on either device."""
    qml = pytest.importorskip("pennylane")
    from pennylane.tape import QuantumScript  # noqa: PLC0415 - PennyLane is optional

    from ketjet.pennylane import KetjetQubit  # noqa: PLC0415 - PennyLane is optional

    ops = [qml.RX(0.3, 0), qml.CNOT([0, 1]), qml.Rot(0.1, -0.7, 1.9, 2), qml.CRY(0.8, [2, 0])]
    exact = QuantumScript(ops, [qml.expval(qml.PauliZ(0) @ qml.PauliX(2)), qml.probs()])
    sampled = QuantumScript(ops, [qml.sample(wires=[0, 1, 2])], shots=200)

    def run(tape: QuantumScript, device: Any) -> Any:
        with jax.default_device(device):
            dev = KetjetQubit(wires=3, backend="jax", c_dtype=dtype, seed=7)
            return qml.execute([tape], dev, diff_method=None)[0]

    gpu, cpu = run(exact, GPU), run(exact, CPU)
    for g, c in zip(gpu, cpu, strict=True):
        assert_close(g, c, dtype)
    np.testing.assert_array_equal(run(sampled, GPU), run(sampled, CPU))
