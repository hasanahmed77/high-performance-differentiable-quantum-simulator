"""The other simulators, as benchmark runners (QS-1103).

Each benchmark circuit is built once as a ketjet circuit (:mod:`benchmarks.circuits`) and
translated here, gate by gate, into each library's own API (ADR-0008 D5): the same gates, the
same angles and the same observable. Every translation is checked against ketjet's NumPy oracle
before it's timed (ADR-0008 D1), which is what catches conventions that differ: Qiskit and Qulacs
number qubits from the least significant bit, and Qulacs's rotations turn the opposite way.

Runners:

``pennylane-default``
    PennyLane's ``default.qubit``, with autograd and backprop for gradients. The QML batch is a
    Python loop: its parameter broadcasting fails under autograd from 12 qubits.
``pennylane-lightning``
    ``lightning.qubit``, PennyLane's C++ simulator, with its adjoint method for gradients and
    parameter broadcasting for the QML batch.
``qiskit-aer``
    Qiskit Aer's statevector simulator. Aer has no gradients of its own, so it runs the state
    benchmarks only.
``qulacs``
    Qulacs, with its ``backprop`` for gradients. Qulacs has complex128 only.

Each library is used as its documentation shows, with its default options and all cores, and
its circuit is built outside the timer.
"""

from collections.abc import Callable
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from benchmarks.circuits import Spec
from ketjet.observables import Observable, Pauli, Prod, Sum
from ketjet.operations import Operation, Parameter

LIBRARY_RUNNERS: Final = ("pennylane-default", "pennylane-lightning", "qiskit-aer", "qulacs")
"""The other simulators' runners, in the order of the results tables."""

PRECISIONS: Final[dict[str, frozenset[str]]] = {
    "pennylane-default": frozenset({"complex128"}),
    "pennylane-lightning": frozenset({"complex128", "complex64"}),
    "qiskit-aer": frozenset({"complex128", "complex64"}),
    "qulacs": frozenset({"complex128"}),
}
"""The precisions each library can run in."""

STATE_ONLY: Final = frozenset({"qiskit-aer"})
"""Runners without gradients of their own."""

PACKAGES: Final[dict[str, str]] = {
    "pennylane-default": "pennylane",
    "pennylane-lightning": "pennylane_lightning",
    "qiskit-aer": "qiskit_aer",
    "qulacs": "qulacs",
}
"""The module each runner needs."""


class UnsupportedGateError(NotImplementedError):
    """A benchmark circuit uses a gate this translation doesn't know."""


def _name(op: Operation) -> str:
    """Return ``op``'s name as the translations know it: ``H``, ``RX``, ``CNOT``, ``CPhase``..."""
    if not op.controls:
        return op.gate
    if len(op.controls) == 1 and all(op.control_values):
        controlled = {"X": "CNOT", "PhaseShift": "CPhase"}
        if op.gate in controlled:
            return controlled[op.gate]
    raise UnsupportedGateError(f"No translation for a controlled {op.gate}.")


def pauli_terms(observable: Observable) -> list[tuple[float, dict[int, str]]]:
    """Return ``observable`` as a list of ``(coefficient, {wire: "X" | "Y" | "Z"})`` terms."""
    if isinstance(observable, Sum):
        return [
            (coeff * c, word) for coeff, term in observable.terms for c, word in pauli_terms(term)
        ]
    if isinstance(observable, Pauli):
        return [(1.0, {w: p for w, p in observable.word if p != "I"})]
    if isinstance(observable, Prod):
        word: dict[int, str] = {}
        coeff = 1.0
        for factor in observable.factors:
            (c, part), *rest = pauli_terms(factor)
            if rest or set(part) & set(word):
                raise UnsupportedGateError(f"No translation for {observable!r}.")
            coeff *= c
            word |= part
        return [(coeff, word)]
    raise UnsupportedGateError(f"No translation for {observable!r}.")


def _indices(spec: Spec) -> dict[Parameter, int]:
    return {p: i for i, p in enumerate(spec.circuit.parameters)}


# --------------------------------------------------------------------------- PennyLane


def _pennylane_ops(spec: Spec, angle: Callable[[float | Parameter], Any]) -> None:
    import pennylane as qml  # noqa: PLC0415 - only these runners need PennyLane

    gates: dict[str, Callable[..., Any]] = {
        "H": qml.Hadamard,
        "RX": qml.RX,
        "RY": qml.RY,
        "RZ": qml.RZ,
        "PhaseShift": qml.PhaseShift,
        "CNOT": qml.CNOT,
        "SWAP": qml.SWAP,
        "CPhase": qml.ControlledPhaseShift,
    }
    for op in spec.circuit:
        name = _name(op)
        if name not in gates:
            raise UnsupportedGateError(f"No PennyLane translation for {name}.")
        gates[name](*(angle(p) for p in op.params), wires=list(op.all_wires))


def _pennylane_observable(observable: Observable) -> Any:
    import pennylane as qml  # noqa: PLC0415

    paulis = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    coeffs, terms = [], []
    for coeff, word in pauli_terms(observable):
        coeffs.append(coeff)
        terms.append(qml.prod(*(paulis[p](w) for w, p in sorted(word.items()))))
    return qml.dot(coeffs, terms)


def _pennylane(runner: str, benchmark: str, spec: Spec, precision: str) -> Callable[[], Any]:
    import pennylane as qml  # noqa: PLC0415
    from pennylane import numpy as pnp  # noqa: PLC0415

    n = spec.circuit.n_wires
    if runner == "pennylane-lightning":
        device = qml.device("lightning.qubit", wires=n, c_dtype=np.dtype(precision).type)
        diff_method = "adjoint"
    else:
        device = qml.device("default.qubit", wires=n)
        diff_method = "backprop"
    index = _indices(spec)

    if spec.observable is None:
        values = spec.values

        def state() -> Any:
            _pennylane_ops(spec, lambda p: values[index[p]] if isinstance(p, Parameter) else p)
            return qml.state()

        qnode = qml.QNode(state, device, interface=None)
        return lambda: np.asarray(qnode())

    observable = _pennylane_observable(spec.observable)

    if benchmark == "vqe-grad":

        def energy(w: Any) -> Any:
            _pennylane_ops(spec, lambda p: w[index[p]] if isinstance(p, Parameter) else p)
            return qml.expval(observable)

        weights = pnp.array(spec.values, requires_grad=True)
        grad = qml.grad(qml.QNode(energy, device, interface="autograd", diff_method=diff_method))
        return lambda: np.asarray(grad(weights))

    assert spec.inputs is not None
    inputs = pnp.array(spec.inputs, requires_grad=False)
    # default.qubit's broadcasting fails under autograd from 12 qubits ("shape-mismatch for sum"
    # in PennyLane 0.45.1), so it loops over the batch, as ketjet's NumPy runner and Qulacs do.
    broadcast = runner == "pennylane-lightning"

    def angle(xs: Any, w: Any, p: float | Parameter) -> Any:
        if not isinstance(p, Parameter):
            return p
        i = index[p]
        if i >= n:
            return w[i - n]
        return xs[:, i] if broadcast else xs[i]

    def output(xs: Any, w: Any) -> Any:
        _pennylane_ops(spec, lambda p: angle(xs, w, p))
        return qml.expval(observable)

    qnode = qml.QNode(output, device, interface="autograd", diff_method=diff_method)

    def loss(xs: Any, w: Any) -> Any:
        if broadcast:
            return pnp.mean(qnode(xs, w))
        return sum(qnode(x, w) for x in xs) / len(xs)

    weights = pnp.array(spec.values, requires_grad=True)
    batch_grad = qml.grad(loss, argnums=1)
    return lambda: np.asarray(batch_grad(inputs, weights))


# --------------------------------------------------------------------------- Qiskit Aer


def _qiskit(spec: Spec, precision: str) -> Callable[[], Any]:
    from qiskit import QuantumCircuit  # noqa: PLC0415
    from qiskit_aer import AerSimulator  # noqa: PLC0415

    if spec.observable is not None:
        raise UnsupportedGateError("Qiskit Aer has no gradients of its own.")
    n = spec.circuit.n_wires
    index = _indices(spec)

    def q(w: int) -> int:
        return n - 1 - w  # Qiskit's qubit 0 is the least significant bit; ketjet's wire 0 the most

    circuit = QuantumCircuit(n)
    for op in spec.circuit:
        name = _name(op)
        angles = [
            float(spec.values[index[p]]) if isinstance(p, Parameter) else p for p in op.params
        ]
        wires = [q(w) for w in op.all_wires]
        method = {
            "H": "h",
            "RX": "rx",
            "RY": "ry",
            "RZ": "rz",
            "PhaseShift": "p",
            "CNOT": "cx",
            "SWAP": "swap",
            "CPhase": "cp",
        }.get(name)
        if method is None:
            raise UnsupportedGateError(f"No Qiskit translation for {name}.")
        getattr(circuit, method)(*angles, *wires)
    circuit.save_statevector()  # added to QuantumCircuit by qiskit_aer
    simulator = AerSimulator(
        method="statevector", precision="double" if precision == "complex128" else "single"
    )
    # Not transpiled: Aer runs these gates natively, and Qiskit's default transpiler removes the
    # QFT's final SWAPs and permutes the returned state instead, which gives a different state.
    return lambda: np.asarray(simulator.run(circuit).result().get_statevector())


# --------------------------------------------------------------------------- Qulacs


def _qulacs_circuit(spec: Spec, gradient: bool) -> tuple[Any, list[int]]:
    """Return the Qulacs circuit, and the ketjet parameter index of each parametric gate."""
    import qulacs  # noqa: PLC0415
    from qulacs import gate as qulacs_gate  # noqa: PLC0415

    n = spec.circuit.n_wires
    index = _indices(spec)

    def q(w: int) -> int:
        return n - 1 - w  # Qulacs's qubit 0 is the least significant bit, as in Qiskit

    circuit = qulacs.ParametricQuantumCircuit(n) if gradient else qulacs.QuantumCircuit(n)
    order: list[int] = []  # ketjet parameter index of each of Qulacs's parametric gates
    for op in spec.circuit:
        name = _name(op)
        wires = [q(w) for w in op.all_wires]
        if name in ("RX", "RY", "RZ"):
            (p,) = op.params
            if gradient and isinstance(p, Parameter):
                # Qulacs's rotations are exp(+i θ P / 2), the opposite of everyone else's.
                getattr(circuit, f"add_parametric_{name}_gate")(wires[0], 0.0)  # set per call
                order.append(index[p])
                continue
            value = float(spec.values[index[p]]) if isinstance(p, Parameter) else p
            getattr(circuit, f"add_Rot{name[1]}_gate")(wires[0], value)
        elif name == "H":
            circuit.add_H_gate(wires[0])
        elif name == "CNOT":
            circuit.add_CNOT_gate(*wires)
        elif name == "SWAP":
            circuit.add_SWAP_gate(*wires)
        elif name == "CPhase":
            control, target = wires
            (phi,) = (
                float(spec.values[index[p]]) if isinstance(p, Parameter) else p for p in op.params
            )
            matrix = qulacs_gate.to_matrix_gate(qulacs_gate.U1(target, phi))
            matrix.add_control_qubit(control, 1)
            circuit.add_gate(matrix)
        else:
            raise UnsupportedGateError(f"No Qulacs translation for {name}.")
    if len(set(order)) != len(order):
        raise UnsupportedGateError("Qulacs's backprop needs each parameter in one gate only.")
    return circuit, order


def _qulacs(benchmark: str, spec: Spec) -> Callable[[], Any]:
    import qulacs  # noqa: PLC0415

    n = spec.circuit.n_wires
    circuit, order = _qulacs_circuit(spec, gradient=spec.observable is not None)
    if spec.observable is None:
        state = qulacs.QuantumState(n)

        def run_state() -> npt.NDArray[np.complex128]:
            state.set_zero_state()
            circuit.update_quantum_state(state)
            return np.asarray(state.get_vector())

        return run_state

    observable = qulacs.Observable(n)
    for coeff, word in pauli_terms(spec.observable):
        paulis = " ".join(f"{p} {n - 1 - w}" for w, p in sorted(word.items()))
        observable.add_operator(coeff, paulis)

    def grad(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        for k, i in enumerate(order):
            circuit.set_parameter(k, -float(values[i]))
        out = np.zeros(len(values))
        out[order] = -np.asarray(circuit.backprop(observable))
        return out

    if benchmark == "vqe-grad":
        return lambda: grad(spec.values)
    assert spec.inputs is not None
    inputs, weights = spec.inputs, spec.values

    def batch_grad() -> npt.NDArray[np.float64]:
        total = np.zeros(len(weights))
        for x in inputs:
            total += grad(np.concatenate([x, weights]))[n:]
        return total / len(inputs)

    return batch_grad


# --------------------------------------------------------------------------- dispatch


def supports(runner: str, benchmark: str, precision: str, gradient: bool) -> bool:
    """Return whether ``runner`` can run ``benchmark`` in ``precision``."""
    return precision in PRECISIONS[runner] and not (gradient and runner in STATE_ONLY)


def prepare_library(runner: str, benchmark: str, spec: Spec, precision: str) -> Callable[[], Any]:
    """Return the call that runs ``spec`` with ``runner``'s library, built outside the timer."""
    if runner in ("pennylane-default", "pennylane-lightning"):
        return _pennylane(runner, benchmark, spec, precision)
    if runner == "qiskit-aer":
        return _qiskit(spec, precision)
    if runner == "qulacs":
        return _qulacs(benchmark, spec)
    raise ValueError(f"Unknown library runner {runner!r}.")


def library_options(runner: str, precision: str) -> dict[str, Any]:
    """Return the options a runner was used with, for the results file (ADR-0008 D5)."""
    options: dict[str, dict[str, Any]] = {
        "pennylane-default": {
            "device": "default.qubit",
            "interface": "autograd",
            "diff_method": "backprop",
            "batch": "Python loop",
        },
        "pennylane-lightning": {
            "device": "lightning.qubit",
            "interface": "autograd",
            "diff_method": "adjoint",
            "c_dtype": precision,
            "batch": "parameter broadcasting",
        },
        "qiskit-aer": {
            "method": "statevector",
            "precision": "double" if precision == "complex128" else "single",
            "transpiled": False,
        },
        "qulacs": {"gradient": "ParametricQuantumCircuit.backprop", "batch": "Python loop"},
    }
    return options[runner]
