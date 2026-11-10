"""Tests for the Circuit builder (QS-302, ADR-0002 D6)."""

import json
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ketjet import Circuit
from ketjet import operations as ops
from ketjet.backends import NumPyBackend
from ketjet.circuit import format_operation
from ketjet.gates import GATES
from ketjet.operations import Operation, Parameter

# --------------------------------------------------------------------------- fluent API

FLUENT_CASES: list[tuple[str, tuple[Any, ...], Operation]] = [
    ("i", (0,), ops.I(0)),
    ("x", (0,), ops.X(0)),
    ("y", (1,), ops.Y(1)),
    ("z", (2,), ops.Z(2)),
    ("h", (0,), ops.H(0)),
    ("s", (0,), ops.S(0)),
    ("sdg", (0,), ops.Sdg(0)),
    ("t", (0,), ops.T(0)),
    ("tdg", (0,), ops.Tdg(0)),
    ("sx", (0,), ops.SX(0)),
    ("rx", (0.1, 0), ops.RX(0.1, 0)),
    ("ry", (0.2, 1), ops.RY(0.2, 1)),
    ("rz", (0.3, 2), ops.RZ(0.3, 2)),
    ("phase", (0.4, 0), ops.PhaseShift(0.4, 0)),
    ("rot", (0.1, 0.2, 0.3, 1), ops.Rot(0.1, 0.2, 0.3, 1)),
    ("u3", (0.1, 0.2, 0.3, 2), ops.U3(0.1, 0.2, 0.3, 2)),
    ("swap", (0, 2), ops.SWAP(0, 2)),
    ("iswap", (1, 0), ops.ISWAP(1, 0)),
    ("cnot", (2, 0), ops.CNOT(2, 0)),
    ("cy", (0, 1), ops.CY(0, 1)),
    ("cz", (0, 1), ops.CZ(0, 1)),
    ("crx", (0.5, 0, 1), ops.CRX(0.5, 0, 1)),
    ("cry", (0.5, 1, 2), ops.CRY(0.5, 1, 2)),
    ("crz", (0.5, 2, 0), ops.CRZ(0.5, 2, 0)),
    ("toffoli", (0, 1, 2), ops.Toffoli(0, 1, 2)),
    ("cswap", (0, 1, 2), ops.CSWAP(0, 1, 2)),
    ("mcx", ([0, 1], 2, [True, False]), ops.MultiControlledX([0, 1], 2, [True, False])),
]


@pytest.mark.parametrize(
    ("method", "args", "expected"), FLUENT_CASES, ids=[c[0] for c in FLUENT_CASES]
)
def test_each_fluent_method_appends_the_right_operation(
    method: str, args: tuple[Any, ...], expected: Operation
) -> None:
    c = Circuit(3)
    returned = getattr(c, method)(*args)
    assert returned is c  # chainable
    assert c.operations == (expected,)


def test_chaining_builds_in_order() -> None:
    c = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
    assert c.operations == (ops.H(0), ops.CNOT(0, 1), ops.CNOT(1, 2))
    assert len(c) == 3
    assert list(c) == list(c.operations)


def test_append_and_extend_accept_operations() -> None:
    c = Circuit(2).append(ops.H(0)).extend([ops.CNOT(0, 1), ops.ctrl(ops.T(0), [1])])
    assert len(c) == 3


def test_constructor_accepts_initial_operations() -> None:
    assert Circuit(2, [ops.H(0), ops.CNOT(0, 1)]) == Circuit(2).h(0).cnot(0, 1)


# --------------------------------------------------------------------------- validation


def test_wire_outside_the_circuit_is_rejected_with_context() -> None:
    with pytest.raises(
        ValueError, match=r"CNOT\(0, 2\) uses wire\(s\) \[2\], but the circuit has 2"
    ):
        Circuit(2).cnot(0, 2)


def test_append_rejects_non_operations() -> None:
    with pytest.raises(TypeError, match="Expected an Operation"):
        Circuit(1).append("H")  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [0, -1, 64])
def test_invalid_wire_counts_are_rejected(n: int) -> None:
    with pytest.raises(ValueError, match="n_wires"):
        Circuit(n)


def test_failed_append_leaves_the_circuit_unchanged() -> None:
    c = Circuit(2).h(0)
    with pytest.raises(ValueError, match="uses wire"):
        c.extend([ops.X(1), ops.X(5)])
    assert c.operations == (ops.H(0), ops.X(1))  # operations before the bad one are kept


# --------------------------------------------------------------------------- snapshots


def test_operations_snapshot_is_not_affected_by_later_appends() -> None:
    c = Circuit(2).h(0)
    snapshot = c.operations
    c.x(1)
    assert snapshot == (ops.H(0),)
    assert isinstance(snapshot, tuple)


def test_iteration_is_safe_while_appending() -> None:
    c = Circuit(1).x(0).x(0)
    for _ in c:
        c.h(0)
    assert len(c) == 4


def test_copy_is_independent() -> None:
    c = Circuit(2).h(0)
    d = c.copy()
    d.x(1)
    assert len(c) == 1
    assert len(d) == 2


def test_circuits_are_unhashable_because_they_are_mutable() -> None:
    with pytest.raises(TypeError):
        hash(Circuit(1))


# --------------------------------------------------------------------------- equality and repr


def test_equality() -> None:
    assert Circuit(2).h(0) == Circuit(2).h(0)
    assert Circuit(2).h(0) != Circuit(3).h(0)
    assert Circuit(2).h(0) != Circuit(2).h(1)
    assert Circuit(2) != "Circuit(2)"


def test_repr_reads_like_the_constructors() -> None:
    c = (
        Circuit(4)
        .h(0)
        .crx(0.25, 0, 1)
        .rx(Parameter("theta"), 2)
        .toffoli(0, 1, 2)
        .append(ops.ctrl(ops.H(3), [0], [False]))
        .mcx([0, 1, 2], 3)
    )
    assert repr(c) == (
        "Circuit(4) [H(0), CRX(0.25, 0, 1), RX(theta, 2), Toffoli(0, 1, 2), "
        "ctrl(H(3), [0], [False]), MultiControlledX([0, 1, 2], 3, [True, True, True])]"
    )


def test_format_operation_for_generic_controlled_gates() -> None:
    assert format_operation(ops.ctrl(ops.RY(0.5, 2), [0, 1])) == "ctrl(RY(0.5, 2), [0, 1])"


# --------------------------------------------------------------------------- inverse


def test_inverse_reverses_and_inverts() -> None:
    c = Circuit(2).h(0).s(1).crx(0.3, 0, 1)
    assert c.inverse().operations == (ops.CRX(-0.3, 0, 1), ops.Sdg(1), ops.H(0))


def test_circuit_times_inverse_is_identity() -> None:
    c = Circuit(3).h(0).cnot(0, 1).rot(0.1, 0.2, 0.3, 2).iswap(1, 2).cswap(0, 1, 2)
    backend = NumPyBackend()
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    psi /= np.linalg.norm(psi)
    out = backend.apply_operations(backend.apply_operations(psi, c), c.inverse())
    np.testing.assert_allclose(out, psi, atol=1e-12)


# --------------------------------------------------------------------------- serialization


def test_serialization_format_is_versioned_and_json_compatible() -> None:
    data = Circuit(2).h(0).crx(Parameter("t"), 0, 1).to_dict()
    assert data["format"] == "ketjet.circuit"
    assert data["version"] == 1
    assert data["operations"][1] == {
        "gate": "RX",
        "wires": [1],
        "params": [{"parameter": "t"}],
        "controls": [0],
        "control_values": [True],
    }
    json.dumps(data)  # must not raise


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"format": "qasm", "version": 1}, "Not a ketjet circuit"),
        ({"format": "ketjet.circuit", "version": 99}, "Unsupported circuit format version 99"),
    ],
)
def test_from_dict_rejects_foreign_or_future_formats(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Circuit.from_dict(data)


@st.composite
def random_circuits(draw: st.DrawFn) -> Circuit:
    n = draw(st.integers(min_value=1, max_value=5))
    c = Circuit(n)
    names = sorted(GATES)
    for _ in range(draw(st.integers(min_value=0, max_value=12))):
        spec = GATES[draw(st.sampled_from(names))]
        if spec.num_wires > n:
            continue
        n_ctrl = draw(st.integers(min_value=0, max_value=n - spec.num_wires))
        wires = draw(st.permutations(range(n)))[: spec.num_wires + n_ctrl]
        params = tuple(
            draw(st.one_of(
                st.floats(-10, 10, allow_nan=False),
                st.sampled_from([Parameter("a"), Parameter("b")]),
            ))
            for _ in range(spec.num_params)
        )  # fmt: skip
        values = tuple(draw(st.lists(st.booleans(), min_size=n_ctrl, max_size=n_ctrl)))
        c.append(Operation(spec.name, tuple(wires[n_ctrl:]), params, tuple(wires[:n_ctrl]), values))
    return c


@settings(max_examples=200)
@given(c=random_circuits())
def test_json_round_trip_preserves_every_circuit(c: Circuit) -> None:
    assert Circuit.from_json(c.to_json()) == c
    assert Circuit.from_dict(c.to_dict()) == c
