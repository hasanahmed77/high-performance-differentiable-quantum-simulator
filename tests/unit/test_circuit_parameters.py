"""Tests for symbolic parameters, binding and trainable slots (QS-303, ADR-0002 D4)."""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ketjet import Circuit
from ketjet import operations as ops
from ketjet.backends import NumPyBackend
from ketjet.operations import Parameter

a, b, c = Parameter("a"), Parameter("b"), Parameter("c")


def ansatz() -> Circuit:
    return Circuit(2).rx(a, 0).ry(b, 1).cnot(0, 1).crz(a, 0, 1).rot(c, 0.5, b, 0)


# --------------------------------------------------------------------------- discovery


def test_parameters_are_unique_in_first_appearance_order() -> None:
    assert ansatz().parameters == (a, b, c)


def test_circuit_without_symbols() -> None:
    circuit = Circuit(1).rx(0.1, 0)
    assert circuit.parameters == ()
    assert circuit.parameter_slots == ()
    assert circuit.is_bound


def test_parameter_slots_point_at_every_symbolic_position() -> None:
    circuit = ansatz()
    assert circuit.parameter_slots == ((0, 0, a), (1, 0, b), (3, 0, a), (4, 0, c), (4, 2, b))
    for op_index, param_index, symbol in circuit.parameter_slots:
        assert circuit.operations[op_index].params[param_index] == symbol


# --------------------------------------------------------------------------- binding


def test_bind_by_parameter_by_name_and_by_keyword_agree() -> None:
    values = {"a": 0.1, "b": 0.2, "c": 0.3}
    by_name = ansatz().bind(values)
    by_symbol = ansatz().bind({a: 0.1, b: 0.2, c: 0.3})
    by_kwargs = ansatz().bind(a=0.1, b=0.2, c=0.3)
    assert by_name == by_symbol == by_kwargs
    assert by_name.is_bound
    assert by_name.operations[0] == ops.RX(0.1, 0)
    assert by_name.operations[4] == ops.Rot(0.3, 0.5, 0.2, 0)


def test_a_symbol_used_twice_gets_the_same_value_in_every_slot() -> None:
    bound = ansatz().bind(a=0.7, b=0, c=0)
    assert bound.operations[0].params == (0.7,)
    assert bound.operations[3].params == (0.7,)


def test_bind_returns_a_new_circuit_and_leaves_the_original_symbolic() -> None:
    original = ansatz()
    bound = original.bind(a=1, b=2, c=3)
    assert bound is not original
    assert original.parameters == (a, b, c)
    assert not original.is_bound


def test_partial_binding_keeps_remaining_symbols() -> None:
    partial = ansatz().bind(a=0.5)
    assert partial.parameters == (b, c)
    assert not partial.is_bound
    assert partial.bind(b=1.0, c=2.0) == ansatz().bind(a=0.5, b=1.0, c=2.0)


def test_keyword_values_override_the_mapping() -> None:
    assert ansatz().bind({"a": 1.0}, a=2.0).operations[0] == ops.RX(2.0, 0)


def test_unknown_names_are_rejected_with_the_known_names() -> None:
    with pytest.raises(ValueError, match=r"Unknown parameter\(s\) theta; the circuit has a, b, c"):
        ansatz().bind(theta=1.0)
    with pytest.raises(ValueError, match="the circuit has none"):
        Circuit(1).x(0).bind(a=1.0)


@pytest.mark.parametrize(
    ("value", "error"),
    [(float("inf"), ValueError), (float("nan"), ValueError), ("1.0", TypeError), (True, TypeError)],
)
def test_invalid_values_are_rejected(value: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        ansatz().bind(a=value)


def test_bound_circuit_runs_and_unbound_one_does_not() -> None:
    backend = NumPyBackend()
    bound = ansatz().bind(a=0.1, b=0.2, c=0.3)
    backend.apply_operations(backend.zero_state(2), bound)
    with pytest.raises(ValueError, match="unbound parameter"):
        backend.apply_operations(backend.zero_state(2), ansatz())


def test_operation_bind_is_identity_when_nothing_matches() -> None:
    op = ops.RX(0.5, 0)
    assert op.bind({"a": 1.0}) is op


@given(values=st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3))
def test_binding_then_serializing_round_trips(values: list[float]) -> None:
    bound = ansatz().bind(dict(zip("abc", values, strict=True)))
    assert Circuit.from_json(bound.to_json()) == bound
    assert (
        Circuit.from_json(ansatz().to_json()).bind(dict(zip("abc", values, strict=True))) == bound
    )
