"""Tests for the gate registry and operations (QS-204, ADR-0002)."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ketjet import operations as ops
from ketjet.gates import GATES, GateSpec, get_gate
from ketjet.operations import Operation, Parameter, ctrl, display_name
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi, allow_nan=False)


def params_for(spec: GateSpec, rng: np.random.Generator) -> tuple[float, ...]:
    return tuple(float(x) for x in rng.uniform(-2 * np.pi, 2 * np.pi, spec.num_params))


def is_phased_permutation(m: npt.NDArray[Any], atol: float = 1e-12) -> bool:
    nonzero = np.abs(m) > atol
    return bool(
        (nonzero.sum(axis=0) == 1).all()
        and (nonzero.sum(axis=1) == 1).all()
        and np.allclose(np.abs(m[nonzero]), 1.0, atol=1e-10)
    )


SPECS = sorted(GATES.values(), key=lambda s: s.name)
SPEC_IDS = [s.name for s in SPECS]


# --------------------------------------------------------------------------- registry


def test_registry_contains_the_standard_base_gates() -> None:
    expected = {"I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "SX", "SXdg", "RX", "RY", "RZ",
                "PhaseShift", "Rot", "U3", "SWAP", "ISWAP", "ISWAPdg"}  # fmt: skip
    assert set(GATES) == expected


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        GATES["NEW"] = GATES["X"]  # type: ignore[index]


def test_unknown_gate_lists_alternatives() -> None:
    with pytest.raises(KeyError, match=r"Unknown gate 'CNOT'\. Available: .*RX"):
        get_gate("CNOT")


# --------------------------------------------------------------------------- matrices and flags


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_gate_is_unitary_with_correct_shape_and_dtype(spec: GateSpec, dtype: Any) -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        m = spec.matrix(params_for(spec, rng), dtype)
        dim = 2**spec.num_wires
        assert m.shape == (dim, dim)
        assert m.dtype == np.dtype(dtype)
        assert_close(m @ m.conj().T, np.eye(dim), dtype)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_structure_flags_match_the_matrix(spec: GateSpec) -> None:
    """A gate claiming a fast-path property must really have it (ADR-0002 D3)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        params = params_for(spec, rng)
        m = spec.matrix(params)
        off_diagonal = m - np.diag(np.diag(m))
        if spec.is_diagonal:
            assert np.allclose(off_diagonal, 0), spec.name
        if spec.is_permutation:
            assert is_phased_permutation(m), spec.name
        if spec.is_self_inverse:
            assert np.allclose(m @ m, np.eye(len(m))), spec.name


@pytest.mark.parametrize(
    "name",
    ["H", "SX", "SXdg", "RX", "RY", "Rot", "U3"],
)
def test_non_diagonal_gates_are_not_flagged_diagonal(name: str) -> None:
    assert not GATES[name].is_diagonal


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_inverse_undoes_the_gate(spec: GateSpec) -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        params = params_for(spec, rng)
        inv_name, inv_params = spec.inverse(params)
        product = GATES[inv_name].matrix(inv_params) @ spec.matrix(params)
        assert_close(product, np.eye(2**spec.num_wires), np.complex128)


def test_matrix_rejects_wrong_parameter_count() -> None:
    with pytest.raises(ValueError, match="RX takes 1 parameter"):
        GATES["RX"].matrix(())


def test_constant_matrices_cannot_be_mutated_through_the_registry() -> None:
    m = GATES["X"].matrix()
    with pytest.raises(ValueError, match="read-only"):
        m[0, 0] = 5


@given(theta=angles)
def test_rotation_identities(theta: float) -> None:
    # RZ(θ) = e^{-iθ/2} PhaseShift(θ); Rot(0, θ, 0) = RY(θ); RX(θ)² = RX(2θ)
    phase = ops.PhaseShift(theta, 0).matrix()
    assert_close(ops.RZ(theta, 0).matrix(), np.exp(-0.5j * theta) * phase, np.complex128)
    assert_close(ops.Rot(0, theta, 0, 0).matrix(), ops.RY(theta, 0).matrix(), np.complex128)
    rx = ops.RX(theta, 0).matrix()
    assert_close(rx @ rx, ops.RX(2 * theta, 0).matrix(), np.complex128)


def test_named_relations_between_fixed_gates() -> None:
    def m(op: Operation) -> npt.NDArray[Any]:
        return op.matrix()

    assert_close(m(ops.S(0)) @ m(ops.S(0)), m(ops.Z(0)), np.complex128)
    assert_close(m(ops.T(0)) @ m(ops.T(0)), m(ops.S(0)), np.complex128)
    assert_close(m(ops.SX(0)) @ m(ops.SX(0)), m(ops.X(0)), np.complex128)
    assert_close(m(ops.ISWAP(0, 1)) @ m(ops.ISWAP(0, 1)), np.diag([1, -1, -1, 1]), np.complex128)


# --------------------------------------------------------------------------- operations


def test_operation_is_immutable_and_hashable() -> None:
    op = ops.RX(0.5, 0)
    with pytest.raises(AttributeError):
        op.wires = (1,)  # type: ignore[misc]
    assert {op, ops.RX(0.5, 0)} == {op}


def test_operation_normalizes_fields() -> None:
    op = Operation("RX", [np.int64(2)], [np.float32(0.5)], [np.int32(0)])  # type: ignore[arg-type]
    assert op.wires == (2,)
    assert op.controls == (0,)
    assert op.params == (0.5,)
    assert type(op.params[0]) is float
    assert op.control_values == (True,)


def test_controlled_gates_are_structure_not_names() -> None:
    assert ops.CNOT(0, 1) == Operation("X", (1,), controls=(0,))
    assert ops.Toffoli(0, 1, 2) == Operation("X", (2,), controls=(0, 1))
    assert ops.CRX(0.3, 2, 0) == Operation("RX", (0,), (0.3,), (2,))
    assert ops.CSWAP(0, 1, 2) == Operation("SWAP", (1, 2), controls=(0,))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"gate": "NOPE", "wires": (0,)}, KeyError, "Unknown gate"),
        ({"gate": "X", "wires": (0, 1)}, ValueError, "acts on 1 wire"),
        ({"gate": "SWAP", "wires": (0, 0)}, ValueError, "distinct"),
        ({"gate": "X", "wires": (0,), "controls": (0,)}, ValueError, "distinct"),
        ({"gate": "X", "wires": (-1,)}, ValueError, "non-negative"),
        ({"gate": "X", "wires": (1.0,)}, TypeError, "Target wires must be ints"),
        ({"gate": "X", "wires": (0,), "controls": (True,)}, TypeError, "Control wires must"),
        ({"gate": "RX", "wires": (0,)}, ValueError, "takes 1 parameter"),
        ({"gate": "RX", "wires": (0,), "params": ("a",)}, TypeError, "real numbers or Parameters"),
        ({"gate": "RX", "wires": (0,), "params": (True,)}, TypeError, "real numbers or Parameters"),
        ({"gate": "RX", "wires": (0,), "params": (float("nan"),)}, ValueError, "finite"),
        ({"gate": "X", "wires": (1,), "controls": (0,), "control_values": (True, False)},
         ValueError, "2 control value"),
        ({"gate": "X", "wires": (1,), "controls": (0,), "control_values": (1,)},
         TypeError, "booleans"),
    ],
)  # fmt: skip
def test_operation_validation(kwargs: dict[str, Any], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        Operation(**kwargs)


def test_controlled_matrix_with_mixed_control_values() -> None:
    # X on wire 2, triggered when wire 0 = 1 and wire 1 = 0: block for controls "10" (rows 4-5).
    op = Operation("X", (2,), controls=(0, 1), control_values=(True, False))
    expected = np.eye(8)
    expected[4:6, 4:6] = [[0, 1], [1, 0]]
    assert_close(op.matrix(), expected, np.complex128)


@pytest.mark.parametrize("dtype", DTYPES)
def test_controlled_matrix_respects_dtype(dtype: Any) -> None:
    assert ops.CRZ(0.4, 0, 1).matrix(dtype).dtype == np.dtype(dtype)


def test_ctrl_composes_and_prepends_controls() -> None:
    op = ctrl(ops.CNOT(1, 2), [0], [False])
    assert op.controls == (0, 1)
    assert op.control_values == (False, True)
    assert ctrl(ops.H(1), [0]).matrix()[2:, 2:] == pytest.approx(ops.H(0).matrix())


@pytest.mark.parametrize(
    "op",
    [ops.CNOT(0, 1), ops.CRY(0.7, 1, 0), ops.Toffoli(0, 1, 2), ops.CSWAP(2, 0, 1),
     ops.MultiControlledX([0, 1, 2], 3, [True, False, True]), ops.Rot(0.1, 0.2, 0.3, 0),
     ops.ISWAP(0, 1)],
    ids=display_name,
)  # fmt: skip
def test_operation_inverse(op: Operation) -> None:
    inv = op.inverse()
    assert inv.controls == op.controls
    assert inv.control_values == op.control_values
    assert_close(inv.matrix() @ op.matrix(), np.eye(2 ** len(op.all_wires)), np.complex128)


def test_symbolic_parameters() -> None:
    theta = Parameter("theta")
    op = ops.CRX(theta, 0, 1)
    assert op.is_parametrized_symbolically
    with pytest.raises(ValueError, match=r"unbound parameter.*theta"):
        op.matrix()
    with pytest.raises(ValueError, match="unbound"):
        op.inverse()
    assert not ops.RX(0.1, 0).is_parametrized_symbolically


@pytest.mark.parametrize("bad", ["", 3])
def test_parameter_name_must_be_non_empty_string(bad: Any) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        Parameter(bad)


@pytest.mark.parametrize(
    ("op", "name"),
    [
        (ops.X(0), "X"),
        (ops.CNOT(0, 1), "CNOT"),
        (ops.CY(0, 1), "CY"),
        (ops.CZ(0, 1), "CZ"),
        (ops.CRZ(0.1, 0, 1), "CRZ"),
        (ops.Toffoli(0, 1, 2), "Toffoli"),
        (ops.CSWAP(0, 1, 2), "CSWAP"),
        (ops.MultiControlledX([0, 1, 2], 3), "MultiControlledX"),
        (ctrl(ops.H(1), [0]), "C-H"),
        (ctrl(ops.T(2), [0, 1]), "CC-T"),
        (Operation("X", (1,), controls=(0,), control_values=(False,)), "C-X"),
    ],
)
def test_display_name(op: Operation, name: str) -> None:
    assert display_name(op) == name


def test_all_wires_puts_controls_first() -> None:
    assert ops.CSWAP(2, 0, 1).all_wires == (2, 0, 1)


CONSTRUCTED: dict[str, Operation] = {
    "I": ops.I(0), "X": ops.X(0), "Y": ops.Y(0), "Z": ops.Z(0), "H": ops.H(0), "S": ops.S(0),
    "Sdg": ops.Sdg(0), "T": ops.T(0), "Tdg": ops.Tdg(0), "SX": ops.SX(0), "RX": ops.RX(0.1, 0),
    "RY": ops.RY(0.2, 0), "RZ": ops.RZ(0.3, 0), "PhaseShift": ops.PhaseShift(0.4, 0),
    "Rot": ops.Rot(0.1, 0.2, 0.3, 0), "U3": ops.U3(0.1, 0.2, 0.3, 0), "SWAP": ops.SWAP(0, 1),
    "ISWAP": ops.ISWAP(0, 1), "CNOT": ops.CNOT(0, 1), "CY": ops.CY(0, 1), "CZ": ops.CZ(0, 1),
    "CRX": ops.CRX(0.1, 0, 1), "CRY": ops.CRY(0.2, 0, 1), "CRZ": ops.CRZ(0.3, 0, 1),
    "Toffoli": ops.Toffoli(0, 1, 2), "CSWAP": ops.CSWAP(0, 1, 2),
    "MultiControlledX": ops.MultiControlledX([0, 1], 2),
}  # fmt: skip


def test_every_public_constructor_is_exercised() -> None:
    public = {n for n in ops.__all__ if n[0].isupper()} - {"Operation", "ParamLike", "Parameter"}
    assert public == set(CONSTRUCTED)


@pytest.mark.parametrize("op", CONSTRUCTED.values(), ids=CONSTRUCTED.keys())
def test_every_constructor_builds_a_unitary_operation(op: Operation) -> None:
    m = op.matrix()
    assert_close(m @ m.conj().T, np.eye(len(m)), np.complex128)


@settings(max_examples=50)
@given(theta=angles, dtype=st.sampled_from(DTYPES))
def test_parametric_controlled_gates_are_unitary(theta: float, dtype: Any) -> None:
    for op in (ops.CRX(theta, 0, 1), ops.CRY(theta, 1, 0), ops.CRZ(theta, 0, 2)):
        m = op.matrix(dtype)
        assert_close(m @ m.conj().T, np.eye(4), dtype)
