"""The other simulators get the same circuits as ketjet and agree with its oracle (QS-1103).

These need the ``bench`` dependency group (``uv sync --group bench``); without it they're
skipped. Each translated circuit is run at small sizes and checked against the NumPy oracle,
exactly as the harness does before it times anything (ADR-0008 D1).
"""

import importlib.util

import numpy as np
import pytest
from benchmarks import harness
from benchmarks.cases import Case, cases, prepare
from benchmarks.circuits import GRADIENT_BENCHMARKS, ising
from benchmarks.libraries import (
    LIBRARY_RUNNERS,
    PACKAGES,
    STATE_ONLY,
    UnsupportedGateError,
    pauli_terms,
    supports,
)
from benchmarks.run import run_case

from ketjet import operations as ops
from ketjet.circuit import Circuit
from ketjet.observables import Hermitian, X, Y

pytestmark = pytest.mark.filterwarnings(
    # Qiskit and PennyLane warn about their own deprecations; not ours to fix.
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
)


def _installed(runner: str) -> bool:
    return importlib.util.find_spec(PACKAGES[runner]) is not None


def _library_cases(precision: str) -> list[Case]:
    return cases(runners=LIBRARY_RUNNERS, sizes=[3, 4], precision=precision)


def _skip_if_missing(case: Case) -> None:
    if not _installed(case.runner):
        pytest.skip(f"{PACKAGES[case.runner]} isn't installed: uv sync --group bench")


@pytest.mark.parametrize("case", _library_cases("complex128"), ids=lambda c: c.id)
def test_every_library_matches_the_oracle(case: Case) -> None:
    _skip_if_missing(case)
    result = run_case(case, harness.SMOKE)
    assert result["first_call_s"] > 0
    assert result["info"]["options"]


@pytest.mark.parametrize("case", _library_cases("complex64"), ids=lambda c: c.id)
def test_every_library_matches_the_oracle_in_complex64(case: Case) -> None:
    _skip_if_missing(case)
    run_case(case, harness.SMOKE)


def test_which_cases_each_library_runs() -> None:
    every = cases(runners=LIBRARY_RUNNERS)
    for runner in STATE_ONLY:
        assert not [c for c in every if c.runner == runner and c.benchmark in GRADIENT_BENCHMARKS]
    assert {c.runner for c in cases(runners=LIBRARY_RUNNERS, precision="complex64")} == {
        "pennylane-lightning",
        "qiskit-aer",
    }
    assert cases(runners=LIBRARY_RUNNERS, device="gpu") == []
    assert supports("qulacs", "vqe-grad", "complex128", gradient=True)
    assert not supports("qulacs", "qft", "complex64", gradient=False)


def test_a_library_case_refuses_the_gpu() -> None:
    with pytest.raises(RuntimeError, match="CPU only"):
        prepare(Case("qft", "qulacs", 3, device="gpu"))


def test_pauli_terms() -> None:
    assert pauli_terms(ising(3)) == [
        (1.0, {0: "Z", 1: "Z"}),
        (1.0, {1: "Z", 2: "Z"}),
        (0.5, {0: "X"}),
        (0.5, {1: "X"}),
        (0.5, {2: "X"}),
    ]
    assert pauli_terms(2.0 * (X(0) @ Y(2))) == [(2.0, {0: "X", 2: "Y"})]
    with pytest.raises(UnsupportedGateError):
        pauli_terms(Hermitian(np.eye(2), [0]))


@pytest.mark.parametrize("runner", ["pennylane-default", "qiskit-aer", "qulacs"])
def test_an_unknown_gate_is_refused_not_mistranslated(runner: str) -> None:
    if not _installed(runner):
        pytest.skip(f"{PACKAGES[runner]} isn't installed")
    from benchmarks.circuits import Spec  # noqa: PLC0415
    from benchmarks.libraries import prepare_library  # noqa: PLC0415

    spec = Spec(Circuit(3, [ops.Toffoli(0, 1, 2)]), np.zeros(0))
    with pytest.raises(UnsupportedGateError):
        prepare_library(runner, "gate-h", spec, "complex128")()
