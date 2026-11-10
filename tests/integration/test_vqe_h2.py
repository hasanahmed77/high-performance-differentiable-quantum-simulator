"""The H₂ VQE example runs and reaches the exact ground-state energy (Sprint 3 demo).

``docs/examples/vqe-h2.md`` is executed block by block, and everything it prints must appear on
the page, so the documented output can't go stale.
"""

import contextlib
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PAGE = Path(__file__).parents[2] / "docs" / "examples" / "vqe-h2.md"
HARTREE_TO_MILLIHARTREE = 1000


@pytest.fixture(scope="module")
def page() -> tuple[str, dict[str, Any], list[str]]:
    """The page's text, the namespace after running its code, and the printed lines."""
    text = PAGE.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for block in re.findall(r"```python\n(.*?)```", text, re.S):
            exec(block, namespace)  # our own docs page
    return text, namespace, out.getvalue().splitlines()


def test_every_printed_line_is_documented(page: tuple[str, dict[str, Any], list[str]]) -> None:
    text, _, printed = page
    assert len(printed) == 8
    for line in printed:
        assert line.strip() in text, f"printed {line!r}, which the page doesn't show"


def test_vqe_reaches_the_fci_energy(page: tuple[str, dict[str, Any], list[str]]) -> None:
    _, ns, _ = page
    # FCI energy of H₂/STO-3G at 0.7414 Å from PennyLane's qchem (dense diagonalization).
    assert ns["exact"] == pytest.approx(-1.1372697289189455, abs=1e-12)
    assert ns["hartree_fock"] == pytest.approx(-1.1166821547327750, abs=1e-12)
    assert abs(ns["energy"] - ns["exact"]) < 1e-12
    assert ns["energy"] >= ns["exact"] - 1e-12  # variational bound
    assert all(np.diff(ns["energies"]) <= 1e-12)  # gradient descent never goes uphill here
    correlation = (ns["hartree_fock"] - ns["exact"]) * HARTREE_TO_MILLIHARTREE
    assert correlation == pytest.approx(20.587, abs=1e-3)


def test_ansatz_spans_the_two_configurations(page: tuple[str, dict[str, Any], list[str]]) -> None:
    _, ns, _ = page
    state = ns["ansatz"].bind({"theta": 1.0}).state()
    expected = np.zeros(16)
    expected[0b0011], expected[0b1100] = np.cos(0.5), np.sin(0.5)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_hamiltonian_matches_pennylane_qchem(page: tuple[str, dict[str, Any], list[str]]) -> None:
    """Regenerate the coefficients, so the pinned ones can't drift from the chemistry."""
    qml = pytest.importorskip("pennylane")
    _, ns, _ = page
    mol = qml.qchem.Molecule(
        ["H", "H"],
        np.array([0.0, 0.0, -0.70056, 0.0, 0.0, 0.70056]),  # bohr: 0.7414 Å bond
        basis_name="sto-3g",
        unit="bohr",
    )
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(mol, method="dhf")
    assert n_qubits == 4
    np.testing.assert_allclose(
        ns["H"].matrix(4), qml.matrix(hamiltonian, wire_order=range(4)), atol=1e-10
    )
