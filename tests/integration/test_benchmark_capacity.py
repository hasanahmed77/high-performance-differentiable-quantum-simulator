"""The capacity tool grows a circuit until it fails, and records why (QS-1108)."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jax")

import jax
from benchmarks import capacity


@pytest.fixture(autouse=True)
def _restore_x64() -> Any:
    x64 = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", x64)  # type: ignore[no-untyped-call]


@pytest.mark.parametrize("workload", capacity.WORKLOADS)
@pytest.mark.parametrize("precision", capacity.PRECISIONS)
def test_one_attempt_runs_and_reports_its_time(workload: str, precision: str) -> None:
    result = capacity.attempt(workload, precision, 3)
    assert result["ok"] is True
    assert result["call_s"] > 0


def test_the_search_stops_at_the_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(workload: str, precision: str, n: int, device: str) -> dict[str, Any]:
        if n < 6:
            return {"ok": True}
        return {"ok": False, "out_of_memory": True, "error": ["RESOURCE_EXHAUSTED"]}

    monkeypatch.setattr(capacity, "_in_subprocess", fake)
    largest, attempts = capacity.search("adjoint", "complex64", 3, 10, "cpu")
    assert largest == 5
    assert [a["n_qubits"] for a in attempts] == [3, 4, 5, 6]
    assert attempts[-1]["out_of_memory"] is True


def test_the_tool_writes_its_record(tmp_path: Path) -> None:
    out = tmp_path / "capacity.json"
    argv = ["--device", "cpu", "--workloads", "state", "--precisions", "complex64"]
    assert capacity.main([*argv, "--start", "3", "--stop", "4", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["machine"]["software"]["ketjet"]
    assert data["results"] == [
        {
            "workload": "state",
            "precision": "complex64",
            "largest": 4,
            "attempts": data["results"][0]["attempts"],
        }
    ]
    assert [a["ok"] for a in data["results"][0]["attempts"]] == [True, True]
