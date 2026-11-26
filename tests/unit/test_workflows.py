"""The benchmark workflows do what ADR-0008 D7 and the GPU rule say (QS-1105).

They're read as YAML: a test can't run a scheduled workflow, but it can check its triggers,
permissions and threshold.
"""

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _load(name: str) -> dict[str, Any]:
    data: dict[Any, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # YAML 1.1 reads the key `on` as True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def test_the_nightly_runs_only_on_schedule_and_by_hand() -> None:
    nightly = _load("nightly-benchmarks.yml")
    assert set(nightly["on"]) == {"schedule", "workflow_dispatch"}


def test_the_nightly_uses_the_approved_threshold_and_reruns() -> None:
    """ADR-0008 D7: 10%, and a regression is rerun once before it counts."""
    nightly = _load("nightly-benchmarks.yml")
    assert nightly["env"]["FACTOR"] == "1.1"
    steps = nightly["jobs"]["regressions"]["steps"]
    compare = next(s for s in steps if s.get("name") == "Compare")["run"]
    assert "for attempt in 1 2" in compare
    assert '--factor "$FACTOR"' in compare


def test_the_nightly_can_write_only_what_it_needs() -> None:
    nightly = _load("nightly-benchmarks.yml")
    assert nightly["permissions"] == {"contents": "read"}
    assert nightly["jobs"]["regressions"]["permissions"] == {
        "contents": "write",
        "issues": "write",
    }


@pytest.mark.parametrize("name", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_no_workflow_runs_on_a_self_hosted_runner(name: str) -> None:
    """The GPU workstation is never reachable from GitHub Actions (ADR-0007 D5, amended)."""
    for job in _load(name)["jobs"].values():
        assert "self-hosted" not in str(job.get("runs-on", ""))


def test_every_pull_request_runs_the_asv_benchmarks_once() -> None:
    ci = _load("ci.yml")
    steps = ci["jobs"]["comparisons"]["steps"]
    assert any("asv run --python=same --quick" in s.get("run", "") for s in steps)
    assert "comparisons" in ci["jobs"]["ci-success"]["needs"]
