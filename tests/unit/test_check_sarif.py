"""The CodeQL gate fails on high-severity results only (QS-110, .github/scripts/check_sarif.py)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "check_sarif.py"


@pytest.fixture(scope="module")
def check_sarif() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_sarif", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sarif(*severities: str | None, in_extension: bool = True) -> dict[str, Any]:
    """A CodeQL-shaped report: rules live in a query-pack extension, results point at them."""
    rules = [
        {"id": f"py/rule-{k}", "properties": {} if s is None else {"security-severity": s}}
        for k, s in enumerate(severities)
    ]
    component = {"name": "codeql/python-queries", "rules": rules}
    tool = {"driver": {"name": "CodeQL"}, "extensions": [component]}
    if not in_extension:
        tool = {"driver": {"name": "CodeQL", "rules": rules}}
    results = [
        {
            "ruleId": f"py/rule-{k}",
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": "src/ketjet/x.py"},
                        "region": {"startLine": 10 + k},
                    }
                }
            ],
        }
        for k in range(len(severities))
    ]
    return {"runs": [{"tool": tool, "results": results}]}


def test_only_results_at_or_above_7_are_high(check_sarif: ModuleType) -> None:
    found = check_sarif.high_severity(_sarif("6.9", "7.0", "9.8", None))
    assert found == [
        "src/ketjet/x.py:11: py/rule-1 (severity 7.0)",
        "src/ketjet/x.py:12: py/rule-2 (severity 9.8)",
    ]


def test_rules_are_found_in_the_driver_too(check_sarif: ModuleType) -> None:
    assert len(check_sarif.high_severity(_sarif("8.1", in_extension=False))) == 1


def test_the_exit_code(check_sarif: ModuleType, tmp_path: Path) -> None:
    assert check_sarif.main(str(tmp_path)) == 1  # no reports: something went wrong upstream
    (tmp_path / "python.sarif").write_text(json.dumps(_sarif("5.0", None)))
    assert check_sarif.main(str(tmp_path)) == 0
    (tmp_path / "actions.sarif").write_text(json.dumps(_sarif("7.5")))
    assert check_sarif.main(str(tmp_path)) == 1
