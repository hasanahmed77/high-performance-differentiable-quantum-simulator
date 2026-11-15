"""Fail if a CodeQL SARIF report has a high- or critical-severity security result (QS-110).

CodeQL's analyze step uploads results but never fails the job. GitHub's ``security-severity``
scale (CVSS) calls 7.0 and above high, and that's the gate here. Lower findings still appear in
the repository's code-scanning alerts.

Usage: python check_sarif.py <directory of .sarif files>
"""

import json
import sys
from pathlib import Path
from typing import Any

HIGH = 7.0


def _rules(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tool = run["tool"]
    components = [tool["driver"], *tool.get("extensions", [])]
    return {rule["id"]: rule for c in components for rule in c.get("rules", [])}


def high_severity(sarif: dict[str, Any]) -> list[str]:
    """Return one line per result whose rule has ``security-severity`` >= 7.0."""
    found = []
    for run in sarif.get("runs", []):
        rules = _rules(run)
        for result in run.get("results", []):
            rule = rules.get(result.get("ruleId", ""), {})
            severity = float(rule.get("properties", {}).get("security-severity", 0))
            if severity >= HIGH:
                location = result.get("locations", [{}])[0].get("physicalLocation", {})
                path = location.get("artifactLocation", {}).get("uri", "?")
                line = location.get("region", {}).get("startLine", "?")
                found.append(f"{path}:{line}: {result['ruleId']} (severity {severity})")
    return found


def main(directory: str) -> int:
    """Print every high-severity result in the directory's reports; return the exit code."""
    reports = sorted(Path(directory).glob("*.sarif"))
    if not reports:
        print(f"No SARIF reports in {directory}.")
        return 1
    found = [hit for report in reports for hit in high_severity(json.loads(report.read_text()))]
    for hit in found:
        print(f"::error::{hit}")
    print(f"{len(found)} high-severity result(s) in {len(reports)} report(s).")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
